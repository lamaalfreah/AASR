import argparse, csv, gzip, io, json, os, re, time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

QWEN_ID = "Qwen/Qwen3-4B"
ROUTER_ID = "xlm-roberta-base"
DATA_BASE = "https://huggingface.co/datasets/ArabicSpatialrReasoning/arabic-spatial-reasoning-v1/resolve/main"

SYSTEM_PROMPT = """أنت نموذج متخصص في الاستدلال المكاني باللغة العربية.
اعتمد فقط على السياق المعطى.
يجب أن تنتهي إجابتك دائمًا بالسطر:
FINAL: <الإجابة>
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--split", choices=["validation","test"], default="validation")
    p.add_argument("--family", choices=["sft","grpo"], default="grpo")
    p.add_argument("--checkpoint-every", type=int, default=25)
    return p.parse_args()

def download_split(split):
    tok = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    r = requests.get(f"{DATA_BASE}/{split}.jsonl.gz", headers=headers, timeout=300)
    r.raise_for_status()
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        for raw in gz:
            s = raw.decode("utf-8").strip()
            if s:
                rows.append(json.loads(s))
    return rows

def normalize_text(text):
    text = str(text or "").strip().lower()
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي").replace("ة","ه")
    text = re.sub(r"[^\w\s\u0600-\u06FF.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_final(text):
    m = re.findall(r"FINAL\s*:\s*(.+)", str(text or ""), flags=re.I)
    return m[-1].strip() if m else ""

def norm_dir(text):
    t = normalize_text(text)
    a = {
        "شمال":"north","الشمال":"north","north":"north",
        "جنوب":"south","الجنوب":"south","south":"south",
        "شرق":"east","الشرق":"east","east":"east",
        "غرب":"west","الغرب":"west","west":"west",
        "شمال شرق":"northeast","شمال شرقي":"northeast","northeast":"northeast",
        "شمال غرب":"northwest","شمال غربي":"northwest","northwest":"northwest",
        "جنوب شرق":"southeast","جنوب شرقي":"southeast","southeast":"southeast",
        "جنوب غرب":"southwest","جنوب غربي":"southwest","southwest":"southwest",
    }
    return a.get(t, t)

def yesno(text):
    t = normalize_text(text)
    if t in {"نعم","yes","ايوه","اي"}: return "yes"
    if t in {"لا","no","كلا"}: return "no"
    return t

def parse_int(text):
    tr = str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789")
    m = re.search(r"-?\d+", str(text or "").translate(tr))
    return int(m.group()) if m else None

def is_correct(output, answer, task):
    pred = extract_final(output)
    if not pred: return False
    if task == "cardinal_direction":
        return norm_dir(pred) == norm_dir(answer)
    if task == "within_radius_yes_no":
        return yesno(pred) == yesno(answer)
    if task == "count_within_radius":
        return parse_int(pred) == parse_int(answer)
    return normalize_text(pred) == normalize_text(answer)

def messages(row):
    return [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"السياق:\n{row['context']}\n\nالسؤال:\n{row['question']}\n\nأجب اعتمادًا على السياق فقط."},
    ]

def truncate_prompt(tok, prompt, max_input):
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_input:
        return prompt
    h = max(1, int(max_input * 0.35))
    t = max_input - h
    return tok.decode(ids[:h] + ids[-t:], skip_special_tokens=True)

@torch.inference_mode()
def generate_one(model, tok, row, adapter, thinking, max_new):
    model.set_adapter(adapter)
    prompt = tok.apply_chat_template(
        messages(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    total_budget = 8192 if thinking else 4096
    prompt = truncate_prompt(tok, prompt, max(512, total_budget - max_new))
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    st = time.perf_counter()
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    sec = time.perf_counter() - st
    new = out[0, inp["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(new.numel()), sec

class ContinuousRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ROUTER_ID)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(h,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,1),
        )
    def forward(self,input_ids,attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.head(out.last_hidden_state[:,0])).squeeze(-1)

def router_text(row):
    c = str(row.get("context") or "")
    q = str(row.get("question") or "")
    return f"السياق:\n{c[:3500]}\n...\n{c[-3500:]}\n\nالسؤال:\n{q}"

def load_router(root, device):
    ckpt = torch.load(
        root/"empirical_router"/"continuous_router"/"model"/"continuous_router.pt",
        map_location="cpu",
    )
    tok = AutoTokenizer.from_pretrained(
        root/"empirical_router"/"continuous_router"/"model"/"tokenizer"
    )
    model = ContinuousRouter()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, tok, float(ckpt["threshold"])

@torch.inference_mode()
def route_score(model, tok, row, device):
    enc = tok(
        router_text(row),
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k:v.to(device) for k,v in enc.items()}
    return float(model(**enc).item())

def require_adapter(path):
    if not (path/"adapter_config.json").exists():
        raise FileNotFoundError(path)
    return path

def load_existing(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def summarize(records):
    variants = {
        "always_short": [],
        "always_long": [],
        "oracle": [],
        "learned_router": [],
    }
    for r in records:
        short = {
            "correct": r["short_correct"],
            "tokens": r["short_tokens"],
            "latency": r["short_latency"],
        }
        long = {
            "correct": r["long_correct"],
            "tokens": r["long_tokens"],
            "latency": r["long_latency"],
        }
        variants["always_short"].append(short)
        variants["always_long"].append(long)

        oracle_choice = long if r["gold_difficulty"] == "hard" else short
        variants["oracle"].append(oracle_choice)

        learned_choice = long if r["learned_route"] == "long" else short
        variants["learned_router"].append(learned_choice)

    out = []
    for name, vals in variants.items():
        n = len(vals)
        out.append({
            "variant": name,
            "n": n,
            "accuracy": sum(int(v["correct"]) for v in vals)/max(1,n),
            "avg_tokens": sum(v["tokens"] for v in vals)/max(1,n),
            "avg_latency_sec": sum(v["latency"] for v in vals)/max(1,n),
        })
    return out

def main():
    a = parse_args()
    root = Path(a.output_root)
    outdir = root/"final_evaluation"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = download_split(a.split)
    rows = [
        r for r in rows
        if str(r.get("difficulty","")).strip().lower() in {"easy","hard"}
    ]
    print(f"Eligible {a.split} Easy+Hard rows: {len(rows)}", flush=True)

    outpath = outdir/f"{a.family}_{a.split}_easy_hard_outputs.jsonl"
    summary_path = outdir/f"{a.family}_{a.split}_easy_hard_summary.csv"

    existing = load_existing(outpath)
    done_ids = {str(r["id"]) for r in existing}
    remaining = [r for r in rows if str(r.get("id")) not in done_ids]

    print(f"Resume: existing={len(existing)} remaining={len(remaining)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router, router_tok, threshold = load_router(root, device)
    print(f"Continuous router threshold={threshold}", flush=True)

    qtok = AutoTokenizer.from_pretrained(QWEN_ID)
    qtok.pad_token = qtok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if a.family == "sft":
        sp = require_adapter(root/"adapters"/"short_sft")
        lp = require_adapter(root/"adapters"/"long_sft")
    else:
        sp = require_adapter(root/"adapters"/"short_grpo_refined")
        lp = require_adapter(root/"adapters"/"long_grpo_refined")

    model = PeftModel.from_pretrained(base, str(sp), adapter_name="short", is_trainable=False)
    model.load_adapter(str(lp), adapter_name="long", is_trainable=False)
    model.eval()

    with outpath.open("a",encoding="utf-8") as f:
        start_n = len(existing)
        for j, row in enumerate(remaining, 1):
            score = route_score(router, router_tok, row, device)
            learned_route = "long" if score >= threshold else "short"

            s_out, s_tok, s_lat = generate_one(model, qtok, row, "short", False, 256)
            l_out, l_tok, l_lat = generate_one(model, qtok, row, "long", True, 1024)

            rec = {
                "id": str(row.get("id")),
                "task_type": row.get("task_type"),
                "gold_difficulty": str(row.get("difficulty","")).lower(),
                "router_score": score,
                "router_threshold": threshold,
                "learned_route": learned_route,
                "short_correct": is_correct(s_out,row["answer"],row.get("task_type")),
                "long_correct": is_correct(l_out,row["answer"],row.get("task_type")),
                "short_tokens": s_tok,
                "long_tokens": l_tok,
                "short_latency": round(s_lat,4),
                "long_latency": round(l_lat,4),
                "short_prediction": extract_final(s_out),
                "long_prediction": extract_final(l_out),
                "gold": row.get("answer"),
            }
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")

            done = start_n + j
            if done % a.checkpoint_every == 0 or j == len(remaining):
                f.flush()
                os.fsync(f.fileno())
                records_now = load_existing(outpath)
                summary = summarize(records_now)
                with summary_path.open("w",newline="",encoding="utf-8") as sf:
                    w = csv.DictWriter(sf,fieldnames=list(summary[0].keys()))
                    w.writeheader()
                    w.writerows(summary)
                print(f"{done}/{len(rows)} completed",flush=True)

    final_records = load_existing(outpath)
    summary = summarize(final_records)

    with summary_path.open("w",newline="",encoding="utf-8") as sf:
        w = csv.DictWriter(sf,fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n=== FINAL SUMMARY ===")
    for r in summary:
        print(json.dumps(r,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
