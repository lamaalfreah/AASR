import argparse, csv, gzip, io, json, os, random, re, time
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

SPATIAL_CUES = {
    "اقرب", "أقرب", "مسافة", "المسافة", "يبعد", "تبعد", "ضمن", "كم", "متر",
    "شمال", "جنوب", "شرق", "غرب", "اتجاه", "احداثيات", "إحداثيات", "خط العرض", "خط الطول",
    "مستشفى", "عيادة", "صيدلية", "مدرسة", "جامعة", "كلية", "مطعم", "مقهى", "بنك", "وقود",
    "hospital", "clinic", "pharmacy", "school", "university", "college", "restaurant", "cafe", "bank", "fuel",
    "latitude", "longitude", "distance", "poi", "wgs84",
}

NAME_TASKS = {
    "nearest_category", "closer_of_two", "nearest_of_two_categories",
    "two_hop_nearest", "spatial_multi_constraint",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--scope", choices=["all", "easy-hard"], default="all",
                   help="all = full split including medium; easy-hard = legacy specialization-only evaluation")
    p.add_argument("--family", choices=["sft", "grpo"], default="sft")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=120,
                   help="0 means all eligible rows; default is a fast stratified pilot.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--beta", type=float, default=0.2)
    return p.parse_args()


def parse_alphas(s):
    vals = sorted({round(float(x.strip()), 4) for x in s.split(",") if x.strip()})
    if 0.0 not in vals or 1.0 not in vals:
        raise ValueError("alphas must include 0 and 1")
    return vals


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


def stratified_sample(rows, n, seed):
    if not n or n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    buckets = {}
    for r in rows:
        buckets.setdefault(str(r.get("task_type", "unknown")), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    out, i = [], 0
    while len(out) < n and keys:
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        keys = [x for x in keys if buckets[x]]
        i += 1
    return out


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
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
    if t in {"نعم", "yes", "ايوه", "اي"}: return "yes"
    if t in {"لا", "no", "كلا"}: return "no"
    return t


def parse_int(text):
    tr = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    m = re.search(r"-?\d+", str(text or "").translate(tr))
    return int(m.group()) if m else None


def correctness(output, answer, task):
    pred = extract_final(output)
    if not pred:
        return False, False
    if task == "cardinal_direction":
        v = norm_dir(pred) == norm_dir(answer)
        return v, v
    if task == "within_radius_yes_no":
        v = yesno(pred) == yesno(answer)
        return v, v
    if task == "count_within_radius":
        v = parse_int(pred) == parse_int(answer)
        return v, v

    p, g = normalize_text(pred), normalize_text(answer)
    strict = p == g
    # Secondary metric only: catches harmless extra words/partial canonical names.
    # Cross-language aliases still require manual review and are NOT silently marked correct.
    lenient = strict
    if task in NAME_TASKS and min(len(p), len(g)) >= 4:
        lenient = strict or p in g or g in p
    return strict, lenient


def split_units(text):
    units = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Break very long Wikipedia-style paragraphs into sentence-like units.
        parts = re.split(r"(?<=[.!؟])\s+|\s*[•●▪]\s*", line)
        units.extend([p.strip() for p in parts if p.strip()])
    return units


def select_relevant_context(tok, context, question, max_tokens):
    ids = tok(str(context or ""), add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return str(context or "")

    qn = normalize_text(question)
    q_words = {w for w in re.findall(r"\w+", qn) if len(w) > 2}
    scored = []
    for idx, unit in enumerate(split_units(context)):
        un = normalize_text(unit)
        u_words = set(re.findall(r"\w+", un))
        score = 3 * len(q_words & u_words)
        score += 2 * sum(1 for cue in SPATIAL_CUES if normalize_text(cue) in un)
        if re.search(r"\d", unit): score += 2
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:كم|km|متر|m)\b", un): score += 3
        # Preserve likely structured evidence / coordinates.
        if any(x in un for x in ["wgs84", "latitude", "longitude", "خط العرض", "خط الطول"]): score += 4
        scored.append((score, idx, unit))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    chosen, used = [], 0
    # Always keep a small amount of framing from the beginning.
    units = split_units(context)
    for idx, unit in enumerate(units[:3]):
        uids = tok(unit, add_special_tokens=False)["input_ids"]
        if used + len(uids) <= max_tokens:
            chosen.append((idx, unit)); used += len(uids)

    seen = {idx for idx, _ in chosen}
    for score, idx, unit in scored:
        if idx in seen or score <= 0:
            continue
        uids = tok(unit, add_special_tokens=False)["input_ids"]
        if used + len(uids) > max_tokens:
            continue
        chosen.append((idx, unit)); seen.add(idx); used += len(uids)
        if used >= int(max_tokens * 0.95):
            break

    # Fallback if scoring was too sparse: use head+tail context tokens, but only on raw context.
    if used < max(128, int(max_tokens * 0.25)):
        h = max_tokens // 3
        t = max_tokens - h
        return tok.decode(ids[:h] + ids[-t:], skip_special_tokens=True)

    chosen.sort(key=lambda x: x[0])
    return "\n".join(unit for _, unit in chosen)


def build_messages(row, selected_context):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"السياق:\n{selected_context}\n\nالسؤال:\n{row['question']}\n\nأجب اعتمادًا على السياق فقط."},
    ]


class ContinuousRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ROUTER_ID)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.head(out.last_hidden_state[:, 0])).squeeze(-1)


def router_text(row):
    # Keep this identical to router training to avoid train/inference mismatch.
    c = str(row.get("context") or "")
    q = str(row.get("question") or "")
    return f"السياق:\n{c[:3500]}\n...\n{c[-3500:]}\n\nالسؤال:\n{q}"


def load_router(root, device):
    model_dir = root / "empirical_router" / "continuous_router" / "model"
    ckpt = torch.load(model_dir / "continuous_router.pt", map_location="cpu")
    tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = ContinuousRouter()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, tok


@torch.inference_mode()
def route_score(model, tok, row, device):
    enc = tok(router_text(row), truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    return float(model(**enc).item())


def nearest_alpha(score, alphas):
    return min(alphas, key=lambda a: abs(float(a) - float(score)))


def adapter_name(alpha):
    if abs(alpha - 0.0) < 1e-8: return "short"
    if abs(alpha - 1.0) < 1e-8: return "long"
    return f"mix_{str(alpha).replace('.', '_')}"


def create_mixed_adapters(model, alphas):
    for a in alphas:
        if a in {0.0, 1.0}:
            continue
        name = adapter_name(a)
        model.add_weighted_adapter(
            adapters=["short", "long"],
            weights=[1.0 - a, a],
            adapter_name=name,
            combination_type="linear",
        )
        print(f"Created {name}: short={1-a:.2f}, long={a:.2f}", flush=True)


@torch.inference_mode()
def generate_alpha(model, tok, row, alpha):
    model.set_adapter(adapter_name(alpha))
    thinking = bool(alpha >= 0.5)
    total_budget = int(round(4096 + alpha * (8192 - 4096)))
    max_new = int(round(256 + alpha * (1024 - 256)))
    # Reserve prompt/template headroom; truncate CONTEXT before chat-template rendering.
    context_budget = max(768, total_budget - max_new - 384)
    selected_context = select_relevant_context(tok, row.get("context", ""), row.get("question", ""), context_budget)
    prompt = tok.apply_chat_template(
        build_messages(row, selected_context),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
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


def require_adapter(path):
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(path)
    return path


def load_existing(path):
    rows = []
    if not path.exists(): return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def candidate_utility(cand, beta):
    # Token-normalization is added per record after all alpha candidates are generated.
    return float(cand["strict_correct"]) - beta * cand["token_ratio"]


def summarize(records, alphas):
    variants = {f"alpha_{a}": [] for a in alphas}
    variants.update({"fixed_alpha_0.5": [], "aasr_router": [], "oracle_alpha": []})
    alpha_counts = {str(a): 0 for a in alphas}

    for r in records:
        cands = r["candidates"]
        for a in alphas:
            variants[f"alpha_{a}"].append(cands[str(a)])
        if "0.5" in cands:
            variants["fixed_alpha_0.5"].append(cands["0.5"])
        ra = str(r["routed_alpha"])
        variants["aasr_router"].append(cands[ra])
        oa = str(r["oracle_alpha"])
        variants["oracle_alpha"].append(cands[oa])
        alpha_counts[ra] = alpha_counts.get(ra, 0) + 1

    out = []
    for name, vals in variants.items():
        if not vals: continue
        n = len(vals)
        out.append({
            "variant": name,
            "n": n,
            "strict_accuracy": sum(int(v["strict_correct"]) for v in vals) / n,
            "lenient_accuracy": sum(int(v["lenient_correct"]) for v in vals) / n,
            "avg_tokens": sum(v["tokens"] for v in vals) / n,
            "avg_latency_sec": sum(v["latency"] for v in vals) / n,
        })
    return out, alpha_counts


def main():
    a = parse_args()
    alphas = parse_alphas(a.alphas)
    root = Path(a.output_root)
    outdir = root / "final_evaluation_adamix_fast"
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = download_split(a.split)
    print(f"Downloaded full {a.split} split: {len(all_rows)} rows", flush=True)
    if a.scope == "easy-hard":
        eligible = [r for r in all_rows if str(r.get("difficulty", "")).strip().lower() in {"easy", "hard"}]
    else:
        eligible = all_rows

    diff_counts = {}
    task_counts = {}
    for r in eligible:
        d = str(r.get("difficulty", "unknown")).strip().lower() or "unknown"
        t = str(r.get("task_type", "unknown"))
        diff_counts[d] = diff_counts.get(d, 0) + 1
        task_counts[t] = task_counts.get(t, 0) + 1
    print(f"Evaluation scope={a.scope}; eligible rows={len(eligible)}", flush=True)
    print("Difficulty counts:", json.dumps(diff_counts, ensure_ascii=False), flush=True)
    print("Task counts:", json.dumps(task_counts, ensure_ascii=False), flush=True)

    rows = stratified_sample(eligible, a.max_samples, a.seed)
    mode = "FULL" if (a.max_samples <= 0 or a.max_samples >= len(eligible)) else "STRATIFIED SUBSET"
    print(f"Evaluation mode={mode}; evaluating {len(rows)} / {len(eligible)} rows", flush=True)

    tag = f"{a.family}_{a.split}_{a.scope}_n{len(rows)}_seed{a.seed}"
    outpath = outdir / f"{tag}_outputs.jsonl"
    summary_path = outdir / f"{tag}_summary.csv"
    alpha_path = outdir / f"{tag}_router_alpha_distribution.json"

    existing = load_existing(outpath)
    done_ids = {str(r["id"]) for r in existing}
    remaining = [r for r in rows if str(r.get("id")) not in done_ids]
    print(f"Resume: existing={len(existing)} remaining={len(remaining)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router, router_tok = load_router(root, device)

    qtok = AutoTokenizer.from_pretrained(QWEN_ID)
    qtok.pad_token = qtok.eos_token
    base = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")

    if a.family == "sft":
        sp = require_adapter(root / "adapters" / "short_sft")
        lp = require_adapter(root / "adapters" / "long_sft")
    else:
        sp = require_adapter(root / "adapters" / "short_grpo_refined")
        lp = require_adapter(root / "adapters" / "long_grpo_refined")

    model = PeftModel.from_pretrained(base, str(sp), adapter_name="short", is_trainable=False)
    model.load_adapter(str(lp), adapter_name="long", is_trainable=False)
    create_mixed_adapters(model, alphas)
    model.eval()

    with outpath.open("a", encoding="utf-8") as f:
        start_n = len(existing)
        for j, row in enumerate(remaining, 1):
            score = route_score(router, router_tok, row, device)
            routed_alpha = nearest_alpha(score, alphas)

            candidates = {}
            max_tok = 1
            for alpha in alphas:
                output, ntok, lat = generate_alpha(model, qtok, row, alpha)
                strict, lenient = correctness(output, row.get("answer"), row.get("task_type"))
                candidates[str(alpha)] = {
                    "alpha": alpha,
                    "strict_correct": bool(strict),
                    "lenient_correct": bool(lenient),
                    "tokens": ntok,
                    "latency": round(lat, 4),
                    "prediction": extract_final(output),
                }
                max_tok = max(max_tok, ntok)

            for cand in candidates.values():
                cand["token_ratio"] = cand["tokens"] / max_tok
                cand["utility"] = round(candidate_utility(cand, a.beta), 6)

            oracle = max(candidates.values(), key=lambda c: (c["utility"], c["strict_correct"], -c["tokens"]))
            rec = {
                "id": str(row.get("id")),
                "task_type": row.get("task_type"),
                "gold_difficulty": str(row.get("difficulty", "")).lower(),
                "router_score": round(score, 6),
                "routed_alpha": routed_alpha,
                "oracle_alpha": oracle["alpha"],
                "gold": row.get("answer"),
                "candidates": candidates,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            done = start_n + j
            if done % a.checkpoint_every == 0 or j == len(remaining):
                f.flush(); os.fsync(f.fileno())
                records_now = load_existing(outpath)
                summary, alpha_counts = summarize(records_now, alphas)
                with summary_path.open("w", newline="", encoding="utf-8") as sf:
                    w = csv.DictWriter(sf, fieldnames=list(summary[0].keys()))
                    w.writeheader(); w.writerows(summary)
                alpha_path.write_text(json.dumps(alpha_counts, indent=2), encoding="utf-8")
                print(f"{done}/{len(rows)} completed", flush=True)

    final_records = load_existing(outpath)
    summary, alpha_counts = summarize(final_records, alphas)
    with summary_path.open("w", newline="", encoding="utf-8") as sf:
        w = csv.DictWriter(sf, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    alpha_path.write_text(json.dumps(alpha_counts, indent=2), encoding="utf-8")

    print("\n=== FINAL SUMMARY ===")
    for r in summary:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    fixed_rows = [r for r in summary if r["variant"].startswith("alpha_")]
    if fixed_rows:
        best_fixed = max(fixed_rows, key=lambda r: (r["strict_accuracy"], -r["avg_tokens"]))
        print("\n=== BEST FIXED ALPHA ON THIS SPLIT ===")
        print(json.dumps(best_fixed, ensure_ascii=False), flush=True)

    print("\n=== ROUTER ALPHA DISTRIBUTION ===")
    print(json.dumps(alpha_counts, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
