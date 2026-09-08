import argparse, gzip, io, json, os, random, re
from pathlib import Path

import requests, torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen3-4B"
DATA_URL = "https://huggingface.co/datasets/ArabicSpatialrReasoning/arabic-spatial-reasoning-v1/resolve/main/train.jsonl.gz"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "v_proj"]
SHORT_MAX_LENGTH, LONG_MAX_LENGTH = 4096, 8192
GRPO_TOTAL_SAMPLES, GRPO_GENERATIONS = 20, 4
SYSTEM_PROMPT = '''أنت نموذج متخصص في الاستدلال المكاني باللغة العربية.
اعتمد فقط على السياق المعطى.
يجب أن تنتهي إجابتك دائمًا بالسطر:
FINAL: <الإجابة>
'''

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare-sft","train-short-sft","train-long-sft","prepare-grpo","refine-short-grpo","refine-long-grpo"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--grpo-total-samples", type=int, default=GRPO_TOTAL_SAMPLES)
    p.add_argument("--grpo-generations", type=int, default=GRPO_GENERATIONS)
    p.add_argument("--short-max-length", type=int, default=SHORT_MAX_LENGTH)
    p.add_argument("--long-max-length", type=int, default=LONG_MAX_LENGTH)
    p.add_argument("--force", action="store_true")
    p.add_argument("--speed-mode", choices=["safe","fast"], default="fast")
    return p.parse_args()

def load_private_rows():
    tok = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    r = requests.get(DATA_URL, headers=headers, timeout=300)
    r.raise_for_status()
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        for raw in gz:
            s = raw.decode("utf-8").strip()
            if s:
                rows.append(json.loads(s))
    return rows

def split_rows(rows):
    return {k:[x for x in rows if x.get("split")==k] for k in ["train","validation","test"]}

def diff(x):
    return str(x.get("difficulty","")).strip().lower()

def save_jsonl(rows, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False)+"\n")

def read_jsonl(path):
    out=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def make_lora():
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")

def messages(row):
    return [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"السياق:\n{row['context']}\n\nالسؤال:\n{row['question']}\n\nأجب اعتمادًا على السياق فقط."}]

def render_prompt(tok, row, thinking=False):
    return tok.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

def head_tail(tok, text, max_tokens):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens: return text
    h = max(1, int(max_tokens*0.35)); t = max_tokens-h
    return tok.decode(ids[:h]+ids[-t:], skip_special_tokens=True)


def prepare_sft(args):
    out = Path(args.output_root) / "data"
    out.mkdir(parents=True, exist_ok=True)
    sf, lf = out / "short_sft.jsonl", out / "long_sft.jsonl"
    if sf.exists() and lf.exists() and not args.force:
        print("SFT data already exists; skipping.")
        return
    train = split_rows(load_private_rows())["train"]
    easy = [x for x in train if diff(x) == "easy"]
    medium = [x for x in train if diff(x) == "medium"]
    hard = [x for x in train if diff(x) == "hard"]
    keep = lambda x: {k: x.get(k) for k in [
        "id","context","question","answer","verified_rationale",
        "task_type","difficulty","difficulty_score"
    ]}
    save_jsonl([keep(x) for x in easy], sf)
    save_jsonl([keep(x) for x in hard], lf)
    summary = {
        "train_total": len(train),
        "easy_used_for_short": len(easy),
        "hard_used_for_long": len(hard),
        "medium_excluded_from_specialization": len(medium),
        "total_used_for_adapter_training": len(easy)+len(hard),
        "rule": "Easy -> Short; Hard -> Long; Medium excluded from primary specialization."
    }
    (out / "sft_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def make_sft_dataset(rows, tok, typ, max_length):
    reserve=256 if typ=="short" else 2048
    max_prompt=max(512,max_length-reserve)
    data=[]
    for r in rows:
        p=head_tail(tok, render_prompt(tok,r,thinking=(typ=="long")), max_prompt)
        if typ=="short":
            c=f"FINAL: {r['answer']}"
        else:
            rat=str(r.get("verified_rationale") or "").strip()
            c=(rat+"\n" if rat else "")+f"FINAL: {r['answer']}"
        data.append({"prompt":p,"completion":c})
    return Dataset.from_list(data)


def find_last_checkpoint(output_dir):
    output_dir = Path(output_dir)
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
            checkpoints.append((step, p))
        except ValueError:
            pass
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda x: x[0])[1])


def train_sft(args, typ):
    set_seed(args.seed)
    root = Path(args.output_root)
    dataf = root / "data" / f"{typ}_sft.jsonl"
    out = root / "adapters" / f"{typ}_sft"
    if not dataf.exists():
        raise FileNotFoundError(f"{dataf} not found. Run prepare-sft first.")
    if (out / "COMPLETE").exists() and (out / "adapter_config.json").exists() and not args.force:
        print("Final adapter exists; skipping:", out)
        return
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    maxlen = args.short_max_length if typ == "short" else args.long_max_length
    ds = make_sft_dataset(read_jsonl(dataf), tok, typ, maxlen)
    if args.speed_mode == "fast":
        per_device_bs, grad_accum, workers = 2, 4, 4
    else:
        per_device_bs, grad_accum, workers = 1, 8, 2
    cfg = SFTConfig(
        output_dir=str(out), learning_rate=1e-4, num_train_epochs=1,
        per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=grad_accum,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=maxlen, completion_only_loss=True, packing=False,
        logging_steps=10, save_strategy="steps", save_steps=25, save_total_limit=2,
        report_to="none", seed=args.seed, dataset_num_proc=4,
        dataloader_num_workers=workers, dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )
    tr = SFTTrainer(model=MODEL_ID, args=cfg, train_dataset=ds, processing_class=tok, peft_config=make_lora())
    last_checkpoint = find_last_checkpoint(out)
    if last_checkpoint:
        print(f"Resuming SFT from checkpoint: {last_checkpoint}", flush=True)
    else:
        print("No SFT checkpoint found; starting from step 0.", flush=True)
    print(f"mode={args.speed_mode} batch={per_device_bs} grad_accum={grad_accum} examples={len(ds)} max_length={maxlen}", flush=True)
    try:
        tr.train(resume_from_checkpoint=last_checkpoint)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM. Checkpoint preserved. Re-run with --speed-mode safe.", flush=True)
        raise
    tr.save_model(str(out)); tok.save_pretrained(str(out))
    (out / "COMPLETE").write_text("ok\n", encoding="utf-8")
    info={"type":typ,"method":"SFT+LoRA","examples":len(ds),"speed_mode":args.speed_mode,
          "batch":per_device_bs,"grad_accum":grad_accum,"checkpoint_every_steps":25,
          "r":16,"alpha":32,"dropout":0.05,"lr":1e-4,"epochs":1,"max_length":maxlen}
    (out / "run_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False,indent=2))

def stratified_pick(rows,n,seed):
    rng=random.Random(seed); buckets={}
    for x in rows: buckets.setdefault(str(x.get("task_type","unknown")),[]).append(x)
    for b in buckets.values(): rng.shuffle(b)
    keys=sorted(buckets); out=[]; i=0
    while len(out)<n and keys:
        k=keys[i%len(keys)]
        if buckets[k]: out.append(buckets[k].pop())
        keys=[k for k in keys if buckets[k]]
        i+=1
    return out

def prepare_grpo(args):
    root=Path(args.output_root)/"data"; root.mkdir(parents=True,exist_ok=True)
    f=root/"grpo20.jsonl"
    if f.exists() and not args.force:
        print("GRPO sample exists; skipping."); return
    train=split_rows(load_private_rows())["train"]
    easy=[x for x in train if diff(x)=="easy"]; hard=[x for x in train if diff(x)=="hard"]
    ne=args.grpo_total_samples//2; nh=args.grpo_total_samples-ne
    picks=stratified_pick(easy,ne,args.seed)+stratified_pick(hard,nh,args.seed+1)
    keep=lambda x:{k:x.get(k) for k in ["id","context","question","answer","task_type","difficulty"]}
    save_jsonl([keep(x) for x in picks],f)
    print(json.dumps({"total":len(picks),"easy":ne,"hard":nh,"sampling":"stratified by task_type"},ensure_ascii=False,indent=2))

def norm(t):
    t=str(t or "").strip().lower().replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي").replace("ة","ه")
    t=re.sub(r"[^\w\s\u0600-\u06FF.-]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

def extract_final(t):
    m=re.findall(r"FINAL\s*:\s*(.+)",str(t or ""),flags=re.I)
    return m[-1].strip() if m else ""

def norm_dir(t):
    x=norm(t); a={"شمال":"north","الشمال":"north","north":"north","جنوب":"south","الجنوب":"south","south":"south","شرق":"east","الشرق":"east","east":"east","غرب":"west","الغرب":"west","west":"west","شمال شرق":"northeast","شمال شرقي":"northeast","northeast":"northeast","شمال غرب":"northwest","شمال غربي":"northwest","northwest":"northwest","جنوب شرق":"southeast","جنوب شرقي":"southeast","southeast":"southeast","جنوب غرب":"southwest","جنوب غربي":"southwest","southwest":"southwest"}
    return a.get(x,x)

def yesno(t):
    x=norm(t)
    if x in {"نعم","yes","ايوه","اي"}: return "yes"
    if x in {"لا","no","كلا"}: return "no"
    return x

def pint(t):
    tr=str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789"); m=re.search(r"-?\d+",str(t or "").translate(tr))
    return int(m.group()) if m else None

def correct(pred,ans,task):
    p=extract_final(pred)
    if not p:return False
    if task=="cardinal_direction":return norm_dir(p)==norm_dir(ans)
    if task=="within_radius_yes_no":return yesno(p)==yesno(ans)
    if task=="count_within_radius":return pint(p)==pint(ans)
    return norm(p)==norm(ans)

def ctext(x):
    if isinstance(x,str):return x
    if isinstance(x,list) and x and isinstance(x[0],dict):return x[0].get("content","")
    if isinstance(x,dict):return x.get("content","")
    return str(x)

def accuracy_reward(completions,answer,task_type,**kwargs):
    return [1.0 if correct(ctext(c),a,t) else 0.0 for c,a,t in zip(completions,answer,task_type)]

def format_reward(completions,**kwargs):
    return [0.1 if extract_final(ctext(c)) else 0.0 for c in completions]

def efficiency_reward(completions,**kwargs):
    out=[]
    for c in completions:
        words=max(1,len(ctext(c).split()))
        out.append(max(0.0,1.0-words/120.0)*0.1)
    return out

def make_grpo_dataset(rows,tok,typ):
    data=[]
    for r in rows:
        p=render_prompt(tok,r,thinking=(typ=="long"))
        p=head_tail(tok,p,3584 if typ=="short" else 6144)
        data.append({"prompt":p,"answer":r["answer"],"task_type":r.get("task_type")})
    return Dataset.from_list(data)

def refine(args,typ):
    set_seed(args.seed); root=Path(args.output_root)
    sft=root/"adapters"/f"{typ}_sft"; refined=root/"adapters"/f"{typ}_grpo_refined"
    if (refined/"adapter_config.json").exists() and not args.force:
        print("Refined adapter exists; skipping",refined); return
    rows=read_jsonl(root/"data"/"grpo20.jsonl")
    want="easy" if typ=="short" else "hard"; rows=[x for x in rows if diff(x)==want]
    tok=AutoTokenizer.from_pretrained(MODEL_ID); tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map="auto")
    model=PeftModel.from_pretrained(base,str(sft),is_trainable=True)
    ds=make_grpo_dataset(rows,tok,typ); g=args.grpo_generations
    cfg=GRPOConfig(output_dir=str(refined),learning_rate=1e-5,num_train_epochs=1,
                   per_device_train_batch_size=1,gradient_accumulation_steps=g,num_generations=g,
                   max_completion_length=512 if typ=="short" else 1536,bf16=True,tf32=True,
                   gradient_checkpointing=True,beta=0.0,logging_steps=1,save_strategy="steps",save_steps=2,
                   save_total_limit=2,report_to="none",seed=args.seed,remove_unused_columns=False)
    rewards=[accuracy_reward,format_reward]+([efficiency_reward] if typ=="short" else [])
    tr=GRPOTrainer(model=model,args=cfg,train_dataset=ds,processing_class=tok,reward_funcs=rewards)
    last_checkpoint = find_last_checkpoint(refined)
    if last_checkpoint:
        print(f"Resuming GRPO from checkpoint: {last_checkpoint}", flush=True)
    else:
        print("No GRPO checkpoint found; starting from step 0.", flush=True)
    tr.train(resume_from_checkpoint=last_checkpoint)
    tr.save_model(str(refined))
    tok.save_pretrained(str(refined))
    print(json.dumps({"type":typ,"method":"small GRPO refinement","examples":len(ds),"generations":g,"lr":1e-5,"epochs":1},ensure_ascii=False,indent=2))

def main():
    a=parse_args()
    if a.stage=="prepare-sft":prepare_sft(a)
    elif a.stage=="train-short-sft":train_sft(a,"short")
    elif a.stage=="train-long-sft":train_sft(a,"long")
    elif a.stage=="prepare-grpo":prepare_grpo(a)
    elif a.stage=="refine-short-grpo":refine(a,"short")
    elif a.stage=="refine-long-grpo":refine(a,"long")

if __name__=="__main__": main()
