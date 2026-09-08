import argparse, gzip, io, json, os, random, re, time, shutil
from collections import defaultdict
from pathlib import Path
import requests, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen3-4B"
DATA_BASE="https://huggingface.co/datasets/ArabicSpatialrReasoning/arabic-spatial-reasoning-v1/resolve/main"
SYSTEM_PROMPT="""أنت نموذج متخصص في الاستدلال المكاني باللغة العربية.
اعتمد فقط على السياق المعطى.
يجب أن تنتهي إجابتك دائمًا بالسطر:
FINAL: <الإجابة>
"""

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--output-root",default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--target-n",type=int,default=2000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--checkpoint-every",type=int,default=25)
    return p.parse_args()

def dl_train():
    tok=os.environ.get("HF_TOKEN")
    h={"Authorization":f"Bearer {tok}"} if tok else {}
    r=requests.get(f"{DATA_BASE}/train.jsonl.gz",headers=h,timeout=300)
    r.raise_for_status()
    rows=[]
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        for raw in gz:
            s=raw.decode("utf-8").strip()
            if s: rows.append(json.loads(s))
    return rows

def stratified(rows,n,seed):
    rng=random.Random(seed)
    buckets=defaultdict(list)
    for r in rows:
        buckets[str(r.get("task_type","unknown"))].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    keys=sorted(buckets)
    out=[]; i=0
    while len(out)<n and keys:
        k=keys[i%len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        keys=[kk for kk in keys if buckets[kk]]
        i+=1
    return out

def normalize_text(text):
    text=str(text or "").strip().lower()
    text=text.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي").replace("ة","ه")
    text=re.sub(r"[^\w\s\u0600-\u06FF.-]"," ",text)
    return re.sub(r"\s+"," ",text).strip()

def extract_final(text):
    m=re.findall(r"FINAL\s*:\s*(.+)",str(text or ""),flags=re.I)
    return m[-1].strip() if m else ""

def norm_dir(text):
    t=normalize_text(text)
    a={"شمال":"north","الشمال":"north","north":"north","جنوب":"south","الجنوب":"south","south":"south","شرق":"east","الشرق":"east","east":"east","غرب":"west","الغرب":"west","west":"west","شمال شرق":"northeast","شمال شرقي":"northeast","northeast":"northeast","شمال غرب":"northwest","شمال غربي":"northwest","northwest":"northwest","جنوب شرق":"southeast","جنوب شرقي":"southeast","southeast":"southeast","جنوب غرب":"southwest","جنوب غربي":"southwest","southwest":"southwest"}
    return a.get(t,t)

def yesno(text):
    t=normalize_text(text)
    if t in {"نعم","yes","ايوه","اي"}: return "yes"
    if t in {"لا","no","كلا"}: return "no"
    return t

def pint(text):
    tr=str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789")
    m=re.search(r"-?\d+",str(text or "").translate(tr))
    return int(m.group()) if m else None

def correct(output,answer,task):
    pred=extract_final(output)
    if not pred: return False
    if task=="cardinal_direction": return norm_dir(pred)==norm_dir(answer)
    if task=="within_radius_yes_no": return yesno(pred)==yesno(answer)
    if task=="count_within_radius": return pint(pred)==pint(answer)
    return normalize_text(pred)==normalize_text(answer)

def messages(r):
    return [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":f"السياق:\n{r['context']}\n\nالسؤال:\n{r['question']}\n\nأجب اعتمادًا على السياق فقط."}]

def truncate_prompt(tok,prompt,max_input):
    ids=tok(prompt,add_special_tokens=False)["input_ids"]
    if len(ids)<=max_input: return prompt
    h=max(1,int(max_input*0.35)); t=max_input-h
    return tok.decode(ids[:h]+ids[-t:],skip_special_tokens=True)

@torch.inference_mode()
def gen(model,tok,row,adapter,thinking,max_new):
    model.set_adapter(adapter)
    prompt=tok.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
    budget=8192 if thinking else 4096
    prompt=truncate_prompt(tok,prompt,max(512,budget-max_new))
    inp=tok(prompt,return_tensors="pt",add_special_tokens=False).to(model.device)
    st=time.perf_counter()
    out=model.generate(**inp,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id)
    sec=time.perf_counter()-st
    new=out[0,inp["input_ids"].shape[1]:]
    return tok.decode(new,skip_special_tokens=True),int(new.numel()),sec

def require_adapter(p):
    if not (p/"adapter_config.json").exists():
        raise FileNotFoundError(p)
    return p

def load_jsonl(path):
    rows=[]
    if not path.exists(): return rows
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def counts(rows):
    c={"short":0,"long":0,"exclude":0}
    for r in rows:
        t=r.get("router_target")
        if t in c: c[t]+=1
    return c

def write_summary(path,target_n,rows):
    c=counts(rows); usable=c["short"]+c["long"]
    s={"target_n":target_n,"completed":len(rows),**c,"usable":usable,"long_rate_among_usable":c["long"]/max(1,usable),"complete":len(rows)==target_n}
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding="utf-8")
    return s

def main():
    a=parse_args()
    root=Path(a.output_root)
    outdir=root/"empirical_router"
    outdir.mkdir(parents=True,exist_ok=True)
    target=outdir/f"empirical_train_{a.target_n}.jsonl"
    summary=outdir/f"empirical_train_{a.target_n}_summary.json"
    pilot=outdir/"empirical_train_500.jsonl"

    if not target.exists() and pilot.exists():
        shutil.copyfile(pilot,target)
        print("Seeded from existing empirical_train_500.jsonl",flush=True)

    existing=load_jsonl(target)
    processed={str(r.get("id")) for r in existing}
    sample=stratified(dl_train(),a.target_n,a.seed)
    target_ids={str(r.get("id")) for r in sample}
    if processed-target_ids:
        raise RuntimeError("Existing file contains IDs outside deterministic target sample.")

    remaining=[r for r in sample if str(r.get("id")) not in processed]
    c=counts(existing)
    print(f"Target={a.target_n} existing={len(existing)} remaining={len(remaining)} short={c['short']} long={c['long']} exclude={c['exclude']}",flush=True)

    if remaining:
        tok=AutoTokenizer.from_pretrained(MODEL_ID); tok.pad_token=tok.eos_token
        base=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map="auto")
        sp=require_adapter(root/"adapters"/"short_sft")
        lp=require_adapter(root/"adapters"/"long_sft")
        model=PeftModel.from_pretrained(base,str(sp),adapter_name="short",is_trainable=False)
        model.load_adapter(str(lp),adapter_name="long",is_trainable=False)
        model.eval()

        with target.open("a",encoding="utf-8") as f:
            start=len(existing)
            for j,r in enumerate(remaining,1):
                so,stok,slat=gen(model,tok,r,"short",False,256)
                lo,ltok,llat=gen(model,tok,r,"long",True,1024)
                sok=correct(so,r["answer"],r.get("task_type"))
                lok=correct(lo,r["answer"],r.get("task_type"))
                rt="short" if sok else ("long" if lok else "exclude")
                rec={"id":r.get("id"),"context":r.get("context"),"question":r.get("question"),"task_type":r.get("task_type"),"gold_difficulty":r.get("difficulty"),"short_correct":sok,"long_correct":lok,"short_tokens":stok,"long_tokens":ltok,"short_latency":round(slat,4),"long_latency":round(llat,4),"router_target":rt}
                f.write(json.dumps(rec,ensure_ascii=False)+"\n")
                done=start+j
                if done%a.checkpoint_every==0 or j==len(remaining):
                    f.flush(); os.fsync(f.fileno())
                    rows=load_jsonl(target)
                    s=write_summary(summary,a.target_n,rows)
                    print(f"{done}/{a.target_n} short={s['short']} long={s['long']} exclude={s['exclude']}",flush=True)

    rows=load_jsonl(target)
    s=write_summary(summary,a.target_n,rows)
    print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
