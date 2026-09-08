import argparse, json
from pathlib import Path
import numpy as np

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--output-root",default="/workspace/artifacts/asar_sft_grpo")
    return p.parse_args()

def pct_rank(values):
    order=np.argsort(values)
    ranks=np.empty_like(order,dtype=np.float32)
    ranks[order]=np.arange(len(values),dtype=np.float32)
    return ranks/max(1,len(values)-1)

def main():
    a=parse_args()
    root=Path(a.output_root)
    src=root/"empirical_router"/"empirical_train_4000.jsonl"
    out=root/"empirical_router"/"continuous_router"
    out.mkdir(parents=True,exist_ok=True)

    rows=[]
    with src.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line)
                if r.get("router_target") in {"short","long"}:
                    rows.append(r)

    st=np.array([float(r.get("short_tokens",0)) for r in rows],dtype=np.float32)
    sl=np.array([float(r.get("short_latency",0)) for r in rows],dtype=np.float32)
    tr=pct_rank(st); lr=pct_rank(sl)

    scored=[]
    for i,r in enumerate(rows):
        pressure=0.5*tr[i]+0.5*lr[i]
        score=(0.05+0.30*pressure) if r["router_target"]=="short" else (0.70+0.30*pressure)
        scored.append({
            "id":r.get("id"),
            "context":r.get("context"),
            "question":r.get("question"),
            "router_target":r.get("router_target"),
            "need_long_score":round(float(score),6),
        })

    with (out/"continuous_train_4000.jsonl").open("w",encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

    vals=np.array([r["need_long_score"] for r in scored],dtype=np.float32)
    summary={
        "usable_examples":len(scored),
        "short_examples":sum(r["router_target"]=="short" for r in scored),
        "long_examples":sum(r["router_target"]=="long" for r in scored),
        "score_min":float(vals.min()),
        "score_max":float(vals.max()),
        "score_mean":float(vals.mean()),
        "definition":{
            "short":"0.05 + 0.30 * short_resource_pressure",
            "long":"0.70 + 0.30 * short_resource_pressure",
            "pressure":"mean percentile of short_tokens and short_latency"
        }
    }
    (out/"continuous_score_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
