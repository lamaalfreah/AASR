import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_ID="xlm-roberta-base"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--output-root",default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--max-length",type=int,default=512)
    return p.parse_args()

def load_rows(path):
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def text_input(r):
    c=str(r.get("context") or ""); q=str(r.get("question") or "")
    return f"السياق:\n{c[:3500]}\n...\n{c[-3500:]}\n\nالسؤال:\n{q}"

class DS(Dataset):
    def __init__(self,rows,tok,max_len):
        self.rows=rows; self.tok=tok; self.max_len=max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        enc=self.tok(text_input(r),truncation=True,max_length=self.max_len,padding=False)
        enc["score"]=float(r["need_long_score"])
        enc["binary"]=0 if r["router_target"]=="short" else 1
        return enc

def collate(batch,tok):
    text=[{k:v for k,v in x.items() if k in {"input_ids","attention_mask"}} for x in batch]
    enc=tok.pad(text,return_tensors="pt")
    enc["score"]=torch.tensor([x["score"] for x in batch],dtype=torch.float32)
    enc["binary"]=torch.tensor([x["binary"] for x in batch],dtype=torch.long)
    return enc

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(MODEL_ID)
        h=self.encoder.config.hidden_size
        self.head=nn.Sequential(nn.Dropout(0.2),nn.Linear(h,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,1))
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        return torch.sigmoid(self.head(out.last_hidden_state[:,0])).squeeze(-1)

def eval_scores(model,loader,device):
    model.eval(); ys=[]; scores=[]; bins=[]
    with torch.no_grad():
        for b in loader:
            ys.extend(b.pop("score").numpy().tolist())
            bins.extend(b.pop("binary").numpy().tolist())
            b={k:v.to(device) for k,v in b.items()}
            scores.extend(model(**b).cpu().numpy().tolist())
    return np.array(ys),np.array(scores),np.array(bins)

def cls_metrics(y,s,t):
    p=(s>=t).astype(int)
    pr,rc,f1,_=precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    return {
        "threshold":float(t),
        "accuracy":float(accuracy_score(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "precision_long":float(pr),
        "recall_long":float(rc),
        "f1_long":float(f1),
        "confusion_matrix":confusion_matrix(y,p).tolist()
    }

def main():
    a=parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    root=Path(a.output_root)
    src=root/"empirical_router"/"continuous_router"/"continuous_train_4000.jsonl"
    out=root/"empirical_router"/"continuous_router"/"model"
    out.mkdir(parents=True,exist_ok=True)

    rows=load_rows(src)
    y=np.array([0 if r["router_target"]=="short" else 1 for r in rows])
    idx=np.arange(len(rows))
    tr_idx,tmp_idx=train_test_split(idx,test_size=0.30,random_state=a.seed,stratify=y)
    tmp_y=y[tmp_idx]
    dev_idx,hold_idx=train_test_split(tmp_idx,test_size=0.50,random_state=a.seed,stratify=tmp_y)

    tr=[rows[i] for i in tr_idx]; dev=[rows[i] for i in dev_idx]; hold=[rows[i] for i in hold_idx]
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    coll=lambda b: collate(b,tok)
    tr_loader=DataLoader(DS(tr,tok,a.max_length),batch_size=12,shuffle=True,collate_fn=coll)
    dev_loader=DataLoader(DS(dev,tok,a.max_length),batch_size=32,shuffle=False,collate_fn=coll)
    hold_loader=DataLoader(DS(hold,tok,a.max_length),batch_size=32,shuffle=False,collate_fn=coll)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Regressor().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
    total=len(tr_loader)*a.epochs
    sched=get_linear_schedule_with_warmup(opt,max(1,int(0.1*total)),total)
    criterion=nn.SmoothL1Loss()

    best_state=None; best_score=-1; best_t=0.5; history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); total_loss=0
        for b in tr_loader:
            gold=b.pop("score").to(device); b.pop("binary")
            b={k:v.to(device) for k,v in b.items()}
            opt.zero_grad(set_to_none=True)
            pred=model(**b)
            loss=criterion(pred,gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step(); total_loss+=loss.item()

        y_score,p_score,y_bin=eval_scores(model,dev_loader,device)
        sweep=[cls_metrics(y_bin,p_score,t) for t in np.arange(0.15,0.86,0.05)]
        chosen=max(sweep,key=lambda m:0.45*m["balanced_accuracy"]+0.35*m["f1_long"]+0.20*m["accuracy"])
        mae=float(mean_absolute_error(y_score,p_score))
        score=0.45*chosen["balanced_accuracy"]+0.35*chosen["f1_long"]+0.20*chosen["accuracy"]
        history.append({"epoch":epoch,"train_loss":total_loss/max(1,len(tr_loader)),"dev_mae":mae,"dev_best":chosen})
        print(json.dumps(history[-1],ensure_ascii=False),flush=True)
        if score>best_score:
            best_score=score; best_t=chosen["threshold"]
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state); model.to(device)
    y_score,p_score,y_bin=eval_scores(model,hold_loader,device)

    summary={
        "model_id":MODEL_ID,
        "source":"continuous_train_4000.jsonl",
        "usable_examples":len(rows),
        "split":{"train":len(tr),"dev":len(dev),"router_holdout":len(hold)},
        "selected_threshold_from_dev":float(best_t),
        "router_holdout_threshold_0.5":cls_metrics(y_bin,p_score,0.5),
        "router_holdout_tuned_threshold":cls_metrics(y_bin,p_score,best_t),
        "holdout_score_mae":float(mean_absolute_error(y_score,p_score)),
        "history":history,
        "input":"context + question only"
    }
    torch.save({"state_dict":model.state_dict(),"model_id":MODEL_ID,"threshold":best_t},out/"continuous_router.pt")
    tok.save_pretrained(out/"tokenizer")
    (out/"continuous_router_metrics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
