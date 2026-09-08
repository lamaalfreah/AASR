import argparse, gzip, io, json, os, random
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROUTER_ID = "xlm-roberta-base"
DATA_BASE = "https://huggingface.co/datasets/ArabicSpatialrReasoning/arabic-spatial-reasoning-v1/resolve/main"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--source-tag", default="sft_validation_all_n120_seed42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=80)
    return p.parse_args()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def download_validation():
    tok = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    r = requests.get(f"{DATA_BASE}/validation.jsonl.gz", headers=headers, timeout=300)
    r.raise_for_status()
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        for raw in gz:
            s = raw.decode("utf-8").strip()
            if s:
                rows.append(json.loads(s))
    return rows


def router_text(row):
    c = str(row.get("context") or "")
    q = str(row.get("question") or "")
    return f"السياق:\n{c[:3500]}\n...\n{c[-3500:]}\n\nالسؤال:\n{q}"


class TextDS(Dataset):
    def __init__(self, rows, tok, max_length):
        self.rows = rows
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(
            router_text(r),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["label"] = int(r["label"])
        enc["id"] = str(r["id"])
        return enc


def collate(batch, tok):
    ids = [x.pop("id") for x in batch]
    labels = torch.tensor([x.pop("label") for x in batch], dtype=torch.long)
    enc = tok.pad(batch, return_tensors="pt")
    enc["labels"] = labels
    enc["row_ids"] = ids
    return enc


class Head(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(ALPHAS)),
        )

    def forward(self, x):
        return self.net(x)


@torch.inference_mode()
def encode_rows(encoder, loader, device):
    encoder.eval()
    xs, ys, ids = [], [], []
    for b in loader:
        labels = b.pop("labels")
        row_ids = b.pop("row_ids")
        b = {k: v.to(device) for k, v in b.items()}
        out = encoder(**b)
        xs.append(out.last_hidden_state[:, 0].cpu())
        ys.append(labels)
        ids.extend(row_ids)
    return torch.cat(xs), torch.cat(ys), ids


def class_weights(labels):
    counts = torch.bincount(labels, minlength=len(ALPHAS)).float()
    w = torch.zeros_like(counts)
    present = counts > 0
    w[present] = labels.numel() / (present.sum() * counts[present])
    return w


def train_head(X, y, train_idx, dev_idx, hidden, epochs, seed):
    torch.manual_seed(seed)
    head = Head(hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=0.01)
    weights = class_weights(y[train_idx])
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_state = None
    best_epoch = 1
    best_score = -1.0
    patience = 12
    stale = 0

    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_idx))
        for start in range(0, len(order), 16):
            idx = train_idx[order[start:start+16]]
            logits = head(X[idx])
            loss = loss_fn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            pred = head(X[dev_idx]).argmax(-1).numpy()
        gold = y[dev_idx].numpy()
        acc = accuracy_score(gold, pred)
        bal = balanced_accuracy_score(gold, pred)
        score = 0.6 * bal + 0.4 * acc

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    head.load_state_dict(best_state)
    return head, best_epoch


def fit_final_head(X, y, hidden, epochs, seed):
    torch.manual_seed(seed)
    head = Head(hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=0.01)
    weights = class_weights(y)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(max(1, epochs)):
        head.train()
        order = torch.randperm(len(y))
        for start in range(0, len(order), 16):
            idx = order[start:start+16]
            logits = head(X[idx])
            loss = loss_fn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    root = Path(a.output_root)
    src = root / "final_evaluation_adamix_fast" / f"{a.source_tag}_outputs.jsonl"
    outdir = root / "alpha_router_v2"
    outdir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_jsonl(src)
    oracle_by_id = {str(r["id"]): float(r["oracle_alpha"]) for r in eval_rows}
    eval_by_id = {str(r["id"]): r for r in eval_rows}

    val_rows = download_validation()
    joined = []
    for r in val_rows:
        rid = str(r.get("id"))
        if rid in oracle_by_id:
            alpha = oracle_by_id[rid]
            joined.append({
                "id": rid,
                "context": r.get("context"),
                "question": r.get("question"),
                "label": ALPHAS.index(alpha),
            })

    if len(joined) != len(eval_rows):
        raise RuntimeError(f"Joined {len(joined)} rows but expected {len(eval_rows)}")

    labels_np = np.array([r["label"] for r in joined], dtype=np.int64)
    counts = {str(ALPHAS[i]): int((labels_np == i).sum()) for i in range(len(ALPHAS))}
    print("Oracle alpha label distribution:", json.dumps(counts, ensure_ascii=False), flush=True)

    idx = np.arange(len(joined))
    binc = np.bincount(labels_np, minlength=len(ALPHAS))
    can_stratify = np.all(binc[binc > 0] >= 2)
    tr_idx, dev_idx = train_test_split(
        idx,
        test_size=0.20,
        random_state=a.seed,
        stratify=labels_np if can_stratify else None,
    )

    tok = AutoTokenizer.from_pretrained(ROUTER_ID)
    ds = TextDS(joined, tok, a.max_length)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate(b, tok))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoModel.from_pretrained(ROUTER_ID).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    X, y, ids = encode_rows(encoder, loader, device)
    hidden = X.shape[1]

    head, best_epoch = train_head(
        X, y,
        torch.tensor(tr_idx, dtype=torch.long),
        torch.tensor(dev_idx, dtype=torch.long),
        hidden, a.epochs, a.seed,
    )

    head.eval()
    with torch.no_grad():
        dev_pred = head(X[dev_idx]).argmax(-1).numpy()
    dev_gold = y[dev_idx].numpy()

    cls_acc = float(accuracy_score(dev_gold, dev_pred))
    cls_bal = float(balanced_accuracy_score(dev_gold, dev_pred))
    cm = confusion_matrix(dev_gold, dev_pred, labels=list(range(len(ALPHAS)))).tolist()

    # End-to-end holdout behavior using already generated candidates: no new Qwen inference.
    strict = []
    toks = []
    lats = []
    for local_i, pred_cls in zip(dev_idx, dev_pred):
        rid = ids[local_i]
        pred_alpha = ALPHAS[int(pred_cls)]
        cand = eval_by_id[rid]["candidates"][str(pred_alpha)]
        strict.append(int(cand["strict_correct"]))
        toks.append(float(cand["tokens"]))
        lats.append(float(cand["latency"]))

    dev_e2e_acc = float(np.mean(strict))
    dev_avg_tokens = float(np.mean(toks))
    dev_avg_latency = float(np.mean(lats))

    # Refit only the small classification head on all 120 examples using selected epoch.
    final_head = fit_final_head(X, y, hidden, best_epoch, a.seed)

    model_dir = outdir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": final_head.state_dict(),
        "router_id": ROUTER_ID,
        "alphas": ALPHAS,
        "hidden_size": int(hidden),
        "best_epoch": int(best_epoch),
        "max_length": int(a.max_length),
    }, model_dir / "alpha_router_v2.pt")
    tok.save_pretrained(model_dir / "tokenizer")

    summary = {
        "source": str(src),
        "examples": len(joined),
        "oracle_alpha_distribution": counts,
        "split": {"train": int(len(tr_idx)), "holdout": int(len(dev_idx))},
        "best_epoch": int(best_epoch),
        "holdout_alpha_class_accuracy": cls_acc,
        "holdout_alpha_balanced_accuracy": cls_bal,
        "holdout_confusion_matrix": cm,
        "holdout_end_to_end_strict_accuracy": dev_e2e_acc,
        "holdout_avg_tokens": dev_avg_tokens,
        "holdout_avg_latency_sec": dev_avg_latency,
        "note": "XLM-R encoder frozen; only a 5-class alpha head is trained on empirical oracle-alpha labels from validation outputs.",
    }
    (outdir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
