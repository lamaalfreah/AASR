import argparse, gzip, io, json, os, random
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROUTER_ID = "xlm-roberta-base"
DATA_BASE = "https://huggingface.co/datasets/ArabicSpatialrReasoning/arabic-spatial-reasoning-v1/resolve/main"
ALPHAS = [0.0, 0.25, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--source-tag", default="sft_validation_all_n120_seed42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=6)
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


def best_restricted_alpha(eval_row):
    # IMPORTANT: choose the best alpha empirically from the already-generated
    # candidates {0.0, 0.25, 1.0}; do not remap the previous 5-class label.
    cands = eval_row["candidates"]
    options = [cands[str(a)] for a in ALPHAS]
    best = max(
        options,
        key=lambda c: (
            float(c["utility"]),
            int(bool(c["strict_correct"])),
            -float(c["tokens"]),
        ),
    )
    return float(best["alpha"])


class RouterDataset(Dataset):
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
        enc["labels"] = int(r["label"])
        enc["row_id"] = str(r["id"])
        return enc


def collate(batch, tok):
    labels = torch.tensor([x.pop("labels") for x in batch], dtype=torch.long)
    row_ids = [x.pop("row_id") for x in batch]
    enc = tok.pad(batch, return_tensors="pt")
    enc["labels"] = labels
    enc["row_ids"] = row_ids
    return enc


class Router3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ROUTER_ID)
        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(h, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(ALPHAS)),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0])


def class_weights(labels):
    counts = torch.bincount(labels, minlength=len(ALPHAS)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = labels.numel() / (present.sum() * counts[present])
    return weights


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    gold, pred, ids = [], [], []
    for b in loader:
        labels = b.pop("labels")
        row_ids = b.pop("row_ids")
        x = {k: v.to(device) for k, v in b.items()}
        logits = model(**x)
        gold.extend(labels.tolist())
        pred.extend(logits.argmax(-1).cpu().tolist())
        ids.extend(row_ids)
    return np.array(gold), np.array(pred), ids


def summarize_variant(eval_by_id, ids, alpha_for_id):
    correct, tokens, latency = [], [], []
    for rid in ids:
        a = float(alpha_for_id[rid])
        cand = eval_by_id[rid]["candidates"][str(a)]
        correct.append(int(bool(cand["strict_correct"])))
        tokens.append(float(cand["tokens"]))
        latency.append(float(cand["latency"]))
    return {
        "strict_accuracy": float(np.mean(correct)),
        "avg_tokens": float(np.mean(tokens)),
        "avg_latency_sec": float(np.mean(latency)),
    }


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    root = Path(a.output_root)
    src = root / "final_evaluation_adamix_fast" / f"{a.source_tag}_outputs.jsonl"
    outdir = root / "alpha_router_v3"
    model_dir = outdir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_jsonl(src)
    eval_by_id = {str(r["id"]): r for r in eval_rows}

    label_alpha_by_id = {
        rid: best_restricted_alpha(r)
        for rid, r in eval_by_id.items()
    }

    val_rows = download_validation()
    joined = []
    for r in val_rows:
        rid = str(r.get("id"))
        if rid in label_alpha_by_id:
            alpha = label_alpha_by_id[rid]
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
    print("Restricted empirical alpha distribution:", json.dumps(counts, ensure_ascii=False), flush=True)

    idx = np.arange(len(joined))
    binc = np.bincount(labels_np, minlength=len(ALPHAS))
    can_stratify = np.all(binc[binc > 0] >= 2)
    tr_idx, hold_idx = train_test_split(
        idx,
        test_size=0.20,
        random_state=a.seed,
        stratify=labels_np if can_stratify else None,
    )

    train_rows = [joined[i] for i in tr_idx]
    hold_rows = [joined[i] for i in hold_idx]

    tok = AutoTokenizer.from_pretrained(ROUTER_ID)
    train_ds = RouterDataset(train_rows, tok, a.max_length)
    hold_ds = RouterDataset(hold_rows, tok, a.max_length)
    coll = lambda b: collate(b, tok)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=coll)
    hold_loader = DataLoader(hold_ds, batch_size=16, shuffle=False, collate_fn=coll)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Router3().to(device)

    y_train = torch.tensor([r["label"] for r in train_rows], dtype=torch.long)
    weights = class_weights(y_train).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * a.epochs)
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_state = None
    best_epoch = 1
    best_score = -1.0
    history = []

    for epoch in range(1, a.epochs + 1):
        model.train()
        running = 0.0
        for b in train_loader:
            labels = b.pop("labels").to(device)
            b.pop("row_ids")
            x = {k: v.to(device) for k, v in b.items()}
            logits = model(**x)
            loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        hold_gold, hold_pred, hold_ids = predict(model, hold_loader, device)
        acc = float(accuracy_score(hold_gold, hold_pred))
        bal = float(balanced_accuracy_score(hold_gold, hold_pred))
        score = 0.6 * bal + 0.4 * acc

        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "holdout_alpha_accuracy": acc,
            "holdout_alpha_balanced_accuracy": bal,
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()
    hold_gold, hold_pred, hold_ids = predict(model, hold_loader, device)

    predicted_alpha_by_id = {
        rid: ALPHAS[int(cls)]
        for rid, cls in zip(hold_ids, hold_pred)
    }

    # Compare on the EXACT SAME holdout rows.
    fixed0 = {rid: 0.0 for rid in hold_ids}
    fixed025 = {rid: 0.25 for rid in hold_ids}
    fixed1 = {rid: 1.0 for rid in hold_ids}
    restricted_oracle = {rid: label_alpha_by_id[rid] for rid in hold_ids}
    full_oracle = {rid: float(eval_by_id[rid]["oracle_alpha"]) for rid in hold_ids}

    metrics = {
        "source": str(src),
        "examples": len(joined),
        "restricted_alpha_set": ALPHAS,
        "restricted_empirical_alpha_distribution": counts,
        "split": {"train": len(train_rows), "holdout": len(hold_rows)},
        "best_epoch": best_epoch,
        "holdout_alpha_class_accuracy": float(accuracy_score(hold_gold, hold_pred)),
        "holdout_alpha_balanced_accuracy": float(balanced_accuracy_score(hold_gold, hold_pred)),
        "holdout_confusion_matrix": confusion_matrix(
            hold_gold,
            hold_pred,
            labels=list(range(len(ALPHAS)))
        ).tolist(),
        "same_holdout_end_to_end": {
            "fixed_alpha_0.0": summarize_variant(eval_by_id, hold_ids, fixed0),
            "fixed_alpha_0.25": summarize_variant(eval_by_id, hold_ids, fixed025),
            "fixed_alpha_1.0": summarize_variant(eval_by_id, hold_ids, fixed1),
            "router_v3": summarize_variant(eval_by_id, hold_ids, predicted_alpha_by_id),
            "restricted_oracle_3alpha": summarize_variant(eval_by_id, hold_ids, restricted_oracle),
            "full_oracle_5alpha": summarize_variant(eval_by_id, hold_ids, full_oracle),
        },
        "history": history,
        "note": "Router V3 fine-tunes XLM-R end-to-end on empirical best alpha labels restricted to {0.0, 0.25, 1.0}. Labels are chosen directly from previously generated validation candidates.",
    }

    torch.save({
        "state_dict": model.state_dict(),
        "router_id": ROUTER_ID,
        "alphas": ALPHAS,
        "best_epoch": best_epoch,
        "max_length": a.max_length,
    }, model_dir / "alpha_router_v3.pt")
    tok.save_pretrained(model_dir / "tokenizer")
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ROUTER V3 FINAL METRICS ===", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
