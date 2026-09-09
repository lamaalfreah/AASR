"""
AASR — E2 English Diagnostic (Student 3)

Implements the task brief exactly.

  Step 1  Reuse the CURRENT trained adapters; do not retrain them for E2.
          -> loads short_sft / long_sft with is_trainable=False.
             No optimizer, no trainer, no training code anywhere in this file.

  Step 2  Run the English diagnostic on MATH500 and record accuracy, tokens,
          and latency.
          -> <tag>_summary.csv reports exactly those three per variant.

  Step 3  Compare the English result with the current Arabic baseline. Treat this
          only as a diagnostic because the domain is different.
          -> pass --arabic-summary <E0 summary csv> to emit <tag>_comparison.csv.
             The diagnostic-only caveat is written into every output file.

  Step 4  Only if English is much better, prepare a small matched English spatial
          subset. Deliberately NOT automated — that judgement is the supervisor's.

  Step 5  Do NOT retrain the Router yet.
          -> the router is never loaded, called, or referenced in this script.

Generation policy is identical to evaluate_full_adamix_fast.py (thinking toggles
at alpha >= 0.5, max_new_tokens scales with alpha), so the English numbers are
directly comparable to the Arabic E0 run.

Usage (via modal_runner_e2_english.py):
    python evaluate_e2_english.py \
        --output-root /workspace/artifacts/asar_sft_grpo \
        --alphas 0,0.25,1 --max-samples 250 --seed 42
"""

import argparse, csv, json, os, random, re, time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_ID = "Qwen/Qwen3-4B"
MATH500_ID = "HuggingFaceH4/MATH-500"

DIAGNOSTIC_CAVEAT = (
    "DIAGNOSTIC ONLY — MATH500 is a different domain from the Arabic spatial "
    "task. An English/Arabic gap cannot be attributed to language alone."
)

SYSTEM_PROMPT = """You are a careful mathematical reasoning assistant.
Solve the problem and give the final answer.
Your response must always end with the line:
FINAL: <answer>
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="/workspace/artifacts/asar_sft_grpo")
    p.add_argument("--family", choices=["sft", "grpo"], default="sft",
                   help="Which existing adapters to reuse. Nothing is trained.")
    p.add_argument("--max-samples", type=int, default=250, help="0 = all 500")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", default="0,0.25,1",
                   help="Must match the Arabic baseline run for a valid comparison.")
    p.add_argument("--beta", type=float, default=0.2,
                   help="Oracle utility efficiency weight; same default as E0.")
    p.add_argument("--max-new-floor", type=int, default=0,
                   help="Minimum max_new_tokens for every alpha. 0 keeps E0's "
                        "policy exactly. Raise it when truncated_rate is high: "
                        "E0's budget was calibrated on ~7-token Arabic answers "
                        "and starves MATH500, which turns the accuracy column "
                        "into a measure of token budget rather than reasoning.")
    p.add_argument("--arabic-summary", default=None,
                   help="Path to the E0 Arabic summary CSV, for step 3.")
    p.add_argument("--checkpoint-every", type=int, default=10)
    return p.parse_args()


def parse_alphas(s):
    vals = sorted({round(float(x.strip()), 4) for x in s.split(",") if x.strip()})
    if 0.0 not in vals or 1.0 not in vals:
        raise ValueError("alphas must include 0 and 1")
    return vals


# ---------------------------------------------------------------------------
# Step 2 — data
# ---------------------------------------------------------------------------

def load_math500():
    from datasets import load_dataset
    ds = load_dataset(MATH500_ID, split="test")
    return [{
        "id": str(r.get("unique_id") or i),
        "problem": r["problem"],
        "answer": str(r["answer"]),
        "level": str(r.get("level", "unknown")),
        "subject": str(r.get("subject", "unknown")),
    } for i, r in enumerate(ds)]


def stratified_sample(rows, n, seed):
    """Same stratification approach as the Arabic evaluator, keyed on level."""
    if not n or n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    buckets = {}
    for r in rows:
        buckets.setdefault(r["level"], []).append(r)
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


# ---------------------------------------------------------------------------
# Answer checking — so that "accuracy" in step 2 is trustworthy
# ---------------------------------------------------------------------------

def extract_final(text):
    """The adapters are SFT-trained to emit 'FINAL: <answer>'. Keeping that
    contract measures reasoning rather than format compliance; \\boxed{} is a
    fallback so a correct answer in the usual MATH format is not lost."""
    m = re.findall(r"FINAL\s*:\s*(.+)", str(text or ""), flags=re.I)
    if m:
        return m[-1].strip()
    b = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", str(text or ""))
    return b[-1].strip() if b else ""


def normalize_math(s):
    t = str(s or "").strip()
    t = re.sub(r"\\boxed\{(.*)\}", r"\1", t)
    t = re.sub(r"\\(?:text|mbox|mathrm)\{([^}]*)\}", r"\1", t)
    t = t.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = t.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "")
    t = t.replace("\\$", "").replace(" ", "")
    t = t.rstrip(".")
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    return t.lower()


def to_float(s):
    t = normalize_math(s).replace(",", "")
    m = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", t)
    if not m:
        m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", t)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(t)
    except ValueError:
        return None


def correctness(output, gold):
    """(strict, lenient, needs_review).

    strict       — normalized strings match.
    lenient      — strings differ, numeric values agree.
    needs_review — neither, but a short non-empty prediction a human should check.

    Nothing is ever silently marked correct.
    """
    pred = extract_final(output)
    if not pred:
        return False, False, False
    np_, ng = normalize_math(pred), normalize_math(gold)
    if np_ == ng:
        return True, True, False
    fp, fg = to_float(pred), to_float(gold)
    lenient = fp is not None and fg is not None and abs(fp - fg) < 1e-6
    return False, lenient, (not lenient) and len(np_) <= 40


# ---------------------------------------------------------------------------
# Step 1 — reuse adapters; alpha mixing identical to E0
# ---------------------------------------------------------------------------

def adapter_name(alpha):
    if abs(alpha - 0.0) < 1e-8: return "short"
    if abs(alpha - 1.0) < 1e-8: return "long"
    return f"mix_{str(alpha).replace('.', '_')}"


def create_mixed_adapters(model, alphas):
    for a in alphas:
        if a in {0.0, 1.0}:
            continue
        model.add_weighted_adapter(
            adapters=["short", "long"], weights=[1.0 - a, a],
            adapter_name=adapter_name(a), combination_type="linear",
        )
        print(f"Created {adapter_name(a)}: short={1-a:.2f}, long={a:.2f}", flush=True)


def build_messages(row):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{row['problem']}\n\nSolve it."},
    ]


@torch.inference_mode()
def generate_alpha(model, tok, row, alpha, max_new_floor=0):
    model.set_adapter(adapter_name(alpha))
    # Identical to evaluate_full_adamix_fast.py so E2 stays comparable to E0,
    # unless max_new_floor raises the budget to keep answers from being cut off.
    thinking = bool(alpha >= 0.5)
    max_new = max(int(round(256 + alpha * (1024 - 256))), max_new_floor)

    prompt = tok.apply_chat_template(
        build_messages(row), tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking,
    )
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    st = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    sec = time.perf_counter() - st
    new = out[0, inp["input_ids"].shape[1]:]
    return (tok.decode(new, skip_special_tokens=True), int(new.numel()), sec,
            bool(new.numel() >= max_new))


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


# ---------------------------------------------------------------------------
# Step 2 metrics and step 3 comparison
# ---------------------------------------------------------------------------

def candidate_utility(cand, beta):
    return float(cand["strict_correct"]) - beta * cand["token_ratio"]


def summarize(records, alphas):
    variants = {f"alpha_{a}": [] for a in alphas}
    variants["oracle_alpha"] = []
    for r in records:
        for a in alphas:
            variants[f"alpha_{a}"].append(r["candidates"][str(a)])
        variants["oracle_alpha"].append(r["candidates"][str(r["oracle_alpha"])])

    out = []
    for name, vals in variants.items():
        n = len(vals)
        if not n: continue
        out.append({
            "variant": name,
            "n": n,
            # --- the three metrics the brief asks for ---
            "strict_accuracy": round(sum(int(v["strict_correct"]) for v in vals) / n, 4),
            "avg_tokens": round(sum(v["tokens"] for v in vals) / n, 2),
            "avg_latency_sec": round(sum(v["latency"] for v in vals) / n, 4),
            # --- integrity columns, so the three above can be trusted ---
            "lenient_accuracy": round(sum(int(v["lenient_correct"]) for v in vals) / n, 4),
            "no_final_line_rate": round(sum(int(v["prediction"] == "") for v in vals) / n, 4),
            "truncated_rate": round(sum(int(v["truncated"]) for v in vals) / n, 4),
            "needs_review_rate": round(sum(int(v["needs_review"]) for v in vals) / n, 4),
        })
    return out


def read_summary_csv(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["variant"]] = r
    return rows


def build_comparison(english, arabic_path):
    """Step 3 — same variants, side by side, with the caveat attached."""
    arabic = read_summary_csv(arabic_path)
    out = []
    for e in english:
        a = arabic.get(e["variant"])
        if not a:
            continue
        out.append({
            "variant": e["variant"],
            "arabic_accuracy": round(float(a["strict_accuracy"]), 4),
            "english_accuracy": e["strict_accuracy"],
            "accuracy_delta": round(e["strict_accuracy"] - float(a["strict_accuracy"]), 4),
            "arabic_avg_tokens": round(float(a["avg_tokens"]), 2),
            "english_avg_tokens": e["avg_tokens"],
            "arabic_avg_latency_sec": round(float(a["avg_latency_sec"]), 4),
            "english_avg_latency_sec": e["avg_latency_sec"],
            "note": "diagnostic only; different domain",
        })
    return out


def write_csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    root = Path(args.output_root)
    outdir = root / "e2_english_diagnostic"
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = load_math500()
    rows = stratified_sample(all_rows, args.max_samples, args.seed)
    lv = {}
    for r in rows:
        lv[r["level"]] = lv.get(r["level"], 0) + 1
    print(f"MATH500 loaded: {len(all_rows)} rows", flush=True)
    print("Level counts:", json.dumps(lv), flush=True)
    print(f"Evaluating {len(rows)} / {len(all_rows)} rows", flush=True)
    print(DIAGNOSTIC_CAVEAT, flush=True)

    tag = f"e2_{args.family}_n{len(rows)}_seed{args.seed}"
    outpath = outdir / f"{tag}_outputs.jsonl"
    summary_path = outdir / f"{tag}_summary.csv"
    comparison_path = outdir / f"{tag}_comparison.csv"
    meta_path = outdir / f"{tag}_meta.json"

    meta_path.write_text(json.dumps({
        "experiment": "E2 English Diagnostic",
        "dataset": MATH500_ID,
        "base_model": QWEN_ID,
        "adapters_reused": args.family,
        "adapters_retrained": False,
        "router_used": False,
        "router_retrained": False,
        "generation_policy": "identical to evaluate_full_adamix_fast.py (E0)",
        "alphas": alphas, "seed": args.seed, "n": len(rows), "beta": args.beta,
        "max_new_floor": args.max_new_floor,
        "generation_budget": ("E0 policy unchanged" if args.max_new_floor == 0
                              else f"E0 policy with a floor of {args.max_new_floor} "
                                   f"new tokens, applied equally to every alpha"),
        "caveat": DIAGNOSTIC_CAVEAT,
    }, indent=2), encoding="utf-8")

    existing = load_existing(outpath)
    done_ids = {str(r["id"]) for r in existing}
    remaining = [r for r in rows if str(r["id"]) not in done_ids]
    print(f"Resume: existing={len(existing)} remaining={len(remaining)}", flush=True)

    tok = AutoTokenizer.from_pretrained(QWEN_ID)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16,
                                                device_map="auto")

    sub = "sft" if args.family == "sft" else "grpo_refined"
    sp = require_adapter(root / "adapters" / f"short_{sub}")
    lp = require_adapter(root / "adapters" / f"long_{sub}")
    # is_trainable=False on both — step 1 is enforced here, not just promised.
    model = PeftModel.from_pretrained(base, str(sp), adapter_name="short", is_trainable=False)
    model.load_adapter(str(lp), adapter_name="long", is_trainable=False)
    create_mixed_adapters(model, alphas)
    model.eval()

    with outpath.open("a", encoding="utf-8") as f:
        start_n = len(existing)
        for j, row in enumerate(remaining, 1):
            candidates, max_tok = {}, 1
            for alpha in alphas:
                text, ntok, lat, trunc = generate_alpha(
                    model, tok, row, alpha, args.max_new_floor)
                strict, lenient, review = correctness(text, row["answer"])
                candidates[str(alpha)] = {
                    "alpha": alpha,
                    "strict_correct": bool(strict),
                    "lenient_correct": bool(lenient),
                    "needs_review": bool(review),
                    "truncated": trunc,
                    "tokens": ntok,
                    "latency": round(lat, 4),
                    "prediction": extract_final(text),
                }
                max_tok = max(max_tok, ntok)

            for c in candidates.values():
                c["token_ratio"] = c["tokens"] / max_tok
                c["utility"] = round(candidate_utility(c, args.beta), 6)

            oracle = max(candidates.values(),
                         key=lambda c: (c["utility"], c["strict_correct"], -c["tokens"]))
            f.write(json.dumps({
                "id": row["id"], "level": row["level"], "subject": row["subject"],
                "gold": row["answer"], "oracle_alpha": oracle["alpha"],
                "candidates": candidates,
            }, ensure_ascii=False) + "\n")

            done = start_n + j
            if done % args.checkpoint_every == 0 or j == len(remaining):
                f.flush(); os.fsync(f.fileno())
                write_csv(summary_path, summarize(load_existing(outpath), alphas))
                print(f"[{done}/{len(rows)}] checkpoint written", flush=True)

    summary = summarize(load_existing(outpath), alphas)
    write_csv(summary_path, summary)

    print("\n=== E2 SUMMARY (accuracy / tokens / latency) ===")
    print(f"{'variant':<18}{'accuracy':>10}{'tokens':>10}{'latency':>10}")
    for r in summary:
        print(f"{r['variant']:<18}{r['strict_accuracy']:>10.4f}"
              f"{r['avg_tokens']:>10.2f}{r['avg_latency_sec']:>10.3f}")

    worst = max(summary, key=lambda r: r["truncated_rate"])
    if worst["truncated_rate"] >= 0.05:
        print("\n*** WARNING: truncation is distorting these numbers ***")
        for r in summary:
            print(f"  {r['variant']:<18} truncated={r['truncated_rate']:>6.1%}  "
                  f"no FINAL line={r['no_final_line_rate']:>6.1%}")
        print("  Generations are being cut off before the FINAL line, so those")
        print("  rows score as wrong for a budget reason, not a reasoning one.")
        print(f"  Re-run with --max-new-floor set above the current cap "
              f"(now {max(256, args.max_new_floor)}).")

    if args.arabic_summary:
        comp = build_comparison(summary, args.arabic_summary)
        write_csv(comparison_path, comp)
        print("\n=== STEP 3: ENGLISH vs ARABIC BASELINE ===")
        print(f"{'variant':<18}{'arabic':>10}{'english':>10}{'delta':>10}")
        for r in comp:
            print(f"{r['variant']:<18}{r['arabic_accuracy']:>10.4f}"
                  f"{r['english_accuracy']:>10.4f}{r['accuracy_delta']:>+10.4f}")
        print(f"\n{DIAGNOSTIC_CAVEAT}")
        print("Step 4 (matched English spatial subset) is a supervisor decision.")

    print(f"\nOutputs in {outdir}")


if __name__ == "__main__":
    main()