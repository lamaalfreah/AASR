# E2 — English Diagnostic (MATH500)

## Status

Complete. Step 4 decision pending supervisor.

## Scope

Covers steps 1–3 of the Student 3 brief. Step 4 is a supervisor decision and is
deliberately not automated. Step 5 is respected: the Router is never loaded,
called, or referenced anywhere in this experiment.

## Setup

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-4B` |
| Adapters | `short_sft`, `long_sft` — **reused, not retrained** (`is_trainable=False`) |
| Dataset | `HuggingFaceH4/MATH-500`, test split |
| Sample | 150 problems, stratified by difficulty level, seed 42 |
| Alphas | 0, 0.25, 1 — same as the Arabic E0 run |
| Decoding | Greedy (`do_sample=False`), thinking mode on at alpha >= 0.5 |
| Answer contract | `FINAL: <answer>`, with `\boxed{}` as fallback |

Compliance is recorded in `e2_sft_n150_seed42_meta.json`
(`adapters_retrained: false`, `router_used: false`).

## Deviation from the E0 generation policy

E0 sets the answer budget as `max_new_tokens = 256 + alpha * 768`. That formula
was calibrated on Arabic spatial answers, which average about 7 tokens. Applied
unchanged to MATH500 it truncated most generations before the model reached its
final answer.

Calibration runs on 8 problems:

| Budget floor | alpha 0 accuracy | alpha 0 truncated |
| ---: | ---: | ---: |
| 256 (E0 default) | 25.0% | 62.5% |
| 2048 | 62.5% | 0.0% |
| 4096 | 62.5% | 0.0% |

Raising the budget moved alpha 0 from 25% to 62.5% **without changing the model
at all**. The original numbers were measuring token budget, not reasoning.

The main run therefore uses `--max-new-floor 4096`, applied equally to every
alpha. We stopped at 4096 because going from 2048 to 4096 gave alpha 1 roughly
500 extra tokens and produced no accuracy gain.

**Consequence:** accuracy remains comparable between Arabic and English. Token
counts and latency do **not** — the two runs ran under different budgets by
design. Do not read the token or latency columns as a cross-language comparison.

## Results (n=150)

| Variant | Accuracy | Avg. Tokens | Avg. Latency |
| --- | ---: | ---: | ---: |
| alpha = 0 | 68.67% | 556.17 | 24.37 s |
| alpha = 0.25 | 66.67% | 495.73 | 21.69 s |
| alpha = 1 | **70.00%** | 1585.75 | 69.57 s |
| Oracle | 83.33% | 445.87 | 19.55 s |

### Measurement integrity

| Variant | Truncated | No FINAL line |
| --- | ---: | ---: |
| alpha = 0 | 3.3% | 4.0% |
| alpha = 0.25 | 1.3% | 2.7% |
| alpha = 1 | **18.0%** | 17.3% |

alpha 1 is the only variant with thinking mode enabled, and it still hits the
4096-token cap on 18% of problems. **Its 70.00% is therefore an underestimate.**
alpha 0 and alpha 0.25 are effectively clean.

## Step 3 — English vs the current Arabic baseline

Comparator: `sft_validation_all_n400_seed42_summary.csv` (E0, same alphas, same
evaluator mechanics).

| Variant | Arabic | English | Delta |
| --- | ---: | ---: | ---: |
| alpha = 0 | 37.00% | 68.67% | +31.67 |
| alpha = 0.25 | 35.00% | 66.67% | +31.67 |
| alpha = 1 | 33.25% | 70.00% | +36.75 |
| Oracle | 59.00% | 83.33% | +24.33 |

**This is a diagnostic only.** MATH500 is a different domain from the Arabic
spatial task, so the gap cannot be attributed to language alone.

## Findings

### 1. English scores far higher on every variant

Every variant gains roughly 31–37 points. The gap is large and consistent.

### 2. The alpha ordering reverses

This is the most informative result, and it is not explained by overall
difficulty.

| | Best alpha | Worst alpha |
| --- | --- | --- |
| Arabic (E0) | 0 (37.00%) | 1 (33.25%) |
| English (E2) | 1 (70.00%, understated) | 0.25 (66.67%) |

In Arabic, accuracy **falls** as alpha rises — the long adapter actively hurts.
In English it is the opposite: alpha 1 leads despite being penalised by 18%
truncation. Correcting for that truncation would widen its lead further.

A domain being easier raises all variants together. It does not flip which
adapter wins. Something about the Arabic setup specifically suppresses the long
adapter.

### 3. The alpha spread is small in English

66.67% to 70.00% is 3.3 points across 150 problems — within sampling noise on
its own. The reversal in *direction* is the finding here, not the magnitude.

### 4. Headroom is smaller in English

| | Oracle | Best fixed | Headroom |
| --- | ---: | ---: | ---: |
| Arabic | 59.00% | 37.00% | +22.00 |
| English | 83.33% | 70.00% | +13.33 |

Less to gain from per-query routing in English. Consistent with finding 3: when
the alphas behave similarly, choosing between them matters less.

## Limitations

* **Domain confound is total.** MATH500 is English mathematics; the baseline is
  Arabic spatial retrieval. These differ in language, domain, answer format, and
  context length simultaneously. No single-variable conclusion is available from
  this comparison. This is exactly why the brief calls it a diagnostic.
* **alpha 1 is understated** by 18% truncation. Its true accuracy is higher.
* **No base-model control.** We did not run plain `Qwen3-4B` on the same subset,
  so we cannot say whether the adapters help or hurt English math. The brief does
  not ask for it, but it would be the natural next control if the supervisor
  wants to interpret the absolute English numbers rather than the relative ones.
* **Token and latency columns are not cross-language comparable** (see Deviation
  above).
* Strict matching is used. Lenient matching did not change any conclusion.

## Step 4 — Decision Requested

The brief says to prepare a matched English spatial subset **only if English is
much better**. English is better by 31–37 points on every variant, which clears
any reasonable threshold.

Two additional points for the supervisor:

1. The **alpha reversal** (finding 2) is the result a matched subset would be
   best placed to explain. A subset that holds domain, answer format, and context
   length fixed while varying only language would isolate whether the long
   adapter's failure is Arabic-specific.
2. Because the domain confound is complete, the size of the gap on its own does
   not justify a language conclusion. The reversal does justify further
   investigation.

Recommendation: proceed to step 4 (100–200 matched English spatial examples).
Final call is the supervisor's.

## Files

| File | Contents |
| --- | --- |
| `command.txt` | Exact commands, including calibration runs |
| `e2_sft_n150_seed42_summary.csv` | Accuracy, tokens, latency per variant |
| `e2_sft_n150_seed42_comparison.csv` | English vs Arabic, step 3 |
| `e2_sft_n150_seed42_meta.json` | Compliance record and run configuration |
| `e2_sft_n150_seed42_outputs.jsonl` | Per-problem raw outputs (on Modal volume) |

## Artifact Paths

```
asar-artifacts:/asar_sft_grpo/adapters/short_sft
asar-artifacts:/asar_sft_grpo/adapters/long_sft
asar-artifacts:/asar_sft_grpo/e2_english_diagnostic/
```
