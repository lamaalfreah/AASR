**Step 3B — selected and frozen Router**

Selected `small_mlp`, threshold `0.30140149024077767`. LONG iff score >= threshold. No refitting on DEV.

Fitted 1,636 binary TRAIN rows (440 SHORT, 1,196 LONG); excluded 284 FAILURE rows. Binary DEV metrics use 413 rows (110 SHORT, 303 LONG). Final-answer selection and analysis use all 480 DEV rows, including 67 FAILURE rows.

All three models use balanced inverse-frequency sample weights. Each has one fixed configuration. Preprocessing is fitted only on binary TRAIN. Full configurations and the predeclared lexicographic selection policy are in `selection_protocol.json`.

| Model | Threshold | Adaptive answer | Long route rate | Unnecessary Long | Regressions |
|---|---|---|---|---|---|
| logistic_regression | 0.231413 | 84.79% | 74.58% | 13 | 1 |
| hist_gradient_boosting | 0.128832 | 87.08% | 78.75% | 18 | 3 |
| small_mlp | 0.301401 | 87.29% | 78.12% | 15 | 3 |

| Selected Router metric | Value |
|---|---|
| routing_accuracy | 98.31% |
| long_precision | 97.74% |
| long_recall | 100.00% |
| short_route_rate | 21.88% |
| long_route_rate | 78.12% |
| adaptive_answer_accuracy | 87.29% |

| Policy, all 480 DEV | Correct | Accuracy |
|---|---|---|
| Always Short | 118 | 24.58% |
| Always Long | 395 | 82.29% |
| Adaptive | 419 | 87.29% |
| Answer oracle (either parser correct) | 422 | 87.92% |

Short/Long route rates describe the selected answer path. Short inference actually runs on 100% of inputs to supply Router features; the Long route rate is the additional Long-call rate. Unnecessary Long means a Long call when Short already has a correct final answer. Regression means that call replaces a correct Short answer with a wrong Long answer.

FAILURE DEV analysis: `{"n": 67, "long_calls": 65, "adaptive_answer_correct": 9, "always_short_correct": 0, "always_long_correct": 9, "oracle_correct": 9}`. FAILURE labels use exact-query-plus-answer correctness, so some can still have a correct final answer despite an inexact query. They are never relabeled or used for fitting.

Selected binary confusion matrix: `{"short_to_short": 103, "short_to_long": 7, "long_to_short": 0, "long_to_long": 303}`. Long rescues: 304; Long regressions: 3.

**Limits and reproducibility**

These are DEV selection results, not an unbiased held-out estimate. Thresholds were compared on DEV only, and DEV was not used for fitting, preprocessing, or refitting. No CORE EVAL or TEST data were read. Gold outcomes are used only for supervision/scoring; task/register labels only for reporting.

Step 3A uses controlled Arabic templates with shared wording families across disjoint TRAIN/DEV anchors. This measures new-anchor generalization rather than unseen-style generalization. The binary Router has no FAILURE/abstention output. Scores are class-weighted classifier outputs, not calibrated probabilities.

All frozen Step 3A data, Short, Long, prompts, schema, and Geo Engine hashes verified unchanged. No Short/Long generations or GPU calls were made in Step 3B. Training warnings, timings, default-threshold metrics, selected metrics and breakdowns are saved in `summary.json`.

Frozen artifact: `router.joblib` contains preprocessing, model, threshold and feature allowlist. `freeze_manifest.json` records hashes. Load only trusted joblib artifacts with `language.lightweight_router.LightweightRouter`. Supply the existing Step 3A feature dictionary; gold fields are rejected.

Stopped after selection and freeze. No CORE evaluation or application integration was launched.
