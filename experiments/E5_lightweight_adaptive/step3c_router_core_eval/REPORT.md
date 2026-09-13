**Final frozen Router CORE EVAL — 128 examples**

Frozen Small MLP, LONG iff score >= `0.30140149024077767`. One Router inference per example. No fitting, threshold selection, or changes after CORE results. Original frozen Short/Long outcomes supply all accuracy scores.

| Policy | Correct / 128 | Accuracy |
|---|---|---|
| Always Short | 30 | 23.44% |
| Always Long | 95 | 74.22% |
| Adaptive Router | 93 | 72.66% |
| Answer oracle | 103 | 80.47% |

| Metric | Result |
|---|---|
| routing_accuracy | 88.00% |
| long_precision | 95.59% |
| long_recall | 87.84% |
| short_route_rate | 28.91% |
| long_route_rate | 71.09% |

Binary routing metrics exclude 28 FAILURE rows and use 100 SHORT/LONG rows, retaining the Step 3A exact-query-plus-answer label definition and SHORT preference. All 128 rows count toward final-answer accuracy and route rates. Binary confusion: `{"short_to_short": 23, "short_to_long": 3, "long_to_short": 9, "long_to_long": 65}`.

Long rescues: 65; regressions: 2; unnecessary Long calls: 6. Regression rate among Short-answer-correct examples: 6.67%. FAILURE analysis: `{"n": 28, "long_calls": 23, "adaptive_answer_correct": 2, "always_short_correct": 2, "always_long_correct": 2, "oracle_correct": 3}`.

Short/Long route rates describe the selected answer path. Short inference is required on 100% of inputs to compute Router features; Long is an additional call only for LONG routes.

**Latency and cost implication**

| Measurement | Mean ms |
|---|---|
| Frozen Short pipeline | 0.577 |
| Saved Long inference | 5979.565 |
| Observed local Router | 0.256 |
| Projected adaptive (saved Short + Router + selected Long) | 4192.066 |
| CPU Short plus feature extraction, observed | 2.239 |
| Projected adaptive with observed feature extraction | 4193.728 |

Projected Long generation calls (including existing retries): 116/162; avoided saved Long inference time: 228.91 seconds (29.91%). These are inference-work savings, not a measured bill reduction. Cost is unavailable. The evaluation itself made zero Long generations and zero GPU calls.

Latency projection from saved CPU Short and L4 Long timings plus local Router timings, not a measured deployed service. Excludes Long cold-start, idle billing, and full orchestration; currencies unavailable.

**Feature provenance and checks**

Original CORE Short predictions did not save entropy/top-two margin. One CPU pass of the unchanged frozen Short model and the exact Step 3A feature extractor supplied the missing inference-safe features. Operation, confidence (tolerance 1e-6), query availability, query exactness, and final-answer correctness matched all original frozen Short predictions. Original outcomes were not overwritten. Only public question/context fields reached the CPU function; only the frozen feature dictionary reached Router inference.

All Router, threshold, Short, Long, schema, Geo Engine, CORE inputs, and saved prediction hashes verified unchanged. No TEST data read. Router retains its recorded DEV training iteration-limit warning; no convergence adjustment or retraining occurred.

Artifacts: `summary.json`, `predictions.jsonl`, `short_feature_response.json`, `pre_eval_freeze.json`, and `freeze_manifest.json`. Stopped after this final evaluation.
