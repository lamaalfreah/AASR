# Step 2B — Modal Qwen3-4B versus frozen Neural Short

Status: **complete**. 

**128/128 CORE EVAL examples completed.** 182 durable generation calls across DEV/EVAL, including schema retries.

The official `Qwen/Qwen3-4B` weights are used without adapters, quantization or training. One L4 container is reused; bfloat16 and non-thinking greedy generation are fixed before EVAL. The existing prompt, schema, Geo Engine, Rule Baseline 0, Neural Short checkpoint/predictions and 320 questions remain frozen.

Only question, exact visible candidate names, anchor name and category vocabulary reach the model. Gold labels and challenge metadata are used locally after inference for scoring. Spatial arithmetic stays in the frozen executor.

## DEV gate

Sixteen DEV items selected before EVAL: first two items for each supported operation in preserved file order. The user accepted the frozen result: 16 valid queries, 15 exact queries, 16 matching answers; one category-comparison operation confusion, four initial responses requiring retry, and verified checkpoint replay. At most one existing generic schema retry; no prompt tuning or EVAL-driven repair.

- core-cardinal_direction-01: status=success, query exact=True, answer correct=True, calls=2.
- core-cardinal_direction-02: status=success, query exact=True, answer correct=True, calls=2.
- core-closer_of_two-01: status=success, query exact=True, answer correct=True, calls=2.
- core-closer_of_two-02: status=success, query exact=True, answer correct=True, calls=1.
- core-count_within_radius-01: status=success, query exact=True, answer correct=True, calls=1.
- core-count_within_radius-02: status=success, query exact=True, answer correct=True, calls=1.
- core-nearest_category-01: status=success, query exact=True, answer correct=True, calls=1.
- core-nearest_category-02: status=success, query exact=True, answer correct=True, calls=1.
- core-nearest_of_two_categories-01: status=success, query exact=False, answer correct=True, calls=1.
- core-nearest_of_two_categories-02: status=success, query exact=True, answer correct=True, calls=1.
- core-spatial_multi_constraint-01: status=success, query exact=True, answer correct=True, calls=2.
- core-spatial_multi_constraint-02: status=success, query exact=True, answer correct=True, calls=1.
- core-two_hop_nearest-01: status=success, query exact=True, answer correct=True, calls=1.
- core-two_hop_nearest-02: status=success, query exact=True, answer correct=True, calls=1.
- core-within_radius_yes_no-01: status=success, query exact=True, answer correct=True, calls=1.
- core-within_radius_yes_no-02: status=success, query exact=True, answer correct=True, calls=1.

## Held-out results

| System | Intent | Query exact match | End-to-end answer |
|---|---:|---:|---:|
| Frozen Neural Short (all 128) | 23.44% | 20.31% (26/128) | 23.44% (30/128) |
| Long (completed subset, N=128) | 75.78 | 72.66 | 74.22 |

Long percentages and paired counts below refer only to completed EVAL examples. Zero completed examples means unmeasured accuracy.

| Level | Both correct | Short only | Long only | Both wrong | Rescue | Regression |
|---|---:|---:|---:|---:|---:|---:|
| query_exact | 19 | 7 | 74 | 28 | 72.55 | 26.92 |
| answer_correct | 22 | 8 | 73 | 25 | 74.49 | 26.67 |

Rescue = Long-only / all Short-wrong; regression = Short-only / all Short-correct. Abstentions count as incorrect. Query EM compares the complete canonical StructuredQuery, preserving exact entity text and order. Answer matching alone can hide wrong queries.

Long answers matching despite a wrong query: 2.

## Validation and failures

Full strict validation: 108/128 (84.38%). Retried examples: 34 (26.56%); retry calls: 34. Invalid responses across attempts: 54; final invalid-output examples: 20. Abstention/failure examples (non-success status): 23.

Invented entity references across all EVAL attempts: 8, affecting 3 examples. Names are checked against the supplied anchor/candidate names; repeated fabricated references are counted separately. Final raw operation-label accuracy before validation: 102/128 (79.69%). Main intent accuracy above requires an accepted query; rejected outputs count as wrong.

Status counts: `{"ambiguous": 3, "invalid_output": 20, "success": 105}`.

## per_task

| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |
|---|---:|---:|---:|---:|---:|---:|---:|
| cardinal_direction | 16 | 25.00 | 87.50 | 25.00 | 87.50 | 100.00 | 100.00 |
| closer_of_two | 16 | 43.75 | 81.25 | 43.75 | 81.25 | 66.67 | 66.67 |
| count_within_radius | 16 | 6.25 | 62.50 | 6.25 | 62.50 | 66.67 | 66.67 |
| nearest_category | 16 | 6.25 | 75.00 | 6.25 | 81.25 | 73.33 | 80.00 |
| nearest_of_two_categories | 16 | 12.50 | 56.25 | 31.25 | 62.50 | 64.29 | 72.73 |
| spatial_multi_constraint | 16 | 6.25 | 68.75 | 12.50 | 68.75 | 66.67 | 64.29 |
| two_hop_nearest | 16 | 56.25 | 62.50 | 56.25 | 62.50 | 42.86 | 42.86 |
| within_radius_yes_no | 16 | 6.25 | 87.50 | 6.25 | 87.50 | 86.67 | 86.67 |

## per_register

| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |
|---|---:|---:|---:|---:|---:|---:|---:|
| conversational | 40 | 17.50 | 77.50 | 20.00 | 80.00 | 78.79 | 81.25 |
| indirect_compositional | 16 | 6.25 | 68.75 | 12.50 | 68.75 | 66.67 | 71.43 |
| light_saudi | 24 | 12.50 | 70.83 | 12.50 | 70.83 | 71.43 | 71.43 |
| natural_msa | 48 | 31.25 | 70.83 | 35.42 | 72.92 | 69.70 | 70.97 |

## per_complexity

| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 24 | 29.17 | 75.00 | 29.17 | 75.00 | 76.47 | 76.47 |
| L2 | 40 | 17.50 | 77.50 | 20.00 | 77.50 | 78.79 | 78.12 |
| L3 | 32 | 28.12 | 71.88 | 31.25 | 78.12 | 69.57 | 77.27 |
| L4 | 32 | 9.38 | 65.62 | 15.62 | 65.62 | 65.52 | 66.67 |

## Runtime and preservation

Observed GPUs: ['NVIDIA L4']. Model loading times and actual runtime versions are recorded per session in `long_metrics.json`. Total measured session wall time: 1442.14 s; image build and local implementation are outside this timer.

CORE EVAL session wall time only: 965.21 s. EVAL generation calls: 162, including 34 retry calls. Frozen Short mean CPU pipeline latency: 0.58 ms.

EVAL inference latency including retries: mean 5979.56 ms, P50 5038.46 ms, P95 9993.80 ms. This uses saved remote measurements, so resumed cache replay does not create artificially fast inference times. Round-trip timings are also saved; replay after a lost response can understate network time.

All DEV/EVAL calls: 362639 input and 18917 output tokens. EVAL alone: 323924 input and 16853 output tokens. Long inference / frozen Short pipeline mean ratio: 10358.27. Timing scopes differ: Short is a saved local CPU pipeline measurement, Long is remote GPU generation including tokenization but excluding network and local Geo execution. This is not a controlled hardware speed comparison. Monetary cost is unavailable; no estimated dollar cost is reported.

CPU-only weight preparation reuses `asar-hf-cache`. Raw generations are atomically persisted and committed in `aasr-step2b-long-checkpoints` before returning; local `modal_calls` and `modal_predictions` preserve progress. Batch size one avoids changing outputs through batching and permits immediate per-call checkpoints. No remote automatic retry, alternate GPU, model substitution or concurrent GPU containers. A 45-minute session guard stops further requests. A process-local file lock prevents duplicate launches from this workspace.

Resume: `modal run --profile joudalrubaish -m language.modal_long_backend --no-smoke-only` from the repository root. Completed predictions and generation calls are reused. The accepted DEV records and inference implementation are pinned by `core_eval_freeze.json`.

## Router opportunity and next step

**ROUTER PROMISING BUT NEEDS MORE DATA**

Long provides measured net exact-query rescue on this fixed challenge, but an oracle paired result does not establish a deployable router. The 128 synthetic EVAL questions are now evaluation evidence and must not be used to select thresholds. Use new TRAIN-derived/DEV paired outcomes and an independently reviewed held-out set for a future router study.

Potential inference signals remain Short confidence, query validation, missing slots, identity ambiguity and question length. Saved Short confidence/status are available in the comparison CSV. Entropy and top-two margin were not saved, so no values or thresholds are invented. Gold task/difficulty, language register and assigned complexity are analysis strata only. No Router has been implemented.

Evidence: `modal_experiment.json`, `modal_predictions/`, `modal_calls/`, `modal_sessions/`, `long_metrics.json`, `short_vs_long.csv`, and the three breakdown CSVs. Historical OpenRouter preflight files are retained as provenance; this run never uses OpenRouter or its credential.
