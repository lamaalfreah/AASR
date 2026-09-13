**Step 2B — Frozen CORE EVAL: 128/128 completed**

The checkpoint contains all 128 unique CORE EVAL examples and zero remaining examples. The original attached Modal run completed in one GPU container and stopped. No inference was restarted when progress was checked. All results below refer to CORE EVAL only, excluding DEV.

**Measured Long results**

| Metric | Count | Rate |
|---|---|---|
| Operation accuracy, accepted query | 97 | 75.78% |
| StructuredQuery exact match | 93 | 72.66% |
| Final Geo Engine answer accuracy | 95 | 74.22% |
| Full strict query validation | 108 | 84.38% |
| Examples requiring the existing retry | 34 | 26.56% |
| Final invalid-output examples | 20 | 15.62% |
| Abstention/failure: non-success status | 23 | 17.97% |

Raw final operation-label accuracy before validation was 102/128 (79.69%). The primary operation metric counts rejected queries as incorrect. Strict validation includes transport fields, operation constraints, and entity grounding; Geo Engine ambiguity can occur after a query passes validation.

128 initial calls plus 34 permitted retry calls = **162 EVAL generation calls**. There were 54 invalid responses across all attempts; 14 examples recovered on retry, and 20 remained invalid. Final statuses: 105 success, 20 invalid_output, and 3 ambiguous. There were zero explicit needs_clarification/unsupported abstentions. A success status means execution completed, not necessarily that the answer was correct.

Entity audit found **8 ungrounded entity-reference occurrences across all attempts, affecting 3 examples**. Repeated references across retries count separately. These were checked against supplied anchor/candidate names. Two answers matched despite an incorrect StructuredQuery.

**Frozen Neural Short versus Long**

| System | Query exact | Final answer |
|---|---|---|
| Frozen Neural Short | 26/128 (20.31%) | 30/128 (23.44%) |
| Frozen Long | 93/128 (72.66%) | 95/128 (74.22%) |

| Classification | StructuredQuery | Final answer |
|---|---|---|
| Short correct / Long correct | 19 | 22 |
| Short correct / Long wrong | 7 | 8 |
| Short wrong / Long correct | 74 | 73 |
| Short wrong / Long wrong | 28 | 25 |

| Metric | StructuredQuery | Final answer |
|---|---|---|
| Long rescue | 74/102 = 72.55% | 73/98 = 74.49% |
| Long regression | 7/26 = 26.92% | 8/30 = 26.67% |

Rescue uses all Short-wrong examples as its denominator. Regression uses all Short-correct examples. Every EVAL example has explicit four-way labels in `core_eval_comparison.csv`. Frozen Short predictions and latency were reused; Neural Short was not rerun.

**Breakdowns**

Accuracy entries below are percentages. Short/Long query exactness and answer accuracy are paired on the same examples. Task, register, and complexity are post-inference analysis labels only.

**Task accuracy**

| Group | N | Long operation | Short query | Long query | Short answer | Long answer |
|---|---|---|---|---|---|---|
| cardinal_direction | 16 | 87.50 | 25.00 | 87.50 | 25.00 | 87.50 |
| closer_of_two | 16 | 81.25 | 43.75 | 81.25 | 43.75 | 81.25 |
| count_within_radius | 16 | 68.75 | 6.25 | 62.50 | 6.25 | 62.50 |
| nearest_category | 16 | 75.00 | 6.25 | 75.00 | 6.25 | 81.25 |
| nearest_of_two_categories | 16 | 62.50 | 12.50 | 56.25 | 31.25 | 62.50 |
| spatial_multi_constraint | 16 | 68.75 | 6.25 | 68.75 | 12.50 | 68.75 |
| two_hop_nearest | 16 | 75.00 | 56.25 | 62.50 | 56.25 | 62.50 |
| within_radius_yes_no | 16 | 87.50 | 6.25 | 87.50 | 6.25 | 87.50 |

**Task rescue and regression**

| Group | Query rescues | Query rescue % | Query regression % | Answer rescues | Answer rescue % | Answer regression % |
|---|---|---|---|---|---|---|
| cardinal_direction | 12 | 100.00 | 50.00 | 12 | 100.00 | 50.00 |
| closer_of_two | 6 | 66.67 | 0.00 | 6 | 66.67 | 0.00 |
| count_within_radius | 10 | 66.67 | 100.00 | 10 | 66.67 | 100.00 |
| nearest_category | 11 | 73.33 | 0.00 | 12 | 80.00 | 0.00 |
| nearest_of_two_categories | 9 | 64.29 | 100.00 | 8 | 72.73 | 60.00 |
| spatial_multi_constraint | 10 | 66.67 | 0.00 | 9 | 64.29 | 0.00 |
| two_hop_nearest | 3 | 42.86 | 22.22 | 3 | 42.86 | 22.22 |
| within_radius_yes_no | 13 | 86.67 | 0.00 | 13 | 86.67 | 0.00 |

**Language register accuracy**

| Group | N | Long operation | Short query | Long query | Short answer | Long answer |
|---|---|---|---|---|---|---|
| conversational | 40 | 80.00 | 17.50 | 77.50 | 20.00 | 80.00 |
| indirect_compositional | 16 | 68.75 | 6.25 | 68.75 | 12.50 | 68.75 |
| light_saudi | 24 | 75.00 | 12.50 | 70.83 | 12.50 | 70.83 |
| natural_msa | 48 | 75.00 | 31.25 | 70.83 | 35.42 | 72.92 |

**Language register rescue and regression**

| Group | Query rescues | Query rescue % | Query regression % | Answer rescues | Answer rescue % | Answer regression % |
|---|---|---|---|---|---|---|
| conversational | 26 | 78.79 | 28.57 | 26 | 81.25 | 25.00 |
| indirect_compositional | 10 | 66.67 | 0.00 | 10 | 71.43 | 50.00 |
| light_saudi | 15 | 71.43 | 33.33 | 15 | 71.43 | 33.33 |
| natural_msa | 23 | 69.70 | 26.67 | 22 | 70.97 | 23.53 |

**Complexity accuracy**

| Group | N | Long operation | Short query | Long query | Short answer | Long answer |
|---|---|---|---|---|---|---|
| L1 | 24 | 79.17 | 29.17 | 75.00 | 29.17 | 75.00 |
| L2 | 40 | 80.00 | 17.50 | 77.50 | 20.00 | 77.50 |
| L3 | 32 | 75.00 | 28.12 | 71.88 | 31.25 | 78.12 |
| L4 | 32 | 68.75 | 9.38 | 65.62 | 15.62 | 65.62 |

**Complexity rescue and regression**

| Group | Query rescues | Query rescue % | Query regression % | Answer rescues | Answer rescue % | Answer regression % |
|---|---|---|---|---|---|---|
| L1 | 13 | 76.47 | 28.57 | 13 | 76.47 | 28.57 |
| L2 | 26 | 78.79 | 28.57 | 25 | 78.12 | 25.00 |
| L3 | 16 | 69.57 | 22.22 | 17 | 77.27 | 20.00 |
| L4 | 19 | 65.52 | 33.33 | 18 | 66.67 | 40.00 |

Breakdown CSVs additionally contain four-way counts, strict validity, retries, failures, and invented-reference counts. Small subgroup denominators can produce large regression percentages.

**Performance**

| Measurement | Observed |
|---|---|
| Model | Qwen/Qwen3-4B, revision 1cfa9a7208912126459214e8b04321603b3df60c |
| Cache | Existing asar-hf-cache; offline local snapshot; no model download |
| GPU/precision | Exactly one NVIDIA L4; BF16; one observed container session |
| Model load | 5.23 seconds |
| Total EVAL session wall time | 965.21 seconds (16 min 5.21 sec) |
| Mean inference per example, including retries | 5.980 seconds |
| P50 inference | 5.038 seconds |
| P95 inference | 9.994 seconds |
| Mean local-to-remote request time, including retries | 7.275 seconds |
| EVAL calls / retry calls | 162 / 34 |
| Frozen Short mean CPU pipeline latency | 0.577 ms |
| Long inference / Short pipeline latency ratio | 10,358x |
| EVAL input / output tokens | 323,924 / 16,853 |
| Modal monetary cost | Not reliably available from the recorded API responses; no estimate |

Wall time is measured from entry into the evaluation harness through its final example, including cache verification, container startup/load and checkpoint requests. It excludes earlier DEV work, CLI/app initialization, and final report rendering. Inference timings include remote tokenization and generation, including retries, but exclude networking and local Geo Engine execution. Short latency is a frozen local CPU pipeline measurement. The ratio is descriptive across different hardware and timing scopes, not a controlled speed benchmark.

**Verification and preservation**

All 128 saved predictions and 162 unique generation checkpoints were replayed locally with a provider that raises if any remote generation is requested. Replay reproduced the saved query, status, retry counts, and scores with zero additional generations; checkpoint bytes remained unchanged. All 128 EVAL IDs were present exactly once and were disjoint from DEV. The 162 responses identify one L4 container session. Remote checkpoint durability and reload had already been verified during accepted DEV; no new GPU was started for final verification.

All 18 original artifact checks and the additional CORE EVAL freeze record pass. The accepted prompt, inference code, generation configuration, StructuredQuery schema, Geo Engine, Neural Short artifacts, challenge files, and DEV records remain unchanged since EVAL authorization. No TEST gold was accessed. No Router was implemented; no push or merge occurred.

**ROUTER PROMISING BUT NEEDS MORE DATA**

Long rescued **74/102 Short-wrong queries (72.55%)** and **73/98 Short-wrong answers (74.49%)**, exceeding its 7 query regressions and 8 answer regressions. This is measured complementary value, with net gains of 67 exact queries and 65 correct answers over Short. However, regressions affect **26.92% of Short-correct queries** and **26.67% of Short-correct answers**. Long also costs about 5.98 seconds of measured inference per question versus Short’s 0.577 ms CPU pipeline, while reliable monetary cost is unavailable.

These paired outcomes establish an opportunity to investigate routing, but they do not demonstrate that a deployable decision rule can identify rescues while avoiding regressions and unnecessary Long calls. The 20 final invalid outputs reinforce this limit. No threshold or prompt is selected from these held-out results. Any future Router must use inference-available signals; gold task, register, and complexity labels are excluded as inference features.

Evidence: `long_metrics.json`, `core_eval_comparison.csv`, `per_task.csv`, `per_register.csv`, `per_complexity.csv`, `core_eval_verification.json`, `core_eval_freeze.json`, and the immutable prediction/generation checkpoints.

Modal run: https://modal.com/apps/joudalrubaish/main/ap-fEGA57H7lFLtKJtTH0m1vW
