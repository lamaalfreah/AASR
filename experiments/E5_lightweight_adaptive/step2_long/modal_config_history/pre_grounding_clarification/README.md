# Step 2B — Modal Qwen3-4B versus frozen Neural Short

Status: **smoke_failed**. DEV did not pass exact-query and frozen-engine answer checks; EVAL remains untouched

**0/128 CORE EVAL examples completed.** 11 durable generation calls across DEV/EVAL, including schema retries.

The official `Qwen/Qwen3-4B` weights are used without adapters, quantization or training. One L4 container is reused; bfloat16 and non-thinking greedy generation are fixed before EVAL. The existing prompt, schema, Geo Engine, Rule Baseline 0, Neural Short checkpoint/predictions and 320 questions remain frozen.

Only question, exact visible candidate names, anchor name and category vocabulary reach the model. Gold labels and challenge metadata are used locally after inference for scoring. Spatial arithmetic stays in the frozen executor.

## DEV gate

Eight DEV items selected before EVAL: first item for each supported operation in preserved file order. All must produce a validated query, exact reference query and matching answer through the frozen executor. At most one existing generic schema retry; no prompt tuning or EVAL-driven repair.

- core-cardinal_direction-01: status=success, query exact=True, answer correct=True, calls=1.
- core-closer_of_two-01: status=success, query exact=True, answer correct=True, calls=2.
- core-count_within_radius-01: status=invalid_output, query exact=False, answer correct=False, calls=2.
- core-nearest_category-01: status=success, query exact=True, answer correct=True, calls=1.
- core-nearest_of_two_categories-01: status=invalid_output, query exact=False, answer correct=False, calls=2.
- core-spatial_multi_constraint-01: status=success, query exact=True, answer correct=True, calls=1.
- core-two_hop_nearest-01: status=success, query exact=False, answer correct=False, calls=1.
- core-within_radius_yes_no-01: status=success, query exact=True, answer correct=True, calls=1.

## Held-out results

| System | Intent | Query exact match | End-to-end answer |
|---|---:|---:|---:|
| Frozen Neural Short (all 128) | 23.44% | 20.31% (26/128) | 23.44% (30/128) |
| Long (completed subset, N=0) | unmeasured | unmeasured | unmeasured |

Long percentages and paired counts below refer only to completed EVAL examples. Zero completed examples means unmeasured accuracy.

| Level | Both correct | Short only | Long only | Both wrong | Rescue | Regression |
|---|---:|---:|---:|---:|---:|---:|
| query_exact | 0 | 0 | 0 | 0 | unmeasured | unmeasured |
| answer_correct | 0 | 0 | 0 | 0 | unmeasured | unmeasured |

Rescue = Long-only / all Short-wrong; regression = Short-only / all Short-correct. Abstentions count as incorrect. Query EM compares the complete canonical StructuredQuery, preserving exact entity text and order. Answer matching alone can hide wrong queries.

Long answers matching despite a wrong query: 0.

## per_task

| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |
|---|---:|---:|---:|---:|---:|---:|---:|

## per_register

| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |
|---|---:|---:|---:|---:|---:|---:|---:|

## per_complexity

| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |
|---|---:|---:|---:|---:|---:|---:|---:|

## Runtime and preservation

Observed GPUs: ['NVIDIA L4']. Model loading times and actual runtime versions are recorded per session in `long_metrics.json`. Total measured session wall time: 289.77 s; image build and local implementation are outside this timer.

EVAL inference latency including retries: mean unmeasured ms, P50 unmeasured ms, P95 unmeasured ms. This uses saved remote measurements, so resumed cache replay does not create artificially fast inference times. Round-trip timings are also saved; replay after a lost response can understate network time.

All DEV/EVAL calls: 20049 input and 1120 output tokens. EVAL alone: 0 input and 0 output tokens. Long inference / frozen Short pipeline mean ratio: unmeasured. Timing scopes differ: Short is a saved local CPU pipeline measurement, Long is remote GPU generation including tokenization but excluding network and local Geo execution. This is not a controlled hardware speed comparison. Monetary cost is unavailable; no estimated dollar cost is reported.

CPU-only weight preparation reuses `asar-hf-cache`. Raw generations are atomically persisted and committed in `aasr-step2b-long-checkpoints` before returning; local `modal_calls` and `modal_predictions` preserve progress. Batch size one avoids changing outputs through batching and permits immediate per-call checkpoints. No remote automatic retry, alternate GPU, model substitution or concurrent GPU containers. A 45-minute session guard stops further requests. A process-local file lock prevents duplicate launches from this workspace.

Resume: `modal run -m language.modal_long_backend` from the repository root. Completed predictions and generation calls are reused. A failed recorded DEV gate stops again without GPU allocation; changing the experiment requires explicit review.

## Router opportunity and next step

**ROUTER NOT YET JUSTIFIED**

The current results do not establish sufficient measured complementary Long benefit. Resolve the recorded blocker or assess completed errors before any Router work.

Potential inference signals remain Short confidence, query validation, missing slots, identity ambiguity and question length. Saved Short confidence/status are available in the comparison CSV. Entropy and top-two margin were not saved, so no values or thresholds are invented. Gold task/difficulty, language register and assigned complexity are analysis strata only. No Router has been implemented.

Evidence: `modal_experiment.json`, `modal_predictions/`, `modal_calls/`, `modal_sessions/`, `long_metrics.json`, `short_vs_long.csv`, and the three breakdown CSVs. Historical OpenRouter preflight files are retained as provenance; this run never uses OpenRouter or its credential.
