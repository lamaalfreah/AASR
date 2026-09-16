**Final Router V2 TEST evaluation — all 2,332 examples**

Frozen calibrated Small MLP ensemble; LONG iff probability >= 0.91. Original TEST questions, answers and split retained. No tuning, retraining, filtering, or repairs after TEST access.

| Policy | Correct / 2332 | Accuracy |
|---|---|---|
| Always Short | 2212 | 94.85% |
| Always Long | 2026 | 86.88% |
| Router V2 | 2119 | 90.87% |
| Answer oracle | 2213 | 94.90% |

| Metric | Result |
|---|---|
| short_route_rate | 47.64% |
| long_route_rate | 52.36% |
| routing_accuracy | 44.87% |
| long_precision | 0.00% |
| long_recall | 0.00% |

Short/Long answer-route counts: 1111/1221. Short inference always runs to provide Router features. Long is an additional call on LONG routes. Rescues: 0; regressions: 93; unnecessary Long calls: 1219.

Routing metrics use 2213 non-FAILURE rows, with 119 both-answer-wrong rows analyzed separately. TEST lacks native gold StructuredQueries, so routing labels use final-answer correctness rather than the development exact-query-plus-answer criterion. Confusion: `{"short_to_short": 993, "short_to_long": 1219, "long_to_short": 1, "long_to_long": 0}`.

**Latency and computation**

| Component/policy | Mean ms | P50 ms | P95 ms |
|---|---|---|---|
| Always Short (parser + Geo) | 2.303 | 2.200 | 3.029 |
| Short with Router features + Geo | 2.423 | 2.321 | 3.145 |
| Router decision | 0.987 | 0.935 | 1.117 |
| Always Long (inference + Geo) | 6909.598 | 5879.179 | 11494.246 |
| Projected adaptive | 3240.965 | 5263.853 | 7911.066 |

Long generation calls: 2739, including 407 retries. Adaptive-only generation calls would be 1306. Avoided Long inference: 8563.17 seconds (53.14%). GPU: NVIDIA L4; BF16, existing asar-hf-cache only. Monetary cost unavailable.

Adaptive latency is a projection from this evaluation’s component timings. It excludes cold start, idle billing and full RPC overhead. Actual evaluation runs Long on all TEST rows for baselines/oracle; savings apply to hypothetical adaptive-only inference.

**Task breakdown**

| Task | N | Short | Long | Router V2 | Oracle | Long route rate |
|---|---|---|---|---|---|---|
| cardinal_direction | 397 | 87.41% | 86.65% | 87.41% | 87.66% | 0.00% |
| closer_of_two | 264 | 77.27% | 75.76% | 77.27% | 77.27% | 0.00% |
| count_within_radius | 89 | 92.13% | 41.57% | 92.13% | 92.13% | 0.00% |
| nearest_category | 759 | 99.74% | 95.92% | 95.92% | 99.74% | 100.00% |
| nearest_of_two_categories | 186 | 100.00% | 65.59% | 65.59% | 100.00% | 100.00% |
| spatial_multi_constraint | 173 | 99.42% | 93.64% | 99.42% | 99.42% | 0.00% |
| two_hop_nearest | 188 | 100.00% | 83.51% | 100.00% | 100.00% | 0.00% |
| within_radius_yes_no | 276 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

**Native difficulty breakdown**

| Native difficulty | N | Short | Long | Router V2 | Oracle | Long route rate |
|---|---|---|---|---|---|---|
| easy | 1432 | 96.37% | 94.13% | 94.34% | 96.44% | 72.28% |
| hard | 361 | 99.72% | 88.37% | 99.72% | 99.72% | 0.00% |
| medium | 539 | 87.57% | 66.60% | 75.70% | 87.57% | 34.51% |

Language register is unavailable in TEST. Native easy/medium/hard difficulty is reported as supplied; CORE complexity labels are not invented. All breakdown labels are used only for scoring/reporting.

Short and Long predictions were committed to the existing Modal checkpoint volume before completion was counted. Local Router decisions were atomically checkpointed before Long outcomes were available. Resume skips completed predictions. Full local replay verified all Long checkpoints without new generation.

The original split is disjoint by wikidata anchor ID from Router development data. The pre-existing split audit reports some coordinate aliases across splits; ID-disjointness is not a guarantee of geographic-coordinate separation. Original template wording also differs from Router development paraphrases. No TEST rows were removed.

All frozen component and dataset hashes passed. No CORE rerun, training, tuning, push, or merge. Artifacts: `summary.json`, `per_example.jsonl`, prediction/checkpoint directories, `pre_test_freeze.json`, `evaluation_code_freeze.json`, `final_manifest.json`.

**Stopped after the final TEST report.**
