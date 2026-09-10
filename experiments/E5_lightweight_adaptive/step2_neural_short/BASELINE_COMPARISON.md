# Rule Baseline 0 versus lightweight Neural Short

This report implements the revised experimental roles using completed results only. No model retraining, dataset regeneration, repeated evaluation, provider calls, or frozen-engine edits occurred during consolidation.

The existing 320 CORE questions are now designated the **Hard Natural Arabic Challenge Set**. All questions, IDs, and split assignments remain unchanged. The old Step-2 label “Short” referred to the frozen rule/template parser; interpret that historical result as **Rule Baseline 0**, not the adaptive neural Short path. Its 0/320 result remains preserved as evidence of template dependence.

## Compared systems

**A — Rule Baseline 0:** unchanged Step-1 eight-template parser, zero learned parameters, deterministic extraction and frozen execution.

**B — Neural Short:** trainable 48-dimensional embeddings → single BiGRU (48 hidden units per direction) → eight-operation classifier → deterministic names/category/radius/direction/hop slots → existing StructuredQuery → unchanged Geo Engine. **32,264 learned parameters**; the neural network never calculates spatial answers. There is no Rule Baseline 0 fallback in neural inference.

The model trained only on original TRAIN material: 12,387 fitting examples and 1,458 internal-holdout examples with disjoint anchors. All 320 challenge source anchors, accounting for 5,752 original TRAIN rows, were excluded from both. CORE DEV and EVAL were not used for fitting, vocabulary construction, or checkpoint selection. Although DEV training is now permitted, this report preserves the already completed experiment; it does not retrain after viewing EVAL.

## Overall results

| Set | N | Rule intent | Rule query EM | Rule answer | Neural intent | Neural query EM | Neural answer |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_validation | 2502 | 100.00% | 100.00%* | 95.24% | 100.00% | 99.92% | 95.20% |
| core_dev | 192 | 0.00% | 0.00% | 0.00% | 22.40% | 20.83% | 22.92% |
| core_eval | 128 | 0.00% | 0.00% | 0.00% | 23.44% | 20.31% | 23.44% |

*Original-validation query reference is the audited Rule Baseline 0 output. Its 100% query agreement is true by definition and is not an independently measured semantic score. The Neural Short's 99.92% measures agreement against that reference. CORE query scores use the existing preserved source-query labels. Original Rule intent accuracy was measured against task annotations; CORE Rule intent accuracy is zero because it produced no parse.

End-to-end answer matching counts abstentions as incorrect. Four DEV and four EVAL answers match gold despite a wrong neural intent/query. Therefore distinguish the neural model's 30/128 matching EVAL answers from its **26/128 exact EVAL queries**.

## Comparison by task

All figures are percentages. Query EM for original-validation Rule Baseline 0 is reference identity, as explained above.

| Set | Group | N | Rule intent | Rule query EM | Rule answer | Neural intent | Neural query EM | Neural answer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original_validation | cardinal_direction | 409 | 100.00 | 100.00 | 91.93 | 100.00 | 100.00 | 91.93 |
| original_validation | closer_of_two | 283 | 100.00 | 100.00 | 74.20 | 100.00 | 100.00 | 74.20 |
| original_validation | count_within_radius | 97 | 100.00 | 100.00 | 86.60 | 100.00 | 97.94 | 85.57 |
| original_validation | nearest_category | 827 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| original_validation | nearest_of_two_categories | 205 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| original_validation | spatial_multi_constraint | 191 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| original_validation | two_hop_nearest | 210 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| original_validation | within_radius_yes_no | 280 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| core_dev | cardinal_direction | 24 | 0.00 | 0.00 | 0.00 | 25.00 | 25.00 | 25.00 |
| core_dev | closer_of_two | 24 | 0.00 | 0.00 | 0.00 | 25.00 | 25.00 | 25.00 |
| core_dev | count_within_radius | 24 | 0.00 | 0.00 | 0.00 | 12.50 | 12.50 | 12.50 |
| core_dev | nearest_category | 24 | 0.00 | 0.00 | 0.00 | 4.17 | 4.17 | 4.17 |
| core_dev | nearest_of_two_categories | 24 | 0.00 | 0.00 | 0.00 | 25.00 | 25.00 | 37.50 |
| core_dev | spatial_multi_constraint | 24 | 0.00 | 0.00 | 0.00 | 16.67 | 16.67 | 20.83 |
| core_dev | two_hop_nearest | 24 | 0.00 | 0.00 | 0.00 | 58.33 | 45.83 | 45.83 |
| core_dev | within_radius_yes_no | 24 | 0.00 | 0.00 | 0.00 | 12.50 | 12.50 | 12.50 |
| core_eval | cardinal_direction | 16 | 0.00 | 0.00 | 0.00 | 25.00 | 25.00 | 25.00 |
| core_eval | closer_of_two | 16 | 0.00 | 0.00 | 0.00 | 43.75 | 43.75 | 43.75 |
| core_eval | count_within_radius | 16 | 0.00 | 0.00 | 0.00 | 6.25 | 6.25 | 6.25 |
| core_eval | nearest_category | 16 | 0.00 | 0.00 | 0.00 | 6.25 | 6.25 | 6.25 |
| core_eval | nearest_of_two_categories | 16 | 0.00 | 0.00 | 0.00 | 12.50 | 12.50 | 31.25 |
| core_eval | spatial_multi_constraint | 16 | 0.00 | 0.00 | 0.00 | 6.25 | 6.25 | 12.50 |
| core_eval | two_hop_nearest | 16 | 0.00 | 0.00 | 0.00 | 81.25 | 56.25 | 56.25 |
| core_eval | within_radius_yes_no | 16 | 0.00 | 0.00 | 0.00 | 6.25 | 6.25 | 6.25 |

## Comparison by register

All figures are percentages. Query EM for original-validation Rule Baseline 0 is reference identity, as explained above.

| Set | Group | N | Rule intent | Rule query EM | Rule answer | Neural intent | Neural query EM | Neural answer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original_validation | original_template | 2502 | 100.00 | 100.00 | 95.24 | 100.00 | 99.92 | 95.20 |
| core_dev | conversational | 56 | 0.00 | 0.00 | 0.00 | 17.86 | 17.86 | 19.64 |
| core_dev | indirect_compositional | 16 | 0.00 | 0.00 | 0.00 | 25.00 | 18.75 | 18.75 |
| core_dev | light_saudi | 40 | 0.00 | 0.00 | 0.00 | 15.00 | 15.00 | 20.00 |
| core_dev | natural_msa | 80 | 0.00 | 0.00 | 0.00 | 28.75 | 26.25 | 27.50 |
| core_eval | conversational | 40 | 0.00 | 0.00 | 0.00 | 22.50 | 17.50 | 20.00 |
| core_eval | indirect_compositional | 16 | 0.00 | 0.00 | 0.00 | 12.50 | 6.25 | 12.50 |
| core_eval | light_saudi | 24 | 0.00 | 0.00 | 0.00 | 12.50 | 12.50 | 12.50 |
| core_eval | natural_msa | 48 | 0.00 | 0.00 | 0.00 | 33.33 | 31.25 | 35.42 |

## Comparison by complexity

All figures are percentages. Query EM for original-validation Rule Baseline 0 is reference identity, as explained above.

| Set | Group | N | Rule intent | Rule query EM | Rule answer | Neural intent | Neural query EM | Neural answer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original_validation | original_template | 2502 | 100.00 | 100.00 | 95.24 | 100.00 | 99.92 | 95.20 |
| core_dev | L1 | 72 | 0.00 | 0.00 | 0.00 | 27.78 | 26.39 | 30.56 |
| core_dev | L2 | 56 | 0.00 | 0.00 | 0.00 | 17.86 | 17.86 | 19.64 |
| core_dev | L3 | 32 | 0.00 | 0.00 | 0.00 | 15.62 | 15.62 | 15.62 |
| core_dev | L4 | 32 | 0.00 | 0.00 | 0.00 | 25.00 | 18.75 | 18.75 |
| core_eval | L1 | 24 | 0.00 | 0.00 | 0.00 | 29.17 | 29.17 | 29.17 |
| core_eval | L2 | 40 | 0.00 | 0.00 | 0.00 | 17.50 | 17.50 | 20.00 |
| core_eval | L3 | 32 | 0.00 | 0.00 | 0.00 | 34.38 | 28.12 | 31.25 |
| core_eval | L4 | 32 | 0.00 | 0.00 | 0.00 | 15.62 | 9.38 | 15.62 |

## CPU latency and training

All times below are saved measurements, in milliseconds per example. Training took **9.7954 seconds**, or **11.7409 seconds** including reading/preparation, with two CPU threads. No GPU, pretrained model, or remote model was used.

| Set | Rule mean | Rule P50 | Rule P95 | Neural mean | Neural P50 | Neural P95 |
|---|---:|---:|---:|---:|---:|---:|
| original_validation | 0.175595 | 0.142354 | 0.382987 | 0.652738 | 0.543083 | 0.970448 |
| core_dev | 0.001337 | 0.001250 | 0.001459 | 0.556026 | 0.556020 | 0.848837 |
| core_eval | 0.001355 | 0.001333 | 0.001500 | 0.577274 | 0.559500 | 0.861944 |

These are not controlled simultaneous speed ratios. Rule CORE timing used pre-parsed contexts and measures rapid rejection; Neural Short timing includes context conversion, intent inference, slots, and execution when possible. Original validation timings include context parsing. No benchmark was repeated during consolidation.

## Interpretation and stopping point

The neural Short now has both successes and failures on natural-language questions: 44/192 matching answers on DEV and 30/128 on EVAL, compared with zero for Rule Baseline 0. That establishes a nontrivial neural comparison path, but **does not establish adequate natural-language robustness**. Only 40 DEV and 26 EVAL queries are exactly correct. It also emits 20 incorrect successful DEV answers and 16 incorrect successful EVAL answers, whereas Rule Baseline 0 abstains on all CORE questions.

Its vocabulary has only 68 entries after masking names/numbers. Unseen words account for approximately 40% of DEV and 47% of EVAL tokens. Original-template supervision is insufficient for broad conversational language. Scores are not monotonic across L1–L4, so there is no evidence yet that it reliably succeeds on all easy questions and fails only on hard ones.

Preserve this checkpoint and its EVAL predictions. Any subsequent refinement should use TRAIN-derived material and DEV, with the current observed EVAL protected from tuning; use a new independently reviewed held-out set for further development decisions if needed.

**Long remains paused and the Router is not implemented.** A later separately authorized Long comparison should use these neural Short outcomes, retain all four paired outcome classes, and report both exact-query and answer-level rescue/regression. No Long benefit or Router justification is claimed here.

## Evidence and preservation

- [Neural implementation/training report](README.md), [saved neural metrics](metrics.json), [training metrics](training_metrics.json).
- [Rule Baseline 0 historical metrics](../step2_short_long/results/short_metrics.json).
- [Overall machine-readable comparison](baseline_comparison.csv), [by task](comparison_by_task.csv), [by register](comparison_by_register.csv), [by complexity](comparison_by_complexity.csv).
- [Challenge designation](../step2_short_long/challenge/designation.json).

The previously completed suite passed 66 tests. This consolidation checked saved aggregates and file hashes; it did not rerun completed tests. Frozen runtime, trained checkpoint, and all four challenge/context hashes match their evaluated versions. Raw data and TEST gold were not accessed.
