# AASR full dataset audit

Audit-only, local CPU. No model loading, training, GPU, Modal, production executor, application changes, derived training dataset, push, or merge. Raw files remain immutable. Test inputs/metadata and schema shape were structurally inspected; test gold values were excluded from all answer and semantic analyses.

This report consolidates completed scans. Completion reused their saved outputs; it did not rescan the dataset.

## A. Dataset status — READY WITH FIXES

Ready for an ambiguity-aware structured-query and deterministic-spatial baseline. Not ready to claim general Arabic understanding or an advantage from adaptive neural depth. All eight question skeletons are recoverable without a learned classifier.

**Reconciliation:** the interim 21,106 reconstructable / 992 ambiguous / 1 discrepancy counts were refined by the final target-versus-context-anchor identity check. Forty-six train/validation target-name collisions exist; two were already ambiguous. Moving the remaining 44 records gives **21,062 / 1,036 / 1**. See [the exact 44 IDs and reason](reconstructability_reconciliation.json). This changes the conservative identity assessment, not raw data or gold labels.

## B. Exact split counts

| Split | Records |
| --- | --- |
| train | 19,597 |
| validation | 2,502 |
| test | 2,332 |
| Total | 24,431 |
| Train + validation | 22,099 |

All integrity checks below are zero:

| Check | Count |
| --- | --- |
| blank_line | 0 |
| candidate_count_mismatch | 0 |
| context_chars_mismatch | 0 |
| duplicate_id_extra_records | 0 |
| invalid_text_input | 0 |
| malformed_json | 0 |
| missing_id | 0 |
| non_object_json | 0 |
| schema_key_set_mismatch | 0 |
| split_field_mismatch | 0 |
| unexpected_field_type | 0 |
| unexpected_task | 0 |

All records agree with their containing split file. Counts were reproduced across completed runs. Raw before/after SHA-256 checks matched.

## C. Task distribution

| Task | Train | Validation | Test | Total |
| --- | --- | --- | --- | --- |
| nearest_category | 6,588 | 827 | 759 | 8,174 |
| cardinal_direction | 3,267 | 409 | 397 | 4,073 |
| within_radius_yes_no | 2,202 | 280 | 276 | 2,758 |
| closer_of_two | 2,167 | 283 | 264 | 2,714 |
| count_within_radius | 701 | 97 | 89 | 887 |
| nearest_of_two_categories | 1,591 | 205 | 186 | 1,982 |
| two_hop_nearest | 1,603 | 210 | 188 | 2,001 |
| spatial_multi_constraint | 1,478 | 191 | 173 | 1,842 |

Largest/smallest task ratios: train **9.40:1**, validation **8.53:1**, test **8.53:1**. Nothing was rebalanced.

## D. Difficulty distribution

| Difficulty | Train | Validation | Test |
| --- | --- | --- | --- |
| easy | 12,057 | 1,516 | 1,432 |
| medium | 4,459 | 585 | 539 |
| hard | 3,081 | 401 | 361 |

Task-to-difficulty mapping is exact: first three tasks Easy; closer/count/two-category tasks Medium; two-hop/multi-constraint Hard. No crossovers. Train Easy:Hard = 3.91:1. Construction difficulty scores are 0.25, 0.45, 0.52, 0.55, 0.78, and 0.82, determined by task. Recommended reasoning is short/mixed/long by difficulty. [Full cross-tab](task_difficulty_distribution.csv).

## E. Schema findings

Exactly **34 fields**, each present in **100%** of records. No extra fields, key-set differences, or unexpected types. Types/null rates below include structural inspection of test fields, without semantic use of test gold values.

| Field | Observed types | Presence % | Null count | Null % |
| --- | --- | --- | --- | --- |
| anchor_latitude | float | 100 | 0 | 0.00 |
| anchor_longitude | float | 100 | 0 | 0.00 |
| anchor_place | str | 100 | 0 | 0.00 |
| answer | str | 100 | 0 | 0.00 |
| answer_position | null, str | 100 | 21,717 | 88.89 |
| article_chars | int | 100 | 0 | 0.00 |
| context | str | 100 | 0 | 0.00 |
| context_chars | int | 100 | 0 | 0.00 |
| context_length_bin_chars | str | 100 | 0 | 0.00 |
| country | str | 100 | 0 | 0.00 |
| difficulty | str | 100 | 0 | 0.00 |
| difficulty_score | float | 100 | 0 | 0.00 |
| direction_constraint | null, str | 100 | 22,589 | 92.46 |
| evidence_poi_ids | list | 100 | 0 | 0.00 |
| first_hop_category | null, str | 100 | 22,430 | 91.81 |
| id | str | 100 | 0 | 0.00 |
| intermediate_answer | null, str | 100 | 22,430 | 91.81 |
| language | str | 100 | 0 | 0.00 |
| long_target | str | 100 | 0 | 0.00 |
| num_context_pois | int | 100 | 0 | 0.00 |
| num_qualifying_pois | float, null | 100 | 22,589 | 92.46 |
| question | str | 100 | 0 | 0.00 |
| radius_constraint_km | float, null | 100 | 22,589 | 92.46 |
| recommended_reasoning | str | 100 | 0 | 0.00 |
| router_target_alpha | null | 100 | 24,431 | 100.00 |
| second_hop_category | null, str | 100 | 22,430 | 91.81 |
| short_target | str | 100 | 0 | 0.00 |
| sources | list | 100 | 0 | 0.00 |
| split | str | 100 | 0 | 0.00 |
| target_category | null, str | 100 | 22,589 | 92.46 |
| task_type | str | 100 | 0 | 0.00 |
| verified_rationale | str | 100 | 0 | 0.00 |
| wikidata_id | str | 100 | 0 | 0.00 |
| wikipedia_title | str | 100 | 0 | 0.00 |

Country is المملكة العربية السعودية, language ar, and sources are OpenStreetMap / Arabic Wikipedia / Wikidata. Candidate source IDs are not present in the visible registry. List elements are strings. Full split-level type counts and task dependence are in [schema_summary.json](schema_summary.json) and [schema_by_task.csv](schema_by_task.csv).

## F. Missing/null findings

No missing fields, empty strings, empty lists, or malformed JSON records were observed. Nullable annotations are deliberate task-dependent fields:

- `router_target_alpha`: null in all 24,431 records.
- `answer_position`: populated only for 2,714 closer-of-two records.
- `intermediate_answer`, `first_hop_category`, `second_hop_category`: populated only for 2,001 two-hop records.
- `target_category`, `direction_constraint`, `radius_constraint_km`, `num_qualifying_pois`: populated only for 1,842 multi-constraint records.
- Other fields are non-null.

A null radius annotation does not mean the question lacks a radius.

## G. Candidate parsing coverage

**24,431/24,431 contexts parsed completely**, recovering **1,553,153 candidate occurrences**. All parsed counts match `num_context_pois`; all context lengths match `context_chars`. The parser preserves ordered, distinct candidate rows.

| Split | Parsed records | Candidate occurrences | Multiline candidate occurrences |
| --- | --- | --- | --- |
| train | 19,597 | 1,253,945 | 4,383 |
| validation | 2,502 | 155,061 | 736 |
| test | 2,332 | 144,147 | 845 |

Candidate blocks use numbered records, `الاسم`, `النوع`, `خط العرض`, and `خط الطول`, separated by Arabic semicolons. Multiline names occur in **2,056 records**; 5,964 candidate occurrences contain name newlines. Edge whitespace occurs in names in **1,604 records**. Preserve literal names, including trailing whitespace, with separate normalization keys if later required. A line-per-candidate assumption and indiscriminate trimming are unsafe.

## H. Candidate-count statistics

| Scope | Min | Mean | P50 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1 | 63.57 | 77 | 100 | 103 | 111 | 114 |
| train | 1 | 63.99 | 77 | 101 | 103 | 111 | 114 |
| validation | 1 | 61.97 | 72 | 98 | 102 | 107 | 113 |
| test | 1 | 61.81 | 79 | 99 | 101 | 111.14 | 113 |

These are per-record statistics; repeated contexts are intentionally counted for each question.

## I. Category findings

Fourteen exact Arabic category labels occur in every split. No unexpected category strings, split-exclusive categories, or Arabic/English category mixtures. Entity names themselves can be multilingual.

| Category | Occurrences |
| --- | --- |
| بنك | 139,648 |
| جامعة | 68,456 |
| روضة أطفال | 25,303 |
| صراف آلي | 59,969 |
| صيدلية | 136,745 |
| عيادة | 113,815 |
| كلية | 65,330 |
| محطة وقود | 134,832 |
| مدرسة | 101,881 |
| مستشفى | 151,466 |
| مطعم | 160,506 |
| مطعم وجبات سريعة | 154,969 |
| مقهى | 146,022 |
| مكان عبادة | 94,211 |

Distribution denominator: candidate occurrences across all records, not globally distinct POIs. [Counts by split](category_distribution.csv); unique-registry counts are also saved. Preserve restaurant versus fast food, bank versus ATM, university versus college, and school versus kindergarten. Any future taxonomy aliases must be explicit and evaluated separately; none were merged.

## J. Duplicate-name findings

| Finding | Train records | Validation records | Test records | Total |
| --- | --- | --- | --- | --- |
| Repeated names / same name at different coordinates | 16,737 | 2,065 | 1,974 | 20,776 |
| Repeated coordinates | 464 | 0 | 109 | 573 |
| Repeated complete candidate rows | 358 | 0 | 69 | 427 |
| Name edge whitespace | 1,227 | 185 | 192 | 1,604 |

Repeated names appear in **85.04%** of records; that alone does not make a query ambiguous. Geometry can uniquely select among identically named candidates. No deduplication was applied. The registry lacks stable visible OSM IDs, so a future derived candidate identity should include record scope and ordinal while preserving provenance.

## K. Entity ambiguity counts

The following is **literal named-candidate lookup ambiguity**, distinct from additional context-anchor collisions and nearest-distance ties. NOT FOUND is zero after exact name preservation.

| Split | UNIQUE | AMBIGUOUS | Ambiguous % of split | NOT FOUND | NOT APPLICABLE |
| --- | --- | --- | --- | --- | --- |
| train | 5,262 | 856 | 4.37% | 0 | 13,479 |
| validation | 675 | 111 | 4.44% | 0 | 1,716 |
| test | 640 | 107 | 4.59% | 0 | 1,585 |
| overall | 6,577 | 1,074 | 4.40% | 0 | 16,780 |

| Task with named candidates | Unique | Ambiguous | Ambiguous % of task | N/A |
| --- | --- | --- | --- | --- |
| cardinal_direction | 3691 | 382 | 9.38% | 0 |
| closer_of_two | 2092 | 622 | 22.92% | 0 |
| count_within_radius | 794 | 70 | 7.89% | 23 |

The other five tasks choose candidates by category/geometry and have no named target lookup. There are separately **23 count-origin/context-anchor collisions** and **54 named-target/context-anchor collisions** across all input splits. These flags can overlap other ambiguity flags; do not add totals blindly.

Example: train `e2f8f921d92cf06f` asks for the direction of الأبواء relative to الأبواء, while the main anchor and same-named candidate have different coordinates. A unique match within the candidate list does not resolve that public-input conflict. Gold evidence IDs must not select a hidden intended referent.

## L. Coordinate issues

No missing, nonnumeric, nonfinite, or out-of-range candidate coordinates; no invalid anchor coordinates were detected. Visible coordinates are rounded relative to some metadata coordinates. Numeric metadata rounding diagnostics are saved in [semantic_summary.json](semantic_summary.json).

Repeated coordinates are present in 573 records. Fifteen train/validation examples encounter an exact nearest tie; five nearest-category cases have different visible names tied for the answer. First-visible-candidate tie breaking fails on some gold labels, so no such rule is justified. Thirty-five examples have a nearest gap within the fixed 0.2-metre diagnostic envelope; three radius examples lie within that envelope. These flags do not authorize changing thresholds or labels.

## M. Context-length findings

| Portion, characters | Min | Mean | P50 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Entire context | 515 | 9,474.79 | 8,858 | 14,981 | 20,113 | 45,120 | 147,305 |
| Question | 40 | 81.54 | 76 | 113 | 132 | 148 | 178 |
| Trimmed article | 79 | 3,638.52 | 1,498 | 7,925 | 12,600 | 37,202 | 139,261 |
| Trimmed registry | 70 | 5,497.10 | 6,565 | 8,803 | 9,037.50 | 9,807 | 10,061 |

Article and registry boundaries were found in all records. The trimmed article length differs from construction `article_chars` in 62 records; these are measurement-boundary differences requiring documentation, not automatic source edits. All eight operations can use structured coordinates and categories; Wikipedia prose need not pass through a neural encoder. No tokenizer or neural model was loaded. [Per-split percentiles](length_summary.json).

## N. Question-pattern/template findings

**All 24,431 questions match exactly one of eight grammatical skeletons**, and the inferred operation agrees with `task_type` in every record. Rules were identified from TRAIN wording, without training a classifier.

| Task | Dominant cue/skeleton |
| --- | --- |
| nearest_category | ما أقرب … إلى … من بين المعالم الواردة في السياق؟ |
| cardinal_direction | في أي اتجاه يقع … بالنسبة إلى …؟ |
| within_radius_yes_no | هل يوجد … ضمن مسافة …؟ |
| closer_of_two | أي الموقعين أقرب إلى …: … أم …؟ |
| count_within_radius | كم عدد المعالم من نوع … الواقعة ضمن مسافة …؟ |
| nearest_of_two_categories | أيهما أقرب إلى …: أقرب … أم أقرب …؟ |
| two_hop_nearest | انطلاقًا من … حدد أولًا … ثم استخدم هذا الموقع … |
| spatial_multi_constraint | ما أقرب … يقع في اتجاه … وضمن مسافة …؟ |

Masking names/numbers alone gives 242 train, 208 validation, and 185 test surface patterns; most remaining variation is categories/directions. Masking all arguments collapses them to eight skeletons. No unmatched free-form questions were observed. Train has 4,567 word types, but named entities inflate this lexical diversity. This is not evidence of paraphrase/dialect coverage. Task prediction from wording is trivial on this benchmark.

## O. Answer-format findings — train/validation only

| Task | Format | Checked records |
| --- | --- | --- |
| nearest_category | candidate entity name | 7,415 |
| cardinal_direction | direction | 3,676 |
| within_radius_yes_no | yes/no | 2,482 |
| closer_of_two | candidate entity name | 2,450 |
| count_within_radius | ASCII integer string | 798 |
| nearest_of_two_categories | candidate entity name | 1,796 |
| two_hop_nearest | candidate entity name | 1,813 |
| spatial_multi_constraint | candidate entity name | 1,669 |

No empty or malformed answers, unexpected direction labels, or non-ASCII count formats were found. All **15,143 entity-name answers** appear literally in their visible registry. Of these, **2,028** names match multiple candidates. Fifty answers contain edge whitespace; preserving names avoids false missing-answer errors. `short_target == answer` and the documented answer-plus-rationale `long_target` template match all 22,099 checked records. No open-ended final answer is required; explanations are separate supervision. Test answer values were not audited.

## P. Split leakage findings

| Shared distinct values | Train–validation | Train–test | Validation–test |
| --- | --- | --- | --- |
| Record IDs | 0 | 0 | 0 |
| Wikidata anchors | 0 | 0 | 0 |
| Anchor names | 0 | 0 | 0 |
| Raw anchor coordinate pairs | 9 | 11 | 0 |
| Visible rounded anchor pairs | 12 | 12 | 2 |
| Exact contexts | 0 | 0 | 0 |
| Exact questions | 28 | 33 | 5 |
| Exact question + context | 0 | 0 | 0 |
| Exact registry text | 6 | 10 | 1 |

Anchor IDs/names and complete contexts are split-disjoint. There are only **1,545 distinct anchors/contexts** for 24,431 questions (1,236 train, 154 validation, 155 test), so examples are clustered, not independent. Group future internal training/calibration partitions by anchor.

However, different anchor names can share coordinates and nearby candidate populations. Complete cross-split candidate-set Jaccard search at >=0.9 found **100 / 148 / 13** nonidentical-registry pairs for train–validation / train–test / validation–test. Affected held-out rows include **997 validation** and **953 test** examples relative to train. This is a spatial-context overlap proxy, not proof of answer leakage or arbitrary-text semantic duplication. Exact identical registries and normalized article overlaps are reported separately. No published split was changed. [Full leakage report](split_integrity_summary.json), [near-registry method and counts](near_duplicate_registry_summary.json).

## Q. Direction semantics — train/validation only

| Label | Count |
| --- | --- |
| جنوب | 930 |
| شرق | 895 |
| شمال | 879 |
| غرب | 972 |

Four-way nearest-cardinal classification agrees with **3,306/3,306 conservatively unambiguous examples** (2,930 train; 376 validation). An eight-way classifier agrees on only **2,074/3,306**. No unambiguous inconsistencies remain.

A conventional partition has boundaries at 45°, 135°, 225°, 315°, but observed unambiguous bearings stay more than 10° away from those boundaries. Therefore boundary behavior itself is not empirically established; the source appears to avoid boundary cases.

**Multi-constraint filtering is different:** angular deviation <=35° from the requested cardinal axis reproduces every train/validation multi-constraint answer and qualifying count. Do not share one sector-classification predicate blindly between these tasks. The existing eight-sector Django verifier was not modified.

## R. Radius/count semantics

| Task | Observed radii km | Min | Mean | P50 | P90 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| within_radius_yes_no | 1.0, 2.0, 3.0, 5.0 | 1 | 2.62 | 2 | 5 | 5 | 5 |
| count_within_radius | 1.5, 2.0, 3.0, 5.0 | 1.50 | 1.57 | 1.50 | 1.50 | 1.50 | 5 |
| spatial_multi_constraint | 3.0, 5.0, 8.0 | 3 | 5.38 | 5 | 8 | 8 | 8 |

Radii above use question text only across all splits. Units are consistently `كم`, with integer or dot-decimal numeric text. `radius_constraint_km` is populated only for multi-constraint records; radius/count questions require text extraction.

Train/validation count-answer distribution: 2: 182, 3: 183, 4: 181, 5: 154, 6: 98. Counts cover **2–6 only**, providing no count-zero/count-one stress coverage.

Haversine distance using visible coordinates is consistent with the reconstructable answers. Constants 6371 and 6371.0088 km did not discriminate relevant count outcomes; categorical labels do not uniquely establish the source Earth model. `<` and `<=` are indistinguishable on the observed exact-boundary cases, because there are none in the audited unambiguous radius subset.

Train `7a7831fabeacb0c8`: hospital distance is approximately **3.000281 km** against a 3-km question; gold is نعم and rationale rounds to 3.0. Rounded-three-decimal comparison explains it, but applying that rounding globally creates a count mismatch elsewhere. Preserve this as a precision/convention review case.

**864 count questions** name an origin different from the context anchor (684/94/86 train/validation/test). Another **23** count questions reuse the context-anchor name at different candidate coordinates. On train/validation those 20 same-name collisions agree with the candidate-origin interpretation, but public-input identity remains conservatively ambiguous. Do not substitute the context anchor automatically. Counting unique names would change **121 otherwise resolvable train/validation answers**; preserve record multiplicity.

## S. Two-hop findings — train/validation only

Both category slots are recovered from every question. They agree with the first/second-hop metadata in all **1,813** checked examples. First-hop geometry selects a unique candidate in every case, and all intermediate and final answers agree with gold. The gold intermediate name appears more than once in **226** registries; using the selected first-hop candidate identity resolves this without gold input.

The observed first/second-hop nearest gaps are about 0.2 km or larger, indicating a comparatively well-separated subset. This task is two deterministic searches after straightforward template extraction; it does not demonstrate a need for deeper neural interpretation. Test intermediate/final labels were not inspected.

## T. Multi-constraint findings — train/validation only

All target category, direction, and radius slots are recoverable and match their annotations. Using exact category matching, **±35° direction filtering**, radius <= requested radius, then nearest selection yields **1,669/1,669** final answers and qualifying counts agreeing with gold.

Training candidate-set sizes: category mean **8.49** (3–10), after direction **5.29** (2–10), after radius **5.02** (2–9). Validation: **8.33**, **5.16**, **4.85** respectively. All cases have at least two qualifying records; zero-match behavior is not covered. Exact equal minima can share the same visible answer; no differing-name final ambiguity occurs here.

Using ±45° instead would change **188 answers** and **575 qualifying counts**. These are resolved convention mismatches, not 575 bad labels. ±35° was identified on train and confirmed on validation; test answers were never used. Constraint intersection order does not change the set; nearest selection must happen after filtering.

## U. Deterministic reconstructability by task — train/validation only

| Task | Total | A reconstructable | B ambiguous | F discrepancy | A % |
| --- | --- | --- | --- | --- | --- |
| nearest_category | 7,415 | 7,410 | 5 | 0 | 99.93% |
| cardinal_direction | 3,676 | 3,306 | 370 | 0 | 89.93% |
| within_radius_yes_no | 2,482 | 2,481 | 0 | 1 | 99.96% |
| closer_of_two | 2,450 | 1,875 | 575 | 0 | 76.53% |
| count_within_radius | 798 | 712 | 86 | 0 | 89.22% |
| nearest_of_two_categories | 1,796 | 1,796 | 0 | 0 | 100.00% |
| two_hop_nearest | 1,813 | 1,813 | 0 | 0 | 100.00% |
| spatial_multi_constraint | 1,669 | 1,669 | 0 | 0 | 100.00% |
| TOTAL | 22,099 | 21,062 | 1,036 | 1 | 95.31% |

| Split | A | B | F |
| --- | --- | --- | --- |
| train | 18,679 | 917 | 1 |
| validation | 2,383 | 119 | 0 |

Categories C (required missing candidate), D (malformed registry), E (taxonomy uncertainty), G (insufficient information), and H (other) are all zero under the documented classification. B includes identity ambiguity even when every interpretation happens to produce the same answer. It also includes five differing-name nearest ties. This conservative readiness metric is not a trained-model accuracy or test score. [Per-split/task percentages](reconstructability_summary.csv), [record-level classes](reconstructability_records.csv).

## V. Suspected label/data-quality issues

Four train `closer_of_two` **answer_position** annotations disagree with the quoted alternative containing the gold answer; the final answer itself is not thereby wrong. Example `bbd01df5e94e5f30` has answer ساحوت in the first alternative but position metadata second.

The single radius discrepancy is consistent with rounding and should not be called a proven incorrect label. Five nearest-name ties cannot be resolved from visible coordinates; examples include Arabic/English university names at the same location. Sixty-two trim-based article-count mismatches and exact duplicate rows need provenance-aware handling, not automatic deletion.

The remaining major problem is input identity ambiguity, not widespread corrupt labels. Representative IDs and limited details are stored in [representative_findings.json](representative_findings.json); no large raw contexts are duplicated in the reports.

## W. Safe inference fields

**A — Direct:** `question`, `context`.

**B — Only if recomputed/extracted or supplied by actual retrieval:** anchor name/coordinates, country/language, Wikidata/title/source provenance, context/article lengths, candidate counts, length bins. Candidate records and ordinals derive from visible context. The origin must be resolved from the question. Predicted operation/category/radius/direction/hop slots are safe; their raw construction annotations are not.

[Field-by-field A/B/C/D policy](field_input_policy.csv). Country/language/provenance do not need to be predictive features merely because they can be available.

## X. Training-target-only fields

`task_type`, `answer`, `short_target`, `long_target`, `verified_rationale`, `answer_position`, `intermediate_answer`, `first_hop_category`, `second_hop_category`, `target_category`, `direction_constraint`, and `radius_constraint_km` are supervised targets/offline references only. On test, use them only at the eventual locked final evaluation. Construction targets may be wrong or identity-ambiguous, so do not blindly train every target field.

## Y. Gold/construction-only fields

Never use `id`, `split`, `difficulty`, `difficulty_score`, `recommended_reasoning`, `router_target_alpha`, `evidence_poi_ids`, or `num_qualifying_pois` as model inputs. IDs/splits support bookkeeping; evidence IDs and qualifying counts can support offline audit. Gold difficulty is not a valid inference-time routing feature, and construction reasoning recommendations are not empirical benefit labels.

## Z. Required fixes before model training

| Rule/finding | Train | Validation | Test inputs | Total |
| --- | --- | --- | --- | --- |
| multiline_candidate_names | 1502 | 251 | 303 | 2056 |
| candidate_name_edge_whitespace | 1227 | 185 | 192 | 1604 |
| query_origin_differs_from_context_anchor | 684 | 94 | 86 | 864 |
| count_origin_name_collides_with_context_anchor | 17 | 3 | 3 | 23 |
| target_name_collides_with_context_anchor | 40 | 6 | 8 | 54 |
| named_identity_ambiguous | 856 | 111 | 107 | 1074 |
| reconstructability_B | 917 | 119 | Not evaluated | 1036 |
| multi_constraint_four_sector_answer_differs | 172 | 16 | Not evaluated | 188 |
| answer_position_mismatch | 4 | 0 | Not evaluated | 4 |
| reconstructability_F | 1 | 0 | Not evaluated | 1 |
| duplicate_candidate_rows | 358 | 0 | 69 | 427 |
| article_chars_mismatch | 38 | 24 | 0 | 62 |

For **every** recommendation, [recommended_remedies.csv](recommended_remedies.csv) gives its exact rule, remedy class A–G, affected task, split, count, and raw-untouched status. Counts overlap. In particular:

1. Preserve multiline/raw names, coordinates, rows, and distinct identities.
2. Separate main-context anchor from query origin and named target identities.
3. Document four-way classification separately from ±35° constraint filtering.
4. Retain ambiguity flags; withhold single-identity labels where unresolved.
5. Review four position annotations, precision boundaries, and tie behavior.
6. Fit no model on gold/construction features. Preserve original splits and use anchor groups for new internal partitions.

**Structured-query supervision readiness:**

| Task | Direct annotation | Derivable from real input | Remaining gap |
| --- | --- | --- | --- |
| All | operation from task_type (target only) | question slots and visible context anchor | new preprocessing; no training dataset created |
| nearest_category | answer only | category, anchor | nearest ties / candidate identity |
| cardinal_direction | direction target | target name, anchor | duplicate/colliding name resolution |
| within_radius_yes_no | Boolean target | category, radius, anchor | precision convention |
| closer_of_two | answer + position target | two names, anchor | ambiguity; four position-label reviews |
| count_within_radius | count target | category, radius, named query origin | origin resolution; same-name anchor collisions |
| nearest_of_two_categories | answer only | two categories, anchor | derive ordered category slots |
| two_hop_nearest | first/second categories, intermediate target | two categories and anchor | first-hop candidate ID must come from geometry |
| spatial_multi_constraint | category/direction/radius/qualifying annotations | category/direction/radius and anchor | document directional window |

Operation and argument extraction supervision is feasible. No cleaned/derived training dataset or model has been created.

## AA. Is the lightweight Transformer still recommended?

**Not as the first implementation on this dataset.** Revise the previous Architecture B recommendation: establish the template/structured-query and deterministic-spatial baseline first. The full dataset supplies eight strong question templates with short questions (median 76, maximum 178 characters), and the audit already resolves their slot structure without a model.

| Family | Assessment after full audit |
| --- | --- |
| GRU | Adequate for short slot sequences if a learned baseline is required; no need to process raw contexts. |
| BiLSTM | Credible question tagging baseline; greater recurrent cost without proven benefit over simpler extraction. |
| BiLSTM + attention | Optional linking/paraphrase experiment; attention cannot invent missing identity. |
| Small Transformer | Viable fixed-depth research baseline, but unnecessary complexity for observed templates. |
| Shared Transformer with early exit | Technically valid future experiment; current data gives no demonstrated Long-benefit population. |
| Pretrained compact encoder | Most useful if independently reviewed free-form Arabic/paraphrases become a requirement; adds size and deployment work. |

Keep typed structured queries, explicit identity handling, exact spatial computation, provenance, and a shared inference interface. Candidate attention and an LLM are unnecessary in the first version. Do not interpret strong template accuracy as general Arabic-language robustness.

## AB. Is Short/Long adaptive routing justified?

**Not yet by this dataset.** Two-hop and multi-constraint tasks have more spatial operations, but their language templates are explicit and their execution is cheap. Candidate ambiguity requires identity information or clarification; deeper layers cannot supply missing evidence. A gold-difficulty router would mainly rediscover task templates.

Binary accept/refine remains a future hypothesis if genuinely varied input yields measurable Short errors that Long fixes. Real inputs could provide operation/constraint/reference counts, missing arguments, candidate-match counts, and later calibrated prediction confidence. No confidence or benefit labels exist before expert evaluation. Three levels have no current justification.

For this dataset, use a transparent deterministic dispatch and ambiguity/unsupported-input branch first. Do not create a learned router merely to preserve the original diagram. Require a measured accuracy–latency improvement over one lightweight model before retaining adaptive neural depth.

## AC. Exact next implementation step

After this audit, implement an independently tested **structured-query extraction and identity-validation baseline**, using immutable raw inputs and separate derived outputs. Its contract must represent operation, explicit origin source, category/target slots, radius/direction parameters, candidate identities, and ambiguity status. Then implement the deterministic executor against the audited semantics and evaluate on train/validation.

Do not begin by implementing the Transformer, experts, router, candidate attention, LLM, or Django wiring. Expand language evaluation first if the product requires free-form Arabic; a compact learned parser can then be compared against the rule baseline.

Audit deliverables: [script](../../../evaluation/audit_full_dataset.py), [summary](dataset_audit_summary.json), [verification](verification.json), [remedies](recommended_remedies.csv), and the linked detailed CSV/JSON reports. Twelve unit tests and ten report consistency checks pass. Raw SHA-256 verification and counts were reused from completed scans. This completion performed no new raw-data scan.

## Reproduction and scope notes

Commands for a future intentional audit rerun (not run again during report completion):

```bash
python3 -B evaluation/audit_full_dataset.py --self-test
python3 -B evaluation/audit_full_dataset.py
```

`semantic_probe` is an offline diagnostic inside the audit script, not a production Geo Engine. It refuses test gold analysis. No production model/executor/router code was added. Structural similarity and identity checks use inputs only. The report-level estimates and readiness decision do not use test answers.

Per-record audit files contain identifiers, flags and brief train/validation diagnostic details, not a cleaned training dataset. Preserve the raw dataset and its published splits.
