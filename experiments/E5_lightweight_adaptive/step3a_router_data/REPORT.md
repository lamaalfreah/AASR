**Step 3A — Router development data (no Router training)**

Completed 2,400 paired frozen Short/Long predictions on new natural-Arabic paraphrases of original TRAIN records. All 320 source anchors used by the CORE challenge were excluded before sampling. CORE EVAL and TEST contents were not read.

Primary correctness criterion: `exact_and_answer`. SHORT takes precedence whenever Short is correct, including when both models are correct; LONG means Short wrong and Long correct; FAILURE means both wrong. Exact-query-plus-answer and answer-only labels are both retained.

**Label distribution**

| Scope | N | SHORT | LONG | FAILURE |
|---|---|---|---|---|
| all | 2400 | 550 | 1499 | 351 |
| train | 1920 | 440 | 1196 | 284 |
| dev | 480 | 110 | 303 | 67 |

**Task distribution**

| Task | N | SHORT | LONG | FAILURE |
|---|---|---|---|---|
| cardinal_direction | 300 | 57 | 238 | 5 |
| closer_of_two | 300 | 38 | 249 | 13 |
| count_within_radius | 300 | 56 | 135 | 109 |
| nearest_category | 300 | 0 | 239 | 61 |
| nearest_of_two_categories | 300 | 66 | 156 | 78 |
| spatial_multi_constraint | 300 | 98 | 185 | 17 |
| two_hop_nearest | 300 | 221 | 50 | 29 |
| within_radius_yes_no | 300 | 14 | 247 | 39 |

**Register distribution**

| Register | N | SHORT | LONG | FAILURE |
|---|---|---|---|---|
| conversational | 600 | 154 | 350 | 96 |
| indirect_compositional | 600 | 88 | 398 | 114 |
| light_saudi | 600 | 150 | 375 | 75 |
| natural_msa | 600 | 158 | 376 | 66 |

Each task/register cell has 75 examples: 60 Router TRAIN and 15 Router DEV. The full cross-tabulation and split-specific labels are in `task_register_distribution.csv`.

Router TRAIN: 1,920 questions across 618 represented anchors. Router DEV: 480 questions across 150 represented anchors. The split is exactly 80/20 by question count and anchor-disjoint. Eligible anchors were assigned approximately 80/20 before record selection; represented-anchor counts differ slightly from that ratio.

**Feature availability**

| Inference-safe feature | Non-null / 2400 |
|---|---|
| constraint_count | 769 |
| entity_ambiguity_count | 2400 |
| missing_slots | 2400 |
| predicted_operation | 2400 |
| query_valid | 2400 |
| question_length_chars | 2400 |
| question_length_words | 2400 |
| short_confidence | 2400 |
| short_entropy | 2400 |
| short_top1_top2_margin | 2400 |

Confidence, entropy (natural-log nats), and top1-top2 margin come from the same frozen Short forward pass, observed with a read-only hook. No classifier weights, logits, or predictions are modified. Query validity means Short produced a query accepted by the frozen validator.

`missing_slots` is [] for valid queries, or the unresolved slot group reported by the frozen extractor (which may include conflicting values, not just omissions). It is null for unrelated failures; exact partial-slot counts cannot be recovered because the frozen parser returns no partial query. Constraint count is the number of targets/categories plus radius/direction/hop fields in the Short query, and is null without a query.

Entity ambiguity counts ambiguous frozen identity resolutions over Short query references, or visible names mentioned in the question when Short has no query. Question length is available in characters and whitespace-delimited words. Predicted operation always comes from Short. No Long outputs, gold task/register labels, expected query, correctness, source IDs, anchors, or gold complexity enter the feature dictionary.

**Long runtime and calls**

| Measurement | Value |
|---|---|
| Frozen model | Qwen/Qwen3-4B; BF16; existing asar-hf-cache; no model download |
| Observed GPU | NVIDIA L4 |
| Generation calls | 2941 |
| Retry calls | 541 |
| Summed remote inference seconds | 17128.35 |
| Long session wall seconds | 23888.05 |
| Monetary cost | Not supplied by inference responses; no estimate |

Load times and individual session metadata are saved in `summary.json` and `sessions/`; CPU Short timing is in `short_runtime.json`. Raw generation checkpoints are committed remotely before responses return and mirrored locally; completed paired predictions are skipped on resume. Full local replay verified all 2,400 examples without new generation.

**Provenance and limits**

There are 2,400 distinct source records and 2,400 distinct rendered questions. Only original TRAIN cases that execute successfully and match the original answer were eligible. Every paraphrase keeps the exact original origin, named targets, category order, radius, direction, hop order and answer. Context compaction preserves the exact frozen parsed spatial context. No original gold answer is replaced.

The questions instantiate 128 new agent-authored generic wording families (four per task/register). They are controlled natural-Arabic paraphrases, not collected user conversations or independently human-reviewed language. Wording families are shared across anchor-disjoint Router TRAIN/DEV; this split measures new-anchor generalization, not unseen-style generalization. Samples are balanced by construction rather than by estimated production frequency.

Artifacts: `router_dataset.jsonl`, `router_train.jsonl`, `router_dev.jsonl`, feature-only `router_train_features.jsonl` and `router_dev_features.jsonl`, `questions.jsonl`, `anchor_split.json`, `manifest.json`, `summary.json`, `verification.json`, and raw/paired checkpoints. Full rows retain provenance and outcomes in separate fields; future Router training should consume only the feature dictionary and label.

**Stopped after Step 3A. No Router was trained.**

**Resume integrity and interruptions**

Completed paired predictions and generation checkpoints recorded at 462, 1,368, and 2,213 completions are byte-for-byte unchanged; see `resume_preservation_verified.json`. The final progress checkpoint records 2,400/2,400 complete. Four sequential Modal sessions each reported exactly one NVIDIA L4 and BF16. Interrupted sessions were resumed using the same frozen pipeline and cache.

Recovered infrastructure errors:

- `ConflictError: function fu-zc6tIpGFIwT4w4eH0tA0SK is stopped`
- `AuthError: failed to parse auth token`
- `ConflictError: function fu-zz8pqOBQMEoSeySowXn894 is stopped`

Session wall time sums active recorded sessions, including startup and interruption handling, and excludes gaps between sessions. It is not total calendar elapsed time. Summed remote inference time counts saved model generation timings. Final Modal execution exited successfully; no additional inference or Router training was launched.
