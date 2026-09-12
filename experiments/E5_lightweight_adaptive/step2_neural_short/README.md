> Protocol clarification: the frozen template parser is **Rule Baseline 0**; the BiGRU is **Neural Short**. See the [consolidated comparison](BASELINE_COMPARISON.md). Existing CORE is the Hard Natural Arabic Challenge Set. Long remains paused.

# Lightweight neural Short: CPU BiGRU evaluation

The requested lightweight neural Short path is implemented and evaluated. **It fits the original task templates well but is not yet a robust natural-Arabic parser.** The frozen Step-1 parser is designated **Rule Baseline 0**; it and the Geo Engine remain unchanged. Existing Step-2 challenges, analyses, and LongParser work were preserved. No OpenRouter/Qwen/Long calls or Router implementation occurred.

## Architecture and files

Arabic question → learned BiGRU intent → deterministic argument extraction → existing `StructuredQuery` → frozen identity validation / Geo Engine.

- [language/neural_short.py](../../../language/neural_short.py): question tokenizer, intent BiGRU, lexical slot extractor, loadable `NeuralShortParser` interface.
- [training script](../../../evaluation/train_neural_short.py): CPU training with original-TRAIN anchor holdout and no challenge/validation fitting.
- [evaluation script](../../../evaluation/evaluate_neural_short.py): one frozen evaluation on original validation, CORE DEV, then CORE EVAL.
- [tests](../../../tests/test_neural_short.py): eight new focused tests, alongside the existing test suite.
- `intent_bigru.pt`: trained CPU state dictionary; present locally and ignored by the repository's existing `*.pt` rule.
- `model_config.json`, `training_metrics.json`, `metrics.json`, `per_task.csv`, `per_register.csv`, `per_complexity.csv`, `per_example.csv`, `failure_summary.csv`, `failure_predictions.jsonl`, `verification.json`, and pinned `requirements.txt`.

No whole-question template parser is used as a fallback in the neural inference path. Rule Baseline 0 is invoked offline only to define original-validation query-reference agreement. The model consumes only question tokens. The slot extractor receives the predicted operation and public question/context. Gold task labels are training targets/evaluation references; they are never inference features.

## Model and training protocol

**32,264 trainable parameters**, comprising a 48-dimensional word embedding, one bidirectional GRU with 48 hidden units per direction, dropout 0.1 and an eight-class linear output. Vocabulary size is **68** including padding and unknown tokens. Quoted proper names and numeric literals are masked for intent classification. Raw names remain intact for deterministic entity resolution.

All 19,597 original TRAIN records were considered. To protect the challenge evaluation further, **320 CORE source anchors and their 5,752 associated TRAIN records** were excluded from both fitting and internal model selection. Remaining data:

- Fitting: **12,387 questions / 818 anchors**.
- Internal holdout: **1,458 questions / 98 anchors**.
- No shared anchor between these two partitions.
- No original validation, CORE DEV, CORE EVAL, or TEST question was used to train the model or vocabulary.

The fixed training configuration used seed 240431, CPU only, two PyTorch threads, batch size 128, AdamW learning rate 0.003, weight decay 0.01, inverse-class-frequency loss weights, and gradient clipping at 1.0. A maximum of 12 epochs and patience 3 with minimum loss improvement 1e-5 were set before evaluation. Epoch **11** was retained: epoch 12's improvement was smaller than the predeclared minimum improvement. Internal holdout intent accuracy was 100% throughout the epoch-end evaluations; this measures original-template recognition, not natural-language robustness.

Training took **9.7954 seconds**; reading/preparing data plus training took **11.7409 seconds**. The checkpoint has a saved SHA-256 digest. PyTorch 2.8.0 and NumPy 2.0.2 were installed in the local ignored `.venv`; no pretrained model or tokenizer was downloaded. GPU/MPS was not used.

## Main results

| Evaluation set | N | Intent accuracy | Structured-query exact match | End-to-end accuracy | Abstentions |
|---|---:|---:|---:|---:|---:|
| original_validation | 2,502 | 100.00% | 99.92% | 95.20% | 120 |
| core_dev | 192 | 22.40% | 20.83% | 22.92% | 128 |
| core_eval | 128 | 23.44% | 20.31% | 23.44% | 82 |

Exact counts:

- Original validation: **2,502/2,502 correct intents**, **2,500/2,502 exact queries**, **2,382/2,502 correct answers**. Accuracy on the 2,383 reconstructable rows is **2,382/2,383 = 99.9580%**.
- CORE DEV: **43/192 correct intents**, **40/192 exact queries**, **44/192 correct answers**.
- CORE EVAL: **30/128 correct intents**, **26/128 exact queries**, **30/128 correct answers**.

Abstentions count as incorrect. Query exact match compares all serialized fields, including origin source and ordered category/entity/hop arguments. Original-validation query labels are the audited Rule Baseline 0 outputs; therefore this is agreement with that frozen reference, not an independently annotated query metric. CORE references are the unchanged source-derived expected queries.

**Answer equality can hide a wrong query:** four DEV and four EVAL answers match gold despite a wrong predicted intent/query. Such cases remain failures under intent and exact-query scoring. Do not describe all 30 matching CORE EVAL answers as correctly interpreted questions; only 26 queries are exact.

For comparison, the previously measured Rule Baseline 0 has **95.2438% original-validation end-to-end accuracy** and **0% on CORE DEV/EVAL**, with full abstention on CORE. The new Short gains some natural-language coverage but produces **20 incorrect successful answers on DEV** and **16 on EVAL**, as well as abstentions. It should not replace the safer baseline indiscriminately.

## Per-task results

### Original validation

| Task | N | Intent | Query exact | End-to-end |
|---|---:|---:|---:|---:|
| cardinal_direction | 409 | 100.00% | 100.00% | 91.93% |
| closer_of_two | 283 | 100.00% | 100.00% | 74.20% |
| count_within_radius | 97 | 100.00% | 97.94% | 85.57% |
| nearest_category | 827 | 100.00% | 100.00% | 100.00% |
| nearest_of_two_categories | 205 | 100.00% | 100.00% | 100.00% |
| spatial_multi_constraint | 191 | 100.00% | 100.00% | 100.00% |
| two_hop_nearest | 210 | 100.00% | 100.00% | 100.00% |
| within_radius_yes_no | 280 | 100.00% | 100.00% | 100.00% |

### CORE DEV

| Task | N | Intent | Query exact | End-to-end |
|---|---:|---:|---:|---:|
| cardinal_direction | 24 | 25.00% | 25.00% | 25.00% |
| closer_of_two | 24 | 25.00% | 25.00% | 25.00% |
| count_within_radius | 24 | 12.50% | 12.50% | 12.50% |
| nearest_category | 24 | 4.17% | 4.17% | 4.17% |
| nearest_of_two_categories | 24 | 25.00% | 25.00% | 37.50% |
| spatial_multi_constraint | 24 | 16.67% | 16.67% | 20.83% |
| two_hop_nearest | 24 | 58.33% | 45.83% | 45.83% |
| within_radius_yes_no | 24 | 12.50% | 12.50% | 12.50% |

### Held-out CORE EVAL

| Task | N | Intent | Query exact | End-to-end |
|---|---:|---:|---:|---:|
| cardinal_direction | 16 | 25.00% | 25.00% | 25.00% |
| closer_of_two | 16 | 43.75% | 43.75% | 43.75% |
| count_within_radius | 16 | 6.25% | 6.25% | 6.25% |
| nearest_category | 16 | 6.25% | 6.25% | 6.25% |
| nearest_of_two_categories | 16 | 12.50% | 12.50% | 31.25% |
| spatial_multi_constraint | 16 | 6.25% | 6.25% | 12.50% |
| two_hop_nearest | 16 | 81.25% | 56.25% | 56.25% |
| within_radius_yes_no | 16 | 6.25% | 6.25% | 6.25% |

## Per-language-register results

### CORE DEV

| Register | N | Intent | Query exact | End-to-end |
|---|---:|---:|---:|---:|
| conversational | 56 | 17.86% | 17.86% | 19.64% |
| indirect_compositional | 16 | 25.00% | 18.75% | 18.75% |
| light_saudi | 40 | 15.00% | 15.00% | 20.00% |
| natural_msa | 80 | 28.75% | 26.25% | 27.50% |

### CORE EVAL

| Register | N | Intent | Query exact | End-to-end |
|---|---:|---:|---:|---:|
| conversational | 40 | 22.50% | 17.50% | 20.00% |
| indirect_compositional | 16 | 12.50% | 6.25% | 12.50% |
| light_saudi | 24 | 12.50% | 12.50% | 12.50% |
| natural_msa | 48 | 33.33% | 31.25% | 35.42% |

Original validation uses the original-template register; its metrics are the overall validation metrics above. Register annotations are unchanged author-provided challenge labels, not independently reviewed dialect annotations.

## Per-complexity results

### CORE DEV

| Level | N | Intent | Query exact | End-to-end |
|---|---:|---:|---:|---:|
| L1 | 72 | 27.78% | 26.39% | 30.56% |
| L2 | 56 | 17.86% | 17.86% | 19.64% |
| L3 | 32 | 15.62% | 15.62% | 15.62% |
| L4 | 32 | 25.00% | 18.75% | 18.75% |

### CORE EVAL

| Level | N | Intent | Query exact | End-to-end |
|---|---:|---:|---:|---:|
| L1 | 24 | 29.17% | 29.17% | 29.17% |
| L2 | 40 | 17.50% | 17.50% | 20.00% |
| L3 | 32 | 34.38% | 28.12% | 31.25% |
| L4 | 32 | 15.62% | 9.38% | 15.62% |

Scores do not decrease monotonically with the assigned L1–L4 levels. Sample sizes are small and phrasing/task interactions matter; do not interpret these levels as calibrated difficulty or infer a routing rule from them.

## Runtime and latency

| Set | Evaluation wall time | Mean pipeline latency | P50 | P95 |
|---|---:|---:|---:|---:|
| original_validation | 1.9844 s | 0.6527 ms | 0.5431 ms | 0.9704 ms |
| core_dev | 0.1158 s | 0.5560 ms | 0.5560 ms | 0.8488 ms |
| core_eval | 0.0802 s | 0.5773 ms | 0.5595 ms | 0.8619 ms |

Measurements use one CPU pass, batch size one at inference, and two PyTorch threads. Pipeline latency includes visible context parsing, neural intent inference, deterministic slots, and execution when a query is produced. The existing 32-entry context-parse cache starts empty for each split. Wall time additionally includes gzip/JSON, offline reference construction and scoring. Model load time was **0.014632 s** and is reported separately. No warmup or repeated benchmark was used.

Parser-only mean latencies, excluding context conversion and Geo Engine, were **0.4659 ms**, **0.3687 ms**, and **0.3917 ms**, respectively. The original Rule Baseline 0 result of about 0.176 ms is historical, not a simultaneous hardware-controlled comparison.

## Failures and limitations

Original validation has **118 identity abstentions** and **two count-slot extraction failures**. One of those two records was already ambiguous in the audit; the other was uniquely reconstructable. Thus the new path loses one correct answer relative to Rule Baseline 0. IDs `1b30df0058b08c66` and `96611c65b870c534` are logged with `category_slot_count`. They were not special-cased after evaluation.

CORE DEV's 148 nonmatching answers comprise **145 intent errors** and **three query errors after a correct intent**. CORE EVAL's 98 nonmatching answers comprise **94 intent errors** and **four query errors after a correct intent**. Hop-order extraction remains a weakness for indirect two-hop phrasing. Stage counts prioritize wrong intent when more than one downstream error occurs, so they are mutually exclusive failure attribution, not every possible defect count.

A central limitation is training diversity: after masking entities and numbers, original TRAIN supplies only **68 vocabulary entries**. Unseen tokens constitute **40.06% of DEV tokens** and **47.48% of EVAL tokens**, versus 0% on original validation. The model has learned template intent signals without sufficient natural-language supervision. The small GRU's capacity alone is not evidence that it can generalize to unseen conversational wording.

This is a closed-set eight-intent classifier. Softmax confidence is exposed for diagnostics but is not calibrated and is not used as a router or rejection threshold. Missing/ambiguous deterministic arguments still abstain. No claim is made that confidence reliably detects unsupported intent.

The original CORE questions were authored by the same agent in earlier work. They were not newly generated or used for fitting here; held-out refers to this training/evaluation protocol, not an independent benchmark author. All source anchors were excluded from this model's training to avoid source-group contamination. No model/slot-rule adjustment was made after inspecting validation, CORE DEV or CORE EVAL results.

## Verification and stopping point

**66 tests pass:** 54 runtime/evaluation/challenge/neural tests plus the existing 12 audit self-tests. The eight new tests cover all original task slot contracts, natural named count origins, names equal to category labels, plural/overlapping categories, Arabic radius expressions, nested/sequential hops, entity masking, and actual CPU BiGRU gradients/output shape. Existing challenge semantics, split checks, and frozen-file checks continue to pass.

All frozen Step-1 hashes and all challenge-file hashes match. No raw data was modified and no TEST file/gold was accessed. No completed dataset audit, challenge generation or previous Short/Long analysis was rerun. Evaluation outputs live in a new directory and preserve all previous results. No LongParser/provider calls, Router, Django changes, merge or push.

Reproduction commands (training and evaluation refuse to overwrite completed artifacts):

```bash
.venv/bin/python -B -m unittest discover -s tests -q
.venv/bin/python -B evaluation/audit_full_dataset.py --self-test
.venv/bin/python -B evaluation/train_neural_short.py
.venv/bin/python -B evaluation/evaluate_neural_short.py
```

**Verdict:** the requested neural Short experiment is complete and reproducible; natural-Arabic readiness is not achieved. Retain Rule Baseline 0 as the frozen comparison. A future, separately versioned experiment should add independently reviewed natural-language TRAIN supervision and consider a character/subword representation to reduce unknown words, using development data for slot refinement while keeping a fresh held-out evaluation untouched. Do not retrain on or patch rules from the current CORE EVAL findings. No Long or Router stage was started.
