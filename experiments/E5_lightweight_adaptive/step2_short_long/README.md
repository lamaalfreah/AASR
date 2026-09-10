> Protocol update: historical “Short” in this report means **Rule Baseline 0**. Existing CORE is preserved as the **Hard Natural Arabic Challenge Set**. The separately implemented **Neural Short** and the current comparison are documented [here](../step2_neural_short/BASELINE_COMPARISON.md). Long evaluation remains paused.

# Step 2: Arabic robustness, Short/Long experiment, exploratory extensions

**Status: local implementation and challenge checks complete; live Long evaluation BLOCKED. Step 2 is not yet an empirically completed Short-versus-Long experiment.**

**Router verdict: ROUTER NOT YET JUSTIFIED.** No live Long result exists from which to estimate rescue. In addition, CORE contains no Short-correct examples, so a learned benefit router cannot be justified from this set alone even if Long eventually rescues many questions.

## 1. Challenge size and split

320 CORE questions, 40 for each of eight original tasks, all sourced from TRAIN audit-class-A records. DEV has 192 (24/task); EVAL has 128 (16/task). All 320 source IDs and source anchors are distinct. DEV/EVAL also use disjoint wording-family IDs. No TEST record or answer was opened. The held-out CORE EVAL questions were not used as prompt examples, tuning cases, or retries; do not use them for future router training.

Construction uses an authored bank of 40 phrasings per task, each filled with a different source problem. This is a controlled language-robustness suite, not independently collected real-user language. Source selection takes eligible distinct anchors from a fixed hash partition; it is not a random population sample. Eight accepted semantic checks accompany every item. A separate inference-only context file preserves visible anchor and exact registry text, omitting irrelevant Wikipedia prose. Original raw data is unchanged.

Files: [all CORE](challenge/core_challenge.jsonl), [DEV](challenge/core_dev.jsonl), [EVAL](challenge/core_eval.jsonl), [manifest](challenge/manifest.json).

## 2–3. Register and complexity distribution

| Register | Count | Share | Short accuracy | Long accuracy |
|---|---:|---:|---:|---|
| natural_msa | 128 | 40% | 0% | Not run |
| conversational | 96 | 30% | 0% | Not run |
| light_saudi | 64 | 20% | 0% | Not run |
| indirect_compositional | 32 | 10% | 0% | Not run |

Register labels describe intended phrasing, not an independent sociolinguistic annotation. Conversational Arabic includes broadly understood informal forms; Saudi wording is moderate. No deliberate spelling corruption was introduced. Category plural inflections and written Arabic radius expressions are exact lexical equivalents, not altered categories or thresholds.

| Level | Count | Short accuracy | Long accuracy | Relative Long benefit |
|---|---:|---:|---|---|
| L1 | 96 | 0% | Not run | Unmeasured |
| L2 | 96 | 0% | Not run | Unmeasured |
| L3 | 64 | 0% | Not run | Unmeasured |
| L4 | 64 | 0% | Not run | Unmeasured |

L1 denotes direct natural reformulation; L2 conversational wording; L3 indirect/reordered information; L4 multiple linked instructions or compositional wording. These author-assigned ordinal labels need independent review before using them as research-grade difficulty measurements. Task composition is unchanged.

## 4. Semantic validation and review

320 final items accepted by author/agent semantic review plus structural checks. **39 initially drafted questions were revised**, principally plural grammar in count questions and awkward category phrasing; all revisions retain their initial/final text and reason in [revision_history.jsonl](challenge/revision_history.jsonl). No source problem or gold answer was revised. No final paraphrase was rejected; all eight invariants and frozen expected-query execution agree on all 320. [Generation review](challenge/generation_review.csv) records each acceptance/revision.

Checks cover origin, named targets, categories, radius, direction, hop structure, constraints, and expected-query execution against the unchanged candidate registry. Mutation tests reject edited questions outside their reviewed wording family. All source questions are parsed by the frozen parser to define their expected contract; oracle execution confirms their audited answer. These checks do not independently prove linguistic equivalence: naturalness and semantic interpretation were reviewed by the authoring agent, **not a separate native-speaker reviewer**. Independent review remains recommended before a research freeze.

## 5. Short on original validation

Reuse the completed frozen Step-1 full-validation result: **2,502 records, 100% parse success, 95.2438% end-to-end accuracy, 119 deliberate identity abstentions, and 100% on 2,383 uniquely reconstructable examples**. Frozen source SHA-256 values were checked, so a repeated raw-data scan was unnecessary.

Original validation has no independent complete structured-query annotation for every task. Its prior operation-annotation agreement is 100%; a full query exact-match score would be a self-comparison against this same parser, so it is **not claimed as an independent metric**. CORE exact match is measured against each source's fixed expected query, including entity reference source, ordered category/target slots and null fields.

## 6–10. Short versus Long and routing opportunity

| Metric | CORE DEV | CORE EVAL |
|---|---:|---:|
| Examples | 192 | 128 |
| Short parse success | 0% | 0% |
| Short structured-query exact match | 0% | 0% |
| Short end-to-end accuracy | 0% | 0% |
| Short abstention | 100% | 100% |
| Long accuracy / query exact match | Not run | Not run |
| Short wrong / Long correct | Unmeasured | Unmeasured |
| Long rescue rate | Unmeasured | Unmeasured |
| Long regression rate | Undefined: zero Short-correct denominator | Undefined: zero Short-correct denominator |

All Short failures are `unsupported` template rejections, not wrong geospatial computations. The Short parser was not expanded. This pronounced drop shows exact-template dependence on the constructed out-of-template inputs; it is not a statistical estimate of performance on naturally sampled Arabic users.

[short_vs_long.csv](results/short_vs_long.csv) retains every example. Long columns are empty and outcomes are `not_evaluated`, never fabricated as wrong predictions. Once real calls complete, it records all four paired outcome classes and rescue/regression metrics. [Short metrics](results/short_metrics.json), [Long metrics and blocked state](results/long_metrics.json).

## 11–13. Results by level, register, and task

Short is 0% in every task, register, and L1–L4 group; Long is unmeasured in every group. Detailed per-split parse/exact-query/answer/abstention and paired-outcome fields are in [per_task.csv](results/per_task.csv), [per_complexity.csv](results/per_complexity.csv), and [per_register.csv](results/per_register.csv).

| Task | DEV / EVAL count | Short DEV / EVAL accuracy | Long |
|---|---:|---:|---|
| nearest_category | 24 / 16 | 0% / 0% | Not run |
| cardinal_direction | 24 / 16 | 0% / 0% | Not run |
| within_radius_yes_no | 24 / 16 | 0% / 0% | Not run |
| closer_of_two | 24 / 16 | 0% / 0% | Not run |
| count_within_radius | 24 / 16 | 0% / 0% | Not run |
| nearest_of_two_categories | 24 / 16 | 0% / 0% | Not run |
| two_hop_nearest | 24 / 16 | 0% / 0% | Not run |
| spatial_multi_constraint | 24 / 16 | 0% / 0% | Not run |

The Short floor effect prevents learning whether relative Long benefit increases monotonically with L1–L4. Even with successful Long evaluation, only differences in Long's own performance across these levels could be observed on CORE; a contrast with surviving Short performance would require a broader evaluation design.

## 14–15. Latency, tokens, and provider availability

| Timing | Mean | P50 | P95 |
|---|---:|---:|---:|
| Short CORE DEV | 0.001337 ms | 0.001250 ms | 0.001459 ms |
| Short CORE EVAL | 0.001355 ms | 0.001333 ms | 0.001500 ms |
| Short original validation (Step 1) | 0.175595 ms | 0.142354 ms | 0.382987 ms |
| Long | Not measured | Not measured | Not measured |

CORE timing uses pre-parsed immutable contexts for both paths and currently measures rapid Short rejection. Original validation included context parsing and successful execution, so its time is not directly comparable. Figures are single final local CPU passes, not a stable microbenchmark or best-of selection. No neural model was loaded locally.

**LLM calls: 0.** No input/output token usage, provider bill, estimated price, latency ratio, rescue rate, or regression observation is available. Unit-test fake-provider responses verify mechanics only and are never included as Long experiment results.

The requested model is pinned to **`qwen3-4b`**, preserving the user's Qwen3-4B experimental continuity. No alternative model was selected. QwenCloud's [public text-model catalogue](https://docs.qwencloud.com/developer-guides/getting-started/text-generation-models) lists that exact ID, but this does not establish access for this account/region. This process has neither `DASHSCOPE_API_KEY` nor `DASHSCOPE_BASE_URL`; account-specific verification is therefore blocked, not failed for model unavailability. [Availability state](results/provider_availability.json).

Before challenge calls, the client requires an authenticated regional [model-list request](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/list-models) to contain the exact ID. If absent it stops, reports returned Qwen3 4B catalogue alternatives when available, and requests explicit model/deployment choice. It never silently substitutes an 8B, Plus, VL, or other model. A different model ID in an inference response also stops execution.

For Qwen3 non-thinking structured output, the provider requests JSON-object mode and applies strict local schema validation. QwenCloud's [structured-output documentation](https://docs.qwencloud.com/developer-guides/text-generation/structured-output) distinguishes JSON-object mode from model-dependent JSON-schema support. The code does not assume strict provider-side JSON Schema is supported for 4B. It uses `enable_thinking=false`, temperature 0, a bounded output, and at most one retry for invalid structure. It never asks the model to compute final answers.

## LongParser interface and inference boundary

`language/long_parser.py` exposes a replaceable `Provider` protocol and `LongParser.parse(question, ParsedContext)`. Output is the same frozen `StructuredQuery`, or an explicit clarification/unsupported/invalid-output result. JSON validation rejects extra keys, duplicate keys, non-finite numbers, wrong types, invented names, and invalid operations/arguments; it also prevents bypassing count-origin ambiguity. The strict validator now compares JSON-normalized lists consistently with dataclass tuples; automated tests caught that serialization mismatch before any live call.

Provider payload contains only question, visible context anchor name, ordered candidate names (duplicates preserved), and category vocabulary. It contains no coordinates, full article, expected query, answer, task label, source record ID, or construction annotation. Identity validation and every spatial calculation remain in the frozen executor. API keys are environment-only, not printed or serialized. HTTP exceptions omit request/response bodies; redirects are refused.

Prompts and code hashes are saved. Paid-call progress is resumable only for matching challenge/prompt/code hashes. Expected queries/gold never enter retry prompts. No prompt revisions based on CORE EVAL results are permitted. The same frozen prompt is used on DEV and EVAL for this first comparison.

## 16–18. Exploratory extension results and examples

80 extension questions are stored separately and excluded from all CORE language scores:

| Intended class | Count | Contract execution |
|---|---:|---|
| Underspecified “best” | 20 | 20 needs_clarification |
| Requires absent population/traffic/land/etc. data | 20 | 20 unsupported |
| Explicit geometric objective | 40 | 32 success, 8 no matching candidate |

`spatial/extensions/site_selection.py` implements a separate operation over **supplied reference points** and **listed facility coordinates**: maximize/minimize nearest-facility distance, cardinal ±35° filtering, minimum-distance exclusion, and top-k ranking with ties retained. No frozen Geo Engine file changed. An unspecified objective never becomes an invented recommendation.

These are **oracle structured-intent contract tests, not end-to-end Arabic parsing results**. The extension evaluator supplies the expected intent to test execution and safe failure behavior; generating expected results with the same implementation is not an independent accuracy benchmark. Separate analytical synthetic tests check nearest-distance ordering, filters, thresholds and ties.

Examples:

- **Geometric success (`extension-041`):** «من النقاط المعروضة حول «ملعب نادي الزلفي»، أي نقطة أبعد عن أقرب مستشفى مذكور؟ اعتبر الأفضلية للبُعد فقط.» Selected reference `candidate:2`; distance to its nearest listed hospital is approximately **2.345589 km**. This ranks supplied points only.
- **Clarification (`extension-001`):** «ما أفضل مكان لإنشاء منشأة صحية شمال «محطة قطار الشمال بالرياض»؟» Returns `needs_clarification`: define “best” with an explicit measurable objective.
- **Unsupported data (`extension-021`):** «أي موقع حول «حلبة كورنيش جدة» يخدم أكبر عدد من السكان؟» Population data is absent; no recommendation is fabricated.
- **No match (`extension-045`):** selecting a point north of قلعة الوجه at least five kilometres from its nearest listed hospital finds no qualifying supplied point.

Inputs use existing POI coordinates as illustrative reference points. They are **not verified available land or genuine proposed healthcare sites**. Hospital coverage is limited to the supplied registry, not a complete regional facility census. Real suitability requires population, demand, demographics, capacity, road accessibility/traffic, land availability and price, zoning, and other domain evidence. [Extension questions](challenge/extension_challenge.jsonl), [contract results](results/extension_results.csv).

## 19. Limitations and research questions

1. **Does natural Arabic variation reduce Short performance?** Yes on this controlled suite: from 100% original-template parsing to 0% on 320 reformulations. Generalization outside this suite is not established.
2. **Does Long rescue failures?** Unknown; provider access is unconfigured and no calls were made.
3. **Does Long benefit grow from L1 to L4?** Unknown; no Long results, and Short is already at zero for all levels.
4. **Is there enough evidence for a learned router?** No. Zero measured rescues, meaning unmeasured opportunity rather than evidence that Long cannot help. Also no Short-correct CORE cases to estimate regressions or learn a meaningful accept/escalate tradeoff.
5. **Can the architecture support more complex geometric requests without LLM arithmetic?** At the structured-contract level, yes: separate extension execution demonstrates explicit objectives, ranking and constraints. Natural-language extension interpretation has not been evaluated.

Additional limits: intended registers/complexity need independent review; synthetic wording families remain templated; CORE deliberately selects reconstructable sources and does not test language-induced recovery of originally ambiguous records; the source-disjoint split is stronger than the requested source-ID separation but does not establish absence of overlapping geographic neighborhoods.

## 20–21. Router verdict and next action

**ROUTER NOT YET JUSTIFIED.** Do not build or train Step 3 now.

First configure the credential locally as `DASHSCOPE_API_KEY` and supply the account's HTTPS regional endpoint as `DASHSCOPE_BASE_URL`. For a nonstandard deployment, also configure `DASHSCOPE_MODELS_URL` on the same provider host. No endpoint/region is guessed. Then run authenticated exact-model discovery and the frozen Long experiment:

```bash
python3 -B evaluation/evaluate_language_challenge.py --live-long
```

If `qwen3-4b` is unavailable, stop and choose an explicitly authorized compatible deployment; do not substitute. Complete independent language review before treating this challenge as a research release. Before any learned router, design a separate preregistered mixed distribution containing both original-template Short successes and natural-language failures, plus genuine ambiguity/unsupported controls. Keep this CORE EVAL held out. Compare any future learned router against the simple deterministic policy “accept a valid Short parse; otherwise escalate,” because that policy already detects every observed CORE Short failure.

Only if live Long results show reliable rescue and a mixed benchmark demonstrates an accuracy/cost benefit beyond that deterministic policy should Step 3 proceed. No learned Router, neural Short/Long layers, Django integration, merge, or push was performed.

## Tests and frozen-file verification

46 runtime/evaluation/challenge/extension tests plus 12 audit self-tests pass: **58 total**. This includes all 320 semantic-preservation cases, source/anchor/wording split disjointness, strict JSON and bounded retries, objective validation, no invented suitability, gold-free inference payloads, TEST source rejection, exact model pinning and substitution refusal. All frozen Step-1 source hashes match. Raw data is read-only and unchanged; prior audit/Step-1 result artifacts were reused, not regenerated.

```bash
python3 -B -m unittest discover -s tests -q
python3 -B evaluation/audit_full_dataset.py --self-test
```

The challenge builders are reproducibility tools. Do not regenerate or alter CORE EVAL after using it for model decisions. Any later linguistic revision requires a new challenge version and a logged review entry. This working version includes 39 logged wording revisions, all completed before Long access.
