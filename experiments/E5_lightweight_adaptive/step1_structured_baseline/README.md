# Step 1: structured spatial baseline

**Ready to freeze as the audited dataset baseline.** The engine executes all eight tasks from `question` and visible `context`, with typed answers, explicit identity resolution, and execution traces. It is trustworthy within these tested templates and documented spatial conventions. This is not a claim of general Arabic interpretation or a final test result.

## Validation results

| Metric | Result |
|---|---:|
| Examples processed | 2,502 |
| Structured query parse success | 2,502 / 2,502 (100%) |
| Inferred operation agrees with annotation | 2,502 / 2,502 |
| Unique identity resolution | 2,383 |
| Ambiguous identity resolution | 119 (4.7562%) |
| Missing identity | 0 |
| Executor successes | 2,383 (95.2438%) |
| End-to-end correct, abstentions scored incorrect | 2,383 / 2,502 (95.2438%) |
| Accuracy among successful answers | 100% |
| Accuracy on saved audit class A | 2,383 / 2,383 (100%) |
| Abstentions | 119 (4.7562%) |
| Parser/context errors | 0 |
| Executor incorrect answers | 0 |

All 119 non-answers are deliberate identity abstentions, not forced predictions. No parser/executor failure was patched using validation gold. The prior audit's class A annotation is used only to calculate the offline subset denominator after inference. It does not select runtime behavior. Reported accuracy is conditional on the existing templated dataset; validation previously helped verify audit semantics, so this is an implementation regression result, not an untouched estimate of convention discovery or generalization.

| Task | Validation examples | Correct | Abstentions | Overall accuracy |
|---|---:|---:|---:|---:|
| cardinal_direction | 409 | 376 | 33 | 91.93% |
| closer_of_two | 283 | 210 | 73 | 74.20% |
| count_within_radius | 97 | 84 | 13 | 86.60% |
| nearest_category | 827 | 827 | 0 | 100.00% |
| nearest_of_two_categories | 205 | 205 | 0 | 100.00% |
| spatial_multi_constraint | 191 | 191 | 0 | 100.00% |
| two_hop_nearest | 210 | 210 | 0 | 100.00% |
| within_radius_yes_no | 280 | 280 | 0 | 100.00% |

Every task has 100% accuracy on its saved reconstructable subset. Validation identity abstentions break down as cardinal direction 33, closer of two 73, count within radius 13. There are no unsupported, invalid-query, or not-found validation results. See [failure_summary.csv](failure_summary.csv) for stage-level counts and [failure_records.csv](failure_records.csv) for IDs and short reasons only; neither stores questions, contexts, or gold answers.

## Separate train diagnostic

Train was evaluated in full solely as a diagnostic: 19,597 records, 100% parse success, 18,679 correct, 917 abstentions, and one successful execution disagreeing with gold. Overall accuracy is 95.3156%; all 18,679 audit-class-A rows agree. The 917 abstentions comprise 912 named-identity cases and five nearest ties with different visible answers.

The sole disagreement remains the known TRAIN radius-boundary case `7a7831fabeacb0c8`: approximately 3.000281 km versus a 3-km radius, gold نعم. No universal rounding or ID-specific exception was introduced. The same issue appeared in the audit; Step 1 found no additional train/validation answer discrepancy.

## Runtime and measurement scope

| Validation timing | Value |
|---|---:|
| Total streaming evaluation wall time | 0.581813 s |
| Total inference time | 0.439339 s |
| Mean inference latency | 0.175595 ms |
| P50 inference latency | 0.142354 ms |
| P95 inference latency | 0.382987 ms |
| Mean wall time per example | 0.232539 ms |

Local CPU, Python 3.9.6, standard library only. Each reported split is one streaming pass in original file order. The 32-entry context cache is cleared at split start; validation had 1,518 hits and 984 misses. Inference includes context parsing/conversion, question parsing, identity, geometry, and trace construction. Wall time additionally includes gzip decoding, JSON, scoring, and audit-class lookup. Python startup, loading the saved audit CSV, and final report writes are outside the timed split. These are local measurements, not guaranteed deployment latency. Train wall time was 4.607446 s.

The evaluator was run once initially and once after completing distance details in traces; the saved figures are the final run, not a best-of benchmark. Answer metrics were identical. No full dataset audit was rerun and no TEST file was opened in this step.

## Public contract

```python
from spatial import parse_context, parse, execute, run

result = run(question, context_text)
# Or replace only the parser later:
context = parse_context(context_text)
query = parse(question, context)
result = execute(query, context)
trace_json = result.to_dict()
```

`ParsedContext` carries the visible anchor plus the ordered candidate tuple. Passing that bundle to `execute` preserves a clean interface without relying on construction anchor metadata. `StructuredQuery` contains operation, origin reference/source, entity targets, category slots, radius, direction, and first/second-hop categories. `Location.identity` is scoped to the supplied context (`anchor` or `candidate:<ordinal>`); it is not a global POI ID. When storing traces for multiple contexts, the caller must retain the context association.

The runtime never accepts a full dataset row. The evaluator's `infer_record` reads exactly `question` and `context`; gold is read only afterward for scoring. TEST evaluation is rejected before file access. No model, router, neural dependency, Django integration, or derived training dataset was added.

`parse` raises `QueryError` for unsupported/invalid input. `run` turns that into an `ExecutionResult`. Statuses are `success`, `ambiguous`, `not_found`, `unsupported`, and `invalid_query`. Answers distinguish entity string, Boolean, integer, and exact direction label. Failure results have no answer. Unknown templates and categories are explicit unsupported cases.

## Identity and semantic decisions

- Context parsing and the eight regex templates were extracted from the existing audit, avoiding duplicate implementations. The audit imports these shared functions; its semantic diagnostics remain separate from production execution.
- Literal names, multiline names, trailing whitespace, row order, and duplicate rows are preserved. Identity lookup performs no normalization and never uses evidence IDs or answers.
- Ordinary anchor templates identify the context anchor when the text agrees. A different question origin resolves against visible candidates. Count questions sharing a name with both anchor and a differently located candidate abstain. Named target/context-anchor collisions also abstain. Same-location anchor representation does not add an artificial coordinate alternative, but duplicate candidate rows still remain separate matches.
- Distances use Haversine with radius 6371.0088 km and visible context coordinates. Radius comparison is `<=` without rounding; this is the audited convention, with exact boundary inclusivity not established empirically by the original data.
- Cardinal classification uses four sectors with conventional boundaries at 45/135/225/315 degrees. Multi-constraint filtering uses angular deviation `<=35` degrees, then radius filtering, then nearest selection. These predicates are intentionally separate.
- Count answers count rows, including identical names and co-located candidates. Zero counts and negative radius-existence results are supported by synthetic tests even though the count dataset lacks zero/one answer coverage.
- Two-hop execution calculates the first candidate and uses its coordinates for the second hop. It never reads intermediate gold metadata. A first-hop identity tie abstains.
- Nearest selection retains all exact minima. Different names tied at minimum produce ambiguity. Identical final answer text can succeed while retaining every minimizing identity and setting `selected_identity_unique=false`; no arbitrary row is chosen. A named reference matching multiple candidates always abstains, even if answers would coincide.
- Bearing at the same location is undefined and produces an invalid-query diagnostic. The observed dataset does not exercise that convention. This engine is scoped to the audited geographic tasks, not a universal geodesic service.
- Traces include parsed arguments, operation/status, origin, resolution alternatives/counts, selected identities, all relevant comparison distances, radius/direction filters, first-hop reference when applicable, typed answer, and precision/tie diagnostics.

## Scoring

Entity answers normalize whitespace only; letter forms, case, punctuation, and names are not collapsed. This is deliberately narrower than historical broad Arabic letter folding. Yes/no permits Arabic combining marks and tatweel removal; count parsing accepts complete integer strings, including Arabic digits, and rejects embedded numbers/decimals. Direction labels match the dataset exactly. Scoring keys never affect identity resolution.

## Files

Created:

- `spatial/__init__.py`: public parse/execute/run interface.
- `spatial/schema.py`: immutable inference contracts and typed results.
- `spatial/context_parser.py`: shared audited grammar and typed context view.
- `spatial/query_parser.py`: shared eight templates, structured extraction and validation.
- `spatial/identity.py`: explicit exact-name resolution and clarification alternatives.
- `spatial/geometry.py`: shared audited distance/bearing/direction primitives.
- `spatial/executor.py`: eight operations and traces.
- `spatial/normalization.py`: task-aware scoring only.
- `evaluation/evaluate_structured_baseline.py`: validation evaluation and optional separate train diagnostic.
- `tests/test_spatial.py`, `tests/test_structured_evaluation.py`: focused runtime/evaluation tests.
- This directory's `README.md`, `metrics.json`, `per_task_metrics.csv`, `failure_summary.csv`, `failure_records.csv`, and `verification.json`.

Modified: `evaluation/audit_full_dataset.py` imports shared parsing/geometry instead of retaining copies; its previous audit output files were not regenerated. Existing audit files were already untracked before this turn, so Git's tracked-file diff does not display this edit.

## Verification and reproducibility

**37 tests pass:** 25 runtime/evaluation tests plus all 12 existing audit self-tests. Tests cover every task, multiline and exact names, duplicated rows/coordinates, missing and ambiguous identities, question-derived origin, anchor collisions, spherical distances, four-sector boundaries, the narrow filter, ties, two-hop reference transfer, zero counts, unrounded radius boundaries, trace serialization and comparison distances, unsupported input, safe scoring, gzip evaluation, denominator correctness, and the test-gold firewall. The initial test run exposed Python 3.9 annotation compatibility; postponed annotations fixed it before dataset evaluation.

```bash
python3 -B -m unittest discover -s tests -v
python3 -B evaluation/audit_full_dataset.py --self-test
python3 -B evaluation/evaluate_structured_baseline.py --include-train
```

The evaluator records source SHA-256 values and checks that runtime files did not change during evaluation. Raw inputs are opened read-only; no cleaning, rewriting, splitting, or TEST gold access occurred. The completed full-audit counts and classifications were reused. No push or merge.

## Freeze decision and exact Step-2 recommendation

**Freeze Step 1 as the reproducible baseline with these limitations documented.** The geometry and identity contract are suitable foundations for parser comparison. The evidence supports correct execution under audited conventions; it does not establish free-form Arabic robustness, source-coordinate precision, unobserved boundary behavior, or single-entity identity in tied-answer cases.

For Step 2, first specify and independently review a challenge set with Arabic paraphrases, changed constraint order, multiple explicit references, recoverable disambiguation, missing slots, unsupported questions, and irreducibly ambiguous cases. Keep immutable published splits and TEST gold locked; partition new internal development/calibration data by anchor and template family. Preserve expected abstention labels when real inputs lack identity information.

Then compare the frozen rule parser against a lightweight Short parser and a stronger Long parser through the same `StructuredQuery`/executor interface. Score slot correctness, end-to-end correctness, ambiguity handling, and latency. Require a measurable population where Long corrects Short errors before proceeding to a learned benefit router in Step 3. Gold difficulty is neither an input nor a routing benefit label. No Step-2 implementation has been started.
