# Step 2B Long evaluation — exact model unavailable

The credential was loaded from local `AASR/.env` without printing, logging, or storing its contents. The refreshed `/api/v1/key` authentication check succeeded with **HTTP 200**. The subsequent `/api/v1/models` catalogue did not contain the exact requested model. The exact model **qwen/qwen3-4b:free** remains pinned; no model substitution occurred.

The earlier authentication blocker is resolved. The current blocker is that the exact model ID is absent from OpenRouter’s current catalogue; this does not establish whether it may return later. No challenge question was sent: DEV smoke completed **0**, CORE EVAL completed **0/128**, remaining **128**. The smoke gate remains closed. No live generation calls were attempted.

## Preserved state

Rule Baseline 0, the Neural Short source/checkpoint and existing predictions, the frozen Geo Engine, and CORE files are unchanged. `freeze_manifest.json` records their hashes. No raw data or TEST gold was accessed. No challenge generation, retraining, tuning, or previous evaluation was repeated.

The frozen Neural Short CORE EVAL results remain **26/128 exact queries (20.3125%)** and **30/128 matching answers (23.4375%)**. Long intent/query/answer accuracy, query- and answer-level paired outcomes, rescue/regression, task/register/complexity comparisons, latency, tokens, provider cost, and latency ratio are **unmeasured**, not zero-accuracy results. `long_metrics.json` records nulls; `api_progress.json` records the blocker and remaining count.

## Work added

- `language/openrouter_provider.py`: explicit local-key loading; exact-model and free-pricing checks; authenticated OpenRouter requests; sanitized errors; stops on HTTP 429 without discarding checkpoints.
- `evaluation/run_long_preflight.py`: safe, read-only provider authentication/model verification.
- `tests/test_openrouter_provider.py`: seven passing focused tests for key isolation, exact-model gate, JSON validation/integration, persistence before retry, replay without duplicate calls, rate-limit preservation, and inference-field separation.

The provider atomically writes and fsyncs each successful JSON generation response before validating model output. This includes invalid structured output, allowing the bounded repair attempt to resume without repeating a completed response. Checkpoints are keyed by local case ID plus full request digest; local case IDs and gold labels never enter the model payload. Provider model output is retained as a prediction artifact, not as a credential log. Unit-test mock responses are not live Long results.

The evaluator's DEV smoke orchestration and 128-example comparison have not run. No completed evaluation is claimed. The frozen Long prompt has not been tuned on EVAL.

## Resume condition

Keep the valid credential private. Resume only when the exact requested model becomes available, or after the user explicitly authorizes a different model. Do not silently drop `:free`, select another model size, or use automatic model routing. To recheck availability later:

```bash
python3 -B evaluation/run_long_preflight.py
```

After authentication and exact-model verification succeed, run a few DEV smoke examples, validate schema and frozen-engine compatibility, then evaluate the 128 frozen EVAL examples with per-call checkpoints and both query/answer paired analysis. Do not bypass the smoke gate or substitute a model.

## Router verdict

**ROUTER NOT YET JUSTIFIED.** No measured Long outcomes establish rescue or regression. Step 3 remains deferred until Step 2B produces real paired evidence. No Router, Django integration, push, or merge was performed.
