# Production integration validation — 2026-09-14

The active Django pipeline is Neural Short → frozen Router V2 (three-member MLP,
threshold 0.91) → Short query or optional frozen Qwen3-4B → strict schema and
identity validation → deterministic Geo Engine → answer, trace and map data.
No model, prompt, threshold, spatial semantics or research evidence was changed.
No research dataset evaluation was run. All questions here use the clearly labeled
synthetic demo catalog. No gold labels enter runtime.

## Passed

- **13 backend tests**, zero failures: all eight operations with explicit expected
  synthetic answers; two-hop intermediate markers; zero count/false radius/no
  spatial matches; duplicate-name ambiguity; unknown identity; Short invalid
  query and Long abstention; backend unavailability; no silent Long fallback;
  timeout cancellation and remote identity mismatch; JSON/CSRF/authentication,
  body limits and busy admission control; real CPU Short/Router feature contract.
- **Browser checks passed** in Chromium: three real Django API → Short → Router →
  Geo → UI requests, selected markers, replay of the saved real Long response
  without another generation, ambiguity/clarification/503/429 fixtures, network
  failure with cleared stale answers/markers, mobile width, no JavaScript
  exceptions, and unavailable Leaflet fallback. The initial browser fixture
  callback had a test-harness error; corrected and rerun successfully.
- Python compilation, JavaScript syntax and `git diff --check` passed.
- All **18 production freeze hashes** matched; no tracked `language/`, `spatial/`
  or experiment files were modified by integration.
- Django normal system checks: no issues. Deployment settings: only intentional
  HSTS subdomain/preload warnings remain, pending a real domain/HTTPS policy.

| Real browser question | Route | Observed answer | Assessment |
|---|---|---|---|
| ما أقرب مستشفى إلى «مركز الحي»؟ | SHORT | مستشفى النخيل | Correct |
| كم عدد الصيدليات ضمن 2 كم من «مركز الحي»؟ | SHORT | 2 | Correct |
| ما اتجاه «مدرسة الرواد» بالنسبة إلى «مركز الحي»؟ | SHORT | شمال | Correct |

## Real Long smoke: transport passed, intended semantics failed

Exactly **one Qwen generation**, no retry, was requested through the real Django
API with actual frozen Router selection. This was not a forced route or a mock.
The production wrapper was deployed as `aasr-production-long`; no other active
Modal app was present before deployment.

- Question: ما أقرب صيدلية إلى أقرب مستشفى من «مركز الحي»؟
- Actual route: LONG (probability 0.9995630977794031).
- Infrastructure: Qwen/Qwen3-4B from existing `asar-hf-cache`, confirmed snapshot
  `1cfa9a7208912126459214e8b04321603b3df60c`, **one NVIDIA L4, BF16**, no download.
- Model load: **9.84 s**. API wall time: **32.12 s**, including local initialization
  and cold remote call. These are one-request observations, not a latency benchmark.
- Complete schema-valid query, unique grounded identities, successful deterministic
  execution, primitive response and trace were received locally.
- Qwen selected **nearest_of_two_categories**, whereas the question requires
  **two_hop_nearest**. It returned **صيدلية الندى**; the intended synthetic answer is
  **صيدلية النور**. Thus language interpretation/final-answer correctness failed.
- No fix to frozen model/prompt, rerouting override or answer substitution was made.
  `PRODUCTION_SMOKE.json` preserves the actual response, metadata and assessment.
  UI explicitly distinguishes calculation under the displayed interpretation from
  confirmation that the interpretation matches the user's intent.

Eight-operation tests cover integration and execution with supplied valid queries;
they do not establish perfect natural-language parsing for all eight operations.

## Remaining limits

This is a working controlled-demo integration. Frozen language interpretation can
still be wrong even when schema/identity checks pass. Catalog data is synthetic,
not a live POI feed. Long requires Modal credentials/network and can cold-start;
no alternative model is substituted on failure. Production Long checkpoints retain
question/name payloads in the existing Modal volume. External map assets need
network access. The single-process admission gate is not a distributed queue or
per-user quota. Internet deployment still needs real HTTPS/static serving, domain
configuration, access/retention policy and operational monitoring. No push or merge.

## Exact local demo command

From the AASR repository root (current environment already installed and migrated):

```sh
AASR_MODAL_APP=aasr-production-long .venv-router/bin/python manage.py runserver 127.0.0.1:8000 --noreload
```

Open http://127.0.0.1:8000/. See `PRODUCTION.md` for setup, configuration and demo scope.
