# AASR application runbook

The existing Django dashboard now executes this path:

Arabic question + visible location catalog → frozen Neural Short (CPU) → frozen
3-member calibrated Router V2 (LONG probability ≥ 0.91) → Short query or frozen
Qwen3-4B LongParser → strict StructuredQuery validation → exact identity validation
→ frozen deterministic Geo Engine → typed answer, trace, selected map identities.

Short runs once on every request. A read-only logits hook supplies the full
8-operation probability vector; feature extraction preserves Step 3A semantics.
No gold labels, evaluation data, training modules, LoRA or GRPO enter runtime.
No fallback overrides a failed LONG decision. Clarification and unavailable
responses contain no fabricated answer. All geometry comes from `spatial.execute`.

## Run locally

Use Python 3.12, from the AASR repository root:

```sh
uv venv .venv-app --python 3.12
uv pip install --python .venv-app/bin/python -r requirements-production.txt
.venv-app/bin/python manage.py migrate
.venv-app/bin/python manage.py test dashboard
.venv-app/bin/python manage.py runserver 127.0.0.1:8000 --noreload
```

The Short weights/config and Router artifact must remain in their existing
`experiments/E5_lightweight_adaptive/step2_neural_short` and `router_v2` folders.
Startup lazily checks `dashboard/data/production_freeze.json` before loading them.
Do not substitute models, recalibrate, or change threshold/prompt to improve demos.
The historical research files remain available and are not invoked by the app.

## Enable the real Long path

Use the existing Modal account/profile and **existing `asar-hf-cache`** volume.
The frozen snapshot is `1cfa9a7208912126459214e8b04321603b3df60c`.

```sh
.venv-app/bin/modal profile activate joudalrubaish
.venv-app/bin/modal deploy -m dashboard.modal_production
export AASR_MODAL_APP=aasr-production-long
.venv-app/bin/python manage.py runserver 127.0.0.1:8000 --noreload
```

Deployment includes the unchanged Qwen class, not a research evaluation invocation.
It uses one L4 at most, BF16, local cached weights only, and the existing checkpoint
volume. No model download. Missing cache or GPU/precision mismatch fails closed.
Only the LONG route calls Modal. Cold starts may take minutes; timeout defaults to
240 seconds per attempt with at most the frozen one parser retry. Infrastructure
errors do not retry silently. The UI keeps the request pending and disables duplicate
submission. Modal persists production request responses under unique request keys;
research checkpoints are not read for serving. Do not log sensitive user catalogs
without an approved retention policy; raw Long messages include names/questions
(no coordinates) and generated responses remain in the existing Modal volume.

## Catalog and demo scope

The bundled **synthetic** neighborhood is explicitly labeled in the UI. Its
coordinates are inputs, never precomputed answers; all displayed spatial outputs
are computed at request time. It is not a live Riyadh POI feed. Set
`AASR_CATALOG_PATH` to a trusted JSON catalog using the same schema. The visible
catalog contains `name`, `anchor: {name,lat,lng}`, and `candidates` containing
`{name,category,lat,lng}`. Duplicate names are preserved and can trigger clarification.

API: `POST /api/analyze/`, JSON `{"question":"..."}`, with Django CSRF token/session.
`AASR_ALLOW_CUSTOM_CATALOG=1` optionally permits a `catalog` object in the same request
for trusted business integrations. It defaults off. Maximum 200 candidates, 2,000
question characters, and 256 KiB request body. Unsupported categories/coordinates
and accidental gold/extra top-level fields are rejected.

Responses expose `status`, `route`, `routing`, `task_type`, `structured_query`,
`validation`, `execution`, `latency`, `trace`, `answer`, and map/evidence fields.
Clarification is HTTP 200 with a non-success status and null answer; malformed
input 400, unsupported media 415, oversized body 413, busy 429, unavailable 503.
No raw exception is returned. Every server log has a request ID. Full DB query logs
are opt-in (`AASR_LOG_REQUESTS=1`) and require migrated tables.

## Investor demo queries

Use the bundled catalog shown beneath the map:

1. ما أقرب مستشفى إلى «مركز الحي»؟
2. كم عدد الصيدليات ضمن 2 كم من «مركز الحي»؟
3. ما اتجاه «مدرسة الرواد» بالنسبة إلى «مركز الحي»؟
4. أيهما أقرب إلى «مركز الحي»: «مستشفى النخيل» أم «مستشفى الواحة»؟
5. ما أقرب صيدلية إلى أقرب مستشفى من «مركز الحي»؟

Show the selected route, interpreted query, identity validation, and deterministic
execution steps rather than claiming the LLM calculated the result. Route and parse
quality are frozen; clarification is a supported outcome, not a reason to retune.
The single real Long smoke on query 5 completed technically but misclassified it
as category comparison and returned صيدلية الندى instead of صيدلية النور.
This is a recorded frozen language-model limitation, not a verified two-hop demo.
No prompt repair or model tuning was applied. See `PRODUCTION_SMOKE.json`.
All eight engine operations remain supported, including category comparison, radius
existence, and direction-plus-radius nearest search.

## Deployment safeguards and limits

For external access set `DJANGO_DEBUG=0`, a strong `DJANGO_SECRET_KEY`, and explicit
`DJANGO_ALLOWED_HOSTS`. HTTPS is required; secure cookies and redirect/HSTS are
then enabled. Production API requests require an authenticated Django session by
default; use `manage.py createsuperuser` and `/admin/login/` for a controlled demo.
Do not disable authentication on an internet-facing paid-GPU endpoint.
Run migrations and `collectstatic`; serve static files through the HTTPS frontend.
Use one Gunicorn worker with a small thread pool, for example:

```sh
gunicorn asar_project.wsgi:application --workers 1 --threads 4 --timeout 600 --bind 127.0.0.1:8000
```

One in-process request gate rejects concurrent expensive requests with 429. It is
not a distributed queue or per-user quota; multi-worker deployment needs a shared
admission-control design. Put this behind an HTTPS reverse proxy with matching
request limits/timeouts. Configure trusted proxy headers only for your actual
infrastructure. Run `manage.py check --deploy` under production settings before
release. CDN Leaflet/fonts/OSM tiles require network; answers remain visible without
the map. Tile-provider licensing/usage and availability need review for scale.

Implemented: real frozen routing/inference adapters, catalog validation, identity
checks, deterministic execution, route/latency/trace API, selected markers,
clarification/error UI, privacy-conscious logging and runtime configuration.

Future roadmap: live licensed POI ingestion, organizational access controls,
per-tenant quotas/retention, distributed request queue, geographic data freshness,
and richer site-selection objectives. Current multi-constraint semantics are only
category + direction + radius, not road travel, zoning, profitability or suitability
optimization. Do not present future workflows as implemented.

Frozen evidence remains unchanged: completed TEST favored Always Short (94.85%)
over Router V2 (90.87%) and Always Long (86.88%). The adaptive architecture is
implemented as requested, not advertised as a measured accuracy improvement.
This is a production-style controlled-demo application, not a claim of audited
large-scale production readiness. No research evaluation is rerun for deployment.

## Validation and current workspace command

From this workspace, dependencies and migrations are already installed in
`.venv-router`, and `aasr-production-long` has been deployed. Start the demo with:

```sh
AASR_MODAL_APP=aasr-production-long .venv-router/bin/python manage.py runserver 127.0.0.1:8000 --noreload
```

Open http://127.0.0.1:8000/. Modal uses the existing `joudalrubaish` profile.
A missing profile/backend fails visibly; there is no mock Long fallback.

Run lightweight backend checks with `.venv-router/bin/python manage.py test dashboard`.
Optional browser checks need Playwright and Chromium (development tools only):

```sh
uv pip install --python .venv-router/bin/python playwright
.venv-router/bin/python -m playwright install chromium
.venv-router/bin/python -m dashboard.browser_smoke
```

The browser check uses three synthetic Short demo queries and fixture-only error
responses. It does not request Long generations. `--long-response PATH` optionally
replays a previously saved API response for UI checks. Neither suite accesses
research datasets. See `PRODUCTION_VALIDATION.md` for measured outcomes and limits.
Production-settings deployment checks retain two intentional HSTS warnings:
subdomain coverage and preload require a real domain/HTTPS deployment decision.
