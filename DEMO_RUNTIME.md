# Trained adaptive demo runtime

The active request path is local OSM grounding → frozen Neural Short → frozen
calibrated Router V2 → optional existing Modal Qwen LongParser → validated
StructuredQuery → the research deterministic `spatial.execute` engine → UI answer.
The Router threshold is read from `router_v2.joblib`; it is not configurable.
There is no OpenAI call, rule-based task classifier, forced Long fallback, or LLM
spatial calculation in this path. Historical integrations remain available.

## Run

Use the existing Python 3.12 environment (`.venv`) or install
`requirements-serving.txt`. The Short weights/config and Router artifact must
exist in their current `experiments/E5_lightweight_adaptive` directories.
Set `ASAR_OSM_PBF_PATH` to the existing PBF. Never commit `.env`, PBFs or `.runtime`.

```sh
.venv/bin/python manage.py check
.venv/bin/python manage.py runserver 127.0.0.1:8000 --noreload
```

Open http://127.0.0.1:8000/. The first request builds a named-POI index if the PBF,
bounding box or exporter version changed. The current workspace cache is already
built. Subsequent requests use the disk/in-memory cache. Restart after settings,
PBF, Python or cached Django template changes.

Modal authentication must point to the account holding the already deployed
`aasr-step2b-qwen3-4b-long` / `QwenLong`, revision
`1cfa9a7208912126459214e8b04321603b3df60c`. This app is referenced, never redeployed
by Django. Only a Router-selected LONG request invokes it. Frozen structured-output
retry behavior remains at most two calls. A busy Long path asks the user to retry;
Short requests do not wait for a GPU. Failed Long calls never silently use Short.
The existing remote checkpoint cache deduplicates identical messages/revision.
It retains question/place-name content; no local precise-coordinate logs are added.

## Grounding and geographic scope

Arabic/English/alternate OSM names are normalized and matched to stable OSM
identities. Matched proper names are quoted for the trained tokenizer; a configured
reference is made explicit when the user omitted one. These are reference-binding
operations, not task classification. Ambiguous names ask for clarification rather
than choosing the first geocoder result. Explicit unknown references fail closed.

Default coverage is the local Riyadh bounding box `46.3,24.3,47.1,25.1`.
`AASR_OSM_BBOX` configures it. Default reference is **وسط الرياض** at the configured
city center, explicitly disclosed in the UI. This is not the user's GPS location.
The API optionally accepts `current_location: {lat, lon}`; coordinates are not
passed to Qwen. Browser GPS acquisition is not added in this focused integration.
Named places take precedence over the optional coordinate reference.

The parser receives a bounded, category-balanced nearby name context. Execution
uses all qualifying local POIs, not the parser subset or an arbitrary first 80
records. Counts/radius checks are not truncated. Nearest searches are explicitly
bounded to 15 km by default; two-hop retrieves the second category around the
first entity selected by the deterministic engine. All straight-line distances
and four-sector directions come from that engine, not the old eight-sector UI
verifier. `AASR_SEARCH_KM` controls the default search extent; radius requests above
`ASAR_GEO_MAX_RADIUS_M` ask for clarification.

Examples (no guaranteed route override):

- كم عدد الصيدليات ضمن 2 كم من وسط الرياض؟
- أيهما أقرب: مستشفى أم صيدلية؟
- ما أقرب مستشفى؟
- ما أقرب صيدلية إلى أقرب مستشفى؟

## Limits

OSM names, tagging and local snapshot coverage are incomplete. Unnamed facilities
are excluded; results describe the indexed named facilities, not an exhaustive
real-world count. No live Google/OSM calls are required. The environment recognizes
`ASAR_GOOGLE_MAPS_API_KEY` (and legacy `GOOGLE_MAPS_API_KEY`), but this path deliberately
uses the available local gazetteer; remote geocoding is not silently substituted.
Entities outside local coverage require different data/configuration. An oversized
radius is not silently reduced. Frozen language models can still misinterpret
valid questions; inspect the visible parsed query/route/trace when demonstrating.
Cold model loading and Modal startup differ substantially from warm Short latency.
This is a local controlled demo, not an authenticated, quota-managed public service.

## Validation on 2026-09-21

- `manage.py check`, JavaScript syntax and `git diff --check`: passed.
- Three small place-grounding regressions: passed; no model/provider calls.
- Actual Short + Router + Geo: pharmacy count, category comparison and a named
  direction question answered successfully. Observed warm request times were
  approximately 3–6 ms; a separate cold Django API request was about 1.31 s.
- One real Router-selected Long request (`ما أقرب مستشفى؟`) succeeded with
  `nearest_category`, one generation, existing Modal Qwen, cached weights,
  NVIDIA L4/BF16. Total request 28.10 s; reported model loading 6.57 s and
  inference 6.11 s. These are individual smoke observations, not benchmarks.
- The 2,914-entity OSM index is derived from the configured local PBF. Rebuilding
  with explicit Arabic/alternate name attributes took 57.18 s, outside warm serving.
- All three frozen artifact hashes matched before/after. No training, threshold
  changes, research evaluation or deployment replacement was performed.
