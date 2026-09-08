## Research & Model Development

The repository also contains the research and machine-learning workflow for AASR.

### Project Documentation

* [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — current project stage, main findings, and planned experiments.
* [`ARTIFACT_REGISTRY.md`](ARTIFACT_REGISTRY.md) — available trained artifacts and their Modal storage locations.
* [`docs/configuration_audit.md`](docs/configuration_audit.md) — configuration review against the original AdaMix framework.

### Machine Learning Code

* [`training/train/`](training/train/) — original Short/Long adapter and continuous-router pipeline.
* [`training/ASAR/`](training/ASAR/) — adaptive alpha-mixing and later Router experiments.

### Experiments

Experimental runs are documented under:

[`experiments/`](experiments/)

Current structure:

```text
E0_current/               Current baseline
E1_balanced_data/         Dataset balancing
E2_english_diagnostic/    English diagnostic
E3_adapter_improvement/   Adapter configuration improvement
E4_router/                Router evaluation
```

See [`experiments/README.md`](experiments/README.md) for the experiment-recording convention.

### Model & Experiment Storage

Large model weights, checkpoints, and generated outputs are stored outside GitHub.

Primary Modal volume:

```text
asar-artifacts
```

Before starting a new training run, check [`ARTIFACT_REGISTRY.md`](ARTIFACT_REGISTRY.md) to avoid unnecessary retraining.



## ASAR Frontend 

واجهة Django عربية أحادية الصفحة، مصممة لتكون سهلة الربط لاحقًا مع مخرجات مودل ASAR.

### التشغيل

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

ثم افتحي:

`http://127.0.0.1:8000/`

### أين أربط المودل؟

الربط مقصود أن يكون في ملف واحد فقط:

`dashboard/services/model_service.py`

الدالة الحالية:

```python
analyze_spatial_question(question)
```

ترجع Mock JSON حتى تعمل الواجهة الآن.

لاحقًا استبدلي محتواها باستدعاء المودل، مع الحفاظ على نفس شكل المخرجات:

```python
{
    "best_option": {...},
    "ranking": [...],
    "reasons": [...],
    "evidence": [...],
    "metrics": {...},
    "map_center": {...},
    "candidates": [...]
}
```

الـ frontend يرسل السؤال إلى:

`POST /api/analyze/`

ثم يحدث كل أجزاء الصفحة تلقائيًا من JSON.

### أهم الملفات

- `dashboard/templates/dashboard/index.html` الواجهة
- `dashboard/static/dashboard/css/style.css` التصميم
- `dashboard/static/dashboard/js/app.js` الديناميكية والربط مع API
- `dashboard/views.py` Django endpoint
- `dashboard/services/model_service.py` مكان ربط مودل ASAR
