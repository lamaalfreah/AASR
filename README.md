# ASAR — Arabic Spatial Adaptive Reasoning

واجهة Django عربية أحادية الصفحة، مصممة لتكون سهلة الربط لاحقًا مع مخرجات مودل ASAR.

## التشغيل

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

## أين أربط المودل؟

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

## أهم الملفات

- `dashboard/templates/dashboard/index.html` الواجهة
- `dashboard/static/dashboard/css/style.css` التصميم
- `dashboard/static/dashboard/js/app.js` الديناميكية والربط مع API
- `dashboard/views.py` Django endpoint
- `dashboard/services/model_service.py` مكان ربط مودل ASAR
