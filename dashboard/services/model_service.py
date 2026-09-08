from .agents.verifier import verify_result
from .agents.logger import log_result


def analyze_spatial_question(question: str) -> dict:
    """Main integration point for the ASAR model.

    Replace the mock builders later with the real model + spatial retrieval.
    Keep the returned JSON schema stable so the UI does not need to change.
    """
    task_type = infer_demo_task_type(question)

    builders = {
        "nearest_category": build_nearest,
        "cardinal_direction": build_direction,
        "within_radius_yes_no": build_radius_yes_no,
        "closer_of_two": build_closer_of_two,
        "count_within_radius": build_count,
        "nearest_of_two_categories": build_two_categories,
        "two_hop_nearest": build_two_hop,
        "spatial_multi_constraint": build_multi_constraint,
    }

    result = builders[task_type]()
    result["question"] = question
    result["task_type"] = task_type
    result = verify_result(result)
    log_result(result)
    return result


def infer_demo_task_type(question: str) -> str:
    """Demo-only router for testing the dynamic UI before the final model is connected."""
    q = question.strip().lower()

    if any(x in q for x in ["كم ", "كم عدد", "عدد "]):
        return "count_within_radius"
    if any(x in q for x in ["اتجاه", "شمال", "جنوب", "شرق", "غرب"]):
        return "cardinal_direction"
    if any(x in q for x in ["هل يوجد", "هل فيه", "ضمن", "نطاق"]):
        return "within_radius_yes_no"
    if any(x in q for x in ["إلى أقرب", "الى اقرب", "ثم أقرب", "ثم اقرب"]):
        return "two_hop_nearest"
    if any(x in q for x in ["أيهما أقرب", "ايهما اقرب", "مقارنة"]):
        return "closer_of_two"
    if any(x in q for x in ["أقرب مدرسة أم", "أقرب مستشفى أم", "فئتين", "نوعين"]):
        return "nearest_of_two_categories"
    if any(x in q for x in ["ويبعد", "وقريب", "شروط", "معايير"]):
        return "spatial_multi_constraint"

    return "nearest_category"


def anchor():
    return {"name": "الموقع المرجعي", "lat": 24.7136, "lng": 46.6753, "role": "anchor"}


def build_nearest():
    return {
        "answer": {"value": "مستشفى الملك خالد الجامعي", "text": "أقرب موقع مطابق هو مستشفى الملك خالد الجامعي."},
        "reasoning": "تمت مقارنة المسافات بين الموقع المرجعي ونقاط الفئة المطلوبة.",
        "visualization": "nearest",
        "anchor": anchor(),
        "locations": [
            anchor(),
            {"name": "مستشفى الملك خالد الجامعي", "lat": 24.7207, "lng": 46.6267, "role": "answer", "category": "مستشفى", "distance_km": 1.4},
            {"name": "مستشفى بديل", "lat": 24.7008, "lng": 46.6902, "role": "alternative", "category": "مستشفى", "distance_km": 2.6},
        ],
        "metrics": {"distance_km": 1.4},
        "evidence": [{"label": "الفئة", "value": "مستشفى"}, {"label": "المسافة", "value": "1.4 كم"}],
        "comparison": [], "constraints": [],
    }


def build_direction():
    return {
        "answer": {"value": "شرق", "text": "الموقع الهدف يقع شرق الموقع المرجعي."},
        "reasoning": "تم تحديد الاتجاه من فرق الإحداثيات بين النقطتين.",
        "visualization": "direction",
        "anchor": anchor(),
        "locations": [anchor(), {"name": "الموقع الهدف", "lat": 24.7140, "lng": 46.7120, "role": "answer"}],
        "metrics": {"direction": "شرق"},
        "evidence": [{"label": "الاتجاه", "value": "شرق"}],
        "comparison": [], "constraints": [],
    }


def build_radius_yes_no():
    return {
        "answer": {"value": "نعم", "text": "نعم، يوجد عنصر مطابق داخل النطاق المحدد."},
        "reasoning": "تم فحص النقاط الواقعة داخل نصف القطر المطلوب.",
        "visualization": "radius_yes_no",
        "anchor": anchor(),
        "locations": [anchor(), {"name": "صيدلية قريبة", "lat": 24.7210, "lng": 46.6810, "role": "answer", "distance_km": 1.1}],
        "metrics": {"radius_km": 2.0, "nearest_distance_km": 1.1},
        "evidence": [{"label": "النطاق المطلوب", "value": "2 كم"}, {"label": "أقرب نقطة مطابقة", "value": "1.1 كم"}],
        "comparison": [], "constraints": [],
    }


def build_closer_of_two():
    return {
        "answer": {"value": "الموقع A", "text": "الموقع A هو الأقرب إلى الموقع المرجعي."},
        "reasoning": "تمت مقارنة المسافة من المرجع إلى الموقعين A و B.",
        "visualization": "comparison",
        "anchor": anchor(),
        "locations": [
            anchor(),
            {"name": "الموقع A", "lat": 24.7200, "lng": 46.6640, "role": "answer", "distance_km": 1.2},
            {"name": "الموقع B", "lat": 24.6990, "lng": 46.7020, "role": "alternative", "distance_km": 2.8},
        ],
        "metrics": {},
        "evidence": [],
        "comparison": [
            {"label": "الموقع A", "value": "1.2 كم", "score": 0.88},
            {"label": "الموقع B", "value": "2.8 كم", "score": 0.52},
        ],
        "constraints": [],
    }


def build_count():
    pts = [
        {"name": f"صيدلية {i}", "lat": 24.7136 + i * 0.004, "lng": 46.6753 + ((-1) ** i) * i * 0.004, "role": "answer"}
        for i in range(1, 7)
    ]
    return {
        "answer": {"value": 6, "text": "يوجد 6 نقاط مطابقة داخل النطاق المحدد."},
        "reasoning": "تم عد جميع النقاط المطابقة الواقعة داخل نصف القطر.",
        "visualization": "count",
        "anchor": anchor(),
        "locations": [anchor(), *pts],
        "metrics": {"radius_km": 3.0, "count": 6},
        "evidence": [{"label": "نصف القطر", "value": "3 كم"}, {"label": "عدد النتائج", "value": "6"}],
        "comparison": [], "constraints": [],
    }


def build_two_categories():
    return {
        "answer": {"value": "مدرسة", "text": "أقرب مدرسة أقرب من أقرب مستشفى."},
        "reasoning": "تم إيجاد أقرب عنصر من كل فئة ثم مقارنة المسافتين.",
        "visualization": "category_comparison",
        "anchor": anchor(),
        "locations": [
            anchor(),
            {"name": "مدرسة قريبة", "lat": 24.7180, "lng": 46.6700, "role": "answer", "distance_km": 0.8},
            {"name": "مستشفى قريب", "lat": 24.7270, "lng": 46.6900, "role": "alternative", "distance_km": 2.1},
        ],
        "metrics": {},
        "evidence": [],
        "comparison": [
            {"label": "مدرسة", "value": "0.8 كم", "score": 0.92},
            {"label": "مستشفى", "value": "2.1 كم", "score": 0.61},
        ],
        "constraints": [],
    }


def build_two_hop():
    return {
        "answer": {"value": "صيدلية الشفاء", "text": "النتيجة النهائية هي صيدلية الشفاء."},
        "reasoning": "تم أولًا إيجاد أقرب مستشفى من المرجع، ثم إيجاد أقرب صيدلية لذلك المستشفى.",
        "visualization": "two_hop",
        "anchor": anchor(),
        "locations": [
            anchor(),
            {"name": "المستشفى الوسيط", "lat": 24.7240, "lng": 46.6820, "role": "intermediate", "step": 1},
            {"name": "صيدلية الشفاء", "lat": 24.7290, "lng": 46.6880, "role": "answer", "step": 2},
        ],
        "metrics": {},
        "evidence": [{"label": "الخطوة 1", "value": "أقرب مستشفى"}, {"label": "الخطوة 2", "value": "أقرب صيدلية للمستشفى"}],
        "comparison": [], "constraints": [],
    }


def build_multi_constraint():
    return {
        "answer": {"value": "الموقع B", "text": "الموقع B يحقق جميع القيود المطلوبة."},
        "reasoning": "تم تقييم جميع المرشحين مقابل أكثر من شرط مكاني في الوقت نفسه.",
        "visualization": "multi_constraint",
        "anchor": anchor(),
        "locations": [
            anchor(),
            {"name": "الموقع A", "lat": 24.7048, "lng": 46.7241, "role": "alternative", "score": 0.72},
            {"name": "الموقع B", "lat": 24.7502, "lng": 46.6603, "role": "answer", "score": 0.86},
            {"name": "الموقع C", "lat": 24.6859, "lng": 46.6248, "role": "alternative", "score": 0.54},
        ],
        "metrics": {"suitability": 0.86},
        "evidence": [],
        "comparison": [
            {"label": "الموقع B", "value": "0.86", "score": 0.86},
            {"label": "الموقع A", "value": "0.72", "score": 0.72},
            {"label": "الموقع C", "value": "0.54", "score": 0.54},
        ],
        "constraints": [
            {"label": "قريب من المدارس", "passed": True},
            {"label": "بعيد عن المراكز الصحية الحالية", "passed": True},
            {"label": "سهولة الوصول عبر الطرق الرئيسية", "passed": True},
        ],
    }