import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .services.model_service import analyze_spatial_question

def dashboard(request):
    return render(request, "dashboard/index.html")

@require_POST
def analyze_api(request):
    try:
        payload = json.loads(request.body or "{}")
        question = (payload.get("question") or "").strip()

        if not question:
            return JsonResponse(
                {"error": "يرجى كتابة سؤال مكاني أولاً."},
                status=400
            )

        result = analyze_spatial_question(question)
        return JsonResponse(result, json_dumps_params={"ensure_ascii": False})

    except json.JSONDecodeError:
        return JsonResponse({"error": "صيغة الطلب غير صحيحة."}, status=400)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
