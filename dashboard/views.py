import json
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .services.model_service import ModelServiceError, analyze_spatial_question


def api_error(code, message, status):
    return JsonResponse(
        {"error": {"code": code, "message": message}},
        status=status,
        json_dumps_params={"ensure_ascii": False},
    )

def dashboard(request):
    return render(request, "dashboard/index.html")

@require_POST
def analyze_api(request):
    if request.content_type != "application/json":
        return api_error("invalid_content_type", "يجب إرسال الطلب بصيغة JSON.", 400)

    try:
        payload = json.loads(request.body or "{}")
        if not isinstance(payload, dict):
            return api_error("invalid_request", "يجب أن يكون الطلب كائن JSON.", 400)

        question = payload.get("question")
        if not isinstance(question, str):
            return api_error("invalid_question", "يجب أن يكون السؤال نصًا.", 400)
        question = question.strip()

        if not question:
            return api_error("empty_question", "يرجى كتابة سؤال مكاني أولاً.", 400)
        if len(question) > settings.ASAR_MAX_QUESTION_CHARS:
            return api_error(
                "question_too_long",
                f"يجب ألا يتجاوز السؤال {settings.ASAR_MAX_QUESTION_CHARS} حرفًا.",
                400,
            )

        result = analyze_spatial_question(question)
        return JsonResponse(result, json_dumps_params={"ensure_ascii": False})

    except json.JSONDecodeError:
        return api_error("invalid_json", "صيغة الطلب غير صحيحة.", 400)
    except UnicodeDecodeError:
        return api_error("invalid_encoding", "يجب ترميز الطلب باستخدام UTF-8.", 400)
    except ModelServiceError as exc:
        return api_error(exc.code, exc.public_message, exc.status_code)
    except Exception:
        return api_error("internal_error", "حدث خطأ غير متوقع.", 500)
