import json
import logging
from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST, require_GET
from .services.model_service import analyze_spatial_question
from .services.catalog import load_catalog, parse_catalog

logger=logging.getLogger(__name__)

@require_GET
def dashboard(request):
    return render(request,'dashboard/index.html',{'catalog':load_catalog()})

@require_POST
def analyze_api(request):
    if settings.AASR_REQUIRE_LOGIN and not request.user.is_authenticated:
        return JsonResponse({"error":"يرجى تسجيل الدخول قبل استخدام خدمة التحليل."},status=401)
    if request.content_type!='application/json':
        return JsonResponse({'error':'يجب إرسال الطلب بصيغة JSON.'},status=415)
    try:
        payload=json.loads(request.body,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('قيمة رقمية غير صالحة.')))
        if not isinstance(payload,dict) or set(payload)-{'question','catalog','current_location'}:
            raise ValueError('يسمح فقط بالسؤال وكتالوج المواقع.')
        if payload.get('catalog') is not None and not settings.AASR_ALLOW_CUSTOM_CATALOG:
            raise ValueError('الكتالوج المخصص غير متاح في هذا النشر.')
        result=analyze_spatial_question(payload.get('question'),payload.get('catalog'),current_location=payload.get('current_location'))
        code=503 if result['status']=='unavailable' else 429 if result['status']=='busy' else 200
        response=JsonResponse(result,status=code,json_dumps_params={'ensure_ascii':False,'allow_nan':False})
        response['Cache-Control']='no-store'
        if code==429:response['Retry-After']='5'
        return response
    except RequestDataTooBig:
        return JsonResponse({"error":"حجم الطلب أكبر من الحد المسموح."},status=413)
    except (ValueError,UnicodeDecodeError) as exc:
        return JsonResponse({'error':str(exc) if not isinstance(exc,json.JSONDecodeError) else 'صيغة JSON غير صالحة.'},status=400)
    except Exception:
        logger.exception('AASR API failure')
        return JsonResponse({'error':'تعذر معالجة الطلب مؤقتًا.'},status=503)
