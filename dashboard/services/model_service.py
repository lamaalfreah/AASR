"""Production Arabic -> frozen parsing/routing -> identity -> deterministic Geo."""
import logging
import threading
import uuid
from dataclasses import asdict
from time import perf_counter
from django.conf import settings
from spatial import execute
from .catalog import load_catalog, parse_catalog
from .registry import initial_context, execution_context, RetrievalIssue
from .current_location import bind_location
from .runtime import get_runtime
from .agents.verifier import validate_identity
from .agents.logger import log_result

logger=logging.getLogger(__name__)
# Bound in-flight work in this process; production uses one worker (see runbook).
_gate=threading.BoundedSemaphore(1)
MESSAGES={
 'ambiguous':'يوجد أكثر من موقع مطابق. وضّح الاسم أو حدّد موقعًا مميزًا في الكتالوج.',
 'not_found':'لم يُعثر على موقع مطابق ضمن البيانات والقيود المتاحة.',
 'needs_clarification':'يرجى توضيح المرجع والفئة أو الأسماء والنطاق المطلوب.',
 'invalid_query':'تعذر تفسير السؤال بدقة. وضّح الموقع المرجعي والقيود المطلوبة.',
 'unsupported':'هذا الطلب خارج العمليات المكانية الثماني المدعومة.',
 'unavailable':'خدمة التحليل غير متاحة مؤقتًا. يرجى المحاولة لاحقًا.',
 'busy':'يوجد طلب قيد المعالجة. يرجى المحاولة بعد قليل.'}
VISUALS={'nearest_category':'nearest','cardinal_direction':'direction','two_hop_nearest':'two_hop',
 'within_radius_yes_no':'radius_yes_no','count_within_radius':'count',
 'closer_of_two':'comparison','nearest_of_two_categories':'category_comparison','spatial_multi_constraint':'multi_constraint'}


def analyze_spatial_question(question, catalog=None, *, runtime=None, current_location=None):
    if not isinstance(question,str) or not question.strip() or len(question)>settings.AASR_MAX_QUESTION_CHARS or any(ord(c)<32 and c not in '\n\t\r' for c in question):
        raise ValueError('يرجى كتابة سؤال صالح بطول لا يتجاوز 2000 حرف.')
    question=question.strip()
    registry_mode=catalog is None and settings.AASR_USE_RUNTIME_REGISTRY
    try:
        context=initial_context(question,current_location) if registry_mode else parse_catalog(load_catalog() if catalog is None else catalog)
        context,needs_location=bind_location(question,context,current_location)
    except RetrievalIssue as issue:
        return dict(question=question,status=issue.status,message=issue.message,answer=None,locations=[],trace=[],latency={'total_ms':0})
    started=perf_counter(); timings={}; trace=[]
    result=dict(request_id=uuid.uuid4().hex, question=question,status='unavailable',route=None,
                task_type=None,structured_query=None,answer=None,execution=None,validation={'status':'not_run'},
                anchor=None,locations=[],metrics={},evidence=[],comparison=[],constraints=[],trace=trace)
    if needs_location:
        result.update(status='needs_clarification',message='شارك موقعك عبر «استخدم موقعي» أو حدّد موقعًا مرجعيًا بالاسم.',latency={'total_ms':0})
        return result
    if not _gate.acquire(blocking=False):
        result.update(status='busy',message=MESSAGES['busy'],latency={'total_ms':0})
        return result
    try:
        t=perf_counter(); runtime=runtime or get_runtime(); timings['load_ms']=(perf_counter()-t)*1000
        t=perf_counter(); short,features=runtime.predict_short(question,context);timings['short_ms']=(perf_counter()-t)*1000
        trace.append({'stage':'short','status':short.status,'operation':short.operation,'confidence':short.confidence})
        t=perf_counter(); route,prob=runtime.route(features);timings['router_ms']=(perf_counter()-t)*1000
        if route not in ('SHORT','LONG'):raise RuntimeError('Invalid router route')
        result.update(route=route, routing={'long_probability':prob,'threshold':.91},task_type=short.operation)
        trace.append({'stage':'router','route':route,'long_probability':prob,'threshold':.91})
        prediction=short
        if route=='LONG':
            t=perf_counter();prediction=runtime.predict_long(question,context);timings['long_ms']=(perf_counter()-t)*1000
            trace.append({'stage':'long','status':prediction.status,'generation_calls':prediction.llm_calls})
        query=prediction.query
        if query is None:
            result['status']=prediction.status if prediction.status in MESSAGES else 'needs_clarification'
        else:
            result.update(structured_query=asdict(query),task_type=query.operation,visualization=VISUALS[query.operation])
            t=perf_counter()
            if registry_mode:context=execution_context(query,context)
            validation=validate_identity(query,context);result['validation']=validation
            trace.append({'stage':'identity','status':validation['status']})
            if validation['status']!='valid':
                result['status']=validation['status']
            else:
                execution=execute(query,context)
                result.update(status=execution.status,execution=execution.to_dict(),answer=asdict(execution.answer) if execution.answer else None)
                trace.append({'stage':'geo_engine','status':execution.status,'steps':execution.trace['steps']})
                present_execution(result, execution.trace)
                visible={p['identity'] for p in result['locations']}
                categories=set(query.categories)|{query.first_hop_category,query.second_hop_category}
                names={ref.name for ref in query.targets}
                for candidate in context.candidates:
                    if len(result['locations'])>=20:break
                    if candidate.identity not in visible and ((candidate.category is not None and candidate.category in categories) or candidate.name in names):
                        result['locations'].append(dict(identity=candidate.identity,name=candidate.name,category=candidate.category,
                                                       lat=candidate.latitude,lng=candidate.longitude,role='candidate'))
            timings['validation_geo_ms']=(perf_counter()-t)*1000
    except RetrievalIssue as issue:
        result.update(status=issue.status,answer=None)
        result['retrieval_message']=issue.message
    except Exception:
        logger.error('AASR request failed request_id=%s route=%s',result['request_id'],result['route'],exc_info=current_location is None)
        result.update(status='unavailable',answer=None,locations=[],anchor=None)
    finally:
        _gate.release()
    result['latency']={**timings,'total_ms':(perf_counter()-started)*1000}
    result['message']=result.pop('retrieval_message',None) or MESSAGES.get(result['status'],'تم تنفيذ الحساب المكاني على الكتالوج المتاح.')
    result['reasoning']='فهم اللغة ثم التحقق من الهوية؛ جميع الحسابات من المحرك الجغرافي الحتمي.' if result['status']=='success' else result['message']
    result['verification']={'passed':result['status']=='success' and result['validation']['status']=='valid','checks':[{'check':'identity','passed':result['validation']['status']=='valid'}]}
    logger.info('AASR request_id=%s route=%s status=%s latency_ms=%.1f',result['request_id'],result['route'],result['status'],result['latency']['total_ms'])
    if settings.AASR_LOG_REQUESTS and current_location is None:log_result(result)
    return result


def present_execution(result, trace):
    """Map and evidence use executor-selected identities, never language guesses."""
    def point(p,role):
        return dict(identity=p['identity'],name=p['name'],category=p['category'],lat=p['latitude'],lng=p['longitude'],role=role)
    origin=trace.get('resolved_origin')
    if origin:
        result['anchor']=point(origin,'anchor');result['locations'].append(result['anchor'])
    seen={origin['identity']} if origin else set()
    for role,key in [('intermediate','first_hop_result'),('answer','selected_candidates')]:
        for p in trace.get(key,[]):
            if p['identity'] not in seen:
                result['locations'].append(point(p,role));seen.add(p['identity'])
    if trace.get('radius_km') is not None:result['metrics']['radius_km']=trace['radius_km']
    for step in trace.get('steps',[]):
        if 'distance_km' in step:
            result['metrics']['distance_km']=round(step['distance_km'],3)
            result['evidence'].append({'label':step['step'],'value':f"{step['distance_km']:.3f} كم"})
        if 'qualifying_count' in step:result['metrics']['count']=step['qualifying_count']
        if 'label' in step:result['metrics']['direction']=step['label']
    result['evidence'].append({'label':'التحقق','value':'هويات المواقع والحسابات من المحرك الحتمي'})
