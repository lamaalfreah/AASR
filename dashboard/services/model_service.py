"""Grounded trained inference, optional Modal Long, deterministic spatial answers."""
import logging
import threading
from dataclasses import asdict
from time import perf_counter
from datetime import datetime,timezone
from django.conf import settings
from spatial import execute
from .adaptive_runtime import get_runtime
from .grounding import prepare,execution_context,Clarification
from .agents.verifier import verify_execution

logger=logging.getLogger(__name__)
_long_gate=threading.BoundedSemaphore(1)

class ModelServiceError(Exception):
    code='model_unavailable';status_code=502
    public_message='تعذر إكمال التحليل حاليًا. يرجى المحاولة مرة أخرى.'
class ModelTimeoutError(ModelServiceError):
    code='model_timeout';status_code=504
    public_message='استغرق التحليل وقتًا أطول من المتوقع. يرجى المحاولة مرة أخرى.'
class ModelConfigurationError(ModelServiceError):
    code='model_not_configured';status_code=503
    public_message='خدمة التحليل غير مهيأة بعد.'

VISUALS={'nearest_category':'nearest','cardinal_direction':'direction','within_radius_yes_no':'radius_yes_no',
 'closer_of_two':'comparison','count_within_radius':'count','nearest_of_two_categories':'category_comparison',
 'two_hop_nearest':'two_hop','spatial_multi_constraint':'multi_constraint'}


def empty_result(question):
    return dict(question=question,status='needs_clarification',task_type='general_spatial',reasoning_depth='concise',
        data_mode='osm_assisted',answer={'value':None,'text':'يرجى توضيح الموقع والفئة المطلوبة.'},
        reasoning='',explanation='',visualization='none',anchor=None,locations=[],metrics=dict(distance_km=None,
        radius_km=None,nearest_distance_km=None,count=None,direction=None,suitability=None),evidence=[],comparison=[],
        constraints=[],sources=[],limitations=['النتائج تخص المعالم المسماة في نسخة OpenStreetMap المحلية داخل منطقة الرياض؛ ليست بيانات حية أو تغطية شاملة.',
        f'البحث عن الأقرب محدود بنطاق {settings.AASR_SEARCH_KM:g} كم؛ المسافات مستقيمة وليست أزمنة قيادة.'],
        route=None,structured_query=None,trace=[],verification={'passed':None,'checks':[]})


def analyze_spatial_question(question,current_location=None):
    began=perf_counter();result=empty_result(question);latency={}
    try:
        t=perf_counter();parser_question,context=prepare(question,current_location);latency['grounding_ms']=(perf_counter()-t)*1000
        result['interpreted_question']=parser_question
        result['reference_name']=context.anchor.name
        t=perf_counter();runtime=get_runtime();latency['model_load_ms']=(perf_counter()-t)*1000
        t=perf_counter();short,route,probability=runtime.short_and_route(parser_question,context);latency['short_router_ms']=(perf_counter()-t)*1000
        result.update(route=route,task_type=short.operation,reasoning_depth='deep' if route=='LONG' else 'concise',
                      routing={'long_probability':probability,'threshold':runtime.threshold})
        result['trace']=[{'stage':'short','operation':short.operation,'status':short.status},{'stage':'router','route':route}]
        prediction=short
        if route=='LONG':
            if not _long_gate.acquire(blocking=False):raise Clarification('يوجد تحليل موسع قيد التنفيذ. حاول بعد قليل.')
            try:
                t=perf_counter();prediction=runtime.long(parser_question,context);latency['long_ms']=(perf_counter()-t)*1000
                result['trace'].append({'stage':'long','status':prediction.status,'calls':prediction.llm_calls,'backend':prediction.backend_metadata})
            finally:_long_gate.release()
        query=prediction.query
        if query is None:raise Clarification('تعذر فهم السؤال بثقة. وضّح اسم المرجع والفئة والنطاق؛ لن تُعرض نتيجة تخمينية.')
        result['structured_query']=asdict(query);result['task_type']=query.operation
        t=perf_counter();context=execution_context(query,context);execution=execute(query,context);latency['geo_ms']=(perf_counter()-t)*1000
        result['trace'].append({'stage':'geo','status':execution.status,'candidate_count':len(context.candidates)})
        result['execution']=execution.to_dict();result['verification']=verify_execution(execution)
        if execution.status!='success':raise Clarification('لم تتوفر نتيجة فريدة ضمن البيانات والقيود. وضّح الأسماء أو عدّل النطاق.')
        result.update(status='answered',answer={'value':execution.answer.value,'text':execution.answer.text},visualization=VISUALS[query.operation])
        present(result,execution)
        result['reasoning']=result['explanation']='فُهم السؤال ثم حُددت المواقع؛ حُسبت النتيجة بمحرك مكاني حتمي.'
    except Clarification as exc:
        result['answer']={'value':None,'text':str(exc)}
        result['status']='needs_clarification'
    except Exception as exc:
        # Exception text may contain provider credentials or location context.
        logger.error('Adaptive request failed (%s)',type(exc).__name__)
        if isinstance(exc,TimeoutError) or 'Timeout' in type(exc).__name__:raise ModelTimeoutError() from None
        raise ModelServiceError() from None
    result['latency']={**latency,'total_ms':(perf_counter()-began)*1000}
    return result


def present(result,execution):
    trace=execution.trace
    def point(raw,role):
        parts=raw['identity'].split(':')
        return dict(name=raw['name'],lat=raw['latitude'],lng=raw['longitude'],role=role,category=raw['category'],
                    distance_km=None,step=None,score=None,source_ref=raw['identity'],
                    source='openstreetmap' if parts[0]=='osm' else 'user',
                    osm_type=parts[1] if parts[0]=='osm' else None,osm_id=parts[2] if parts[0]=='osm' else None)
    origin=point(trace['resolved_origin'],'anchor');result['anchor']=origin;result['locations']=[origin]
    seen={origin['source_ref']}
    for role,key in [('intermediate','first_hop_result'),('answer','selected_candidates')]:
        points=trace.get(key,[])
        if len(points)>40:result['limitations'].append('الخريطة تعرض أول 40 موقعًا فقط؛ الحساب يشمل جميع المواقع المطابقة في البيانات المحلية.')
        for raw in points[:40]:
            if raw['identity'] not in seen:
                result['locations'].append(point(raw,role));seen.add(raw['identity'])
    result['metrics']['radius_km']=trace.get('radius_km')
    for step in trace.get('steps',[]):
        if 'distance_km' in step:
            result['metrics']['distance_km']=step['distance_km']
            result['evidence'].append({'label':'المسافة المستقيمة','value':f"{step['distance_km']:.3f} كم"})
        if 'qualifying_count' in step:result['metrics']['count']=step['qualifying_count']
        if 'label' in step:result['metrics']['direction']=step['label']
    result['sources']=[dict(provider='OpenStreetMap (local PBF)',attribution='© OpenStreetMap contributors',
                           url='https://www.openstreetmap.org/copyright',retrieved_at=datetime.now(timezone.utc).isoformat(),data_timestamp=None)]
