"""Small production contract tests; no research datasets or gold labels."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
from django.test import SimpleTestCase, Client, override_settings
from spatial.schema import StructuredQuery, EntityReference, OPERATIONS
from .services.catalog import load_catalog, parse_catalog
from .services.model_service import analyze_spatial_question
from .services.runtime import verify_frozen


@override_settings(AASR_USE_RUNTIME_REGISTRY=False,AASR_LOG_REQUESTS=False,AASR_REQUIRE_LOGIN=False)
class PipelineTests(SimpleTestCase):
    def setUp(self):
        self.catalog=load_catalog()
        self.origin=EntityReference('مركز الحي','context_anchor')

    def query(self,op):
        kw={}
        if op=='cardinal_direction':kw['targets']=(EntityReference('مدرسة الرواد'),)
        if op=='closer_of_two':kw['targets']=(EntityReference('مستشفى النخيل'),EntityReference('مستشفى الواحة'))
        if op in ('nearest_category','within_radius_yes_no','count_within_radius','spatial_multi_constraint'):kw['categories']=('صيدلية',)
        if op=='nearest_of_two_categories':kw['categories']=('مستشفى','مدرسة')
        if op=='two_hop_nearest':kw.update(first_hop_category='مستشفى',second_hop_category='صيدلية')
        if op in ('within_radius_yes_no','count_within_radius','spatial_multi_constraint'):kw['radius_km']=3
        if op=='spatial_multi_constraint':kw['direction']='شمال'
        return StructuredQuery(op,self.origin,**kw)

    def runtime(self,q,route='SHORT'):
        runtime=Mock()
        runtime.predict_short.return_value=(SimpleNamespace(query=q,operation=q.operation,status='success',confidence=.9),{})
        runtime.route.return_value=(route,.95 if route=='LONG' else .2)
        runtime.predict_long.return_value=SimpleNamespace(query=q,status='success',llm_calls=1)
        return runtime

    def test_all_eight_execute_real_engine_and_return_primitive_trace(self):
        expected={'nearest_category':'صيدلية الندى','cardinal_direction':'شمال',
                  'within_radius_yes_no':True,'closer_of_two':'مستشفى النخيل',
                  'count_within_radius':2,'nearest_of_two_categories':'مدرسة الرواد',
                  'two_hop_nearest':'صيدلية النور','spatial_multi_constraint':'صيدلية النور'}
        for op in OPERATIONS:
            with self.subTest(op=op):
                runtime=self.runtime(self.query(op))
                result=analyze_spatial_question('سؤال تجريبي',self.catalog,runtime=runtime)
                self.assertEqual(result['status'],'success')
                self.assertEqual(result['execution']['operation'],op)
                self.assertTrue(result['locations'])
                self.assertEqual(result['answer'],result['execution']['answer'])
                self.assertEqual(result['answer']['value'],expected[op])
                self.assertEqual(result['validation']['status'],'valid')
                json.dumps(result,allow_nan=False)
                runtime.predict_long.assert_not_called()

    def test_long_is_selected_before_execution(self):
        runtime=self.runtime(self.query('nearest_category'),'LONG')
        runtime.predict_long.return_value.query=self.query('two_hop_nearest')
        result=analyze_spatial_question('سؤال',self.catalog,runtime=runtime)
        self.assertEqual(result['route'],'LONG')
        self.assertEqual(result['task_type'],'two_hop_nearest')
        self.assertEqual([p['role'] for p in result['locations'] if p['role']!='candidate'],['anchor','intermediate','answer'])
        runtime.predict_long.assert_called_once()

    def test_ambiguity_blocks_execution(self):
        self.catalog['candidates'].append(dict(self.catalog['candidates'][4],lat=24.8))
        runtime=self.runtime(self.query('cardinal_direction'))
        with patch('dashboard.services.model_service.execute') as execute:
            result=analyze_spatial_question('سؤال',self.catalog,runtime=runtime)
            execute.assert_not_called()
        self.assertEqual(result['status'],'ambiguous');self.assertIsNone(result['answer'])

    def test_unavailable_long_never_silently_falls_back(self):
        runtime=self.runtime(self.query('nearest_category'),'LONG')
        runtime.predict_long.side_effect=RuntimeError('private backend credential')
        result=analyze_spatial_question('سؤال',self.catalog,runtime=runtime)
        self.assertEqual(result['status'],'unavailable');self.assertIsNone(result['answer'])
        self.assertNotIn('credential',json.dumps(result))

    def test_invalid_short_requires_clarification(self):
        runtime=self.runtime(self.query('nearest_category'))
        runtime.predict_short.return_value[0].query=None
        runtime.predict_short.return_value[0].status='invalid_query'
        result=analyze_spatial_question('سؤال',self.catalog,runtime=runtime)
        self.assertEqual(result['status'],'invalid_query');self.assertIsNone(result['execution'])
        runtime.predict_long.assert_not_called()

    def test_catalog_rejects_gold_invalid_coordinates_and_unsupported_category(self):
        for change in ({'gold_answer':'x'},{'candidates':[]},{'anchor':{'name':'x','lat':True,'lng':0}}):
            with self.assertRaises(ValueError):parse_catalog({**self.catalog,**change})

    def test_input_boundary_csrf_and_api_wiring(self):
        client=Client()
        for payload in ([],{'question':12},{'question':''},{'question':'x'*2001},{'question':'x','task_type':'nearest_category'}):
            self.assertEqual(client.post('/api/analyze/',json.dumps(payload),content_type='application/json').status_code,400)
        self.assertEqual(Client(enforce_csrf_checks=True).post('/api/analyze/','{}',content_type='application/json').status_code,403)
        runtime=self.runtime(self.query('count_within_radius'))
        with patch('dashboard.services.model_service.get_runtime',return_value=runtime):
            response=client.post('/api/analyze/',json.dumps({'question':'كم عدد الصيدليات؟'}),content_type='application/json')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['answer']['value'],2)
        self.assertContains(client.get('/'),'location-catalog')

    def test_frozen_artifact_integrity(self):
        verify_frozen()

    def test_long_abstention_and_unknown_entity_do_not_execute(self):
        runtime=self.runtime(self.query('nearest_category'),'LONG')
        runtime.predict_long.return_value=SimpleNamespace(query=None,status='needs_clarification',llm_calls=1)
        with patch('dashboard.services.model_service.execute') as execute:
            result=analyze_spatial_question('سؤال',self.catalog,runtime=runtime)
            execute.assert_not_called()
        self.assertEqual(result['status'],'needs_clarification')
        self.assertIsNone(result['answer'])
        unknown=StructuredQuery('cardinal_direction',self.origin,targets=(EntityReference('اسم غير موجود'),))
        with patch('dashboard.services.model_service.execute') as execute:
            result=analyze_spatial_question('سؤال',self.catalog,runtime=self.runtime(unknown))
            execute.assert_not_called()
        self.assertEqual(result['status'],'not_found')
        self.assertIsNone(result['answer'])

    def test_api_auth_limits_unavailability_and_admission_control(self):
        from .services.model_service import _gate
        client=Client()
        with override_settings(AASR_REQUIRE_LOGIN=True):
            self.assertEqual(client.post('/api/analyze/','{}',content_type='application/json').status_code,401)
        self.assertEqual(client.post('/api/analyze/','{}',content_type='text/plain').status_code,415)
        self.assertEqual(client.post('/api/analyze/','x'*262145,content_type='application/json').status_code,413)
        with patch('dashboard.services.model_service.get_runtime',side_effect=RuntimeError('offline')):
            response=client.post('/api/analyze/',json.dumps({'question':'سؤال'}),content_type='application/json')
        self.assertEqual(response.status_code,503)
        self.assertIsNone(response.json()['answer'])
        _gate.acquire()
        try:
            response=client.post('/api/analyze/',json.dumps({'question':'سؤال'}),content_type='application/json')
        finally:
            _gate.release()
        self.assertEqual(response.status_code,429)
        self.assertEqual(response['Retry-After'],'5')
        self.assertIsNone(response.json()['answer'])

    @override_settings(AASR_MODAL_APP='test-only')
    def test_remote_timeout_cancels_call_and_identity_mismatch_fails_closed(self):
        from .services.runtime import ModalProvider, Unavailable
        call=Mock();call.get.side_effect=TimeoutError()
        with patch('modal.Cls.from_name') as remote:
            remote.return_value.return_value.generate.spawn.return_value=call
            with self.assertRaises(Unavailable):ModalProvider().complete([])
        call.cancel.assert_called_once()
        call=Mock();call.get.return_value={'gpu':'wrong device'}
        with patch('modal.Cls.from_name') as remote:
            remote.return_value.return_value.generate.spawn.return_value=call
            with self.assertRaises(Unavailable):ModalProvider().complete([])

    def test_real_cpu_short_router_feature_contract(self):
        from .services.runtime import get_runtime
        from language.router_v2 import encode_v2
        runtime=get_runtime()
        before=len(runtime.short.model._forward_hooks)
        prediction,features=runtime.predict_short('كم عدد الصيدليات ضمن 2 كم من «مركز الحي»؟',parse_catalog(self.catalog))
        self.assertEqual(len(runtime.short.model._forward_hooks),before)
        self.assertEqual(set(features['operation_probabilities']),set(OPERATIONS))
        self.assertAlmostEqual(sum(features['operation_probabilities'].values()),1,places=5)
        self.assertEqual(prediction.operation,'count_within_radius')
        encode_v2(features)
        route,probability=runtime.route(features)
        self.assertEqual(route,'LONG' if probability>=.91 else 'SHORT')
        self.assertEqual(runtime.router.threshold,.91)
        self.assertEqual(len(runtime.router.ensemble.components),3)

    def test_empty_spatial_matches_are_not_backend_failures(self):
        for op,expected in [('count_within_radius',0),('within_radius_yes_no',False)]:
            query=StructuredQuery(op,self.origin,categories=('صيدلية',),radius_km=.01)
            result=analyze_spatial_question('سؤال',self.catalog,runtime=self.runtime(query))
            self.assertEqual(result['status'],'success')
            self.assertEqual(result['answer']['value'],expected)
            self.assertEqual(len([p for p in result['locations'] if p['role']!='candidate']),1)
        query=StructuredQuery('spatial_multi_constraint',self.origin,categories=('صيدلية',),radius_km=.01,direction='شمال')
        result=analyze_spatial_question('سؤال',self.catalog,runtime=self.runtime(query))
        self.assertEqual(result['status'],'not_found')
        self.assertIsNone(result['answer'])
