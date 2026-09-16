"""Location integration checks with stub parsing and the real deterministic engine."""
import json
from types import SimpleNamespace
from unittest.mock import Mock,patch
from django.test import SimpleTestCase,Client,override_settings
from spatial.schema import StructuredQuery,EntityReference
from .services.model_service import analyze_spatial_question
from .services.catalog import load_catalog,parse_catalog
from .services.current_location import bind_location

@override_settings(AASR_USE_RUNTIME_REGISTRY=False,AASR_REQUIRE_LOGIN=False,AASR_LOG_REQUESTS=True)
class LocationTests(SimpleTestCase):
    def runtime(self):
        runtime=Mock()
        def predict(question,context):
            q=StructuredQuery('nearest_category',EntityReference(context.anchor.name,'context_anchor'),categories=('مستشفى',))
            return SimpleNamespace(query=q,operation=q.operation,status='success',confidence=.9),{}
        runtime.predict_short.side_effect=predict
        runtime.route.return_value=('SHORT',.2)
        return runtime

    def test_coordinates_change_engine_answer_without_persistence(self):
        runtime=self.runtime()
        with patch('dashboard.services.model_service.log_result') as log:
            a=analyze_spatial_question('وين أقرب مستشفى؟',runtime=runtime,current_location={'lat':24.724,'lon':46.682})
            b=analyze_spatial_question('وش أقرب مستشفى لي؟',runtime=runtime,current_location={'lat':24.704,'lon':46.69})
            log.assert_not_called()
        self.assertEqual(a['answer']['value'],'مستشفى النخيل')
        self.assertEqual(b['answer']['value'],'مستشفى الواحة')
        self.assertEqual(a['anchor']['name'],'موقعي الحالي')
        runtime.predict_long.assert_not_called()

    def test_missing_location_requires_clarification_and_original_request_works(self):
        runtime=self.runtime()
        result=analyze_spatial_question('وش أقرب صيدلية لي؟',runtime=runtime)
        self.assertEqual(result['status'],'needs_clarification')
        runtime.predict_short.assert_not_called()
        with patch('dashboard.services.model_service.log_result'):
            result=analyze_spatial_question('ما أقرب مستشفى إلى «مركز الحي»؟',runtime=runtime)
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['anchor']['name'],'مركز الحي')

    def test_validation_and_explicit_reference(self):
        ctx=parse_catalog(load_catalog())
        for value in ({'lat':True,'lon':0},{'lat':91,'lon':0},{'lat':0,'lon':float('nan')},{'lat':0},[]):
            with self.assertRaises(ValueError):bind_location('سؤال',ctx,value)
        unchanged,needs=bind_location('قل لي ما اتجاه مدرسة الرواد؟',ctx,None)
        self.assertFalse(needs)
        self.assertIs(unchanged,ctx)
        actual,_=bind_location('ما أقرب مستشفى إلى «مركز الحي»؟',ctx,{'lat':0,'lon':0})
        self.assertIs(actual,ctx)

    def test_api_optional_location(self):
        client=Client()
        with patch('dashboard.services.model_service.get_runtime',return_value=self.runtime()),patch('dashboard.services.model_service.log_result'):
            response=client.post('/api/analyze/',json.dumps({'question':'وين أقرب مستشفى؟','current_location':{'lat':24.704,'lon':46.69}}),content_type='application/json')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['anchor']['name'],'موقعي الحالي')
        self.assertEqual(response['Cache-Control'],'no-store')
