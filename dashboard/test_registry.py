"""Cheap production registry tests. Frozen parsers are stubbed, geometry is real."""
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock,patch
from django.test import SimpleTestCase,override_settings
from spatial import execute
from spatial.geometry import distance
from spatial.schema import StructuredQuery,EntityReference,ParsedContext,Location,OPERATIONS
from .services import registry
from .services.model_service import analyze_spatial_question

@override_settings(AASR_USE_RUNTIME_REGISTRY=True,AASR_LOG_REQUESTS=False)
class RegistryTests(SimpleTestCase):
    def setUp(self):
        self.points=registry.entities()
        p=self.points[0]
        self.anchor=Location('current_location','موقعي الحالي',p.latitude,p.longitude,None)
        self.context=ParsedContext(self.anchor,())
        self.origin=EntityReference(self.anchor.name,'context_anchor')

    def test_load_spatial_only_stable_identities(self):
        self.assertEqual(len(self.points),540)
        self.assertEqual(len({p.identity for p in self.points}),540)
        self.assertEqual(len(registry.preview_catalog()['candidates']),12)

    def test_category_and_location_selection(self):
        q=StructuredQuery('nearest_category',self.origin,categories=('مستشفى',))
        a=registry.execution_context(q,self.context)
        far=max(self.points,key=lambda p:distance(p.coordinates,self.anchor.coordinates))
        b=registry.execution_context(q,ParsedContext(Location('current_location','موقعي الحالي',*far.coordinates,None),()))
        self.assertTrue(all(p.category=='مستشفى' for p in a.candidates))
        self.assertNotEqual({p.identity for p in a.candidates},{p.identity for p in b.candidates})
        self.assertLessEqual(len(a.candidates),registry.LIMIT)

    def test_named_anchor_and_duplicate_name_ambiguity(self):
        counts=Counter(p.name for p in self.points)
        p=next(p for p in self.points if counts[p.name]==1 and p.name not in registry.CATEGORIES)
        ctx=registry.initial_context('ما أقرب صيدلية إلى «'+p.name+'»؟',None)
        self.assertEqual(ctx.anchor.identity,p.identity)
        other=Location('other',p.name,p.latitude+.01,p.longitude,p.category)
        with patch.object(registry,'entities',return_value=(p,other)):
            with self.assertRaises(registry.RetrievalIssue) as caught:registry.initial_context('أقرب صيدلية إلى «'+p.name+'»',None)
        self.assertEqual(caught.exception.status,'ambiguous')

    def test_radius_is_complete_and_overflow_abstains(self):
        q=StructuredQuery('count_within_radius',self.origin,categories=('صيدلية',),radius_km=2)
        context=registry.execution_context(q,self.context)
        expected=[p for p in self.points if p.category=='صيدلية' and distance(p.coordinates,self.anchor.coordinates)<=2]
        self.assertEqual({p.identity for p in context.candidates},{p.identity for p in expected})
        many=tuple(Location(str(i),'موقع '+str(i),*self.anchor.coordinates,'صيدلية') for i in range(201))
        with patch.object(registry,'entities',return_value=many):
            with self.assertRaises(registry.RetrievalIssue):registry.execution_context(q,self.context)

    def test_all_eight_retrieval_answers_match_full_registry_engine(self):
        for op in OPERATIONS:
            kw={}
            if op=='cardinal_direction':kw['targets']=(EntityReference(self.points[0].name),)
            if op=='closer_of_two':kw['targets']=tuple(EntityReference(p.name) for p in self.points[:2])
            if op in ('nearest_category','count_within_radius','within_radius_yes_no','spatial_multi_constraint'):kw['categories']=('صيدلية',)
            if op=='nearest_of_two_categories':kw['categories']=('صيدلية','مستشفى')
            if op=='two_hop_nearest':kw.update(first_hop_category='مستشفى',second_hop_category='صيدلية')
            if op in ('count_within_radius','within_radius_yes_no','spatial_multi_constraint'):kw['radius_km']=2
            if op=='spatial_multi_constraint':kw['direction']='شمال'
            q=StructuredQuery(op,self.origin,**kw)
            with self.subTest(operation=op):
                expected=execute(q,ParsedContext(self.anchor,self.points))
                actual=execute(q,registry.execution_context(q,self.context))
                self.assertEqual(actual.status,expected.status)
                self.assertEqual(actual.answer,expected.answer)

    def test_map_bounded_and_final_engine_is_used(self):
        runtime=Mock()
        def predict(question,ctx):
            q=StructuredQuery('nearest_category',EntityReference(ctx.anchor.name,'context_anchor'),categories=('مستشفى',))
            return SimpleNamespace(query=q,operation=q.operation,status='success',confidence=.9),{}
        runtime.predict_short.side_effect=predict;runtime.route.return_value=('SHORT',.2)
        with patch('dashboard.services.model_service.execute',wraps=execute) as engine:
            result=analyze_spatial_question('وين أقرب مستشفى؟',runtime=runtime,current_location={'lat':self.anchor.latitude,'lon':self.anchor.longitude})
        engine.assert_called_once()
        self.assertLessEqual(len(result['locations']),20)
        self.assertTrue(all(p['category']=='مستشفى' for p in result['locations'] if p['role']!='anchor'))
        runtime.predict_long.assert_not_called()
