import math
import unittest
from language.neural_short import ShortPrediction
from spatial.schema import ParsedContext, Location, StructuredQuery, EntityReference
from evaluation.router_data import safe_features, label, slots


class RouterDataTests(unittest.TestCase):
    def test_label_precedence_and_failure(self):
        self.assertEqual(label(True,True),'SHORT')
        self.assertEqual(label(True,False),'SHORT')
        self.assertEqual(label(False,True),'LONG')
        self.assertEqual(label(False,False),'FAILURE')

    def test_features_measure_uncertainty_and_public_identity(self):
        ctx=ParsedContext(Location('a','مرجع',0,0),(
            Location('1','متكرر',1,1),Location('2','متكرر',2,2)))
        query=StructuredQuery('cardinal_direction',EntityReference('مرجع','context_anchor'),
                              (EntityReference('متكرر','unspecified'),))
        prediction=ShortPrediction('cardinal_direction',.5,query,'success')
        f=safe_features('أين متكرر بالنسبة إلى مرجع؟',ctx,prediction,[.5,.5,0,0,0,0,0,0])
        self.assertAlmostEqual(f['short_entropy'],math.log(2))
        self.assertEqual(f['short_top1_top2_margin'],0)
        self.assertEqual(f['entity_ambiguity_count'],1)
        self.assertEqual(f['constraint_count'],1)
        self.assertEqual(f['missing_slots'],[])
        self.assertNotIn('task_type',f)
        self.assertNotIn('source_anchor',f)

    def test_unavailable_partial_slots_are_not_fabricated(self):
        ctx=ParsedContext(Location('a','مرجع',0,0),())
        p=ShortPrediction('count_within_radius',1,None,'invalid_query','missing_or_conflicting_radius')
        f=safe_features('مرجع',ctx,p,[1,0,0,0,0,0,0,0])
        self.assertEqual(f['missing_slots'],['radius_km'])
        self.assertIsNone(f['constraint_count'])
        p.reason='invalid_context'
        self.assertIsNone(safe_features('مرجع',ctx,p,[1,0,0,0,0,0,0,0])['missing_slots'])

    def test_hop_slot_order_remains_original(self):
        q=StructuredQuery('two_hop_nearest',EntityReference('مرجع','context_anchor'),
            first_hop_category='مستشفى',second_hop_category='صيدلية')
        self.assertEqual(slots(q),{'a':'«مرجع»','c1':'مستشفى','c2':'صيدلية'})
