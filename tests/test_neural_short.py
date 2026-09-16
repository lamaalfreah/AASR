import unittest
from dataclasses import asdict
from language.neural_short import (extract_slots,radius_from_question,category_mentions,
                                   tokenize,make_model,encode)
from spatial import parse_context,parse
from spatial.schema import QueryError
from test_spatial import context,ROWS,QUESTIONS


class NeuralShortTests(unittest.TestCase):
    def test_all_original_slot_contracts(self):
        ctx=parse_context(context(ROWS))
        for op,q in QUESTIONS.items():
            with self.subTest(operation=op):
                self.assertEqual(extract_slots(op,q,ctx),parse(q,ctx))

    def test_natural_named_count_origin(self):
        ctx=parse_context(context(ROWS))
        q=extract_slots('count_within_radius','ما عدد المدارس ضمن كيلومتر ونصف من «بنك قريب»؟',ctx)
        self.assertEqual(q.origin.name,'بنك قريب');self.assertEqual(q.origin.source,'candidate')
        self.assertEqual(q.radius_km,1.5);self.assertEqual(q.categories,('مدرسة',))

    def test_name_that_equals_category_is_still_a_target(self):
        ctx=parse_context(context([('محطة وقود','محطة وقود',1,2)]))
        q=extract_slots('cardinal_direction','وش جهة «محطة وقود» من «أ»؟',ctx)
        self.assertEqual(q.targets[0].name,'محطة وقود')

    def test_plural_categories_and_overlap(self):
        self.assertEqual([c for _,_,c in category_mentions('عدد المستشفيات ومحطات الوقود')],['مستشفى','محطة وقود'])
        self.assertEqual([c for _,_,c in category_mentions('أقرب مطعم وجبات سريعة')],['مطعم وجبات سريعة'])
        self.assertEqual(category_mentions('«مستشفى الملك»'),[])

    def test_numeric_and_word_radii(self):
        for q,expected in [('ضمن ١.٥ كم',1.5),('ثلاثة كيلومترات',3),('كيلومترين',2),('كيلومتر واحد',1),('كيلومتر ونصف',1.5)]:
            self.assertEqual(radius_from_question(q),expected)
        with self.assertRaises(QueryError):radius_from_question('ثلاثة كيلومترات أو خمسة كيلومترات')

    def test_nested_and_sequential_hops(self):
        ctx=parse_context(context(ROWS))
        q=extract_slots('two_hop_nearest','ما أقرب بنك إلى أقرب مدرسة من «أ»؟',ctx)
        self.assertEqual((q.first_hop_category,q.second_hop_category),('مدرسة','بنك'))
        q=extract_slots('two_hop_nearest','أولًا أقرب مدرسة من «أ»، ثم أقرب بنك من المكان الأول.',ctx)
        self.assertEqual((q.first_hop_category,q.second_hop_category),('مدرسة','بنك'))

    def test_tokenizer_masks_entity_names_only(self):
        a=tokenize('أقرب مستشفى إلى «اسم سري» ضمن 3 كم')
        b=tokenize('أقرب مستشفى إلى «اسم آخر» ضمن 5 كم')
        self.assertEqual(a,b);self.assertIn('مستشفي',a);self.assertIn('entity',a)

    def test_neural_model_cpu_shape_and_gradients(self):
        import torch
        model=make_model(20)
        x=torch.tensor([[2,3,4],[5,6,0]]);lengths=torch.tensor([3,2])
        output=model(x,lengths)
        self.assertEqual(tuple(output.shape),(2,8))
        output.sum().backward()
        self.assertIsNotNone(model.gru.weight_ih_l0.grad)
        self.assertEqual(next(model.parameters()).device.type,'cpu')


if __name__=='__main__':unittest.main()
