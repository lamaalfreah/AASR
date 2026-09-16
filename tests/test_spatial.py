import json
import math
import unittest
from dataclasses import replace
from spatial import execute, parse, parse_context, run
from spatial.context_parser import ART_MARK, REG_MARK
from spatial.geometry import bearing, direction, distance
from spatial.identity import resolve
from spatial.normalization import answer_matches, integer
from spatial.schema import EntityReference, Location, ParsedContext, StructuredQuery, TypedAnswer


def context(rows, anchor='أ', latitude=0, longitude=0):
    return (f'المكان المرجعي هو «{anchor}»\nخط العرض: {latitude}\nخط الطول: {longitude}\n'
            + ART_MARK + '\nنص\n' + REG_MARK + '\n' + '\n'.join(
                f'{i}. الاسم: {name}؛ النوع: {cat}؛ خط العرض: {lat}؛ خط الطول: {lon}.'
                for i, (name, cat, lat, lon) in enumerate(rows, 1)))


ROWS = [('مدرسة أولى', 'مدرسة', .01, 0), ('مدرسة ثانية', 'مدرسة', .03, 0),
        ('بنك قريب', 'بنك', 0, .02), ('مقهى بعيد', 'مقهى', .03, .03)]
QUESTIONS = {
    'nearest_category': 'ما أقرب مدرسة إلى «أ» من بين المعالم الواردة في السياق؟',
    'cardinal_direction': 'في أي اتجاه يقع «بنك قريب» بالنسبة إلى «أ»؟',
    'within_radius_yes_no': 'هل يوجد مدرسة ضمن مسافة 2 كم تقريبًا من «أ» من بين المعالم الواردة في السياق؟',
    'closer_of_two': 'أي الموقعين أقرب إلى «أ»: «مدرسة أولى» أم «بنك قريب»؟',
    'count_within_radius': 'كم عدد المعالم من نوع «مدرسة» الواقعة ضمن مسافة 2 كم تقريبًا من «أ» وفق الإحداثيات المتاحة؟',
    'nearest_of_two_categories': 'أيهما أقرب إلى «أ»: أقرب مدرسة أم أقرب بنك؟ اذكر اسم المعلم الأقرب.',
    'two_hop_nearest': 'انطلاقًا من «أ»، حدد أولًا أقرب مدرسة، ثم استخدم هذا الموقع كنقطة مرجعية جديدة لتحديد أقرب بنك. ما اسم المعلم النهائي؟',
    'spatial_multi_constraint': 'ما أقرب مدرسة يقع في اتجاه شمال من «أ» وضمن مسافة 5 كم تقريبًا؟',
}


class SpatialTests(unittest.TestCase):
    def test_all_eight_templates_and_answers(self):
        expected = ['مدرسة أولى', 'شرق', 'نعم', 'مدرسة أولى', '1', 'مدرسة أولى', 'بنك قريب', 'مدرسة أولى']
        for (task, question), answer in zip(QUESTIONS.items(), expected):
            with self.subTest(task=task):
                result = run(question, context(ROWS))
                self.assertEqual(result.operation, task)
                self.assertEqual(result.status, 'success')
                self.assertEqual(result.answer.text, answer)
                self.assertEqual(result.trace['final_answer']['text'], answer)
                self.assertTrue(result.trace['steps'])
                self.assertIsNotNone(result.trace['resolved_origin'])
                json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_multiline_whitespace_and_duplicate_rows_preserved(self):
        rows = [('اسم\nمتعدد ', 'مدرسة', 1, 2)] * 2
        parsed = parse_context(context(rows))
        self.assertFalse(parsed.errors)
        self.assertEqual([p.name for p in parsed.candidates], ['اسم\nمتعدد '] * 2)
        self.assertNotEqual(parsed.candidates[0].identity, parsed.candidates[1].identity)

    def test_invalid_coordinate_and_malformed_registry_rejected(self):
        for text in (context([('س', 'مدرسة', 91, 0)]), context(ROWS) + '\ntrailing junk',
                     context(ROWS).replace('خط العرض: 0.01', 'خط العرض: '), 'bad context'):
            with self.subTest(text=text[:30]):
                self.assertEqual(run(QUESTIONS['nearest_category'], text).status, 'invalid_query')

    def test_identity_states_and_exact_names(self):
        parsed = parse_context(context(ROWS + [('مدرسة أولى', 'مدرسة', 1, 2), ('ذيل ', 'بنك', 2, 2)]))
        for name, state, count in [('بنك قريب', 'UNIQUE', 1), ('مدرسة أولى', 'AMBIGUOUS', 2),
                                   ('غائب', 'NOT_FOUND', 0), ('ذيل', 'NOT_FOUND', 0), ('ذيل ', 'UNIQUE', 1)]:
            r = resolve(EntityReference(name), parsed)
            self.assertEqual((r.status, r.candidate_count), (state, count))

    def test_haversine_and_bearing(self):
        self.assertEqual(distance((0, 0), (0, 0)), 0)
        self.assertAlmostEqual(distance((0, 0), (0, 1)), 111.1950802335, places=7)
        self.assertAlmostEqual(distance((0, 179), (0, -179)), 222.390160467, places=6)
        self.assertAlmostEqual(distance((0, 0), (0, 180)), math.pi * 6371.0088)
        self.assertEqual(bearing((0, 0), (0, 1)), 90)

    def test_four_sector_boundaries(self):
        self.assertEqual([direction(b) for b in (0, 44.99, 45, 135, 225, 315, 359)],
                         ['شمال', 'شمال', 'شرق', 'جنوب', 'غرب', 'شمال', 'شمال'])

    def test_question_derived_origin(self):
        rows = [('بنك قريب', 'بنك', 1, 1), ('س', 'مدرسة', 1.001, 1)]
        question = QUESTIONS['count_within_radius'].replace('«أ»', '«بنك قريب»')
        result = run(question, context(rows))
        self.assertEqual(result.answer.value, 1)
        self.assertEqual(result.trace['resolved_origin']['identity'], 'candidate:1')

    def test_anchor_candidate_origin_collision(self):
        result = run(QUESTIONS['count_within_radius'], context(ROWS + [('أ', 'بنك', 1, 1)]))
        self.assertEqual(result.status, 'ambiguous')
        self.assertIsNone(result.answer)
        self.assertEqual(result.trace['identity_resolutions'][0]['candidate_count'], 2)

    def test_target_context_anchor_collision(self):
        q = QUESTIONS['cardinal_direction'].replace('«بنك قريب»', '«أ»')
        self.assertEqual(run(q, context(ROWS + [('أ', 'بنك', 1, 1)])).status, 'ambiguous')

    def test_duplicates_abstain_even_same_coordinates(self):
        r = run(QUESTIONS['cardinal_direction'], context(ROWS + [ROWS[2]]))
        self.assertEqual((r.status, r.stage), ('ambiguous', 'identity'))

    def test_comparison_trace_contains_both_distances(self):
        r = run(QUESTIONS['closer_of_two'], context(ROWS))
        ds = r.trace['steps'][0]['distances_km']
        self.assertEqual(set(ds), {'candidate:1', 'candidate:3'})
        self.assertLess(ds['candidate:1'], ds['candidate:3'])

    def test_named_target_missing_and_multiline_question(self):
        q = QUESTIONS['cardinal_direction'].replace('بنك قريب', 'غائب')
        self.assertEqual(run(q, context(ROWS)).status, 'not_found')
        name = 'بنك\nمتعدد '
        q = QUESTIONS['cardinal_direction'].replace('بنك قريب', name)
        r = run(q, context([(name, 'بنك', 0, .01)]))
        self.assertEqual(r.answer.text, 'شرق')
        self.assertEqual(r.trace['selected_candidates'][0]['name'], name)

    def test_count_rows_not_names(self):
        r = run(QUESTIONS['count_within_radius'], context([ROWS[0]] * 3))
        self.assertEqual(r.answer.value, 3)

    def test_zero_radius_and_missing_category(self):
        self.assertEqual(run(QUESTIONS['count_within_radius'], context([ROWS[2]])).answer.value, 0)
        self.assertFalse(run(QUESTIONS['within_radius_yes_no'], context([ROWS[2]])).answer.value)
        self.assertEqual(run(QUESTIONS['nearest_category'], context([ROWS[2]])).status, 'not_found')
        ctx = parse_context(context([('س', 'مدرسة', 0, 0)]))
        query = replace(parse(QUESTIONS['count_within_radius'], ctx), radius_km=0)
        self.assertEqual(execute(query, ctx).answer.value, 1)

    def test_radius_inclusive_without_rounding(self):
        ctx = parse_context(context([ROWS[0]]))
        d = distance(ctx.anchor.coordinates, ctx.candidates[0].coordinates)
        query = parse(QUESTIONS['count_within_radius'], ctx)
        self.assertEqual(execute(replace(query, radius_km=d), ctx).answer.value, 1)
        self.assertEqual(execute(replace(query, radius_km=d-.0000001), ctx).answer.value, 0)

    def test_multi_constraint_narrow_window_and_nearest_after_filter(self):
        rows = [('خارج', 'مدرسة', .01, .01*math.tan(math.radians(40))), ('داخل', 'مدرسة', .02, 0)]
        self.assertEqual(run(QUESTIONS['spatial_multi_constraint'], context(rows)).answer.text, 'داخل')
        self.assertEqual(run(QUESTIONS['spatial_multi_constraint'], context([rows[0]])).status, 'not_found')

    def test_two_hop_references_computed_candidate(self):
        rows = [('قريب', 'مدرسة', 1, 0), ('بنك الأصل', 'بنك', 0, .01), ('بنك المرحلة', 'بنك', 1, .01)]
        r = run(QUESTIONS['two_hop_nearest'], context(rows))
        self.assertEqual(r.answer.text, 'بنك المرحلة')
        self.assertEqual(r.trace['first_hop_result'][0]['name'], 'قريب')
        self.assertEqual(r.trace['steps'][1]['reference']['name'], 'قريب')

    def test_tie_policies(self):
        rows = [('س', 'مدرسة', .01, 0), ('ص', 'مدرسة', .01, 0), ROWS[2]]
        self.assertEqual(run(QUESTIONS['nearest_category'], context(rows)).status, 'ambiguous')
        self.assertEqual(run(QUESTIONS['two_hop_nearest'], context(rows)).reason, 'first_hop_identity_tie')
        r = run(QUESTIONS['nearest_category'], context([rows[0]] * 2))
        self.assertEqual(r.status, 'success')
        self.assertFalse(r.trace['selected_identity_unique'])
        self.assertEqual(len(r.trace['selected_candidates']), 2)

    def test_unsupported_and_invalid_queries(self):
        self.assertEqual(run('سؤال حر', context(ROWS)).status, 'unsupported')
        self.assertEqual(run(42, context(ROWS)).status, 'invalid_query')
        ctx = parse_context(context(ROWS)); q = parse(QUESTIONS['count_within_radius'], ctx)
        for radius in (-1, float('nan'), float('inf'), True):
            self.assertEqual(execute(replace(q, radius_km=radius), ctx).status, 'invalid_query')
        self.assertEqual(execute(replace(q, operation='unknown'), ctx).status, 'unsupported')
        self.assertEqual(execute(replace(q, categories=('unknown',)), ctx).status, 'unsupported')
        self.assertEqual(execute(replace(q, targets=(EntityReference('س'),)), ctx).status, 'invalid_query')
        self.assertEqual(run(QUESTIONS['count_within_radius'].replace('2 كم', '1.2.3 كم'), context(ROWS)).status, 'invalid_query')

    def test_same_location_has_no_direction(self):
        r = run(QUESTIONS['cardinal_direction'], context([('بنك قريب', 'بنك', 0, 0)]))
        self.assertEqual(r.reason, 'bearing_undefined_at_same_location')

    def test_scoring_does_not_conflate_names_or_extract_partial_numbers(self):
        a = TypedAnswer('entity', 'مدرسة أ', 'مدرسة أ')
        self.assertTrue(answer_matches(a, ' مدرسة\nأ '))
        self.assertFalse(answer_matches(a, 'مدرسه ا'))
        self.assertEqual(integer('٣'), 3)
        self.assertIsNone(integer('3.0')); self.assertIsNone(integer('count 3'))
        self.assertTrue(answer_matches(TypedAnswer('boolean', True, 'نعم'), 'نَعَم'))
        self.assertFalse(answer_matches(TypedAnswer('direction', 'شمال', 'شمال'), 'الشمال'))

    def test_public_api_accepts_only_question_and_context(self):
        class Forbidden(dict):
            def __getitem__(self, key):
                if key not in ('question', 'context'):
                    raise AssertionError('gold accessed')
                return super().__getitem__(key)
        row = Forbidden(question=QUESTIONS['nearest_category'], context=context(ROWS))
        self.assertEqual(run(row['question'], row['context']).answer.text, 'مدرسة أولى')


if __name__ == '__main__':
    unittest.main()
