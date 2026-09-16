import unittest

import numpy as np

from evaluation.train_lightweight_router import evaluate_routes, select_threshold
from language.lightweight_router import FEATURES, encode_features


def row(label, short, long):
    return dict(label=label, outcomes=dict(short=dict(answer_correct=short), long=dict(answer_correct=long)))


class RouterSelectionTests(unittest.TestCase):
    def test_failure_is_not_binary_but_counts_in_final_accuracy(self):
        rows = [row('SHORT', True, False), row('LONG', False, True), row('FAILURE', False, False)]
        result = evaluate_routes(rows, [False, True, True])
        self.assertEqual(result['binary_n'], 2)
        self.assertEqual(result['routing_accuracy'], 1.0)
        self.assertEqual(result['adaptive_answer_correct'], 2)
        self.assertEqual(result['adaptive_answer_accuracy'], 2/3)
        self.assertEqual(result['long_route_count'], 2)
        self.assertEqual(result['failure_analysis']['long_calls'], 1)

    def test_regression_and_unnecessary_calls(self):
        rows = [row('SHORT', True, True), row('SHORT', True, False), row('LONG', False, True)]
        result = evaluate_routes(rows, [True, True, True])
        self.assertEqual(result['unnecessary_long_calls'], 2)
        self.assertEqual(result['long_regressions'], 1)
        self.assertEqual(result['long_rescues'], 1)
        self.assertEqual(result['oracle_correct'], 3)

    def test_threshold_uses_accuracy_then_avoids_unnecessary_long(self):
        rows = [row('SHORT', True, True), row('LONG', False, True), row('FAILURE', False, False)]
        selected, curve = select_threshold(rows, np.array([.3, .8, .5]))
        self.assertEqual(selected['threshold'], .8)
        self.assertEqual(selected['long_route_count'], 1)
        self.assertEqual(selected['adaptive_answer_correct'], 2)
        self.assertTrue(any(r['long_route_count'] == 0 for r in curve))
        self.assertTrue(any(r['long_route_count'] == 3 for r in curve))

    def test_feature_allowlist_and_missingness(self):
        features = {name: None for name in FEATURES}
        encoded = encode_features(features)
        self.assertTrue(np.isnan(encoded['constraint_count']))
        self.assertEqual(encoded['missing_slots'], '__unavailable__')
        features['gold_task'] = 'nearest_category'
        with self.assertRaises(ValueError):
            encode_features(features)


if __name__ == '__main__':
    unittest.main()
