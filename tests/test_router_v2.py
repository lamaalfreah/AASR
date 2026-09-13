import unittest
import numpy as np
from evaluation.train_router_v2 import split_groups,robust_threshold,fit_calibrated
from language.lightweight_router import FEATURES
from language.router_v2 import encode_v2,logit
from spatial.schema import OPERATIONS


def example(i):
    label=('SHORT','LONG','FAILURE')[i%3]
    features={k:0 for k in FEATURES}
    features.update(predicted_operation=OPERATIONS[0],missing_slots=[],constraint_count=None,
        operation_probabilities={op:1/8 for op in OPERATIONS})
    return dict(label=label,features=features,provenance=dict(source_anchor='anchor-'+str(i//3)),
        outcomes=dict(short=dict(answer_correct=label=='SHORT'),long=dict(answer_correct=label=='LONG')))


class RouterV2Tests(unittest.TestCase):
    def test_group_cv_has_no_anchor_overlap_and_full_coverage(self):
        data=[example(i) for i in range(120)]
        folds=split_groups(data,5,123)
        for train,test in folds:
            self.assertFalse({data[i]['provenance']['source_anchor'] for i in train}&
                             {data[i]['provenance']['source_anchor'] for i in test})
        self.assertEqual(sorted(np.concatenate([t for _,t in folds]).tolist()),list(range(120)))

    def test_named_probabilities_and_feature_allowlist(self):
        features=example(0)['features']
        encoded=encode_v2(features)
        self.assertEqual(sum(k.startswith('operation_probability:') for k in encoded),8)
        self.assertTrue(np.isnan(encoded['constraint_count']))
        self.assertTrue(np.isfinite(logit([0,1])).all())
        features['task_type']='gold'
        with self.assertRaises(ValueError):encode_v2(features)

    def test_threshold_rule_preserves_accuracy_and_reduces_calls(self):
        data=[example(i) for i in range(12)]
        p=np.array([.1,.9,.3]*4)
        selection=robust_threshold(data,p,[np.arange(i,i+3) for i in range(0,12,3)])
        selected=selection['selected']
        self.assertEqual(selected['metrics']['adaptive_answer_correct'],8)
        self.assertEqual(selected['metrics']['long_route_count'],4)
        self.assertGreater(selected['threshold'],.3)
        self.assertLessEqual(selected['threshold'],.9)

    def test_calibration_holdout_is_disjoint_and_failure_not_fitted(self):
        data=[example(i) for i in range(120)]
        ensemble,audit=fit_calibrated(data,'logistic_regression',123)
        for component in audit:
            self.assertEqual(component['fit_n']+component['calibration_n'],80)
            self.assertFalse(set(component['fit_anchors'])&set(component['calibration_anchors']))
        p=ensemble.probabilities([r['features'] for r in data])
        self.assertEqual(len(p),120)
        self.assertTrue(np.isfinite(p).all())


if __name__=='__main__':unittest.main()
