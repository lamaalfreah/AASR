"""Frozen calibrated Router V2 runtime: Short features in, routing probability out."""
from dataclasses import dataclass
import joblib
import numpy as np
from language.lightweight_router import FEATURES, NUMERIC_FEATURES
from spatial.schema import OPERATIONS

V2_FEATURES = FEATURES | {'operation_probabilities'}


def encode_v2(features):
    if set(features)!=V2_FEATURES:
        raise ValueError('Router V2 accepts exactly the frozen inference-safe feature keys')
    p=features['operation_probabilities']
    if set(p)!=set(OPERATIONS) or not all(np.isfinite(v) and 0<=v<=1 for v in p.values()) or abs(sum(p.values())-1)>1e-5:
        raise ValueError('Expected the full eight-operation probability vector')
    result={k:np.nan if features[k] is None else float(features[k]) for k in NUMERIC_FEATURES}
    result.update({'operation_probability:'+op:float(p[op]) for op in OPERATIONS})
    result['predicted_operation']=features['predicted_operation'] or '__missing__'
    missing=features['missing_slots']
    result['missing_slot_group_count']=np.nan if missing is None else float(len(set(missing)))
    for slot in ['__unavailable__'] if missing is None else sorted(set(missing)) or ['__none__']:
        result['missing_slot:'+slot]=1.0
    return result


def logit(values):
    p=np.clip(np.asarray(values,dtype=float),1e-7,1-1e-7)
    return np.log(p/(1-p)).reshape(-1,1)


@dataclass
class CalibratedRouterEnsemble:
    components: list

    def probabilities(self,feature_rows,calibrated=True):
        encoded=[encode_v2(f) for f in feature_rows]
        all_probabilities=[]
        for component in self.components:
            x=component['preprocessor'].transform(encoded)
            model=component['model']
            p=model.predict_proba(x)[:,list(model.classes_).index(1)]
            if calibrated:p=component['calibrator'].predict_proba(logit(p))[:,1]
            all_probabilities.append(p)
        return np.mean(all_probabilities,axis=0)


class RouterV2:
    def __init__(self,path):
        artifact=joblib.load(path)
        self.ensemble=artifact['ensemble'];self.threshold=artifact['threshold']

    def long_probabilities(self,feature_rows):
        return self.ensemble.probabilities(feature_rows)

    def route(self,feature_rows):
        return np.where(self.long_probabilities(feature_rows)>=self.threshold,'LONG','SHORT').tolist()
