"""Feature-only CPU Router. No parser calls, gold labels, or spatial execution."""
from pathlib import Path

import joblib
import numpy as np


NUMERIC_FEATURES = (
    'short_confidence', 'short_entropy', 'short_top1_top2_margin',
    'query_valid', 'entity_ambiguity_count', 'question_length_chars',
    'question_length_words', 'constraint_count',
)
FEATURES = frozenset((*NUMERIC_FEATURES, 'predicted_operation', 'missing_slots'))


def encode_features(features):
    """Encode only the existing feature dictionary; reject accidental gold input."""
    if set(features) != FEATURES:
        raise ValueError('Router requires exactly the frozen Step 3A feature keys')
    encoded = {key: np.nan if features[key] is None else float(features[key])
               for key in NUMERIC_FEATURES}
    encoded['predicted_operation'] = features['predicted_operation'] or '__missing__'
    slots = features['missing_slots']
    encoded['missing_slots'] = '__unavailable__' if slots is None else '|'.join(sorted(set(slots))) or '__none__'
    return encoded


class LightweightRouter:
    """Load a trusted frozen artifact and route saved/inference-time Short features."""

    def __init__(self, path):
        self.artifact = joblib.load(Path(path))
        self.threshold = self.artifact['threshold']

    def long_probabilities(self, feature_rows):
        matrix = self.artifact['preprocessor'].transform([encode_features(f) for f in feature_rows])
        classifier = self.artifact['model']
        index = list(classifier.classes_).index(1)
        return classifier.predict_proba(matrix)[:, index]

    def route(self, feature_rows):
        return np.where(self.long_probabilities(feature_rows) >= self.threshold, 'LONG', 'SHORT').tolist()
