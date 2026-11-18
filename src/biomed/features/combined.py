import numpy as np

from .base import BaseFeatureExtractor


class CombinedFeatures(BaseFeatureExtractor):
    """Runs several extractors and stacks their features side by side."""

    def __init__(self, extractors):
        self.extractors = list(extractors)

    @property
    def feature_names(self):
        names = []
        for e in self.extractors:
            names += e.feature_names
        return names

    def transform(self, X, groups=None):
        return np.hstack([e.transform(X, groups) for e in self.extractors])
