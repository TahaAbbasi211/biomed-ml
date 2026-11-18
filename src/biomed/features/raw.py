import numpy as np

from .base import BaseFeatureExtractor


class RawFeatures(BaseFeatureExtractor):
    """Passes the columns through unchanged (for tabular data)."""

    def __init__(self, names=None):
        self.names = list(names) if names else []

    @property
    def feature_names(self):
        return self.names

    def transform(self, X, groups=None):
        return np.asarray(X, dtype=float)
