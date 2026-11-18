import numpy as np
from scipy.stats import skew, kurtosis

from .base import BaseFeatureExtractor


class StatsFeatures(BaseFeatureExtractor):
    """Time-domain statistics of each segment."""

    @property
    def feature_names(self):
        return ["mean", "std", "min", "max", "ptp", "rms", "skew",
                "kurtosis", "line_length", "zero_crossings"]

    def transform(self, X, groups=None):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1)
        centered = X - mean[:, None]
        zero_cross = np.sum(np.diff(np.sign(centered), axis=1) != 0, axis=1)
        return np.column_stack([
            mean,
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            X.max(axis=1) - X.min(axis=1),
            np.sqrt((X ** 2).mean(axis=1)),
            skew(X, axis=1),
            kurtosis(X, axis=1),
            np.sum(np.abs(np.diff(X, axis=1)), axis=1),
            zero_cross,
        ])
