from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    """Turns a 2D array (n_samples, n_points) into a 2D feature array."""

    @property
    @abstractmethod
    def feature_names(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X, groups=None):
        raise NotImplementedError
