import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class WeightedDecisionClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a classifier and predicts argmax(probability * class weight).

    A weight above 1 makes a class easier to predict, 0 means the class is
    never predicted. predict_proba returns one column per weight, so a class
    missing from the training data gets a zero column.
    """

    def __init__(self, base=None, weights=None):
        self.base = base
        self.weights = weights

    def fit(self, X, y, **fit_params):
        self.base.fit(X, y, **fit_params)
        self.classes_ = np.asarray(self.base.classes_)
        return self

    def predict_proba(self, X):
        p = self.base.predict_proba(X)
        w = np.asarray(self.weights, dtype=float)
        full = np.zeros((len(p), len(w)))
        full[:, self.classes_.astype(int)] = p
        return full

    def predict(self, X):
        w = np.asarray(self.weights, dtype=float)
        return (self.predict_proba(X) * w).argmax(axis=1)
