import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from biomed import pipeline
from biomed.datasets import BaseDataset
from biomed.models import WeightedDecisionClassifier


def three_class(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    X = rng.normal(size=(n, 2)) + y[:, None] * 2
    return X, y


def test_zero_weight_class_is_never_predicted():
    X, y = three_class()
    plain = LogisticRegression(max_iter=1000).fit(X, y).predict(X)
    assert (plain == 2).sum() > 0
    clf = WeightedDecisionClassifier(LogisticRegression(max_iter=1000),
                                     [1.0, 1.0, 0.0]).fit(X, y)
    assert 2 not in set(clf.predict(X))


def test_large_weight_makes_a_class_more_likely():
    X, y = three_class()
    plain = (LogisticRegression(max_iter=1000).fit(X, y).predict(X) == 0).sum()
    clf = WeightedDecisionClassifier(LogisticRegression(max_iter=1000),
                                     [20.0, 1.0, 1.0]).fit(X, y)
    assert (clf.predict(X) == 0).sum() > plain


def test_missing_class_gets_a_zero_column():
    X, y = three_class()
    keep = y != 2
    clf = WeightedDecisionClassifier(LogisticRegression(max_iter=1000),
                                     [1.0, 1.0, 1.0]).fit(X[keep], y[keep])
    p = clf.predict_proba(X)
    assert p.shape == (60, 3)
    assert (p[:, 2] == 0).all()


def test_factory_wraps_model_when_decision_weights_given():
    cfg = {"model": {"name": "logreg", "decision_weights": [1, 1, 1]}}
    assert isinstance(pipeline.make_model_factory(cfg)(), WeightedDecisionClassifier)
    cfg = {"model": {"name": "logreg"}}
    assert not isinstance(pipeline.make_model_factory(cfg)(), WeightedDecisionClassifier)


def test_ecg_components_are_registered():
    assert "ecg_beat" in pipeline.FEATURES
    assert "mitbih" in pipeline.DATASETS


class SplitDataset(BaseDataset):
    name = "split"
    named_splits = {"A": [f"g{i}" for i in range(30)],
                    "B": [f"g{i}" for i in range(30, 40)]}

    def load(self):
        raise NotImplementedError


def grouped(n_groups=40, per=4, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_groups) % 2, per)
    g = np.repeat([f"g{i}" for i in range(n_groups)], per)
    X = rng.normal(size=(len(y), 3)) + y[:, None] * 3
    return X, y, g


def test_named_splits_in_holdout():
    X, y, g = grouped()
    cfg = {"evaluation": {"type": "holdout", "train_groups": "A",
                          "test_groups": "B"}}
    res = pipeline.run_evaluation(cfg, lambda: LogisticRegression(), X, y, g,
                                  SplitDataset())
    assert res.loc[0, "n_train"] == 120
    assert res.loc[0, "n_test"] == 40


def test_unknown_named_split_raises():
    X, y, g = grouped()
    cfg = {"evaluation": {"type": "holdout", "test_groups": "Z"}}
    with pytest.raises(KeyError):
        pipeline.run_evaluation(cfg, lambda: LogisticRegression(), X, y, g,
                                SplitDataset())
