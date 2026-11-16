import numpy as np
from sklearn.linear_model import LogisticRegression

from biomed.evaluation import cross_validate, out_of_fold_proba


def make_grouped_data(n_groups=40, per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    y_group = np.array([0, 1] * (n_groups // 2))
    y = np.repeat(y_group, per_group)
    groups = np.repeat([f"rec{i}" for i in range(n_groups)], per_group)
    X = rng.normal(size=(len(y), 3)) + y[:, None] * 2
    return X, y, groups


def test_cross_validate_returns_one_row_per_fold():
    X, y, g = make_grouped_data()
    res = cross_validate(lambda: LogisticRegression(), X, y, g, n_splits=4)
    assert len(res) == 4
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(res.columns)
    assert res["roc_auc"].mean() > 0.9


def test_out_of_fold_proba_covers_every_sample():
    X, y, g = make_grouped_data()
    p = out_of_fold_proba(lambda: LogisticRegression(), X, y, g, n_splits=4)
    assert p.shape == y.shape
    assert ((p > 0) & (p < 1)).all()
