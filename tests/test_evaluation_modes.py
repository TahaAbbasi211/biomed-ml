import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from biomed import pipeline
from biomed.datasets import BaseDataset
from biomed.evaluation import compute_metrics, cross_validate, holdout_evaluate


def make_groups_data(n_groups=40, per_group=4, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_groups) % n_classes, per_group)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per_group)
    X = rng.normal(size=(len(y), 3)) + y[:, None] * 3
    return X, y, groups


def factory():
    return LogisticRegression()


def test_binary_metrics_values():
    m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]),
                        np.array([0.1, 0.6, 0.8, 0.9]))
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_multiclass_metrics_values():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 2, 2, 2])
    m = compute_metrics(y, pred, task="multiclass", labels=[0, 1, 2])
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["f1_0"] == pytest.approx(1.0)
    assert m["recall_1"] == pytest.approx(0.5)
    assert m["precision_2"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((1 + 2 / 3 + 0.8) / 3)


def test_macro_labels_leave_a_class_out():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 2, 2, 2])
    m = compute_metrics(y, pred, task="multiclass", labels=[0, 1, 2],
                        macro_labels=[0, 1])
    assert m["macro_f1"] == pytest.approx((1 + 2 / 3) / 2)


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0, 1]), task="nope")


def test_holdout_explicit_test_groups():
    X, y, g = make_groups_data()
    res = holdout_evaluate(factory, X, y, g,
                           test_groups=["g0", "g1", "g2", "g3", "g4"])
    assert len(res) == 1
    assert res.loc[0, "n_test"] == 20
    assert res.loc[0, "n_train"] == 140


def test_holdout_explicit_train_and_test_groups():
    X, y, g = make_groups_data()
    res = holdout_evaluate(
        factory, X, y, g,
        train_groups=[f"g{i}" for i in range(10, 40)],
        test_groups=[f"g{i}" for i in range(5)])
    assert res.loc[0, "n_train"] == 120
    assert res.loc[0, "n_test"] == 20


def test_holdout_test_size_splits_by_group():
    X, y, g = make_groups_data()
    res = holdout_evaluate(factory, X, y, g, test_size=0.25)
    assert res.loc[0, "n_test"] == 40


def test_holdout_overlapping_groups_raise():
    X, y, g = make_groups_data()
    with pytest.raises(ValueError):
        holdout_evaluate(factory, X, y, g, train_groups=["g0", "g1"],
                         test_groups=["g0"])


def test_holdout_needs_a_split_rule():
    X, y, g = make_groups_data()
    with pytest.raises(ValueError):
        holdout_evaluate(factory, X, y, g)


def test_holdout_unknown_group_names_raise():
    X, y, g = make_groups_data()
    with pytest.raises(ValueError):
        holdout_evaluate(factory, X, y, g, test_groups=["nope"])


def test_holdout_requires_groups():
    X, y, _ = make_groups_data()
    with pytest.raises(ValueError):
        holdout_evaluate(factory, X, y, None, test_size=0.2)


def test_multiclass_cross_validate():
    X, y, g = make_groups_data(n_classes=3)
    res = cross_validate(factory, X, y, g, n_splits=4, task="multiclass")
    assert {"macro_f1", "f1_2"} <= set(res.columns)
    assert res["macro_f1"].mean() > 0.8


class ThreeClassDataset(BaseDataset):
    name = "three"

    def load(self):
        rng = np.random.default_rng(0)
        n_groups, per = 45, 4
        labels = np.repeat(np.arange(n_groups) % 3 + 1, per)
        groups = np.repeat([f"r{i}" for i in range(n_groups)], per)
        scale = np.array([1.0, 3.0, 8.0])[labels - 1][:, None]
        X = rng.normal(size=(len(labels), 178)) * scale
        return X, labels, groups


def test_multiclass_holdout_through_pipeline(monkeypatch):
    monkeypatch.setitem(pipeline.DATASETS, "three", ThreeClassDataset)
    cfg = {
        "experiment": "t3", "task": "multiclass",
        "labels": [1, 2, 3], "macro_labels": [1, 2],
        "dataset": {"name": "three"},
        "features": [{"name": "stats"}],
        "model": {"name": "logreg"},
        "evaluation": {"type": "holdout", "test_size": 0.3, "seed": 0},
    }
    folds, summary = pipeline.run_experiment(cfg, save=False)
    assert len(folds) == 1
    assert {"f1_1", "f1_2", "f1_3", "macro_f1"} <= set(folds.columns)
    assert "n_test" not in summary
    assert summary["macro_f1"] > 0.7


def test_unknown_evaluation_type_raises():
    X, y, g = make_groups_data()
    with pytest.raises(ValueError):
        pipeline.run_evaluation({"evaluation": {"type": "nope"}}, factory, X, y, g)
