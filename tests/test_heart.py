import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from biomed import pipeline
from biomed.datasets import HeartDiseaseDataset
from biomed.datasets.heart import FEATURE_COLUMNS
from biomed.evaluation import cross_validate, out_of_fold_proba


def make_heart_csv(path, n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    df = pd.DataFrame({
        "age": rng.normal(55, 8, n) + 6 * y,
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.normal(130, 15, n),
        "chol": rng.normal(240, 40, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.normal(150, 20, n) - 20 * y,
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.abs(rng.normal(1, 1, n)) + 1.5 * y,
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    df.loc[[3, 7], "ca"] = np.nan      # missing values, like the real data
    df.loc[[11], "thal"] = np.nan
    df["target"] = y
    df.to_csv(path, index=False)
    return path


def test_loads_features_target_and_no_groups(tmp_path):
    csv = make_heart_csv(tmp_path / "heart.csv")
    X, y, groups = HeartDiseaseDataset(path=csv).load()
    assert X.shape == (120, 13)
    assert groups is None
    assert set(np.unique(y)) == {0, 1}
    assert int(np.isnan(X).sum()) == 3


def test_missing_column_raises(tmp_path):
    csv = make_heart_csv(tmp_path / "heart.csv")
    pd.read_csv(csv).drop(columns="chol").to_csv(csv, index=False)
    with pytest.raises(ValueError):
        HeartDiseaseDataset(path=csv).load()


def test_real_heart_csv_is_valid():
    X, y, groups = HeartDiseaseDataset().load()
    assert X.shape[1] == len(FEATURE_COLUMNS) == 13
    assert len(X) > 200
    assert set(np.unique(y)) <= {0, 1}
    assert groups is None


def test_cross_validate_without_groups():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 60)
    X = rng.normal(size=(120, 3)) + y[:, None] * 2
    res = cross_validate(lambda: LogisticRegression(), X, y, None, n_splits=4)
    assert len(res) == 4
    assert res["roc_auc"].mean() > 0.9


def test_out_of_fold_proba_without_groups():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 60)
    X = rng.normal(size=(120, 3)) + y[:, None] * 2
    p = out_of_fold_proba(lambda: LogisticRegression(), X, y, None, n_splits=4)
    assert p.shape == y.shape
    assert ((p > 0) & (p < 1)).all()


def test_heart_experiment_end_to_end(tmp_path):
    csv = make_heart_csv(tmp_path / "heart.csv")
    cfg = {
        "experiment": "heart_test",
        "dataset": {"name": "heart_cleveland", "positive_labels": [1],
                    "params": {"path": str(csv)}},
        "features": [{"name": "raw"}],
        "preprocess": "heart_tabular",
        "model": {"name": "logreg"},
        "cv": {"n_splits": 4, "seed": 0, "grouped": False},
        "output_dir": str(tmp_path / "out"),
    }
    folds, summary = pipeline.run_experiment(cfg)
    assert len(folds) == 4
    assert summary["roc_auc"] > 0.8
    assert (tmp_path / "out" / "heart_test" / "summary.json").exists()


def test_unknown_preprocessor_raises():
    cfg = {"model": {"name": "logreg"}, "preprocess": "nope"}
    with pytest.raises(KeyError):
        pipeline.make_model_factory(cfg)
