import numpy as np

from biomed import pipeline
from biomed.datasets import BaseDataset


class FakeDataset(BaseDataset):
    name = "fake"

    def load(self):
        rng = np.random.default_rng(0)
        n_groups, per = 40, 4
        labels = np.repeat(np.array([1, 2] * (n_groups // 2)), per)
        groups = np.repeat([f"r{i}" for i in range(n_groups)], per)
        scale = np.where(labels == 1, 5.0, 1.0)[:, None]
        X = rng.normal(size=(len(labels), 178)) * scale
        return X, labels, groups


def test_run_experiment_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(pipeline.DATASETS, "fake", FakeDataset)
    cfg = {
        "experiment": "test_run",
        "dataset": {"name": "fake", "positive_labels": [1]},
        "features": [{"name": "stats"}],
        "model": {"name": "logreg"},
        "cv": {"n_splits": 4, "seed": 0, "grouped": True},
        "output_dir": str(tmp_path),
    }
    folds, summary = pipeline.run_experiment(cfg)
    assert len(folds) == 4
    assert summary["roc_auc"] > 0.9
    assert (tmp_path / "test_run" / "summary.json").exists()
