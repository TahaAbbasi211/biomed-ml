import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.pipeline import make_pipeline

from .datasets import BonnEEGDataset, HeartDiseaseDataset
from .evaluation import cross_validate
from .features import (BandPowerFeatures, CombinedFeatures, RawFeatures,
                       StatsFeatures)
from .models import build_model
from .preprocessing import PREPROCESSORS

# Registries: to plug in something new, write its class and add one line here.
DATASETS = {"eeg_bonn": BonnEEGDataset, "heart_cleveland": HeartDiseaseDataset}
FEATURES = {"stats": StatsFeatures, "band_power": BandPowerFeatures,
            "raw": RawFeatures}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg):
    name = cfg["name"]
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name](**cfg.get("params", {}))


def build_features(items, dataset=None):
    extractors = []
    for item in items:
        name = item["name"]
        if name not in FEATURES:
            raise KeyError(f"Unknown feature set '{name}'. Available: {list(FEATURES)}")
        params = dict(item.get("params", {}))
        if name == "raw" and "names" not in params and hasattr(dataset, "feature_names"):
            params["names"] = list(dataset.feature_names)
        extractors.append(FEATURES[name](**params))
    return CombinedFeatures(extractors)


def make_model_factory(cfg):
    """Returns a function that builds a fresh model. If the config has a
    'preprocess' entry, the model is wrapped in a Pipeline so the
    preprocessing is fitted again inside every cross-validation fold."""
    model_name = cfg["model"]["name"]
    prep_name = cfg.get("preprocess")
    if prep_name is not None and prep_name not in PREPROCESSORS:
        raise KeyError(f"Unknown preprocessor '{prep_name}'. "
                       f"Available: {list(PREPROCESSORS)}")

    def factory():
        model = build_model(model_name)
        if prep_name is None:
            return model
        return make_pipeline(PREPROCESSORS[prep_name](), model)

    return factory


def run_experiment(cfg, save=True):
    """Run one experiment described by a config dict. Returns (folds, summary)."""
    dataset = build_dataset(cfg["dataset"])
    X, y, groups = dataset.load()

    positive = cfg["dataset"].get("positive_labels", [1])
    target = np.isin(y, positive).astype(int)

    fe = build_features(cfg["features"], dataset)
    F = fe.transform(X)

    cv_cfg = cfg.get("cv", {})
    folds = cross_validate(
        make_model_factory(cfg), F, target, groups,
        n_splits=cv_cfg.get("n_splits", 5),
        seed=cv_cfg.get("seed", 42),
        grouped=cv_cfg.get("grouped", True),
    )
    summary = folds.drop(columns="fold").mean().round(4).to_dict()

    if save:
        out = Path(cfg.get("output_dir", "outputs")) / cfg["experiment"]
        out.mkdir(parents=True, exist_ok=True)
        folds.to_csv(out / "folds.csv", index=False)
        payload = {
            "experiment": cfg["experiment"],
            "model": cfg["model"]["name"],
            "preprocess": cfg.get("preprocess"),
            "n_samples": int(len(target)),
            "n_features": int(F.shape[1]),
            "summary": summary,
        }
        (out / "summary.json").write_text(json.dumps(payload, indent=2))
    return folds, summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m biomed.pipeline configs/<name>.yaml")
        return 1
    folds, summary = run_experiment(load_config(argv[0]))
    print(folds.round(3).to_string(index=False))
    print("mean:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
