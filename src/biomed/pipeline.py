import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.pipeline import make_pipeline

from .datasets import BonnEEGDataset, HeartDiseaseDataset, MITBIHDataset
from .evaluation import cross_validate, holdout_evaluate
from .features import (BandPowerFeatures, CombinedFeatures, ECGBeatFeatures,
                       RawFeatures, StatsFeatures)
from .models import WeightedDecisionClassifier, build_model
from .preprocessing import PREPROCESSORS

# Registries: to plug in something new, write its class and add one line here.
DATASETS = {"eeg_bonn": BonnEEGDataset, "heart_cleveland": HeartDiseaseDataset,
            "mitbih": MITBIHDataset}
FEATURES = {"stats": StatsFeatures, "band_power": BandPowerFeatures,
            "raw": RawFeatures, "ecg_beat": ECGBeatFeatures}


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
    """Returns a function that builds a fresh model.

    model.decision_weights wraps the model so that predictions are
    argmax(probability * weight). 'preprocess' puts a preprocessing step in
    front, inside a Pipeline, so it is fitted again in every fold.
    """
    mcfg = cfg["model"]
    model_name = mcfg["name"]
    weights = mcfg.get("decision_weights")
    prep_name = cfg.get("preprocess")
    if prep_name is not None and prep_name not in PREPROCESSORS:
        raise KeyError(f"Unknown preprocessor '{prep_name}'. "
                       f"Available: {list(PREPROCESSORS)}")

    def factory():
        model = build_model(model_name)
        if weights is not None:
            model = WeightedDecisionClassifier(model, weights)
        if prep_name is None:
            return model
        return make_pipeline(PREPROCESSORS[prep_name](), model)

    return factory


def make_target(cfg, y):
    """Binary task: 1 for the positive labels, 0 for the rest.
    Multiclass task: the labels as they are."""
    if cfg.get("task", "binary") == "multiclass":
        return np.asarray(y).astype(int)
    positive = cfg["dataset"].get("positive_labels", [1])
    return np.isin(y, positive).astype(int)


def _resolve_groups(spec, dataset):
    """A list of group names, or the name of a split the dataset defines."""
    if isinstance(spec, str):
        splits = getattr(dataset, "named_splits", {})
        if spec not in splits:
            raise KeyError(f"Unknown group set '{spec}'. Available: {list(splits)}")
        return list(splits[spec])
    return spec


def run_evaluation(cfg, factory, F, target, groups, dataset=None):
    """Evaluate with cross validation (default) or a group holdout split."""
    common = dict(task=cfg.get("task", "binary"),
                  labels=cfg.get("labels"),
                  macro_labels=cfg.get("macro_labels"))
    ev = cfg.get("evaluation") or {}
    kind = ev.get("type", "cv")
    if kind == "holdout":
        return holdout_evaluate(
            factory, F, target, groups,
            test_groups=_resolve_groups(ev.get("test_groups"), dataset),
            train_groups=_resolve_groups(ev.get("train_groups"), dataset),
            test_size=ev.get("test_size"),
            seed=ev.get("seed", 42), **common)
    if kind != "cv":
        raise ValueError(f"Unknown evaluation type '{kind}'. Use 'cv' or 'holdout'.")
    cv_cfg = cfg.get("cv", {})
    return cross_validate(
        factory, F, target, groups,
        n_splits=cv_cfg.get("n_splits", 5),
        seed=cv_cfg.get("seed", 42),
        grouped=cv_cfg.get("grouped", True), **common)


def run_experiment(cfg, save=True):
    """Run one experiment described by a config dict. Returns (folds, summary)."""
    dataset = build_dataset(cfg["dataset"])
    X, y, groups = dataset.load()
    target = make_target(cfg, y)

    fe = build_features(cfg["features"], dataset)
    F = fe.transform(X, groups)

    folds = run_evaluation(cfg, make_model_factory(cfg), F, target, groups,
                           dataset)
    skip = {"fold", "n_train", "n_test"}
    metric_cols = [c for c in folds.columns if c not in skip]
    summary = folds[metric_cols].mean().round(4).to_dict()

    if save:
        out = Path(cfg.get("output_dir", "outputs")) / cfg["experiment"]
        out.mkdir(parents=True, exist_ok=True)
        folds.to_csv(out / "folds.csv", index=False)
        payload = {
            "experiment": cfg["experiment"],
            "task": cfg.get("task", "binary"),
            "evaluation": (cfg.get("evaluation") or {}).get("type", "cv"),
            "model": cfg["model"]["name"],
            "decision_weights": cfg["model"].get("decision_weights"),
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
