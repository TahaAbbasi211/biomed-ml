import sys
from pathlib import Path

import joblib

from .pipeline import (build_dataset, build_features, load_config,
                       make_model_factory, make_target)


def train_final_model(cfg):
    """Fit the configured model on ALL data. Its score on this same data is
    meaningless; use the evaluation results for performance claims."""
    dataset = build_dataset(cfg["dataset"])
    X, y, _ = dataset.load()
    target = make_target(cfg, y)

    features = build_features(cfg["features"], dataset)
    model = make_model_factory(cfg)()
    model.fit(features.transform(X), target)
    return {
        "model": model,
        "features": features,
        "experiment": cfg["experiment"],
        "segment_len": int(X.shape[1]),
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m biomed.train configs/<name>.yaml")
        return 1
    cfg = load_config(argv[0])
    bundle = train_final_model(cfg)
    out = Path("models")
    out.mkdir(exist_ok=True)
    path = out / f"{cfg['experiment']}.joblib"
    joblib.dump(bundle, path, compress=3)
    print("saved", path, f"({path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
