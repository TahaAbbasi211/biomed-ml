import sys
from pathlib import Path

import joblib
import numpy as np

from .models import build_model
from .pipeline import build_dataset, build_features, load_config


def train_final_model(cfg):
    """Fit the configured model on ALL data. Its score on this same data is
    meaningless; use the cross-validation results for performance claims."""
    X, y, _ = build_dataset(cfg["dataset"]).load()
    positive = cfg["dataset"].get("positive_labels", [1])
    target = np.isin(y, positive).astype(int)

    features = build_features(cfg["features"])
    model = build_model(cfg["model"]["name"])
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
