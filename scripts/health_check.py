"""Project health check: prints PASS/FAIL for each part of biomed-ml."""
import importlib
import json
import os
import py_compile
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

results = []
state = {}


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            info, ok = fn(), True
        except Exception as e:
            tb = traceback.format_exc().splitlines()[-6:]
            info, ok = f"{type(e).__name__}: {e}\n" + "\n".join(tb), False
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({time.time() - t0:.1f}s)")
        for line in str(info or "").splitlines():
            print("       " + line)
        return fn
    return wrap


print("python:", sys.executable)
print("version:", sys.version.split()[0])
for pkg in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "PyYAML",
            "joblib", "streamlit", "pytest", "pytest-cov", "nbformat", "ipykernel"]:
    try:
        v = metadata.version(pkg)
    except metadata.PackageNotFoundError:
        v = "NOT INSTALLED"
    print(f"  {pkg:14s} {v}")
print()


@check("imports")
def _():
    for m in ["biomed", "biomed.datasets", "biomed.features", "biomed.models",
              "biomed.evaluation", "biomed.pipeline", "biomed.train",
              "biomed.inference"]:
        importlib.import_module(m)
    from biomed.evaluation import cross_validate, out_of_fold_proba  # noqa: F401
    return "all modules import"


@check("dataset loads (Bonn)")
def _():
    from biomed.datasets import BonnEEGDataset
    X, y, g = BonnEEGDataset().load()
    state.update(X=X, y=y, g=g)
    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    assert X.shape == (11500, 178), X.shape
    assert len(counts) == 5 and all(v == 2300 for v in counts.values()), counts
    assert len(set(g)) == 500, len(set(g))
    assert np.isfinite(X).all()
    return f"X={X.shape}, classes={counts}, recordings={len(set(g))}"


@check("features")
def _():
    from biomed.features import BandPowerFeatures, CombinedFeatures, StatsFeatures
    X, y = state["X"], state["y"]
    fe = CombinedFeatures([StatsFeatures(), BandPowerFeatures()])
    F = fe.transform(X)
    state.update(F=F, fe=fe)
    names = fe.feature_names
    assert F.shape == (11500, 21), F.shape
    assert np.isfinite(F).all()
    assert len(set(names)) == 21
    rel = [i for i, n in enumerate(names) if n.endswith("_rel")]
    assert (F[:, rel] >= 0).all() and (F[:, rel] <= 1.0001).all()
    i = names.index("ptp")
    a, b = np.median(F[y == 1, i]), np.median(F[y != 1, i])
    assert a > b, "seizure ptp should be larger"
    return f"F={F.shape}; median ptp: seizure={a:.0f} vs others={b:.0f}"


@check("no group leakage in splits")
def _():
    from sklearn.model_selection import StratifiedGroupKFold
    y, g, F = state["y"], state["g"], state["F"]
    target = (y == 1).astype(int)
    state["target"] = target
    fr = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(F, target, g):
        assert set(g[tr]).isdisjoint(g[te]), "recording appears in train and test"
        fr.append(round(float(target[te].mean()), 3))
    return f"test-fold seizure fractions: {fr}"


@check("experiments from configs (grouped CV)")
def _():
    from biomed.pipeline import load_config, run_experiment
    paths = sorted(Path("configs").glob("*.yaml"))
    assert paths, "no configs found"
    lines = []
    for p in paths:
        cfg = load_config(p)
        _, s = run_experiment(cfg, save=False)
        lines.append(f"{cfg['experiment']}: {s}")
        assert s["roc_auc"] > 0.5, "worse than chance"
    lines.append("reference: always answering 'no seizure' gives accuracy 0.8 and recall 0")
    return "\n".join(lines)


@check("leakage effect (random forest)")
def _():
    from biomed.evaluation import cross_validate
    from biomed.models import MODELS
    F, t, g = state["F"], state["target"], state["g"]
    a = cross_validate(MODELS["random_forest"], F, t, g, grouped=True)
    b = cross_validate(MODELS["random_forest"], F, t, g, grouped=False)
    return (f"grouped: f1={a.f1.mean():.4f} auc={a.roc_auc.mean():.4f}\n"
            f"naive  : f1={b.f1.mean():.4f} auc={b.roc_auc.mean():.4f}")


@check("final model file and prediction")
def _():
    from biomed.inference import load_bundle, predict_proba
    p = Path("models/eeg_random_forest.joblib")
    assert p.exists(), "missing: run python -m biomed.train configs/eeg_rf.yaml"
    bundle = load_bundle(p)
    X, y = state["X"], state["y"]
    neg_idx = np.random.default_rng(0).choice(np.where(y != 1)[0], 200, replace=False)
    pos = predict_proba(bundle, X[y == 1][:50]).mean()
    neg = predict_proba(bundle, X[neg_idx]).mean()
    assert pos > neg
    return (f"size={p.stat().st_size / 1e6:.2f} MB; mean seizure probability: "
            f"seizure samples={pos:.3f}, other samples={neg:.3f} "
            "(in-sample, not a performance estimate)")


@check("streamlit app compiles")
def _():
    py_compile.compile("app/streamlit_app.py", doraise=True)
    return "syntax OK"


@check("notebooks saved without errors")
def _():
    problems, lines = [], []
    for p in sorted(Path("notebooks").glob("*.ipynb")):
        text = p.read_text(encoding="utf-8")
        nb = json.loads(text)
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        errs = sum(1 for c in code for o in c.get("outputs", [])
                   if o.get("output_type") == "error")
        notrun = sum(1 for c in code if c.get("execution_count") is None)
        local = ("Users\\\\" in text) or ("AppData" in text)
        lines.append(f"{p.name}: code cells={len(code)}, errors={errs}, "
                     f"not run={notrun}, local paths in outputs={local}")
        if errs or notrun:
            problems.append(p.name)
    assert not problems, "problems in: " + ", ".join(problems) + "\n" + "\n".join(lines)
    return "\n".join(lines)


print("\n==== SUMMARY ====")
failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", ", ".join(failed))
    print("(a failure in an early check can cause later ones to fail too)")
sys.exit(1 if failed else 0)
