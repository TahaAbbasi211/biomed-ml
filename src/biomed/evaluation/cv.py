import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .metrics import compute_metrics


def cross_validate(model_factory, X, y, groups=None, n_splits=5, seed=42,
                   grouped=True, task="binary", labels=None, macro_labels=None):
    """Cross-validate a model and return one row of metrics per fold.

    grouped=True keeps all segments of one recording on the same side of the
    split (no leakage). grouped=False is a plain stratified split, kept only
    to show how much leakage inflates the score. If groups is None (one row
    per subject) a plain stratified split is used.
    task is 'binary' or 'multiclass' (see compute_metrics).
    """
    grouped = grouped and groups is not None
    if task == "multiclass" and labels is None:
        labels = np.unique(y)

    if grouped:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)
        splits = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        splits = splitter.split(X, y)

    rows = []
    for fold, (tr, te) in enumerate(splits):
        if grouped:
            assert set(groups[tr]).isdisjoint(groups[te]), "group leakage!"
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])
        if task == "binary":
            proba = proba[:, 1]
        rows.append({"fold": fold,
                     **compute_metrics(y[te], pred, proba, task, labels,
                                       macro_labels)})
    return pd.DataFrame(rows)
