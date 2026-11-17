import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def cross_validate(model_factory, X, y, groups=None, n_splits=5, seed=42,
                   grouped=True):
    """Cross-validate a model and return one row of metrics per fold.

    grouped=True keeps all segments of one recording on the same side of the
    split (no leakage). grouped=False is a plain stratified split, kept only
    to show how much leakage inflates the score. If groups is None (one row
    per subject) a plain stratified split is used.
    """
    grouped = grouped and groups is not None
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
        proba = model.predict_proba(X[te])[:, 1]
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[te], pred),
            "precision": precision_score(y[te], pred, zero_division=0),
            "recall": recall_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[te], proba),
        })
    return pd.DataFrame(rows)
