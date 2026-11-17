import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def out_of_fold_proba(model_factory, X, y, groups=None, n_splits=5, seed=42):
    """Probability of class 1 for every sample, predicted by a model that
    never saw that sample (group-aware when groups are given)."""
    proba = np.zeros(len(y))
    if groups is None:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=seed).split(X, y)
    else:
        splits = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed).split(X, y, groups)
    for tr, te in splits:
        if groups is not None:
            assert set(groups[tr]).isdisjoint(groups[te]), "group leakage!"
        model = model_factory()
        model.fit(X[tr], y[tr])
        proba[te] = model.predict_proba(X[te])[:, 1]
    return proba
