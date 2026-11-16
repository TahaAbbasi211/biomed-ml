import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def out_of_fold_proba(model_factory, X, y, groups, n_splits=5, seed=42):
    """Probability of class 1 for every sample, predicted by a model that
    never saw that sample's recording (group-aware cross validation)."""
    proba = np.zeros(len(y))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    for tr, te in splitter.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(groups[te]), "group leakage!"
        model = model_factory()
        model.fit(X[tr], y[tr])
        proba[te] = model.predict_proba(X[te])[:, 1]
    return proba
