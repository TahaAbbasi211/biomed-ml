import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .metrics import compute_metrics


def holdout_evaluate(model_factory, X, y, groups, test_groups=None,
                     train_groups=None, test_size=None, seed=42,
                     task="binary", labels=None, macro_labels=None):
    """Train once on some groups and test once on other groups.

    The split is by group (for example by patient), never by row.
    Give explicit test_groups (train_groups is optional; by default the
    training part is every other group), or give test_size to pick random
    test groups. Returns a one-row DataFrame.
    """
    if groups is None:
        raise ValueError("holdout evaluation needs groups")
    g = np.asarray(groups).astype(str)

    if test_groups is not None:
        test_mask = np.isin(g, [str(t) for t in test_groups])
        if train_groups is not None:
            train_mask = np.isin(g, [str(t) for t in train_groups])
        else:
            train_mask = ~test_mask
    elif test_size is not None:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                        random_state=seed).split(X, y, g))
        train_mask = np.zeros(len(y), dtype=bool)
        test_mask = np.zeros(len(y), dtype=bool)
        train_mask[tr] = True
        test_mask[te] = True
    else:
        raise ValueError("Give test_groups or test_size")

    if not train_mask.any() or not test_mask.any():
        raise ValueError("The train or test part is empty; check the group names")
    if not set(g[train_mask]).isdisjoint(g[test_mask]):
        raise ValueError("A group is in both the train and the test part")

    if task == "multiclass" and labels is None:
        labels = np.unique(y)

    model = model_factory()
    model.fit(X[train_mask], y[train_mask])
    pred = model.predict(X[test_mask])
    proba = model.predict_proba(X[test_mask])
    if task == "binary":
        proba = proba[:, 1]
    row = {"fold": 0, "n_train": int(train_mask.sum()),
           "n_test": int(test_mask.sum()),
           **compute_metrics(y[test_mask], pred, proba, task, labels,
                             macro_labels)}
    return pd.DataFrame([row])
