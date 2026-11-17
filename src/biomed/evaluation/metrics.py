import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)


def compute_metrics(y_true, pred, proba=None, task="binary", labels=None,
                    macro_labels=None):
    """One flat dict of metrics for a single evaluation.

    binary     : accuracy, precision, recall, f1, roc_auc (class 1 is the
                 positive class; proba is the probability of class 1)
    multiclass : accuracy, precision_/recall_/f1_ for every label, and
                 macro_* averages over macro_labels (default: all labels).
                 Use macro_labels to leave a class out of the average.
    """
    if task == "binary":
        return {
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
        }
    if task != "multiclass":
        raise ValueError(f"Unknown task '{task}' (use 'binary' or 'multiclass')")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, pred]))
    labels = [int(l) for l in labels]
    macro_labels = [int(l) for l in (labels if macro_labels is None else macro_labels)]

    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=labels, zero_division=0)
    out = {"accuracy": accuracy_score(y_true, pred)}
    for lab, pi, ri, fi in zip(labels, p, r, f):
        out[f"precision_{lab}"] = float(pi)
        out[f"recall_{lab}"] = float(ri)
        out[f"f1_{lab}"] = float(fi)
    idx = [labels.index(l) for l in macro_labels]
    out["macro_precision"] = float(np.mean(p[idx]))
    out["macro_recall"] = float(np.mean(r[idx]))
    out["macro_f1"] = float(np.mean(f[idx]))
    return out
