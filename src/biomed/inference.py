import re

import joblib
import numpy as np


def load_bundle(path):
    return joblib.load(path)


def parse_signal(text):
    """Parse numbers separated by spaces, commas, semicolons or newlines."""
    tokens = [t for t in re.split(r"[,\s;]+", text.strip()) if t]
    try:
        return np.array([float(t) for t in tokens])
    except ValueError as e:
        raise ValueError(
            "The file must contain only numbers separated by spaces, "
            "commas or newlines."
        ) from e


def split_segments(values, segment_len=178):
    """Cut a long signal into non-overlapping segments; drop the remainder."""
    n = len(values) // segment_len
    if n == 0:
        raise ValueError(
            f"Need at least {segment_len} values, got {len(values)}."
        )
    return values[: n * segment_len].reshape(n, segment_len)


def predict_proba(bundle, segments):
    """Probability of the positive class (seizure) for each segment."""
    feats = bundle["features"].transform(segments)
    return bundle["model"].predict_proba(feats)[:, 1]
