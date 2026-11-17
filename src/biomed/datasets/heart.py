from pathlib import Path

import pandas as pd

from .base import BaseDataset

FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "heart" / "heart.csv"


class HeartDiseaseDataset(BaseDataset):
    """Cleveland heart disease data: 13 clinical features and a binary target.

    One row is one patient, so there are no groups (load() returns None).
    Missing values stay as NaN; imputation happens inside the model pipeline.
    """

    name = "heart_cleveland"
    feature_names = FEATURE_COLUMNS

    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)

    def load(self):
        df = pd.read_csv(self.path)
        missing = [c for c in FEATURE_COLUMNS + ["target"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {self.path}: {missing}")
        X = (df[FEATURE_COLUMNS]
             .apply(pd.to_numeric, errors="coerce")
             .to_numpy(dtype=float))
        y = df["target"].to_numpy().astype(int)
        return X, y, None
