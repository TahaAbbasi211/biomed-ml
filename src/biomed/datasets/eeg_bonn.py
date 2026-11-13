from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset

# Bonn folder -> label (same labels as the Kaggle version of the dataset)
SET_TO_LABEL = {"S": 1, "F": 2, "N": 3, "O": 4, "Z": 5}
SEGMENT_LEN = 178
SAMPLING_RATE = 173.61  # Hz
SUPPORTED = {".mat", ".txt"}

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "eeg" / "raw"


class BonnEEGDataset(BaseDataset):
    """Bonn University EEG data, cut into 178-sample segments."""

    name = "eeg_bonn"

    def __init__(self, root=DEFAULT_ROOT, segment_len=SEGMENT_LEN):
        self.root = Path(root)
        self.segment_len = segment_len

    def _read_signal(self, path):
        if path.suffix.lower() == ".txt":
            return np.loadtxt(path, dtype=float).ravel()
        mat = loadmat(path)
        arrays = [
            v for k, v in mat.items()
            if not k.startswith("__")
            and isinstance(v, np.ndarray)
            and v.dtype.kind in "fiu"
            and v.size > 1000
        ]
        if not arrays:
            raise ValueError(f"No signal array found in {path}")
        return arrays[0].astype(float).ravel()

    def load(self):
        X, y, groups = [], [], []
        L = self.segment_len
        for set_name, label in SET_TO_LABEL.items():
            folder = self.root / set_name
            files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)
            if not files:
                raise FileNotFoundError(f"No signal files found in {folder}")
            for f in files:
                signal = self._read_signal(f)
                for i in range(len(signal) // L):
                    X.append(signal[i * L:(i + 1) * L])
                    y.append(label)
                    groups.append(f"{set_name}_{f.stem}")
        return np.array(X), np.array(y), np.array(groups)
