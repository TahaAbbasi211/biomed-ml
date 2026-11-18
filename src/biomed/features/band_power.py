import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from .base import BaseFeatureExtractor

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}


class BandPowerFeatures(BaseFeatureExtractor):
    """Absolute and relative power in EEG frequency bands (Welch PSD)."""

    def __init__(self, fs=173.61, bands=None, nperseg=128):
        self.fs = fs
        self.bands = bands or BANDS
        self.nperseg = nperseg

    @property
    def feature_names(self):
        names = []
        for b in self.bands:
            names += [f"{b}_abs", f"{b}_rel"]
        return names + ["dominant_freq"]

    def transform(self, X, groups=None):
        X = np.asarray(X, dtype=float)
        freqs, psd = welch(X, fs=self.fs, nperseg=self.nperseg, axis=-1)
        total = trapezoid(psd, freqs, axis=-1) + 1e-12
        feats = []
        for lo, hi in self.bands.values():
            mask = (freqs >= lo) & (freqs < hi)
            power = trapezoid(psd[:, mask], freqs[mask], axis=-1)
            feats += [power, power / total]
        feats.append(freqs[np.argmax(psd, axis=1)])
        return np.column_stack(feats)
