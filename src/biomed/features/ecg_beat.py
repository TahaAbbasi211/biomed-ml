import numpy as np
from scipy.stats import kurtosis, skew

from .base import BaseFeatureExtractor

RR_NAMES = ["prev_rr", "next_rr", "rr_ratio", "prev_rr_rel", "next_rr_rel"]
MORPH_NAMES = ["amp_r", "peak", "trough", "p2p", "t_peak", "t_trough", "width",
               "energy", "area", "sd", "skew", "kurtosis"]


class ECGBeatFeatures(BaseFeatureExtractor):
    """57 features per beat, ported from the ECG arrhythmia project.

    X columns: the beat window (beat_len samples) followed by the R-R
    interval before and after the beat. groups gives the record of every beat
    (the R-R ratios use the median R-R interval of the beat's own record);
    without groups all beats count as one record.
    """

    def __init__(self, beat_len=200, r_index=90, width_half=40, shape_step=5):
        self.beat_len = beat_len
        self.r_index = r_index
        self.width_half = width_half
        self.shape_step = shape_step

    @property
    def feature_names(self):
        n_shape = len(range(0, self.beat_len, self.shape_step))
        return RR_NAMES + MORPH_NAMES + [f"shape_{i}" for i in range(n_shape)]

    def transform(self, X, groups=None):
        X = np.asarray(X, dtype=float)
        beats = X[:, :self.beat_len]
        rr = X[:, self.beat_len:self.beat_len + 2]
        g = np.zeros(len(X), dtype=int) if groups is None else np.asarray(groups)
        return np.hstack([self._rr(rr, g), self._morph(beats)])

    @staticmethod
    def _rr(rr, g):
        prev_rr = rr[:, 0]
        next_rr = rr[:, 1]
        med_rr = np.zeros_like(prev_rr)
        for rec in np.unique(g):
            m = g == rec
            med_rr[m] = np.median(prev_rr[m])
        return np.column_stack([prev_rr, next_rr, prev_rr / next_rr,
                                prev_rr / med_rr, next_rr / med_rr])

    def _morph(self, X):
        R = self.r_index
        X = X - np.median(X, axis=1, keepdims=True)
        peak = X.max(axis=1)
        trough = X.min(axis=1)
        p2p = peak - trough
        amp_r = X[:, R]
        t_peak = X.argmax(axis=1) - R
        t_trough = X.argmin(axis=1) - R
        thr = 0.3 * np.abs(X).max(axis=1, keepdims=True)
        width = (np.abs(X[:, R - self.width_half:R + self.width_half]) > thr).sum(axis=1)
        energy = (X ** 2).sum(axis=1)
        area = np.abs(X).sum(axis=1)
        sd = X.std(axis=1)
        sk = skew(X, axis=1)
        ku = kurtosis(X, axis=1)
        shape = X[:, ::self.shape_step] / (p2p[:, None] + 1e-8)
        return np.column_stack([amp_r, peak, trough, p2p, t_peak, t_trough,
                                width, energy, area, sd, sk, ku, shape])
