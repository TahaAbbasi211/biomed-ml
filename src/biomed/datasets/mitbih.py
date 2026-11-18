import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from .base import BaseDataset

FS = 360  # Hz
PRE, POST = 90, 110  # samples before / after the R peak

# AAMI classes as label codes 0..3. Q (unclassifiable) is dropped.
CLASSES = ["N", "S", "V", "F"]
LABEL_OF = {c: i for i, c in enumerate(CLASSES)}

# MIT-BIH beat annotation symbol -> AAMI class. Symbols that are not beats
# (rhythm changes, noise marks, ...) are not in this map and are ignored.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# Records with a pacemaker are excluded (AAMI recommendation).
PACED = ["102", "104", "107", "217"]

# Inter-patient split of de Chazal et al. (2004): 22 records each.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "ecg" / "mitdb"


def _bandpass(sig, fs=FS, low=0.5, high=40, order=3):
    b, a = butter(order, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, sig)


class MITBIHDataset(BaseDataset):
    """MIT-BIH Arrhythmia Database cut into single beats.

    load() returns:
      X      (n, pre + post + 2) float32: the beat window (pre + post samples
             around the annotated R peak) followed by two columns, the R-R
             interval before and after the beat (in samples)
      y      label codes 0..3 for the AAMI classes N, S, V, F
      groups record id of every beat (use it to split by patient)
    """

    name = "mitbih"

    def __init__(self, root=DEFAULT_ROOT, records=None, exclude_paced=True,
                 lead="MLII", pre=PRE, post=POST, use_cache=True):
        self.root = Path(root)
        self.records = None if records is None else [str(r) for r in records]
        self.exclude_paced = exclude_paced
        self.lead = lead
        self.pre = pre
        self.post = post
        self.use_cache = use_cache

    def _record_ids(self):
        if self.records is not None:
            return self.records
        ids = sorted(p.stem for p in self.root.glob("*.hea"))
        if not ids:
            raise FileNotFoundError(
                f"No MIT-BIH records (.hea files) found in {self.root}")
        if self.exclude_paced:
            ids = [i for i in ids if i not in PACED]
        return ids

    def _beats_of_record(self, rec):
        import wfdb  # imported here so the rest of the package works without it

        base = str(self.root / rec)
        record = wfdb.rdrecord(base)
        names = list(record.sig_name)
        ch = names.index(self.lead) if self.lead in names else 0
        sig = _bandpass(record.p_signal[:, ch].astype(float))
        ann = wfdb.rdann(base, "atr")

        samples, classes = [], []
        for s, sym in zip(ann.sample, ann.symbol):
            cls = AAMI_MAP.get(sym)
            if cls is not None:
                samples.append(int(s))
                classes.append(cls)

        X, y = [], []
        for k in range(1, len(samples) - 1):
            p = samples[k]
            if classes[k] not in LABEL_OF:
                continue
            if p - self.pre < 0 or p + self.post > len(sig):
                continue
            window = sig[p - self.pre:p + self.post]
            rr = [samples[k] - samples[k - 1], samples[k + 1] - samples[k]]
            X.append(np.concatenate([window, rr]))
            y.append(LABEL_OF[classes[k]])
        if not X:
            return (np.empty((0, self.pre + self.post + 2), dtype=np.float32),
                    np.empty(0, dtype=int))
        return np.array(X, dtype=np.float32), np.array(y, dtype=int)

    def load(self):
        ids = self._record_ids()
        meta = json.dumps({"v": 1, "records": ids, "lead": self.lead,
                           "pre": self.pre, "post": self.post})
        cache = self.root.parent / "beats_cache.npz"
        if self.use_cache and cache.exists():
            d = np.load(cache)
            if str(d["meta"]) == meta:
                return d["X"], d["y"], d["groups"]

        Xs, ys, gs = [], [], []
        for rec in ids:
            X, y = self._beats_of_record(rec)
            Xs.append(X)
            ys.append(y)
            gs.append(np.full(len(y), rec))
        X, y, groups = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)

        if self.use_cache:
            np.savez_compressed(cache, X=X, y=y, groups=groups,
                                meta=np.array(meta))
        return X, y, groups
