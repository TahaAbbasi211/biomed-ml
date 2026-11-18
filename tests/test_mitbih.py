import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")

from biomed.datasets import MITBIHDataset
from biomed.datasets.mitbih import AAMI_MAP, DS1, DS2, PACED

FS = 360


def write_record(root, name, pairs, sig_names=("MLII", "V1"), n=4000):
    """Write a tiny synthetic MIT-BIH style record: noise plus a sharp spike
    at every annotated sample, only in the channel called MLII."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(name))
    sig = rng.normal(0, 0.01, size=(n, 2))
    samples = np.array([s for s, _ in pairs])
    symbols = [sym for _, sym in pairs]
    sig[samples, list(sig_names).index("MLII")] += 1.0
    wfdb.wrsamp(name, fs=FS, units=["mV", "mV"], sig_name=list(sig_names),
                p_signal=sig, fmt=["16", "16"], write_dir=str(root))
    wfdb.wrann(name, "atr", sample=samples, symbol=symbols, write_dir=str(root))


PAIRS = [(400, "N"), (800, "N"), (1000, "+"), (1300, "V"), (1700, "A"),
         (2100, "N"), (2500, "F"), (2900, "Q"), (3300, "N")]


def test_split_constants():
    assert len(DS1) == len(DS2) == 22
    assert set(DS1).isdisjoint(DS2)
    assert set(DS1 + DS2).isdisjoint(PACED)
    assert len(set(DS1 + DS2 + PACED)) == 48


def test_aami_map():
    for sym, cls in {"L": "N", "R": "N", "e": "N", "j": "N", "a": "S",
                     "J": "S", "S": "S", "E": "V", "f": "Q"}.items():
        assert AAMI_MAP[sym] == cls
    assert "+" not in AAMI_MAP


def test_beats_labels_rr_and_groups(tmp_path):
    root = tmp_path / "mitdb"
    write_record(root, "100", PAIRS)
    X, y, g = MITBIHDataset(root=root, use_cache=False).load()
    # beats: N800 V1300 A1700 N2100 F2500 (first, last and the Q are dropped,
    # the '+' rhythm mark is not a beat)
    assert X.shape == (5, 202)
    assert y.tolist() == [0, 2, 1, 0, 3]
    assert set(g) == {"100"}
    assert X[:, 200:].tolist() == [[400, 500], [500, 400], [400, 400],
                                   [400, 400], [400, 400]]
    peak = np.abs(X[:, :200]).argmax(axis=1)
    assert (np.abs(peak - 90) <= 2).all()


def test_lead_is_chosen_by_name(tmp_path):
    root = tmp_path / "mitdb"
    write_record(root, "101", PAIRS, sig_names=("V1", "MLII"))
    X, _, _ = MITBIHDataset(root=root, use_cache=False).load()
    peak = np.abs(X[:, :200]).argmax(axis=1)
    assert (np.abs(peak - 90) <= 2).all()


def test_paced_records_are_excluded_by_default(tmp_path):
    root = tmp_path / "mitdb"
    for name in ["100", "101", "102"]:
        write_record(root, name, PAIRS)
    _, _, g = MITBIHDataset(root=root, use_cache=False).load()
    assert set(g) == {"100", "101"}
    _, _, g_all = MITBIHDataset(root=root, use_cache=False,
                                exclude_paced=False).load()
    assert set(g_all) == {"100", "101", "102"}


def test_records_argument_selects_records(tmp_path):
    root = tmp_path / "mitdb"
    for name in ["100", "101"]:
        write_record(root, name, PAIRS)
    _, _, g = MITBIHDataset(root=root, records=["101"], use_cache=False).load()
    assert set(g) == {"101"}


def test_beats_too_close_to_the_start_are_dropped(tmp_path):
    root = tmp_path / "mitdb"
    write_record(root, "100", [(30, "N"), (60, "N"), (500, "N"), (900, "N")],
                 n=1500)
    X, _, _ = MITBIHDataset(root=root, use_cache=False).load()
    assert X.shape[0] == 1   # only the beat at 500 has a full window


def test_cache_gives_identical_data(tmp_path):
    root = tmp_path / "mitdb"
    write_record(root, "100", PAIRS)
    a = MITBIHDataset(root=root, use_cache=True).load()
    assert list(tmp_path.glob("beats_cache_*.npz"))
    b = MITBIHDataset(root=root, use_cache=True).load()
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MITBIHDataset(root=tmp_path / "nope", use_cache=False).load()
