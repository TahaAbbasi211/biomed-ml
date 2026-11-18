import pytest

pytest.importorskip("wfdb")

from biomed.datasets import MITBIHDataset

from .test_mitbih import PAIRS, write_record


def test_subset_loads_only_the_named_records(tmp_path, monkeypatch):
    root = tmp_path / "mitdb"
    for name in ["100", "101"]:
        write_record(root, name, PAIRS)
    monkeypatch.setattr(MITBIHDataset, "named_splits",
                        {"DS1": ["101"], "DS2": ["100"]})
    _, _, g = MITBIHDataset(root=root, subset="DS1", use_cache=False).load()
    assert set(g) == {"101"}
    _, _, g2 = MITBIHDataset(root=root, subset="DS2", use_cache=False).load()
    assert set(g2) == {"100"}


def test_unknown_subset_raises(tmp_path):
    with pytest.raises(KeyError):
        MITBIHDataset(root=tmp_path, subset="DS9", use_cache=False).load()


def test_different_record_sets_use_different_cache_files(tmp_path):
    root = tmp_path / "mitdb"
    for name in ["100", "101"]:
        write_record(root, name, PAIRS)
    MITBIHDataset(root=root, records=["100"], use_cache=True).load()
    MITBIHDataset(root=root, records=["101"], use_cache=True).load()
    assert len(list(tmp_path.glob("beats_cache_*.npz"))) == 2
