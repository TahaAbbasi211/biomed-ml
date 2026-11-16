import numpy as np
import pytest

from biomed.inference import parse_signal, split_segments


def test_parse_signal_handles_mixed_separators():
    assert parse_signal("1, 2;3\n4 5").tolist() == [1, 2, 3, 4, 5]


def test_parse_signal_rejects_text():
    with pytest.raises(ValueError):
        parse_signal("1 2 abc")


def test_split_segments_drops_remainder():
    seg = split_segments(np.arange(400.0))
    assert seg.shape == (2, 178)


def test_split_segments_too_short():
    with pytest.raises(ValueError):
        split_segments(np.arange(10.0))
