import numpy as np

from biomed.features import ECGBeatFeatures


def make_x(n=6, seed=0, rr=None):
    rng = np.random.default_rng(seed)
    beats = rng.normal(size=(n, 200))
    rr = np.tile([[400.0, 400.0]], (n, 1)) if rr is None else np.asarray(rr, float)
    return np.hstack([beats, rr])


def test_feature_count_and_names():
    f = ECGBeatFeatures()
    out = f.transform(make_x(30, rr=np.random.default_rng(1).integers(250, 500, (30, 2))))
    assert out.shape == (30, 57)
    assert len(f.feature_names) == 57
    assert len(set(f.feature_names)) == 57
    assert np.isfinite(out).all()


def test_rr_relative_to_median_of_each_record():
    prev = [400, 400, 800, 300, 300, 300]
    nxt = [400, 400, 400, 300, 300, 300]
    X = make_x(rr=np.column_stack([prev, nxt]))
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    f = ECGBeatFeatures()
    out = f.transform(X, groups)
    names = f.feature_names
    assert out[:, names.index("prev_rr_rel")].tolist() == [1, 1, 2, 1, 1, 1]
    assert out[:, names.index("rr_ratio")][2] == 2.0


def test_without_groups_all_beats_are_one_record():
    prev = [400, 400, 800, 300, 300, 300]
    X = make_x(rr=np.column_stack([prev, [400] * 6]))
    out = ECGBeatFeatures().transform(X)
    rel = out[:, ECGBeatFeatures().feature_names.index("prev_rr_rel")]
    assert np.allclose(rel, np.array(prev) / np.median(prev))


def test_amplitude_at_r_peak_and_normalised_shape():
    X = make_x()
    f = ECGBeatFeatures()
    out = f.transform(X)
    names = f.feature_names
    x = X[0, :200] - np.median(X[0, :200])
    assert np.isclose(out[0, names.index("amp_r")], x[90])
    p2p = x.max() - x.min()
    first = names.index("shape_0")
    assert np.allclose(out[0, first:first + 40], x[::5] / (p2p + 1e-8))


def test_wrong_width_still_works_with_custom_window():
    f = ECGBeatFeatures(beat_len=100, r_index=40, width_half=20)
    rng = np.random.default_rng(0)
    X = np.hstack([rng.normal(size=(5, 100)), np.full((5, 2), 300.0)])
    assert f.transform(X).shape == (5, 5 + 12 + 20)
