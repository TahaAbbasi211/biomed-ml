import numpy as np

from biomed.features import BandPowerFeatures, CombinedFeatures, StatsFeatures


def make_signals(n=20, length=178, seed=0):
    return np.random.default_rng(seed).normal(size=(n, length))


def test_stats_shape_and_names():
    f = StatsFeatures()
    out = f.transform(make_signals())
    assert out.shape == (20, len(f.feature_names))
    assert np.isfinite(out).all()


def test_relative_band_power_is_a_fraction():
    f = BandPowerFeatures()
    out = f.transform(make_signals())
    rel_idx = [i for i, n in enumerate(f.feature_names) if n.endswith("_rel")]
    rel = out[:, rel_idx]
    assert (rel >= 0).all()
    assert (rel.sum(axis=1) <= 1.0001).all()


def test_scaling_changes_absolute_but_not_relative_power():
    f = BandPowerFeatures()
    X = make_signals()
    a, b = f.transform(X), f.transform(2 * X)
    abs_idx = [i for i, n in enumerate(f.feature_names) if n.endswith("_abs")]
    rel_idx = [i for i, n in enumerate(f.feature_names) if n.endswith("_rel")]
    assert np.allclose(b[:, abs_idx], 4 * a[:, abs_idx])
    assert np.allclose(b[:, rel_idx], a[:, rel_idx], atol=1e-6)


def test_combined_features_concatenate():
    c = CombinedFeatures([StatsFeatures(), BandPowerFeatures()])
    out = c.transform(make_signals())
    assert out.shape == (20, 21)
    assert len(c.feature_names) == 21
