import numpy as np
import pandas as pd
import pytest

from sortinghat.models import FoldPreprocessor, HeadConfig, PreprocConfig, fit_model
from sortinghat.models.head import LabelCalibrator, ShallowHead, prevalence_prior


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    lp = np.column_stack([1.5 * X[:, 0], -1.0 * X[:, 1], 0.0 * X[:, 2]])
    y = (rng.random((n, 3)) < 1 / (1 + np.exp(-lp))).astype(float)
    return pd.DataFrame(X, columns=list("abcde")), y, np.ones_like(y, bool)


def test_preprocessor_uses_training_statistics_only():
    rng = np.random.default_rng(1)
    Xtr = rng.normal(5, 2, size=(200, 3))
    Xtr[::10, 1] = np.nan
    pre = FoldPreprocessor().fit(Xtr, ["a", "b", "c"])
    np.testing.assert_allclose(pre.mean_, np.nanmean(np.where(np.isnan(Xtr), np.nanmedian(Xtr, 0), Xtr), 0)[pre.keep_])
    Z1 = pre.transform(np.array([[5.0, 5.0, 5.0]]))
    Z2 = pre.transform(np.array([[5.0, 5.0, 5.0], [1e9, np.nan, -1e9]]))   # other rows do not change row 0
    np.testing.assert_array_equal(Z1[0], Z2[0])
    assert any(c.endswith("__missing") for c in pre.out_columns_)           # indicator only where train had NaN
    assert np.abs(Z2).max() <= 5.0 + 1.0                                     # clipped (indicator is 0/1)


def test_preprocessor_drops_constant_and_selects_top_k():
    rng = np.random.default_rng(2)
    n = 400
    y = (rng.random((n, 1)) < 0.4).astype(float)
    X = np.column_stack([y[:, 0] + rng.normal(0, 0.5, n), rng.normal(size=(n, 6)), np.full(n, 3.0)])
    cols = ["sig"] + [f"n{i}" for i in range(6)] + ["const"]
    pre = FoldPreprocessor(PreprocConfig(max_eeg_features=2)).fit(X, cols, y, np.ones_like(y, bool))
    names = [cols[j] for j in pre.keep_]
    assert "const" not in names and "sig" in names and len(names) == 2


def test_selection_never_drops_protected_columns():
    rng = np.random.default_rng(3)
    n = 300
    y = (rng.random((n, 1)) < 0.4).astype(float)
    X = rng.normal(size=(n, 6))
    cols = ["base0", "base1", "e0", "e1", "e2", "e3"]
    pre = FoldPreprocessor(PreprocConfig(max_eeg_features=1)).fit(X, cols, y, np.ones_like(y, bool), protected=["base0", "base1"])
    kept = [cols[j] for j in pre.keep_]
    assert "base0" in kept and "base1" in kept and len(kept) == 3


def test_head_learns_and_handles_degenerate_label():
    X, y, m = _toy()
    pre = FoldPreprocessor().fit(X.values, list(X.columns))
    Z = pre.transform(X.values)
    y[:, 2] = 0.0                                   # label with no positives -> smoothed prevalence
    h = ShallowHead(HeadConfig(), 1.0).fit(Z, y, m)
    P = h.predict_proba(Z)
    assert P[:, 0][y[:, 0] == 1].mean() > P[:, 0][y[:, 0] == 0].mean() + 0.2
    assert np.allclose(P[:, 2], P[0, 2]) and P[0, 2] < 0.01
    # MLP variant works with the same interface
    hm = ShallowHead(HeadConfig(kind="mlp", max_iter=200), 1.0).fit(Z, y, m)
    assert hm.predict_proba(Z).shape == P.shape


def test_calibrators_fix_miscalibration_and_fall_back_to_identity():
    rng = np.random.default_rng(4)
    n = 2000
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < p_true).astype(float)[:, None]
    p_bad = np.clip(p_true ** 3, 1e-3, 1 - 1e-3)[:, None]       # distorted
    m = np.ones_like(y, bool)
    for method in ("platt", "isotonic"):
        cal = LabelCalibrator(method, 1e-4, 8).fit(p_bad, y, m)
        out = cal.transform(p_bad)
        assert abs(out.mean() - y.mean()) < abs(p_bad.mean() - y.mean())
    thin = LabelCalibrator("platt", 1e-4, 8).fit(p_bad[:20], y[:20] * 0, m[:20])
    np.testing.assert_array_equal(thin.transform(p_bad), p_bad)


def test_fit_model_uses_dev_only_for_selection_and_falls_back_without_it():
    X, y, m = _toy(800)
    Xtr, ytr, mtr = X.iloc[:500], y[:500], m[:500]
    Xdev, ydev, mdev = X.iloc[500:], y[500:], m[500:]
    a = fit_model(Xtr, ytr, mtr, Xdev, ydev, mdev)
    assert a.info["used_dev"] and a.info["calibrated"]
    b = fit_model(Xtr, ytr, mtr, None, None, None)
    assert not b.info["used_dev"] and not b.info["calibrated"] and b.info["selected_c"] == HeadConfig().default_c
    P = a.predict(Xdev)
    assert P.shape == (300, 3) and np.all((P > 0) & (P < 1))
    with pytest.raises(ValueError):
        a.predict(Xdev.rename(columns={"a": "z"}))


def test_prevalence_prior():
    y = np.array([[1.0, 0], [1, 0], [0, 0], [0, 1]])
    m = np.array([[1, 1], [1, 1], [1, 1], [1, 0]], bool)
    p = prevalence_prior(y, m)
    assert np.allclose(p, [(2 + .5) / 5, (0 + .5) / 4])
