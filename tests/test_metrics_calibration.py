import numpy as np
import pytest

from sortinghat.metrics import (
    auroc,
    brier,
    calibration_slope_intercept,
    ece,
    per_label_report,
    per_label_risk_coverage,
    risk_coverage,
)
from sortinghat.metrics.calibration import fit_logistic, logit


def _sim(n, a=0.0, b=1.0, seed=0, sd=1.2, center=-1.0):
    rng = np.random.default_rng(seed)
    lp = rng.normal(center, sd, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(a + b * lp)))).astype(float)
    p = 1 / (1 + np.exp(-lp))
    return y, p


def test_logistic_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40000)
    y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(0.4 - 0.9 * x)))).astype(float)
    beta, cov, ok = fit_logistic(np.column_stack([np.ones_like(x), x]), y)
    assert ok
    assert beta == pytest.approx([0.4, -0.9], abs=0.04)
    assert np.sqrt(cov[1, 1]) < 0.02


def test_well_calibrated_predictions_have_slope_one_and_zero_citl():
    y, p = _sim(60000, seed=1)
    r = calibration_slope_intercept(y, p)
    assert r["slope"] == pytest.approx(1.0, abs=0.05)
    assert r["intercept"] == pytest.approx(0.0, abs=0.06)
    assert r["citl"] == pytest.approx(0.0, abs=0.05)
    assert r["oe"] == pytest.approx(1.0, abs=0.03)
    assert r["slope_lo"] < 1.0 < r["slope_hi"]


def test_overconfident_model_has_slope_below_one():
    # truth uses 0.5 * LP: a model that reports LP is overfit; known slope 0.5, intercept 0.3
    y, p = _sim(80000, a=0.3, b=0.5, seed=2)
    r = calibration_slope_intercept(y, p)
    assert r["slope"] == pytest.approx(0.5, abs=0.05)
    assert r["intercept"] == pytest.approx(0.3, abs=0.07)


def test_citl_sign_for_overestimated_risk():
    y, p = _sim(60000, a=-1.0, b=1.0, seed=3)  # truth lower than predicted
    r = calibration_slope_intercept(y, p)
    assert r["citl"] == pytest.approx(-1.0, abs=0.08)
    assert r["oe"] < 0.8


def test_degenerate_inputs_return_nan_not_error():
    r = calibration_slope_intercept(np.zeros(50), np.full(50, 0.1))
    assert np.isnan(r["slope"]) and r["n_events"] == 0
    # perfect separation: not converged -> NaN
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9])
    r = calibration_slope_intercept(y, p)
    assert not r["converged"] and np.isnan(r["slope"])


def test_ece_known_values():
    assert ece(np.array([1, 0, 0, 0.0]), np.full(4, 0.25)) == pytest.approx(0.0)
    assert ece(np.zeros(10), np.full(10, 0.9)) == pytest.approx(0.9)
    # two bins: half at p=0.1 (10% events), half at p=0.9 (50% events)
    y = np.array([1] + [0] * 9 + [1] * 5 + [0] * 5, dtype=float)
    p = np.array([0.1] * 10 + [0.9] * 10)
    assert ece(y, p) == pytest.approx(0.5 * 0.0 + 0.5 * 0.4)
    assert ece(y, p, strategy="quantile", n_bins=2) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        ece(y, p, strategy="x")


def test_ece_near_zero_for_calibrated_large_sample():
    y, p = _sim(100000, seed=4)
    assert ece(y, p) < 0.01


def test_brier_and_auroc_known():
    y = np.array([1, 0, 1, 0.0])
    p = np.array([0.9, 0.2, 0.6, 0.4])
    assert brier(y, p) == pytest.approx(np.mean([0.01, 0.04, 0.16, 0.16]))
    assert auroc(y, p) == 1.0
    assert auroc(y, 1 - p) == 0.0
    assert auroc(y, np.full(4, 0.5)) == 0.5
    assert np.isnan(auroc(np.zeros(4), p))


def test_per_label_report_respects_mask():
    rng = np.random.default_rng(0)
    n = 5000
    y = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 2, n)]).astype(float)
    p = np.column_stack([0.2 + 0.6 * y[:, 0], rng.uniform(size=n)])
    mask = np.column_stack([np.ones(n, bool), rng.uniform(size=n) < 0.5])
    rep = per_label_report(y, p, mask, ["E1", "E2"])
    assert rep["E1"]["auroc"] == 1.0 and rep["E1"]["n"] == n
    assert rep["E2"]["n"] == int(mask[:, 1].sum())
    assert rep["E2"]["auroc"] == pytest.approx(0.5, abs=0.05)


def test_risk_coverage_known_curve():
    y = np.array([1, 1, 0, 0.0])
    p = np.array([0.95, 0.3, 0.05, 0.55])  # errors: index 1 and 3; confidence .95,.7,.95,.55
    rc = risk_coverage(y, p, risk="error")
    assert rc.coverage == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert rc.risk == pytest.approx([0, 0, 1 / 3, 1 / 2])
    assert rc.aurc == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)


def test_risk_coverage_monotone_for_informative_confidence():
    y, p = _sim(20000, seed=5, sd=2.0, center=0.0)
    rc = risk_coverage(y, p, risk="logloss")
    assert rc.risk[1999] < rc.risk[-1]  # abstaining on low-confidence cases lowers risk
    # a random confidence score is flat in expectation
    rng = np.random.default_rng(0)
    rr = risk_coverage(y, p, risk="logloss", confidence=rng.uniform(size=y.size))
    assert abs(rr.risk[1999] - rr.risk[-1]) < 0.03
    assert rc.aurc < rr.aurc
    assert rc.risk[-1] == pytest.approx(rr.risk[-1])  # full coverage identical


def test_risk_coverage_bad_args():
    with pytest.raises(ValueError):
        risk_coverage(np.array([1.0]), np.array([0.5]), risk="nope")
    with pytest.raises(ValueError):
        risk_coverage(np.array([]), np.array([]))


def test_per_label_risk_coverage_shapes():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, (100, 3)).astype(float)
    p = rng.uniform(size=(100, 3))
    mask = rng.uniform(size=(100, 3)) < 0.8
    out = per_label_risk_coverage(y, p, mask, ["a", "b", "c"], risk="brier")
    assert set(out) == {"a", "b", "c"}
    assert len(out["a"].risk) == int(mask[:, 0].sum())


def test_logit_clipping_is_finite():
    assert np.all(np.isfinite(logit(np.array([0.0, 1.0]))))
