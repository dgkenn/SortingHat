import numpy as np
import pytest

from sortinghat.metrics import (
    DEFAULT_EPS,
    delta_log_loss,
    favorable_at_every_site,
    mean_masked_log_loss,
    per_label_delta,
    per_patient_delta,
    per_patient_loss,
    per_site_delta,
    site_weighted_delta,
)
from sortinghat.metrics.labels import PRIMARY_LABELS, e7_eligible, primary_label_indices


def test_hand_computed_masked_loss_is_patient_then_label_mean():
    # patient 0: three assessable labels; patient 1: only label 2 assessable.
    y = np.array([[1, 0, 1], [0, 0, 1]])
    p = np.array([[0.8, 0.1, 0.5], [0.3, 0.9, 0.25]])
    mask = np.array([[True, True, True], [False, False, True]])
    l0 = np.mean([-np.log(0.8), -np.log(0.9), -np.log(0.5)])
    l1 = -np.log(0.25)
    pl = per_patient_loss(y, p, mask)
    assert pl == pytest.approx([l0, l1])
    assert mean_masked_log_loss(y, p, mask) == pytest.approx((l0 + l1) / 2)
    # pooled-over-cells mean would differ: guards against the wrong aggregation
    pooled = (3 * l0 + l1) / 4
    assert abs(mean_masked_log_loss(y, p, mask) - pooled) > 0.05


def test_masked_cells_cannot_influence_loss():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (50, 4)).astype(float)
    p = rng.uniform(0.05, 0.95, (50, 4))
    mask = rng.uniform(size=(50, 4)) < 0.7
    base = mean_masked_log_loss(y, p, mask)
    y2, p2 = y.copy(), p.copy()
    y2[~mask] = np.nan
    p2[~mask] = np.nan
    assert mean_masked_log_loss(y2, p2, mask) == pytest.approx(base)


def test_patient_with_no_assessable_label_is_excluded_not_zero():
    y = np.array([[1, 0], [1, 1]])
    p = np.array([[0.5, 0.5], [0.9, 0.9]])
    mask = np.array([[True, True], [False, False]])
    assert np.isnan(per_patient_loss(y, p, mask)[1])
    assert mean_masked_log_loss(y, p, mask) == pytest.approx(-np.log(0.5))


def test_clipping_bounds_the_loss():
    y = np.array([[1.0, 0.0]])
    p = np.array([[0.0, 1.0]])  # maximally wrong
    mask = np.ones((1, 2), dtype=bool)
    assert mean_masked_log_loss(y, p, mask) == pytest.approx(-np.log(DEFAULT_EPS))
    assert mean_masked_log_loss(y, p, mask, eps=1e-3) == pytest.approx(-np.log(1e-3))


def test_delta_sign_antisymmetry_and_zero():
    rng = np.random.default_rng(1)
    n, K = 400, 5
    truth = rng.uniform(0.1, 0.5, (n, K))
    y = (rng.uniform(size=(n, K)) < truth).astype(float)
    mask = rng.uniform(size=(n, K)) < 0.85
    good = truth
    base = np.full((n, K), y.mean())
    d_gb = delta_log_loss(y, good, base, mask)
    d_bg = delta_log_loss(y, base, good, mask)
    assert d_gb < 0  # informative predictions beat the prevalence baseline: EEG helps = negative
    assert d_gb == pytest.approx(-d_bg)
    assert delta_log_loss(y, good, good, mask) == pytest.approx(0.0)


def test_delta_equals_difference_of_losses_on_common_patients():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, (100, 3)).astype(float)
    pm = rng.uniform(0.1, 0.9, (100, 3))
    pb = rng.uniform(0.1, 0.9, (100, 3))
    mask = rng.uniform(size=(100, 3)) < 0.6
    mask[:5] = False  # five patients with nothing assessable
    assert delta_log_loss(y, pm, pb, mask) == pytest.approx(
        mean_masked_log_loss(y, pm, mask) - mean_masked_log_loss(y, pb, mask)
    )


def test_validation_errors():
    y = np.array([[1, 0]])
    p = np.array([[1.2, 0.5]])
    with pytest.raises(ValueError):
        mean_masked_log_loss(y, p, np.ones((1, 2), dtype=bool))
    with pytest.raises(ValueError):
        mean_masked_log_loss(np.array([[2, 0]]), np.array([[0.5, 0.5]]), np.ones((1, 2), dtype=bool))
    with pytest.raises(ValueError):
        mean_masked_log_loss(y, np.array([[0.5]]), np.ones((1, 2), dtype=bool))


def test_per_label_delta_matches_manual():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    pm = np.array([[0.9, 0.2], [0.2, 0.8], [0.7, 0.6]])
    pb = np.full((3, 2), 0.5)
    mask = np.array([[True, True], [True, False], [True, True]])
    got = per_label_delta(y, pm, pb, mask)
    l0 = np.mean([-np.log(0.9), -np.log(0.8), -np.log(0.7)]) - (-np.log(0.5))
    l1 = np.mean([-np.log(0.8), -np.log(0.6)]) - (-np.log(0.5))
    assert got == pytest.approx([l0, l1])


def test_per_site_and_favorable_at_every_site():
    d = np.array([-0.2, -0.1, -0.05, -0.15, 0.1, 0.05])
    sites = np.array(["A", "A", "B", "B", "C", "C"])
    ps = per_site_delta(d, sites)
    assert ps["A"]["delta"] == pytest.approx(-0.15)
    assert ps["C"]["delta"] == pytest.approx(0.075)
    r = favorable_at_every_site(d, sites)
    assert not r["all_favorable"] and r["unfavorable"] == ["C"]
    d2 = d.copy()
    d2[4:] = [-0.01, -0.02]
    assert favorable_at_every_site(d2, sites)["all_favorable"]
    # overall pooled Delta can be negative while a site is unfavorable: why the check exists
    assert np.mean(d) < 0
    # exactly 0 is not favorable
    assert not favorable_at_every_site(np.array([0.0, 0.0]), np.array(["A", "A"]))["all_favorable"]


def test_site_weighted_vs_patient_weighted():
    d = np.array([-0.3] * 10 + [0.1] * 90)
    sites = np.array(["A"] * 10 + ["B"] * 90)
    assert d.mean() == pytest.approx(-0.03 + 0.09)
    assert site_weighted_delta(d, sites) == pytest.approx(-0.1)


def test_primary_label_set_excludes_e3_and_e4b():
    names = ["E1", "E2", "E3", "E4a", "E4b", "E5", "E6", "E7"]
    idx = primary_label_indices(names)
    assert [names[i] for i in idx] == list(PRIMARY_LABELS)
    assert "E3" not in PRIMARY_LABELS and "E4b" not in PRIMARY_LABELS
    assert names[primary_label_indices(names, include_e7=True)[-1]] == "E7"


def test_e7_eligibility_rule():
    assert e7_eligible({"A": 60, "B": 45})
    assert not e7_eligible({"A": 120, "B": 0})  # one site only
    assert not e7_eligible({"A": 50, "B": 40})  # < 100 total
