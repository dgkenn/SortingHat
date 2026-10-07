import numpy as np
import pytest

from sortinghat.metrics import (
    NESTED_WINDOWS_S,
    first_window_reaching_fraction,
    h3_severity_stratified,
    h5_interaction,
    h6_ratio,
    kendall_ranking,
    pairwise_auroc,
    per_label_delta_ci,
)


def _strata_data(means, n=200, sd=0.2, seed=0):
    rng = np.random.default_rng(seed)
    d = np.concatenate([rng.normal(m, sd, n) for m in means])
    strata = np.repeat([f"S{i}" for i in range(len(means))], n)
    sites = np.tile(np.array(["A", "B", "C", "D"]), len(means) * n // 4)
    return d, strata, sites


def test_h3_met_when_two_of_three_strata_favorable():
    d, st, si = _strata_data([-0.10, -0.05, 0.06])
    r = h3_severity_stratified(d, st, si, n_boot=300)
    assert r.n_favorable == 2 and r.met
    assert r.strata["S2"]["favorable"] is False
    assert r.strata["S0"]["hi"] < 0  # CI reported


def test_h3_not_met_when_only_one_stratum_favorable():
    d, st, si = _strata_data([-0.10, 0.05, 0.06])
    r = h3_severity_stratified(d, st, si, n_boot=300)
    assert r.n_favorable == 1 and not r.met


def test_h3_small_stratum_not_estimable_counts_against():
    d, st, si = _strata_data([-0.1, -0.1, -0.1], n=200)
    keep = ~((st == "S2") & (np.arange(len(d)) % 200 >= 20))  # S2 left with 20 patients
    r = h3_severity_stratified(d[keep], st[keep], si[keep], n_boot=200, min_stratum_n=50)
    assert r.strata["S2"]["estimable"] is False and np.isnan(r.strata["S2"]["delta"])
    assert r.n_favorable == 2 and r.met


def test_h5_interaction_detects_larger_early_gain():
    rng = np.random.default_rng(0)
    n = 1500
    early = rng.uniform(size=n) < 0.3
    sites = np.tile(["A", "B", "C"], n // 3)
    d = rng.normal(-0.02, 0.2, n) + np.where(early, -0.10, 0.0)
    r = h5_interaction(d, early, sites, n_boot=800, seed=1)
    assert r.interaction.estimate == pytest.approx(-0.10, abs=0.04)
    assert r.met and r.interaction.hi < 0
    assert r.delta_early < r.delta_late and r.n_early + r.n_late == n


def test_h5_null_is_not_met():
    rng = np.random.default_rng(1)
    n = 1500
    early = rng.uniform(size=n) < 0.3
    sites = np.tile(["A", "B", "C"], n // 3)
    d = rng.normal(-0.05, 0.2, n)
    r = h5_interaction(d, early, sites, n_boot=800, seed=1)
    assert not r.met
    assert r.interaction.lo < 0 < r.interaction.hi


def test_h5_covariate_adjustment_removes_confounding():
    rng = np.random.default_rng(2)
    n = 6000
    sev = rng.normal(size=n)
    early = rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * sev))  # sicker patients scanned later... or earlier
    d = -0.10 * early + 0.15 * sev + rng.normal(0, 0.1, n)
    sites = np.tile(["A", "B", "C"], n // 3)
    raw = h5_interaction(d, early, sites, n_boot=200, seed=0)
    adj = h5_interaction(d, early, sites, covariates=sev, n_boot=200, seed=0)
    assert abs(raw.interaction.estimate - (-0.10)) > 0.08  # confounded
    assert adj.interaction.estimate == pytest.approx(-0.10, abs=0.02)
    assert adj.adjusted and adj.met


def test_h5_empty_subgroup_replicates_are_nan_not_crash():
    d = np.array([-0.1, -0.2, 0.0, 0.1])
    early = np.array([True, False, False, False])
    r = h5_interaction(d, early, np.array(["A", "A", "B", "B"]), n_boot=200)
    assert r.interaction.n_valid < 200


def test_h6_ratio_known_value_and_threshold():
    rng = np.random.default_rng(0)
    n = 3000
    d10 = rng.normal(-0.10, 0.12, n)
    sites = np.tile(["A", "B", "C"], n // 3)
    d2_hi = 0.80 * d10 + rng.normal(0, 0.03, n)
    d2_lo = 0.40 * d10 + rng.normal(0, 0.03, n)
    hi = h6_ratio(d2_hi, d10, sites, n_boot=600)
    lo = h6_ratio(d2_lo, d10, sites, n_boot=600)
    assert hi.ratio == pytest.approx(0.8, abs=0.05) and hi.met
    assert hi.lo < hi.ratio < hi.hi
    assert lo.ratio == pytest.approx(0.4, abs=0.05) and not lo.met


def test_h6_uninterpretable_when_full_window_has_no_gain():
    rng = np.random.default_rng(0)
    d10 = rng.normal(0.02, 0.1, 500)
    d2 = rng.normal(0.0, 0.1, 500)
    r = h6_ratio(d2, d10, np.tile(["A", "B"], 250), n_boot=200)
    assert not r.met and r.n_valid == 0 and np.isnan(r.lo)


def test_first_window_reaching_fraction():
    deltas = dict(zip(NESTED_WINDOWS_S, [-0.01, -0.04, -0.075, -0.095, -0.10]))
    assert first_window_reaching_fraction(deltas) == 120
    assert first_window_reaching_fraction(deltas, fraction=0.95) == 300
    assert first_window_reaching_fraction({20: 0.0, 600: 0.01}) is None


def test_per_label_delta_ci_known_labels():
    rng = np.random.default_rng(0)
    n = 3000
    truth = np.column_stack([np.full(n, 0.3), np.full(n, 0.3), np.full(n, 0.3)])
    y = (rng.uniform(size=(n, 3)) < truth).astype(float)
    base = np.full((n, 3), 0.3)
    model = base.copy()
    model[:, 0] = np.where(y[:, 0] == 1, 0.7, 0.15)  # informative on label 0
    model[:, 2] = np.where(y[:, 2] == 1, 0.15, 0.7)  # anti-informative on label 2
    mask = np.ones((n, 3), dtype=bool)
    sites = np.tile(["A", "B", "C"], n // 3)
    out = per_label_delta_ci(y, model, base, mask, ["E1", "E2", "E5"], sites, n_boot=400)
    assert out["E1"]["improved"] and out["E1"]["delta"] < -0.1
    assert not out["E2"]["improved"] and abs(out["E2"]["delta"]) < 1e-9
    assert out["E5"]["delta"] > 0 and not out["E5"]["improved"]
    assert out["E1"]["p_one_sided"] < 0.01 and out["E5"]["p_one_sided"] > 0.9


def test_pairwise_auroc_perfect_and_chance():
    rng = np.random.default_rng(0)
    n = 2000
    which = rng.integers(0, 2, n)  # 0: label a, 1: label b
    y = np.column_stack([(which == 0), (which == 1)]).astype(float)
    mask = np.ones((n, 2), dtype=bool)
    good = np.column_stack([np.where(which == 0, 0.9, 0.1), np.where(which == 1, 0.9, 0.1)])
    assert pairwise_auroc(y, good, mask, 0, 1) == 1.0
    noise = rng.uniform(0.05, 0.95, (n, 2))
    assert pairwise_auroc(y, noise, mask, 0, 1) == pytest.approx(0.5, abs=0.04)


def test_kendall_ranking():
    order = ["E3", "E2", "E1", "E4a", "E5", "E6"]
    perfect = {l: -1.0 + 0.1 * i for i, l in enumerate(order)}  # most identifiable = most negative
    assert kendall_ranking(perfect, order) == pytest.approx(1.0)
    reverse = {l: 1.0 - 0.1 * i for i, l in enumerate(order)}
    assert kendall_ranking(reverse, order) == pytest.approx(-1.0)
    assert np.isnan(kendall_ranking({"E3": -1}, order))
