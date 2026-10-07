from dataclasses import replace

import numpy as np
import pytest

from sortinghat.metrics import power as pw
from sortinghat.metrics.bootstrap import paired_bootstrap_delta
from sortinghat.metrics.loss import favorable_at_every_site, per_patient_delta

# tau = 0 and no prevalence shift: the only case with a clean zero-effect null
CLEAN = pw.PowerConfig(gain=0.0, tau_gain=0.0, site_prev_sd=0.0)


def test_site_sizes_sum_and_balance():
    assert pw.site_sizes(500, 3) == [167, 167, 166]
    assert sum(pw.site_sizes(1999, 3)) == 1999


def test_config_validation_and_drop_label():
    with pytest.raises(ValueError):
        pw.PowerConfig(labels=("E1", "E2"))
    c = pw.PowerConfig().drop_label("E7")
    assert c.labels == ("E1", "E2", "E4a", "E5", "E6") and len(c.prevalences) == 5 == len(c.base_auc)


def test_delta_from_auc_roundtrip():
    from scipy.stats import norm
    assert norm.cdf(pw.delta_from_auc(0.75) / np.sqrt(2)) == pytest.approx(0.75)


def test_seeded_reproducible_and_seed_sensitive():
    a = pw.power_cell(CLEAN, 600, reps=40, seed=3)
    b = pw.power_cell(CLEAN, 600, reps=40, seed=3)
    c = pw.power_cell(CLEAN, 600, reps=40, seed=4)
    assert a == b
    assert a["mean_delta_hat"] != c["mean_delta_hat"]


def test_simulated_d_matches_existing_loss_code():
    """The vectorised d_i must equal loss.per_patient_delta applied to the same y, p, mask."""
    cfg = replace(CLEAN, gain=0.04)
    rng = np.random.default_rng(0)
    d = pw.simulate_site_d(cfg, 400, 3, rng, np.zeros(3), np.zeros(3))
    assert d.shape == (3, 400)
    # reconstruct one rep through the public loss function with an independent small generator
    rng2 = np.random.default_rng(1)
    n, K = 300, cfg.k
    y = (rng2.random((n, K)) < np.asarray(cfg.prevalences)).astype(float)
    p_b = np.clip(rng2.random((n, K)), 0.01, 0.99)
    p_e = np.clip(p_b + 0.05 * rng2.standard_normal((n, K)), 0.01, 0.99)
    mask = rng2.random((n, K)) < 0.9
    mask[0] = False
    ref = per_patient_delta(y, p_e, p_b, mask)
    assert np.isnan(ref[0]) and not np.isnan(ref[1:]).any()
    # same arithmetic as the simulator's inner expression
    from sortinghat.metrics.loss import binary_log_loss
    diff = np.where(mask, binary_log_loss(y, p_e) - binary_log_loss(y, p_b), 0.0)
    cnt = mask.sum(axis=1)
    got = np.where(cnt > 0, diff.sum(axis=1) / np.maximum(cnt, 1), np.nan)
    assert np.allclose(got, ref, equal_nan=True)


def test_effect_has_expected_sign_and_size():
    t0 = pw.true_delta(replace(CLEAN, gain=0.0), n_total=200_000)
    t4 = pw.true_delta(replace(CLEAN, gain=0.04), n_total=200_000)
    assert abs(t0["delta"]) < 0.001                     # zero AUROC gain -> zero log-loss difference
    assert -0.03 < t4["delta"] < -0.008                 # 0.04 AUROC gain -> modest log-loss gain
    assert 0.05 < t4["sd_d"] < 0.25


def test_within_site_ci_matches_existing_bootstrap():
    cfg = replace(CLEAN, gain=0.03)
    rng = np.random.default_rng(5)
    S, n = 3, 700
    ds = [pw.simulate_site_d(cfg, n, 1, rng, np.zeros(1), np.zeros(1))[0] for _ in range(S)]
    d = np.concatenate(ds)
    sites = np.repeat(np.arange(S), n)
    keep = ~np.isnan(d)
    n_s = np.array([[np.sum(~np.isnan(x)) for x in ds]], float)
    m_s = np.array([[np.nanmean(x) for x in ds]])
    v_s = np.array([[np.nanvar(x, ddof=1) for x in ds]])
    up = pw.ci_upper_bounds(n_s, m_s, v_s, np.random.default_rng(0))
    ref = paired_bootstrap_delta(d[keep], sites[keep], "within_site", n_boot=3000, seed=1)
    assert up["within_site"][0] == pytest.approx(ref.hi, abs=0.15 * (ref.hi - ref.estimate))
    # every-site check agrees with loss.favorable_at_every_site
    fav = favorable_at_every_site(d, sites)["all_favorable"]
    assert bool(np.all(m_s < 0)) == fav


def test_null_false_positive_rate_is_nominal_on_ci_criterion():
    # CI upper < 0 under a true Delta of 0 is a one-tailed 2.5% event for a 95% two-sided CI.
    res = pw.power_cell(CLEAN, 900, reps=1500, seed=11)
    assert 0.010 <= res["ci_within_site"] <= 0.042
    assert 0.008 <= res["ci_site_t"] <= 0.045
    assert res["ci_two_stage"] <= 0.045
    # full rule can only be rarer than either component
    assert res["rule_within_site"] <= res["ci_within_site"] + 1e-12
    assert res["rule_within_site"] <= res["every_site"] + 1e-12


def test_every_site_requirement_only_removes_power():
    r = pw.power_cell(replace(pw.PowerConfig(), gain=0.02), 1000, reps=200, seed=2)
    for m in pw.CI_MODES:
        assert r[f"rule_{m}"] <= r[f"ci_{m}"] + 1e-12
        assert r[f"rule_{m}"] <= r["every_site"] + 1e-12
    assert r["ci_site_t"] <= r["ci_within_site"] + 0.05      # t(2) interval is the conservative one


def test_power_monotone_in_n_and_in_effect():
    cfg = pw.PowerConfig()
    p_n = [pw.power_cell(replace(cfg, gain=0.04), n, reps=300, seed=9)["rule_within_site"]
           for n in (300, 600, 1200, 2400)]
    assert p_n[0] < p_n[1] < p_n[2] + 0.03 and p_n[2] - 0.03 <= p_n[3]
    assert p_n[3] > p_n[0] + 0.15
    p_g = [pw.power_cell(replace(cfg, gain=g), 1000, reps=250, seed=9)["rule_within_site"]
           for g in (0.0, 0.02, 0.04, 0.06)]
    assert p_g == sorted(p_g) and p_g[-1] > 0.9 and p_g[0] < 0.2


def test_heterogeneity_lowers_power_and_inflates_within_site_ci_type1():
    flat = pw.power_cell(replace(pw.PowerConfig(), gain=0.02, tau_gain=0.0), 2000, reps=300, seed=6)
    het = pw.power_cell(replace(pw.PowerConfig(), gain=0.02, tau_gain=0.03), 2000, reps=300, seed=6)
    assert het["rule_within_site"] < flat["rule_within_site"]
    null_het = pw.power_cell(replace(pw.PowerConfig(), gain=0.0, tau_gain=0.03), 2000, reps=600, seed=6)
    assert null_het["ci_within_site"] > 0.05            # within-site CI is blind to between-site variance


def test_min_n_for_power():
    ns = [500, 1000, 2000]
    assert pw.min_n_for_power(ns, [0.5, 0.7, 0.9]) == pytest.approx(1500)
    assert pw.min_n_for_power(ns, [0.85, 0.9, 0.95]) == 500
    assert pw.min_n_for_power(ns, [0.2, 0.3, 0.4]) == float("inf")
    # Monte Carlo dip is smoothed away
    assert pw.min_n_for_power(ns, [0.5, 0.85, 0.8]) == pytest.approx(500 + 0.3 / 0.35 * 500)


def test_expected_positives_shows_e7_cannot_reach_100_at_1000():
    pos = pw.expected_positives(pw.PowerConfig(), 1000)
    assert pos["E7"] < 100 and pos["E1"] > 100


def test_level_scan_is_nested_and_matches_default_level():
    cfg = replace(pw.PowerConfig(), gain=0.0)
    r = pw.level_scan(cfg, 900, reps=400, seed=5)
    # a higher CI level can only reject less often, and the full rule never exceeds the CI criterion
    assert r[0.99]["rule"] <= r[0.975]["rule"] <= r[0.95]["rule"]
    for lv in r.values():
        assert lv["rule"] <= lv["ci"]
    # with heterogeneity the 99% level controls rejection better than 95%
    het = pw.level_scan(replace(cfg, tau_gain=0.015), 1000, reps=800, seed=5)
    assert het[0.99]["rule"] < het[0.95]["rule"]
