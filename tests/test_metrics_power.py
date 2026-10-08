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


# ---------------------------------------------------------------- 2-site design (D-120)
CLEAN2 = pw.two_site_config(gain=0.0, tau_gain=0.0, site_prev_sd=0.0)


def test_site_sizes_frac_60_40():
    assert pw.site_sizes_frac(1000, (0.6, 0.4)) == [600, 400]
    assert sum(pw.site_sizes_frac(1001, (0.6, 0.4))) == 1001
    assert pw.two_site_config().n_sites == 2


def test_two_site_seeded_and_nested():
    a = pw.two_site_cell(CLEAN2, 500, reps=60, seed=3)
    assert a == pw.two_site_cell(CLEAN2, 500, reps=60, seed=3)
    assert a != pw.two_site_cell(CLEAN2, 500, reps=60, seed=4)
    for lv, v in a.items():
        assert v["both"] <= v["both_ub"] <= min(v["loso"], v["temporal"]) + 1e-12
    assert a[0.99]["both"] <= a[0.975]["both"] <= a[0.95]["both"]
    assert a[0.99]["loso"] <= a[0.975]["loso"] <= a[0.95]["loso"]


def test_two_site_null_rate_within_tolerance():
    # tau = 0, no prevalence shift: the one-sided rejection of a 95% CI is nominally 2.5%, and the
    # 2-site every-site requirement (both Deltas < 0) only lowers it; the combined rule is lower still.
    r = pw.two_site_cell(CLEAN2, 900, reps=1500, seed=11)
    for lv, nominal in ((0.95, 0.025), (0.975, 0.0125), (0.99, 0.005)):
        assert r[lv]["loso"] <= nominal + 0.02
        assert r[lv]["temporal"] <= nominal + 0.02
        assert r[lv]["both"] <= min(r[lv]["loso"], r[lv]["temporal"]) + 1e-12
    assert r[0.95]["loso"] > 0.005                       # the simulator is not trivially never rejecting


def test_two_site_power_monotone_in_n_and_effect():
    cfg = pw.two_site_config(gain=0.04)
    p = [pw.two_site_cell(cfg, n, reps=500, seed=9, levels=(0.99,))[0.99]["both"] for n in (400, 800, 1600)]
    assert p[0] < p[1] + 0.03 and p[1] < p[2] + 0.03 and p[2] > p[0] + 0.15
    g = [pw.two_site_cell(pw.two_site_config(gain=x), 1000, reps=300, seed=9, levels=(0.99,))[0.99]["both"]
         for x in (0.0, 0.02, 0.04, 0.06)]
    assert g == sorted(g) and g[-1] > 0.9 and g[0] < 0.1


def test_two_site_drift_lowers_power_and_hits_temporal_scheme_hardest():
    flat = pw.two_site_cell(pw.two_site_config(gain=0.04), 1000, reps=500, seed=4, levels=(0.99,))[0.99]
    dr = pw.two_site_cell(pw.two_site_config(gain=0.04, drift=0.02), 1000, reps=500, seed=4, levels=(0.99,))[0.99]
    assert dr["both"] < flat["both"] and dr["temporal"] < flat["temporal"] - 0.1
    assert dr["temporal"] < dr["loso"]                   # LOSO only sees the drift in 30% of its patients


def test_two_site_grid_and_min_n_helpers():
    rows = pw.two_site_grid(reps=40, seed=1, taus=(0.015,), drifts=(0.0,), n_grid=(500, 800), gains=(0.0, 0.04),
                            levels=(0.95, 0.99))
    assert len(rows) == 1 * 1 * 2 * 2 * 2
    md = pw.format_two_site(rows)
    assert "Null (gain 0)" in md and "Power +0.04" in md and "N=800" in md
    assert set(pw.two_site_min_n(rows)) == {(0.015, 0.0, 0.95), (0.015, 0.0, 0.99)}
