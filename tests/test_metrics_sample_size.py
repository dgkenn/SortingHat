import numpy as np
import pytest

from sortinghat.metrics import sample_size as ss
from sortinghat.metrics.calibration import fit_logistic


def test_oe_closed_form():
    # SE(ln O/E) = sqrt((1-phi)/(n phi)); width 0.2 -> SE 0.0510
    n = ss.n_for_oe(0.10, 0.2)
    assert n == pytest.approx(0.9 / (0.1 * (0.2 / (2 * 1.959964)) ** 2), rel=1e-5)
    assert ss.oe_width_at_n(0.10, n) == pytest.approx(0.2, rel=1e-6)


def test_matches_riley_published_covid_example_within_tolerance():
    # Riley et al. BMJ 2024 (PMID 38253388): prevalence 0.43, C 0.77, beta-distributed LP.
    # Published: O/E (width .22) 423; slope (.3) 949; slope (.2) 2137; C (.1) 347.
    # We use the binormal last-resort LP, so require agreement within 10%.
    assert ss.n_for_oe(0.43, 0.22) == pytest.approx(423, rel=0.03)
    assert ss.n_for_slope(0.43, 0.77, 0.3) == pytest.approx(949, rel=0.10)
    assert ss.n_for_slope(0.43, 0.77, 0.2) == pytest.approx(2137, rel=0.10)
    assert ss.n_for_cstat(0.43, 0.77, 0.1) == pytest.approx(347, rel=0.10)


def test_monotonicity():
    assert ss.n_for_oe(0.05, 0.2) > ss.n_for_oe(0.10, 0.2) > ss.n_for_oe(0.20, 0.2)
    for phi in ss.PREVALENCES:
        assert ss.n_for_slope(phi, 0.70, 0.2) > ss.n_for_slope(phi, 0.75, 0.2) > ss.n_for_slope(phi, 0.80, 0.2)
    for c in ss.C_STATS:
        assert ss.n_for_slope(0.05, c, 0.2) > ss.n_for_slope(0.20, c, 0.2)
    # halving the target width quadruples N
    assert ss.n_for_slope(0.1, 0.75, 0.1) == pytest.approx(4 * ss.n_for_slope(0.1, 0.75, 0.2))


def test_binormal_lp_is_calibrated_and_has_target_c():
    phi, c = 0.1, 0.75
    mu0, mu1, s = ss.binormal_lp_params(phi, c)
    # slope 1: log LR = (mu1-mu0)/s^2 * (x - mid) = x - logit(phi)  ->  (mu1 - mu0) = s^2
    assert mu1 - mu0 == pytest.approx(s**2)
    from scipy.stats import norm

    assert norm.cdf((mu1 - mu0) / (np.sqrt(2) * s)) == pytest.approx(c)
    # mean predicted risk equals prevalence
    rng = np.random.default_rng(0)
    y = rng.uniform(size=400000) < phi
    lp = np.where(y, rng.normal(mu1, s, y.size), rng.normal(mu0, s, y.size))
    assert (1 / (1 + np.exp(-lp))).mean() == pytest.approx(phi, rel=0.03)


def test_predicted_slope_se_matches_monte_carlo():
    phi, c, n = 0.2, 0.75, 1500
    mu0, mu1, s = ss.binormal_lp_params(phi, c)
    rng = np.random.default_rng(0)
    slopes = []
    for _ in range(300):
        y = rng.uniform(size=n) < phi
        lp = np.where(y, rng.normal(mu1, s, n), rng.normal(mu0, s, n))
        beta, _, ok = fit_logistic(np.column_stack([np.ones(n), lp]), y.astype(float))
        assert ok
        slopes.append(beta[1])
    predicted = np.sqrt(ss.slope_var_per_n(phi, c) / n)
    assert np.mean(slopes) == pytest.approx(1.0, abs=0.03)
    assert np.std(slopes, ddof=1) == pytest.approx(predicted, rel=0.12)


def test_cstat_inverse_consistent():
    n = ss.n_for_cstat(0.2, 0.75, 0.1)
    assert 2 * 1.959964 * ss.c_stat_se(0.75, n, 0.2) == pytest.approx(0.1, rel=1e-3)


def test_row_binding_and_tables_render():
    rows = ss.build_table()
    assert len(rows) == 9
    r = next(r for r in rows if r.prevalence == 0.10 and r.c_stat == 0.75)
    assert r.binding == "slope" and r.n_required == r.n_slope
    txt = ss.render()
    assert "Table S1" in txt and "Table S3" in txt and "5%" in txt


def test_evaluation_set_of_1000_does_not_reach_slope_width_02():
    # Headline: at N = 1000 assessable patients no prevalence/C combination in the grid reaches width 0.2.
    for phi in ss.PREVALENCES:
        for c in ss.C_STATS:
            assert ss.slope_width_at_n(phi, c, 1000) > 0.2


def test_delta_precision_helpers():
    assert ss.delta_halfwidth_at_n(0.2, 1000) == pytest.approx(1.959964 * 0.2 / np.sqrt(1000))
    assert ss.n_for_delta_halfwidth(0.2, 0.0124) == pytest.approx(1000, rel=0.02)


def test_cli_prints_tables(capsys):
    assert ss.main([]) == 0
    assert "Table S1" in capsys.readouterr().out


def test_power_rule_properties():
    # large true effect -> power near 1; zero effect -> power well below alpha-ish bound
    assert ss.power_h1_rule(-0.10, 0.2, 333, n_sim=20000) > 0.99
    assert ss.power_h1_rule(0.0, 0.2, 333, n_sim=20000) < 0.03  # <= one-sided 2.5%
    # closed form for the per-site condition
    assert ss.p_all_sites_favorable(0.0, 0.2, 333) == pytest.approx(0.125)
    # between-site heterogeneity lowers power for a modest effect
    assert ss.power_h1_rule(-0.02, 0.2, 333, tau=0.02, n_sim=40000) < ss.power_h1_rule(-0.02, 0.2, 333, n_sim=40000)
    # power increases with n
    assert ss.power_h1_rule(-0.02, 0.2, 600, n_sim=40000) > ss.power_h1_rule(-0.02, 0.2, 333, n_sim=40000)
