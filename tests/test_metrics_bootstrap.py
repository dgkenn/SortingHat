import numpy as np
import pytest

from sortinghat.metrics import (
    delta_ci_all_modes,
    delta_log_loss,
    holm_adjust,
    one_sided_p_below_zero,
    paired_bootstrap_delta,
    per_patient_delta,
    site_mean_t_interval,
)


def _homog(n_per=400, mean=-0.05, sd=0.2, sites=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    d = np.concatenate([rng.normal(mean, sd, n_per) for _ in sites])
    s = np.repeat(list(sites), n_per)
    return d, s


def test_within_site_ci_matches_normal_theory():
    d, s = _homog()
    r = paired_bootstrap_delta(d, s, "within_site", n_boot=3000, seed=1)
    se = d.std(ddof=1) / np.sqrt(d.size)
    assert r.estimate == pytest.approx(d.mean())
    assert (r.hi - r.lo) == pytest.approx(2 * 1.96 * se, rel=0.1)
    assert r.lo < r.estimate < r.hi


def test_deterministic_given_seed():
    d, s = _homog(n_per=100)
    a = paired_bootstrap_delta(d, s, "two_stage", n_boot=300, seed=7)
    b = paired_bootstrap_delta(d, s, "two_stage", n_boot=300, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_ci_excludes_zero_for_true_effect_and_covers_for_null():
    d, s = _homog(mean=-0.05, sd=0.2, n_per=400)
    assert paired_bootstrap_delta(d, s, n_boot=1500, seed=2).excludes_zero_below
    d0, s0 = _homog(mean=0.0, sd=0.2, n_per=400, seed=5)
    r0 = paired_bootstrap_delta(d0, s0, n_boot=1500, seed=2)
    assert r0.lo < 0 < r0.hi


def test_cluster_and_two_stage_are_wider_when_sites_differ():
    rng = np.random.default_rng(3)
    means = {"A": -0.15, "B": -0.02, "C": 0.08, "D": -0.10}
    d = np.concatenate([rng.normal(m, 0.1, 300) for m in means.values()])
    s = np.repeat(list(means), 300)
    res = delta_ci_all_modes(d, s, n_boot=1500, seed=4)
    w = {m: res[m].hi - res[m].lo for m in ("within_site", "cluster", "two_stage")}
    assert w["cluster"] > 3 * w["within_site"]
    assert w["two_stage"] >= w["cluster"] * 0.9
    assert w["two_stage"] > w["within_site"]
    # the within-site CI excludes 0 even though one site is unfavorable
    assert res["within_site"].hi < 0
    assert np.isfinite(res["site_t"]["lo"]) and res["site_t"]["df"] == 3


def test_cluster_mode_degenerate_with_three_sites_is_documented():
    # With S=3, 3/27 of cluster draws use a single site: replicates must take few distinct values.
    d, s = _homog(n_per=50, sd=0.2)
    from sortinghat.metrics import bootstrap_replicates

    reps = bootstrap_replicates(lambda idx: d[idx].mean(), s, "cluster", n_boot=2000, seed=0)
    assert len(np.unique(np.round(reps, 12))) <= 10


def test_bad_mode_and_single_site():
    d, s = _homog(n_per=20)
    with pytest.raises(ValueError):
        paired_bootstrap_delta(d, s, "nope")
    with pytest.raises(ValueError):
        paired_bootstrap_delta(np.ones(5), np.array(["A"] * 5), "cluster")


def test_nan_patients_dropped():
    d, s = _homog(n_per=50)
    d[:10] = np.nan
    r = paired_bootstrap_delta(d, s, n_boot=200, seed=0)
    assert r.estimate == pytest.approx(np.nanmean(d))


def test_pairing_is_preserved_end_to_end():
    # Model and baseline share a strong patient-level component; a paired CI on Delta must be much
    # narrower than an unpaired comparison would give.
    rng = np.random.default_rng(11)
    n, K = 900, 4
    risk = rng.uniform(0.05, 0.6, (n, 1)) * np.ones((1, K))
    y = (rng.uniform(size=(n, K)) < risk).astype(float)
    mask = np.ones((n, K), dtype=bool)
    base = np.clip(risk + rng.normal(0, 0.05, (n, K)), 0.02, 0.98)
    model = np.clip(risk + rng.normal(0, 0.03, (n, K)), 0.02, 0.98)
    sites = np.repeat(["A", "B", "C"], n // 3)
    d = per_patient_delta(y, model, base, mask)
    r = paired_bootstrap_delta(d, sites, n_boot=800, seed=0)
    from sortinghat.metrics import per_patient_loss

    la, lb = per_patient_loss(y, model, mask), per_patient_loss(y, base, mask)
    unpaired_se = np.sqrt(la.var(ddof=1) / n + lb.var(ddof=1) / n)
    assert r.se < 0.5 * unpaired_se
    assert r.estimate == pytest.approx(delta_log_loss(y, model, base, mask))


def test_site_mean_t_interval_known():
    d = np.array([-0.3, -0.3, 0.0, 0.0, -0.6, -0.6])
    s = np.array(["A", "A", "B", "B", "C", "C"])
    r = site_mean_t_interval(d, s)
    assert r["estimate"] == pytest.approx(-0.3)
    se = np.std([-0.3, 0.0, -0.6], ddof=1) / np.sqrt(3)
    assert r["hi"] - r["lo"] == pytest.approx(2 * 4.302652729911275 * se)


def test_one_sided_p_and_holm():
    reps = np.concatenate([np.full(990, -1.0), np.full(10, 1.0)])
    assert one_sided_p_below_zero(reps) == pytest.approx(11 / 1001)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm_adjust([0.5, np.nan, 0.001])
    assert np.isnan(out[1]) and out[2] == pytest.approx(0.002)
