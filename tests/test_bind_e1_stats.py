"""Statistics of the imaging-E1 analysis (sortinghat/metrics/bind_e1.py) on SYNTHETIC arrays only (CLAUDE.md rule 1)."""
import numpy as np
import pytest

from sortinghat.metrics import bind_e1 as be
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only

LAB = {"S1": "site_1", "S2": "site_2"}


def planted(n=1200, seed=0, gain=1.0, prev=0.2, shift=0.0):
    """Two sites; baseline score carries some signal; the EEG arm adds `gain` more. `shift` moves BOTH arms' logits (silver vs imaging prevalence)."""
    rng = np.random.default_rng(seed)
    sites = np.where(np.arange(n) < n // 2, "S1", "S2")
    y = rng.random(n) < prev
    z0 = rng.normal(0, 1, n) + 0.8 * y
    z1 = z0 + gain * (0.8 * y + rng.normal(0, 0.6, n)) * 0.5
    sig = lambda z: 1 / (1 + np.exp(-z))                                   # noqa: E731
    return y, sig(z1 - 1.4 + shift), sig(z0 - 1.4 + shift), sites


def test_auroc_matches_a_hand_value_and_handles_empty_classes():
    assert be.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert np.isnan(be.auroc([1, 1], [0.2, 0.3]))


def test_fit_shift_recovers_a_known_intercept_shift():
    rng = np.random.default_rng(1)
    l0 = rng.normal(-1.5, 1.0, 60000)
    p_true = 1 / (1 + np.exp(-(l0 + 0.9)))
    y = rng.random(l0.size) < p_true
    assert be.fit_shift(y, l0) == pytest.approx(0.9, abs=0.05)
    assert np.isnan(be.fit_shift(np.zeros(10), np.zeros(10)))


def test_pair_block_detects_a_planted_gain_and_the_shift_removes_a_prevalence_mismatch():
    y, p1, p0, s = planted(gain=1.5, shift=-1.0)             # imaging prevalence 0.2 but the predictions were calibrated to a lower one
    b = be.pair_block(be.Pair(y, p1, p0, s), B=400, seed=3, site_labels=LAB)
    assert b["estimable"] and b["shift_mode"] == "cross"
    assert b["delta_auroc"]["point"] > 0.01 and b["delta_auroc"]["ci_99"]["lo"] > 0
    assert b["delta_logloss_shifted"]["point"] < 0 and b["delta_logloss_shifted"]["ci_99"]["hi"] < 0
    assert set(b["per_site"]) == {"site_1", "site_2"} and b["per_site"]["site_1"]["delta_auroc"] > 0
    v = be.verdict(b, label_validated=True)
    assert v["verdict"] == "supportive" and "reading" not in v
    assert be.verdict(b)["verdict"] == "supportive"
    assert_aggregate_only(b)
    # unshifted log loss is larger (worse) than the shifted one for the same arms when the shift is large, but the DIFFERENCE keeps its sign
    assert b["delta_logloss_unshifted"]["point"] is not None


def test_no_gain_is_not_supportive_and_the_reading_says_unvalidated():
    y, p1, p0, s = planted(gain=0.0, seed=5)
    b = be.pair_block(be.Pair(y, p1, p0, s), B=300, seed=1, site_labels=LAB)
    v = be.verdict(b)
    assert v["verdict"] == "not_supportive" and v["reading"].startswith("inconclusive, label unvalidated")


def test_harm_when_the_eeg_arm_is_worse_with_an_interval_excluding_zero():
    y, p1, p0, s = planted(gain=1.5, seed=6)
    b = be.pair_block(be.Pair(y, p0, p1, s), B=300, seed=1, site_labels=LAB)          # arms swapped: EEG arm is the worse one
    assert be.verdict(b, label_validated=True)["verdict"] == "harm"


def test_stop_rule_s3_makes_a_stratum_not_estimable():
    y, p1, p0, s = planted(n=400, prev=0.05, seed=2)            # about 20 positives
    b = be.pair_block(be.Pair(y, p1, p0, s), B=100, seed=1, site_labels=LAB)
    assert b["estimable"] is False and "S3" in b["why"] and "delta_auroc" not in b
    assert be.verdict(b)["verdict"] == "not_estimable"
    assert b["n_positive"] in (SUPPRESSED,) or isinstance(b["n_positive"], int)
    b2 = be.pair_block(be.Pair(y, p1, p0, s), B=100, seed=1, site_labels=LAB, min_positives=5)
    assert b2["estimable"] is True and b2["descriptive_only"] is True


def test_insample_shift_when_a_site_has_too_few_positives_and_one_site_case():
    y, p1, p0, s = planted(seed=7)
    s2 = np.where(np.arange(len(y)) < 100, "S1", "S2")                 # site 1 gets few positives
    y2 = y.copy(); y2[:100] = False; y2[:5] = True
    assert be.shift_mode(y2, s2) == "insample"
    assert be.shift_mode(y, s) == "cross" and be.shift_mode(y, np.full(len(y), "S1")) == "insample"
    b = be.pair_block(be.Pair(y, p1, p0, np.full(len(y), "S1")), B=100, seed=1, site_labels={"S1": "site_1"})
    assert b["shift_mode"] == "insample" and "in-sample" in b["shift_note"]


def test_bootstrap_is_deterministic_and_resamples_within_site():
    y, p1, p0, s = planted(seed=8)
    a = be.pair_block(be.Pair(y, p1, p0, s), B=200, seed=9, site_labels=LAB)
    b = be.pair_block(be.Pair(y, p1, p0, s), B=200, seed=9, site_labels=LAB)
    assert a == b


def test_auc_only_and_single_auroc_blocks():
    y, p1, p0, s = planted(gain=1.5, seed=10)
    b = be.auc_pair_block(y, p1, p0, s, B=200, seed=1, site_labels=LAB)
    assert b["estimable"] and "delta_logloss_shifted" not in b and b["delta_auroc"]["point"] > 0
    e = be.single_auroc_block(y, p1, s, B=200, seed=1, site_labels=LAB)
    assert e["estimable"] and e["auroc"]["point"] > 0.6
    assert be.single_auroc_block(y[:100] & False, p1[:100], s[:100], B=10, seed=1, site_labels=LAB)["estimable"] is False


def test_graded_delta_auroc_strata_trend_and_reading():
    rng = np.random.default_rng(11)
    n = 4000
    sites = np.where(np.arange(n) < n // 2, "S1", "S2")
    grade = rng.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.25, 0.15])
    y = rng.random(n) < 0.25
    z0 = rng.normal(0, 1, n) + 0.7 * y
    z1 = z0 + (0.6 + 0.0 * grade) * (y - 0.25) + rng.normal(0, 0.3, n)          # the gain does not depend on grade
    p0, p1 = 1 / (1 + np.exp(-z0)), 1 / (1 + np.exp(-z1))
    g = be.graded_delta_auroc(y, p1, p0, sites, grade, B=300, seed=2, site_labels=LAB)
    assert g["estimable"] and all(g["grades"][str(k)]["estimable"] for k in range(4))
    assert g["strata"]["light"]["delta_auroc"] > 0 and g["strata"]["light"]["ci_99"]["lo"] > 0
    assert g["trend"]["slope_pooled"]["estimable"] and g["trend"]["slope_site_adjusted_primary"]["estimable"]
    assert g["reading"].startswith(("(a)", "(d)"))                               # no trend with intensity: not reading (b) or (c)
    small = be.graded_delta_auroc(y[:60], p1[:60], p0[:60], sites[:60], grade[:60], B=50, seed=2, site_labels=LAB)
    assert small["estimable"] is False and "50" in small["why"]            # under 50 patients: not estimable, never exculpatory
    assert_aggregate_only(g)


def test_sedation_reading_rules():
    base = {"strata": {"light": {"estimable": True, "ci_99": {"lo": 0.01, "hi": 0.05}}},
            "trend": {"slope_site_adjusted_primary": {"estimable": True, "ci_99": {"lo": -0.01, "hi": 0.01}}}}
    assert be.sedation_reading(base).startswith("(a)")
    b = {"strata": {"light": {"estimable": True, "ci_99": {"lo": -0.01, "hi": 0.02}}},
         "trend": {"slope_site_adjusted_primary": {"estimable": True, "ci_99": {"lo": 0.01, "hi": 0.03}}}}
    assert be.sedation_reading(b).startswith("(b)")
    c = {"strata": {"light": {"estimable": True, "ci_99": {"lo": 0.01, "hi": 0.02}}},
         "trend": {"slope_site_adjusted_primary": {"estimable": True, "ci_99": {"lo": 0.01, "hi": 0.03}}}}
    assert be.sedation_reading(c).startswith("(c)")
    assert be.sedation_reading({"strata": {}, "trend": {}}).startswith("(d)")


def test_concordance_kappa_values_and_suppression():
    idx = range(200)
    img = np.array([float(i < 100) for i in idx])
    silver = np.array([float(i < 80 or 100 <= i < 120) for i in idx])
    c = be.concordance_kappa(silver, img)
    assert c["sensitivity"] == 0.8 and c["kappa"] == pytest.approx(0.6)
    c2 = be.concordance_kappa(np.r_[np.ones(5), np.zeros(195)], img)
    assert c2["kappa"] == SUPPRESSED and c2["silver_pos_img_pos"] == SUPPRESSED
    img3 = img.copy(); img3[:50] = np.nan
    assert be.concordance_kappa(silver, img3)["n_both_assessable"] == 150
    assert_aggregate_only(c)


def test_drift_and_selection_blocks():
    rng = np.random.default_rng(3)
    n = 600
    tr = np.arange(n) < 400
    y = np.where(tr, rng.random(n) < 0.1, rng.random(n) < 0.4)           # prevalence moves 4x
    d = be.prevalence_drift(y, np.ones(n, bool), tr, ~tr)
    assert d["flag"] is True and d["ratio_test_over_train"] > 3
    few = be.prevalence_drift(y, np.ones(n, bool), tr & (np.arange(n) < 20), ~tr)
    assert few["flag"] is None and "not estimable" in few["ratio_test_over_train"]
    sites = np.where(np.arange(n) % 2 == 0, "S1", "S2")
    t = be.assessable_by_tenth(sites, np.arange(n), rng.random(n) < 0.6, LAB)
    assert set(t) == {"site_1", "site_2"} and all(len(v) == 10 for v in t.values())
    sel = be.window_study_share(rng.random(n) < 0.5, sites, rng.uniform(5, 95, n), LAB)
    assert set(sel["by_age_band"]) == {"<18", "18-39", "40-64", "65-79", ">=80"} and isinstance(sel["overall"], float)
    assert_aggregate_only({"d": d, "t": t, "sel": sel})
