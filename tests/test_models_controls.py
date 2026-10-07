import numpy as np
import pytest

from sortinghat.models import (DEFAULT_RUNGS, LadderConfig, default_exposure_registry, exposure_accounting,
                               leakage_probes, negative_control, permute_eeg_within_strata,
                               permute_labels_within_site, run_ladder, run_mandatory_controls,
                               sedative_excluded_rerun, severity_stratified_delta, stratified_delta)
from sortinghat.models.controls import control_failed, delta_concentration_by_site, exposed_site_delta
from sortinghat.models.synthetic import make_synthetic_study

CFG = LadderConfig(n_boot=200, seed=2, per_label=False)
RUNG = [s for s in DEFAULT_RUNGS if s.name == "combined"]


@pytest.fixture(scope="module")
def planted():
    d = make_synthetic_study(n_per_site=500, signal=1.0, seed=21)
    return d, run_ladder(d, None, RUNG, CFG)


def test_leakage_probes_flag_site_but_not_duration_or_channels():
    d = make_synthetic_study(n_per_site=300, signal=0.0, site_shift=1.5, seed=1)
    p = leakage_probes(d.eeg, d.sites, d.covariates["duration_s"], d.covariates["n_channels"])
    assert p["probes"]["site"]["auroc"] > 0.8 and p["probes"]["site"]["flagged"]
    assert p["probes"]["duration"]["auroc"] < 0.65 and not p["probes"]["duration"]["flagged"]
    assert p["probes"]["channel_count"]["estimable"] is False          # all recordings have 19 channels
    assert p["any_flagged"]


def test_leakage_probes_clean_features_and_channel_signal():
    d = make_synthetic_study(n_per_site=300, signal=0.0, site_shift=0.0, seed=2, channel_variation=True)
    p = leakage_probes(d.eeg, d.sites, d.covariates["duration_s"], d.covariates["n_channels"])
    assert p["probes"]["site"]["auroc"] < 0.62 and not p["any_flagged"]
    # a feature column that encodes channel count is detected
    e = d.eeg.copy()
    e["qeeg.global.leak"] = d.covariates["n_channels"] + np.random.default_rng(0).normal(0, 0.5, d.n)
    p2 = leakage_probes(e, d.sites, d.covariates["duration_s"], d.covariates["n_channels"])
    assert p2["probes"]["channel_count"]["flagged"]


def test_sedative_excluded_rerun_keeps_the_gain(planted):
    d, full = planted
    out = sedative_excluded_rerun(d, rungs=RUNG, cfg=CFG)
    r = out["result"].get("combined")
    assert r.delta < -0.01 and r.all_sites_favorable and r.ci_primary["hi"] < 0
    n_sed = int(d.covariates["sedated"].sum())
    assert out["result"].n_eval < full.n_eval and n_sed > 11


def test_severity_stratified_h3(planted):
    d, res = planted
    h3 = severity_stratified_delta(res, d, "combined", n_boot=100)
    assert h3["met"] and h3["n_favorable"] == 3 and not h3["cut_points_prespecified"]
    h3b = severity_stratified_delta(res, d, "combined", cut_points=[-0.4, 0.4], n_boot=100)
    assert h3b["cut_points_prespecified"] and h3b["met"]
    # strata too small -> not estimable -> not favorable
    h3c = severity_stratified_delta(res, d, "combined", min_stratum_n=10_000, n_boot=50)
    assert not h3c["met"]


def test_stratified_delta_by_duration(planted):
    d, res = planted
    dur = d.covariates["duration_s"][res.eval_idx]
    strata = np.digitize(dur, np.quantile(dur, [1 / 3, 2 / 3])).astype(str)
    s = stratified_delta(res.get("combined").d, strata, res.eval_sites, n_boot=50)
    assert s["n_strata"] == 3 and s["n_favorable"] == 3


def test_negative_control_labels_and_eeg_give_no_gain():
    d = make_synthetic_study(n_per_site=500, signal=1.0, seed=31)
    for kind in ("labels", "eeg"):
        out = negative_control(d, kind, RUNG, CFG, seed=4)
        v = out["rungs"]["all/combined"]
        assert abs(v["delta"]) < 0.03, (kind, v)
        assert not out["any_spurious_gain"]


def test_permutations_preserve_structure():
    d = make_synthetic_study(n_per_site=100, signal=1.0, seed=1)
    p = permute_labels_within_site(d, 0)
    assert (p.gold_role == d.gold_role).all() and (p.sites == d.sites).all()
    assert p.y_gold.sum() == d.y_gold.sum() and not np.array_equal(p.y_gold, d.y_gold)
    for s in np.unique(d.sites):                       # rows only move within a site
        assert p.y_silver[p.sites == s].sum() == d.y_silver[d.sites == s].sum()
    q = permute_eeg_within_strata(d, strata=d.baseline["gcs"] > 0, seed=0)
    assert np.isclose(q.eeg.sum().sum(), d.eeg.sum().sum()) and not q.eeg.equals(d.eeg)


def test_site_concentration_and_failed_control(planted):
    d, res = planted
    r = res.get("combined")
    c = delta_concentration_by_site(r, res.site_labels)
    assert 0.3 < c["top_share"] < 0.6 and not c["carried_by_one_site"]
    p = leakage_probes(d.eeg, d.sites)
    assert not control_failed(p, r, res.site_labels)
    # synthetic: all gain at one site and a flagged site probe -> failed control
    r2 = type(r)(**{**r.__dict__, "per_site": {"a": {"delta": -0.3, "n": 300}, "b": {"delta": -0.01, "n": 300},
                                               "c": {"delta": 0.0, "n": 300}}, "delta": -0.1})
    assert control_failed({"probes": {"site": {"flagged": True}}}, r2)


def test_exposure_accounting_stub(planted):
    _d, res = planted
    reg = default_exposure_registry()
    tab = exposure_accounting(reg, ["SITE_A", "SITE_B"], {"SITE_A": ["TUEG"]}, {"SITE_A": "site_1", "SITE_B": "site_2"})
    assert tab["cbramod"]["sites"] == {"site_1": "exposed", "site_2": "not_exposed"}
    assert set(tab["morgoth"]["sites"].values()) == {"unknown"} and tab["morgoth"]["status"] == "UNVERIFIED"
    row = {"site_1": "exposed", "site_2": "not_exposed", "site_3": "not_exposed"}
    by = exposed_site_delta(res.get("combined"), row, res.site_labels)
    assert set(by) == {"exposed", "not_exposed"}


def test_run_mandatory_controls_bundle_writes_aggregate_json(planted, tmp_path):
    d, res = planted
    c = run_mandatory_controls(d, res, "combined", CFG, seed=3)
    assert set(c) >= {"leakage_probes", "sedative_excluded", "severity_stratified_h3", "negative_control_labels",
                      "negative_control_eeg", "exposure_accounting", "site_probe_control_failed"}
    assert c["sedative_excluded"]["pooled_delta_negative"] and not c["negative_control_labels"]["any_spurious_gain"]
    from sortinghat.models import write_results
    import json
    p = write_results(tmp_path / "r.json", res, controls=c)
    assert "controls" in json.loads(p.read_text())
