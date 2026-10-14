"""D-155 presentation-based intended-use strata and transport weighting, SYNTHETIC ONLY.

Pure-function tests use made-up arrays (strata definitions, independence from diagnosis codes, delta blocks, domain weights);
the end-to-end ones run scripts/run_silver_feasibility.py on the synthetic inputs of test_silver_feasibility_inputs, once without
OMOP proxies (the empty test cache: strata defined without them) and once with proxies planted through ``attach_omop``."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sortinghat.metrics import intended_use as iu
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_silver_feasibility import run_main
from test_silver_feasibility_inputs import make_inputs, rsf

LABELS = {"S1": "site_1", "S2": "site_2"}
NEW_TITLE = "Presentation-based intended-use strata"


# --------------------------------------------------------------------------------------------- definitions
def _sed(on, n=None):
    on = np.asarray(on, bool)
    return {"any_on_t0": on, "any_6h": on.copy()}


def _case(hours, on=False, dur=1800.0, ceeg=False, vent=False, press=False):
    """One row per entry of ``hours`` (all other inputs broadcast)."""
    h = np.atleast_1d(np.asarray(hours, float))
    n = len(h)
    b = lambda v: np.broadcast_to(np.asarray(v, bool), n).copy()          # noqa: E731
    return dict(hours=h, sed=_sed(b(on)), dur=np.broadcast_to(np.asarray(dur, float), n).copy(), ceeg=b(ceeg),
                omop={"vent_24h": b(vent), "vasopressor_24h": b(press)})


def test_calendar_day_from_a_date_only_visit_start():
    d = iu.calendar_day([0.0, 23.9, 24.0, 47.9, 48.0, 100.0, -1.0, np.nan])
    assert d[:6].tolist() == [0, 0, 1, 1, 2, 4] and np.isnan(d[6]) and np.isnan(d[7])


def test_iu_large_and_small_follow_their_definitions():
    #       day0  day1  day2  day1+infusion  day0+vent  day0+pressor  day0+long cEEG  day1+short cEEG  day1 long non-cEEG  day0 cEEG short
    c = _case([5, 30, 50, 30, 5, 5, 5, 30, 30, 5],
              on=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], vent=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], press=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              dur=[1800, 1800, 1800, 1800, 1800, 1800, 20 * 3600, 3 * 3600, 20 * 3600, 3600],
              ceeg=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1])
    r = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], c["ceeg"], c["omop"])
    large, small = r["masks"][iu.IU_LARGE], r["masks"][iu.IU_SMALL]
    assert large.tolist() == [1, 1, 0, 0, 0, 0, 0, 1, 1, 1] and small.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert not (small & ~large).any()                                       # IU-small is a subset of IU-large
    assert r["proxies_in"] and not r["missing"] and not r["unavailable"]
    assert "no_ventilation_proxy_24h" in r["applied"][iu.IU_LARGE] and "not_ceeg_task_folder" in r["applied"][iu.IU_SMALL]


def test_prn_does_not_exclude_but_a_running_infusion_does():
    c = _case([3, 3], on=[0, 1])
    c["sed"]["any_6h"] = np.array([True, True])                             # a PRN dose in the 6 h, nothing running vs a running drug
    r = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], c["ceeg"], c["omop"])
    assert r["masks"][iu.IU_LARGE].tolist() == [True, False] and r["masks"][iu.IU_SMALL].tolist() == [True, False]


def test_strata_are_missing_data_safe_and_define_without_unavailable_proxies():
    c = _case([3, np.nan, -2, 3], on=False)
    r = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], c["ceeg"], c["omop"])
    assert r["masks"][iu.IU_LARGE].tolist() == [True, False, False, True]    # unknown or negative timing is not met
    # no OMOP: defined without the vent / pressor conditions and says so
    r2 = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], c["ceeg"], None)
    assert not r2["proxies_in"] and "ventilation_vasopressor" in r2["missing"]
    assert "no_ventilation_proxy_24h" not in r2["applied"][iu.IU_LARGE]
    cv = _case([3, 3], vent=[0, 1])
    assert iu.intended_use_strata(cv["hours"], cv["sed"], cv["dur"], cv["ceeg"], None)["masks"][iu.IU_LARGE].all()
    assert iu.intended_use_strata(cv["hours"], cv["sed"], cv["dur"], cv["ceeg"], cv["omop"])["masks"][iu.IU_LARGE].tolist() == [True, False]
    # without the Baseline A sedation columns or the visit start the strata are unavailable; without the cEEG folder only IU-small is
    ra = iu.intended_use_strata(c["hours"], None, c["dur"], c["ceeg"], c["omop"])
    assert ra["masks"] == {iu.IU_LARGE: None, iu.IU_SMALL: None} and "sed__" in ra["unavailable"]["both"]
    rb = iu.intended_use_strata(np.full(4, np.nan), c["sed"], c["dur"], c["ceeg"], c["omop"])
    assert rb["masks"] == {iu.IU_LARGE: None, iu.IU_SMALL: None}
    rc = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], None, c["omop"])
    assert rc["masks"][iu.IU_LARGE] is not None and rc["masks"][iu.IU_SMALL] is None and iu.IU_SMALL in rc["unavailable"]


def test_recording_condition_is_short_or_not_ceeg():
    c = _case([3] * 4, dur=[3600, 13 * 3600, 13 * 3600, np.nan], ceeg=[1, 1, 0, 0])
    m = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], c["ceeg"], c["omop"])["masks"][iu.IU_LARGE]
    assert m.tolist() == [True, False, True, True]                           # unknown length but a routine folder passes; cEEG needs <= 12 h


def test_strata_do_not_depend_on_diagnosis_codes_labels_or_eeg():
    """Both entry points take no diagnosis, label or EEG argument; the report module computes them from presentation columns only."""
    import inspect
    for fn in (iu.intended_use_strata, iu.domain_design):
        names = " ".join(p for p in inspect.signature(fn).parameters if p != "site_labels")      # site_labels = pseudonyms only
        assert not any(w in names for w in ("dx", "label", "eeg_", "silver", "icd"))
    c = _case([3, 40, 3, 60])
    base = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], c["ceeg"], c["omop"])["masks"]
    # a frame with diagnosis / label columns changes nothing: the function never sees them
    assert base[iu.IU_LARGE].tolist() == [True, True, True, False]


def test_assembly_strata_ignore_the_diagnosis_code_column(tmp_path):
    """Flipping every row's label-family diagnosis flag moves U0 but leaves IU-large / IU-small untouched."""
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=4, n_per_site=250)
    outs = []
    for dx in (0.0, 1.0):
        bl = pd.read_parquet(paths["baselines"])
        bl["meta__label_dx_before_t0"] = dx
        bl.to_parquet(paths["baselines"], index=False)
        a = rsf.build_parser().parse_args(["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver",
                                           str(paths["silver"]), "--baselines", str(paths["baselines"])])
        A = rsf.assemble(a)
        sed = iu.sedation_features(A.baseline)
        r = iu.intended_use_strata(A.hours, sed, A.covariates["duration_s"], iu.ceeg_task_flag(A.frame), None)
        outs.append((A.subgroup.sum(), r["masks"][iu.IU_LARGE].copy(), r["masks"][iu.IU_SMALL].copy()))
    assert outs[0][0] > 0 and outs[1][0] == 0                                # U0 depends on the codes ...
    assert np.array_equal(outs[0][1], outs[1][1]) and np.array_equal(outs[0][2], outs[1][2])      # ... the new strata do not
    assert outs[0][1].sum() > 0


# ------------------------------------------------------------------------------------------ counts / suppression
def test_count_rows_hide_small_cells_and_the_pooled_total_that_would_reveal_them():
    row = iu.count_row({"S1": 5, "S2": 500}, LABELS)
    assert row == {"site_1": SUPPRESSED, "site_2": 500, "all_sites": SUPPRESSED}
    assert iu.count_row({"S1": 40, "S2": 500}, LABELS) == {"site_1": 40, "site_2": 500, "all_sites": 540}
    assert iu.count_row({"S1": 3, "S2": 4}, LABELS)["all_sites"] == SUPPRESSED
    # a complement below 11 is a small cell too (rows that FAIL a condition): 395 of 400 meet it -> 5 do not
    assert iu.count_row({"S1": 395, "S2": 400}, LABELS, {"S1": 400, "S2": 400}) == {"site_1": SUPPRESSED, "site_2": SUPPRESSED, "all_sites": SUPPRESSED}
    row = iu.count_row({"S1": 300, "S2": 395}, LABELS, {"S1": 400, "S2": 400})
    assert row == {"site_1": 300, "site_2": SUPPRESSED, "all_sites": SUPPRESSED}
    assert iu.count_row({"S1": 300, "S2": 399}, LABELS, {"S1": 400, "S2": 400})["all_sites"] == SUPPRESSED


def test_stratum_summary_has_sizes_and_per_label_positives_suppressed():
    n = 400
    sites = np.repeat(["S1", "S2"], n // 2)
    c = _case(np.where(np.arange(n) % 2 == 0, 3.0, 60.0))
    res = iu.intended_use_strata(c["hours"], c["sed"], c["dur"], c["ceeg"], c["omop"])
    rng = np.random.default_rng(0)
    y = np.column_stack([rng.random(n) < 0.3, np.arange(n) < 4]).astype(float)
    m = np.ones((n, 2), bool)
    m[::6, 0] = False                                                       # a third of the stratum rows have no E1 label
    m[::10, 1] = False
    out = iu.stratum_summary(res, y, m, ["E1", "E2"], sites, ["S1", "S2"], LABELS)
    st = out["strata"][iu.IU_LARGE]
    assert st["size"]["all_sites"] == 200 and st["per_label"]["E1"]["positive"]["all_sites"] > 11
    assert st["per_label"]["E2"]["positive"]["all_sites"] == SUPPRESSED            # 2 positives in the stratum: below the floor
    assert isinstance(st["per_label"]["E1"]["assessable"]["all_sites"], int) and 100 < st["per_label"]["E1"]["assessable"]["all_sites"] < 200
    # everything assessable and assessable == stratum size: the 0 unassessable rows are a small complement, hidden
    out2 = iu.stratum_summary(res, y, np.ones((n, 2), bool), ["E1", "E2"], sites, ["S1", "S2"], LABELS)
    assert out2["strata"][iu.IU_LARGE]["per_label"]["E1"]["assessable"]["all_sites"] == SUPPRESSED
    # components that nearly everyone meets are hidden (their complement is below 11)
    assert out["components_met"]["no_ventilation_proxy_24h"]["all_sites"] == SUPPRESSED
    assert set(out["components_met"]) >= {"calendar_day_0_1", "no_infusion_at_t0", "no_ventilation_proxy_24h"}


# --------------------------------------------------------------------------------------------- delta blocks
def _d(n_per_site=400, mu=-0.1, sd=0.3, seed=0):
    rng = np.random.default_rng(seed)
    sites = np.repeat(["S1", "S2"], n_per_site)
    return mu + rng.normal(0, sd, len(sites)), sites


def test_delta_block_matches_the_mean_and_has_both_intervals():
    d, s = _d()
    b = iu.delta_block(d, s, LABELS, boot=400, seed=1)
    assert b["estimable"] and b["delta"] == pytest.approx(d.mean()) and b["n"] == len(d)
    assert b["ci_99_within_site"]["lo"] < b["ci_95_within_site"]["lo"] < b["delta"] < b["ci_95_within_site"]["hi"] < b["ci_99_within_site"]["hi"]
    assert b["pattern_met"] is True and b["all_sites_favorable"] is True and b["boot_reps"] == 400


def test_delta_block_not_estimable_below_50_or_with_a_tiny_site():
    d, s = _d(20)
    b = iu.delta_block(d, s, LABELS, boot=100)
    assert not b["estimable"] and "fewer than 50" in b["not_estimable"] and b["n"] == 40
    d2 = np.r_[np.full(8, -0.1), np.random.default_rng(1).normal(-0.1, 0.3, 120)]
    s2 = np.r_[["S1"] * 8, ["S2"] * 120]
    b2 = iu.delta_block(d2, s2, LABELS, boot=100)
    assert not b2["estimable"] and "at a site" in b2["not_estimable"]
    assert b2["n"] == SUPPRESSED and b2["per_site_n"]["site_1"] == SUPPRESSED         # the pooled count would reveal the hidden one


def test_stratum_delta_uses_only_the_stratum_rows_and_no_refit():
    d, s = _d(300, mu=-0.02, seed=2)
    mask = np.zeros(len(d), bool)
    mask[::3] = True
    d = np.where(mask, d - 0.2, d)                                           # a planted larger gain inside the stratum
    units = {"loso": {"P": (d, np.arange(len(d)), s)}}
    out = iu.strata_delta(units, {iu.IU_LARGE: mask, iu.IU_SMALL: None}, LABELS, boot=300, seed=3)
    assert set(out["loso"]) == {iu.IU_LARGE}                                  # an unavailable stratum is skipped
    blk = out["loso"][iu.IU_LARGE]["P"]
    assert blk["delta"] == pytest.approx(d[mask].mean()) and blk["delta"] < -0.15 and blk["n"] == int(mask.sum())
    assert blk["ci_99_within_site"]["hi"] < 0


def test_stratum_delta_maps_scored_rows_through_eval_idx():
    d, s = _d(150, seed=5)
    idx = np.arange(len(d)) * 2                                              # scored rows are every second analysed row
    mask = np.zeros(len(d) * 2, bool)
    mask[idx[::2]] = True
    out = iu.strata_delta({"temporal": {"baseline_A": (d, idx, s)}}, {iu.IU_LARGE: mask}, LABELS, boot=100)
    assert out["temporal"][iu.IU_LARGE]["baseline_A"]["delta"] == pytest.approx(d[::2].mean())


# ------------------------------------------------------------------------------------------------ transport
def _domain(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    sites = rng.choice(["S1", "S2"], n)
    age = rng.normal(60, 15, n)
    day = rng.integers(0, 5, n)
    infusion = rng.random(n) < 0.5
    z = (day <= 1) & ~infusion & (rng.random(n) < 0.95)                     # noisy membership so the weights stay informative
    X, names = iu.domain_design(age=age, sex=(rng.random(n) < 0.5).astype(float), sites=sites, site_labels=LABELS,
                                hours=day * 24.0 + 5.0, sed={"any_on_t0": infusion, "any_6h": infusion},
                                dur=np.full(n, 1800.0), ceeg=np.zeros(n, bool), omop=None)
    return X, names, z, sites


def test_domain_design_uses_pseudonyms_and_only_presentation_covariates():
    X, names, _z, _s = _domain()
    assert "site_site_2" in names and not any("S1" in c or "S2" in c for c in names)
    assert {"age_years", "sex_male", "calendar_day", "infusion_at_t0"} <= set(names)
    assert not any(w in c for c in names for w in ("dx", "label", "qeeg", "conn", "silver"))
    assert np.isfinite(X).all()


def test_transport_weights_are_finite_trimmed_stabilised_and_report_ess():
    X, names, z, s = _domain()
    tw = iu.transport_weights(X, names, z, s, LABELS, seed=1)
    info, w = tw["info"], tw["w"]
    assert info["estimable"] and np.isfinite(w).all() and (w > 0).all()
    assert (w[z] == 1.0).all()                                               # IU rows keep weight 1
    assert info["ess_all"] == pytest.approx(iu.ess(w), abs=0.01) and 0 < info["ess_other"] <= (~z).sum()
    assert info["crossfit_auc"] > 0.8 and info["crossfit_folds"] == 5
    other = w[~z]
    assert other.max() / other.min() < 1e4 and other.max() <= np.percentile(other, 100)    # trimmed: bounded spread
    assert info["rows"]["iu"] == int(z.sum()) and "weight_quantiles_other" in info and "balance_iu_minus_other" in info
    assert info["max_abs_smd_after"] < info["max_abs_smd_before"]            # weighting improves balance
    assert_aggregate_only(info)


def test_inverse_odds_weights_trim_at_the_1st_and_99th_percentile():
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(0.5, 3, 2000), 1e-4, 1 - 1e-4)
    z = rng.random(2000) < p
    wt = iu.inverse_odds_weights(p, z)
    lo, hi = wt["bounds"]
    assert wt["w"][~z].min() >= lo - 1e-12 and wt["w"][~z].max() <= hi + 1e-12 and np.isfinite(wt["w"]).all()
    assert wt["trimmed"].sum() >= 0.015 * (~z).sum() * 0.5
    assert iu.inverse_odds_weights(np.zeros(2000), z)["w"].max() < np.inf       # p = 0 is clipped, never infinite


def test_transport_not_estimable_with_few_iu_rows():
    X, names, z, s = _domain(300)
    z[:] = False
    z[:20] = True
    tw = iu.transport_weights(X, names, z, s, LABELS)
    assert tw["w"] is None and not tw["info"]["estimable"] and tw["info"]["rows"]["iu"] == 20


def test_weighted_delta_recovers_the_target_effect_and_reports_ess():
    rng = np.random.default_rng(3)
    n = 1500
    sites = rng.choice(["S1", "S2"], n)
    z = rng.random(n) < 0.3
    d = np.where(z, -0.30, -0.05) + rng.normal(0, 0.2, n)                    # a larger gain in the target population
    w = np.where(z, 1.0, 0.0) + np.where(z, 0.0, 1.0) * 0.0
    w = np.where(z, 1.0, 0.05)                                               # almost all weight on the IU rows
    b = iu.delta_block(d, sites, LABELS, w, boot=300, seed=1)
    assert b["estimable"] and b["delta"] < -0.25 and b["unweighted_delta"] > -0.2
    assert 0 < b["ess"] <= len(d) and b["ess_over_scored"] <= 1
    assert b["delta"] == pytest.approx(float((w * d).sum() / w.sum()))
    out = iu.transport_delta({"loso": {"A": (d, np.arange(n), sites)}}, w, LABELS, boot=100, seed=1)
    assert out["loso"]["A"]["delta"] == pytest.approx(b["delta"])


# --------------------------------------------------------------------------------------------- end to end
def _flat(o, key=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _flat(v, k)
    elif isinstance(o, list):
        for v in o:
            yield from _flat(v, key)
    else:
        yield key, o


def _check_report(run):
    iu_blk = run.J["intended_use"]
    pr = iu_blk["presentation"]
    assert "failed" not in pr and pr["title"].startswith(NEW_TITLE)
    ids = {str(p) for p in run.cohort["person_id"]} | set(rsf.recording_ids(run.cohort))
    assert_aggregate_only(iu_blk, ids)
    assert_aggregate_only(run.md, ids)
    assert "S0001" not in json.dumps(pr) and "S0002" not in json.dumps(pr)
    # counts keyed n_* (patient counts) are suppressed or >= 11
    seen = [(k, v) for k, v in _flat(pr) if isinstance(v, int) and not isinstance(v, bool) and (k.startswith("n_") or k in ("n", "assessable", "positive"))]
    assert seen and all(v >= 11 for _, v in seen)
    return pr


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    d = tmp_path_factory.mktemp("iu_plain")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=3, n_per_site=450)
    J, md, stdout = run_main(paths, d / "out", "--splits", "loso", "temporal", "--skip-controls",
                             "--omop-cache", str(d / "no_omop_cache"))      # hermetic: never the repo's local cache
    return SimpleNamespace(d=d, paths=paths, ms=ms, cohort=cohort, J=J, md=md, stdout=stdout)


def test_report_has_the_new_sections_and_labels_u0_as_superseded(plain):
    md = plain.md
    assert "### Presentation-based intended-use strata (IU-large, IU-small) and transport weighting (D-155)" in md
    assert "### Undifferentiated subgroup: U0 (code-conditioned; superseded, see D-155)" in md
    assert "| loso | U0 (code-conditioned; superseded, see D-155) |" in md
    assert "**Sizes**" in md and "Silver-label positives / assessable per stratum" in md and "**Delta by comparison within the strata**" in md
    assert "**Transport: re-weighting toward IU-large**" in md and "effective sample size" in md
    assert md.index("## Intended use: undifferentiated AMS") < md.index("Presentation-based intended-use strata") < md.index("## Data flow")
    assert plain.J["intended_use"]["subgroup_label"].startswith("U0 (code-conditioned; superseded")
    assert "D-155" in json.dumps(plain.J["decisions"]) and any("D-155" in x for x in plain.J["limitations"])
    assert any("IU-large" in line for line in plain.stdout.splitlines())


def test_strata_and_comparisons_in_the_json(plain):
    pr = _check_report(plain)
    st = pr["summary"]["strata"]
    assert set(st) == {"IU_large", "IU_small"} and all(v["available"] for v in st.values())
    assert set(pr["set_labels"]) == {"prior", "agesex", "P", "baseline_A", "baseline_C"}
    assert pr["summary"]["proxies_included"] is False and "ventilation_vasopressor" in pr["summary"]["missing"]
    assert "WITHOUT" in pr["summary"]["missing"]["ventilation_vasopressor"] and "NOT included" in plain.md
    large, small = st["IU_large"]["size"]["all_sites"], st["IU_small"]["size"]["all_sites"]
    assert isinstance(large, int) and large >= 11 and (small == SUPPRESSED or (isinstance(small, int) and small <= large))
    for lab, c in st["IU_large"]["per_label"].items():
        assert set(c) == {"assessable", "positive"}
    for split in ("loso", "temporal"):
        for nm in ("IU_large", "IU_small"):
            blk = pr["delta"][split][nm]
            assert set(blk) == {"prior", "agesex", "P", "baseline_A", "baseline_C"}
            for k, r in blk.items():
                assert r["estimable"] or "not_estimable" in r
                if r["estimable"]:
                    assert r["ci_99_within_site"]["lo"] <= r["delta"] <= r["ci_99_within_site"]["hi"] and r["ci_95_within_site"]
    # scored within the stratum only: the IU-large Delta of A is the mean of the stored d_i over those rows (no refit)
    assert pr["delta"]["loso"]["IU_large"]["baseline_A"]["boot_reps"] == 300


def test_stratum_delta_equals_the_mean_of_the_held_out_d_inside_it(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=6, n_per_site=300)
    a = rsf.build_parser().parse_args(["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver", str(paths["silver"]),
                                       "--baselines", str(paths["baselines"]), "--n-boot", "100", "--splits", "loso", "--skip-controls",
                                       "--omop-source", "none", "--out", str(tmp_path / "o")])
    A = rsf.assemble(a)
    cfg = rsf.LadderConfig(n_boot=100, seed=0, include_e7=A.include_e7, temporal_test_fraction=0.2)
    iu_res: dict = {}
    rsf.run_intended_use(A, a, cfg, "combined", ["loso"], keep=iu_res)
    res = iu_res["loso"]
    rr = res.get("combined", "P")
    sed = iu.sedation_features(A.baseline)
    masks = iu.intended_use_strata(A.hours, sed, A.covariates["duration_s"], iu.ceeg_task_flag(A.frame), None)["masks"]
    sel = masks[iu.IU_LARGE][res.eval_idx]
    d = np.asarray(rr.d, float)
    want = np.nanmean(d[sel])
    out = iu.strata_delta({"loso": {"P": (d, res.eval_idx, np.asarray(res.eval_sites).astype(str))}}, masks,
                          {s: l for s, l in A.site_labels.items()}, boot=100)
    assert out["loso"][iu.IU_LARGE]["P"]["delta"] == pytest.approx(want)


def test_transport_section_has_ess_and_finite_weights(plain):
    pr = _check_report(plain)
    tr = pr["transport"]
    assert tr["estimable"] and tr["info"]["estimable"]
    info = tr["info"]
    assert np.isfinite(info["ess_all"]) and info["ess_all"] > 0 and info["ess_other"] > 0 and 0 < info["ess_other_over_rows"] <= 1
    assert "calendar_day" in info["covariates"] and not any("dx" in c or "label" in c for c in info["covariates"])
    for split in ("loso", "temporal"):
        assert set(tr["delta"][split]) == {"prior", "agesex", "P", "baseline_A", "baseline_C"}
        for r in tr["delta"][split].values():
            if r["estimable"]:
                assert np.isfinite(r["delta"]) and r["ess"] > 0 and 0 < r["ess_over_scored"] <= 1
                assert r["ci_99_within_site"] and r["ci_95_within_site"] and "unweighted_delta" in r


@pytest.fixture(scope="module")
def with_omop(tmp_path_factory):
    """Plants ventilation / vasopressor proxies through ``attach_omop`` (a deterministic function of the fake ids)."""
    d = tmp_path_factory.mktemp("iu_omop")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=3, n_per_site=450)

    def fake_attach(A, a, store=None):
        pid = A.frame["person_id"].to_numpy("int64")
        A.omop = pd.DataFrame({"vent_24h": pid % 5 == 0, "vasopressor_24h": pid % 7 == 0})
        A.omop_info = {"omop_source": "test planted"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rsf, "attach_omop", fake_attach)
        J, md, stdout = run_main(paths, d / "out", "--splits", "loso", "--skip-controls")
    return SimpleNamespace(d=d, paths=paths, ms=ms, cohort=cohort, J=J, md=md, stdout=stdout)


def test_omop_proxies_enter_the_strata_when_available(plain, with_omop):
    pr = _check_report(with_omop)
    sm = pr["summary"]
    assert sm["proxies_included"] is True and not sm["missing"]
    assert "no_ventilation_proxy_24h" in sm["applied_components"]["IU_large"] and "no_vasopressor_24h" in sm["applied_components"]["IU_small"]
    base = plain.J["intended_use"]["presentation"]["summary"]["strata"]["IU_large"]["size"]["all_sites"]
    with_p = sm["strata"]["IU_large"]["size"]["all_sites"]
    assert with_p < base                                                      # requiring no vent / no pressor shrinks the stratum
    assert "test planted" in json.dumps(pr["sources"]) and "included (OMOP" in with_omop.md
    assert "ventilation_proxy_24h" in pr["transport"]["info"]["covariates"]


def test_attach_omop_reads_in_memory_tables_and_degrades_gracefully(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=7, n_per_site=120)
    a = rsf.build_parser().parse_args(["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver", str(paths["silver"]),
                                       "--baselines", str(paths["baselines"])])
    A = rsf.assemble(a)
    ns = SimpleNamespace(omop_source="none", omop_cache=None)
    rsf.attach_omop(A, ns)
    assert A.omop is None and A.omop_info == {"omop_source": "none"}
    ns = SimpleNamespace(omop_source="cache", omop_cache=str(tmp_path / "empty_cache"))
    rsf.attach_omop(A, ns)
    assert A.omop is None and "empty or absent" in A.omop_info["omop_source"]
    # in-memory tables: a ventilator row inside the 24 h before t0 for the first 10 rows, a vasopressor for rows 10-19
    t0 = pd.to_datetime(A.frame["t0"])
    pid = A.frame["person_id"].to_numpy("int64")
    meas = pd.DataFrame({"person_id": pid[:10], "measurement_datetime": t0.iloc[:10] - pd.Timedelta(hours=3), "measurement_date": None,
                         "measurement_source_value": "Ventilator Mode"})
    drug = pd.DataFrame({"person_id": pid[10:20], "drug_exposure_start_datetime": t0.iloc[10:20] - pd.Timedelta(hours=3),
                         "drug_exposure_end_datetime": t0.iloc[10:20] - pd.Timedelta(hours=2), "drug_exposure_start_date": None,
                         "drug_exposure_end_date": None, "drug_source_value": "NOREPINEPHRINE 8 MG IN D5W 250 ML"})
    tables = {"omop_measurement": meas, "omop_drug_exposure": drug, "omop_procedure_occurrence": pd.DataFrame(
        {"person_id": [], "procedure_datetime": [], "procedure_date": [], "procedure_source_value": []})}
    A.omop, A.omop_info = None, {}
    rsf.attach_omop(A, SimpleNamespace(omop_source="store", omop_cache=None), tables)
    assert A.omop is not None and A.omop["vent_24h"].to_numpy()[:10].all() and not A.omop["vent_24h"].to_numpy()[10:].any()
    assert A.omop["vasopressor_24h"].to_numpy()[10:20].all() and not A.omop["vasopressor_24h"].to_numpy()[:10].any()
    assert A.omop_info["omop_source"] == "in-memory tables"


def test_small_strata_are_not_estimable_and_suppressed(tmp_path):
    """Everyone is on a running sedative at t0: IU-large and IU-small are empty, every Delta is not estimable, sizes are '<11'."""
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=8, n_per_site=300)
    bl = pd.read_parquet(paths["baselines"])
    bl["sed__sedative__on_t0"] = 1.0
    bl.to_parquet(paths["baselines"], index=False)
    J, md, _ = run_main(paths, tmp_path / "o", "--splits", "loso", "--skip-controls",
                       "--omop-cache", str(tmp_path / "no_omop_cache"))
    pr = J["intended_use"]["presentation"]
    for nm in ("IU_large", "IU_small"):
        assert pr["summary"]["strata"][nm]["size"]["all_sites"] == SUPPRESSED
        assert pr["summary"]["strata"][nm]["share_of_analysed"] == SUPPRESSED
        for r in pr["delta"]["loso"][nm].values():
            assert r["estimable"] is False and "fewer than 50" in r["not_estimable"]
    assert pr["transport"]["info"]["estimable"] is False and "not_estimable" in pr["transport"]["info"]
    assert "Not estimable" in md or "Not run" in md
    ids = {str(p) for p in cohort["person_id"]}
    assert_aggregate_only(pr, ids)
