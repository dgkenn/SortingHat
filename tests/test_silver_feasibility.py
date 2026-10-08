"""scripts/run_silver_feasibility.py end to end on SYNTHETIC inputs: planted signal recovered, null near zero, controls
present, circularity audit wired, and nothing record-level in any output (D-143)."""
import contextlib
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sortinghat.metrics.loss import per_patient_delta
from sortinghat.safe_output import assert_aggregate_only
from test_silver_feasibility_inputs import (argv_for, fake_reports_findings, make_inputs,  # noqa: F401  (tests/ on sys.path)
                                            rsf)

BANNER = "EXPLORATORY — silver-label feasibility; not a test of preregistered hypotheses"
HEADLINE = "combined"


def run_main(paths, out, *extra, ms=None, cohort=None, rf_mode="silver", store_dir=None, small=True):
    argv = argv_for(paths, out, *extra, small=small)
    if ms is not None:
        argv += ["--data", str(store_dir)]
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        if ms is not None:
            mp.setattr(rsf, "load_reports_findings", lambda store, sites: fake_reports_findings(cohort, ms, rf_mode))
        with contextlib.redirect_stdout(buf):
            rc = rsf.main(argv)
    assert rc == 0
    out = Path(out)
    return json.loads((out / "report.json").read_text()), (out / "report.md").read_text(), buf.getvalue()


@pytest.fixture(scope="module")
def sig(tmp_path_factory):
    d = tmp_path_factory.mktemp("sig")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=3)
    (d / "store").mkdir()
    J, md, stdout = run_main(paths, d / "out", ms=ms, cohort=cohort, store_dir=d / "store")
    return SimpleNamespace(d=d, paths=paths, ms=ms, cohort=cohort, J=J, md=md, stdout=stdout)


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    d = tmp_path_factory.mktemp("null")
    paths, ms, cohort = make_inputs(d, signal=0.0, seed=4)
    J, md, stdout = run_main(paths, d / "out")
    return SimpleNamespace(d=d, paths=paths, ms=ms, cohort=cohort, J=J, md=md, stdout=stdout)


def rung(J, split, base, name=HEADLINE):
    return J["ladder"][split]["rungs"][base][name]


# ------------------------------------------------------------------------------------------- planted signal
@pytest.mark.parametrize("split", ["loso", "temporal"])
@pytest.mark.parametrize("base", ["A", "C"])
def test_planted_signal_is_recovered(sig, split, base):
    r = rung(sig.J, split, base)
    assert r["delta"] < -0.02
    assert r["ci_primary_within_site"]["alpha"] == 0.01 and r["ci_primary_within_site"]["hi"] < 0   # 99% interval, D-095
    assert r["all_sites_favorable"] and r["h1h2_rule_met"]
    assert all(v["delta"] < 0 for v in r["per_site"].values())
    for rn in ("qeeg", "connectivity"):
        assert rung(sig.J, split, base, rn)["delta"] < 0


def test_signal_labels_improve_and_signal_free_label_does_not(sig):
    pl = rung(sig.J, "loso", "A")["per_label"]
    for lab in ("E1", "E2", "E5", "E6"):
        assert pl[lab]["improved"] and pl[lab]["delta"] < -0.03
    assert pl["E4a"]["delta"] > -0.01                              # the synthetic study plants no EEG signal for E4a
    d_auc = sig.J["discrimination_calibration"]["loso"]["A"]["delta_auroc_vs_baseline"][HEADLINE]
    assert d_auc["E1"]["lo"] > 0 and d_auc["E2"]["lo"] > 0


def test_label_policy_excludes_e3_e4b_and_applies_the_e7_rule(sig):
    lab = sig.J["labels"]
    assert not {"E3", "E4b"} & set(lab["analysed"])
    assert set(lab["primary_for_delta"]) >= {"E1", "E2", "E4a", "E5", "E6"}
    assert "excluded_reason" in lab["detail"]["E3"] and "excluded_reason" in lab["detail"]["E4b"]
    # E7 is primary only with >= 100 positives over >= 2 sites
    n7 = lab["detail"]["E7"]["n_positive"]
    assert lab["e7_primary"] == (isinstance(n7, int) and n7 >= 100)
    assert ("E7" in lab["primary_for_delta"]) == lab["e7_primary"]


def test_primary_ci_level_matches_the_sample_size_note(sig):
    note = (Path(__file__).resolve().parents[1] / "docs" / "research" / "evaluation_sample_size.md").read_text()
    assert "99%" in note and "within-site" in note                  # the recommendation the 0.01 default follows
    assert sig.J["settings"]["ci_policy"].startswith("99% within-site")


def test_both_comparators_and_both_schemes_are_reported(sig):
    assert set(sig.J["ladder"]) == {"loso", "temporal"}
    for split in sig.J["ladder"]:
        assert set(sig.J["ladder"][split]["rungs"]) == {"A", "C"}
    assert sig.J["headline_rung"] == HEADLINE
    assert set(sig.J["settings"]["skipped_rungs"]) == {"morgoth", "embeddings (CBraMod)", "dynamics"}


def test_discrimination_and_calibration_blocks(sig):
    dc = sig.J["discrimination_calibration"]["loso"]["A"]
    assert set(dc["auroc"]) >= {"baseline_only", "qeeg", "connectivity", HEADLINE}
    for lab in sig.J["labels"]["analysed"]:
        assert dc["auroc"][HEADLINE][lab]["pooled"] != "<11"
    cal = dc["calibration"][HEADLINE]
    for lab, v in cal.items():
        if isinstance(v["n_events"], int) and v["n_events"] >= 100:
            assert v["slope"] is not None
        else:
            assert v["slope"] is None                              # D-094: O/E only below 100 events
    assert cal["E1"]["slope"] is not None and 0.5 < cal["E1"]["slope"] < 1.5


# ------------------------------------------------------------------------------------------------ null
@pytest.mark.parametrize("split", ["loso", "temporal"])
@pytest.mark.parametrize("base", ["A", "C"])
def test_null_delta_is_near_zero_and_rule_not_met(null, split, base):
    for rn in ("qeeg", "connectivity", HEADLINE):
        r = rung(null.J, split, base, rn)
        assert abs(r["delta"]) < 0.02
    r = rung(null.J, split, base)
    assert not r["h1h2_rule_met"]


# --------------------------------------------------------------------------------------------- controls
def test_mandatory_controls_are_present_and_behave(sig, null):
    c = sig.J["controls"]
    assert set(c["leakage_probes"]["probes"]) >= {"site", "duration"}
    for split in ("loso", "temporal"):
        for base in ("A", "C"):
            v = c["by_scheme"][split][base]
            assert v["sedative_excluded"]["delta"] < 0 and v["sedative_excluded"]["pooled_delta_negative"]
            assert v["negative_control_shuffled_labels"]["n_reps"] == 1
            assert abs(v["negative_control_shuffled_labels"]["mean_delta"]) < 0.02
            assert abs(v["negative_control_permuted_eeg"]["delta"]) < 0.02
            assert v["severity_strata"]["cut_points_prespecified"] is True
    sev = c["by_scheme"]["loso"]["A"]["severity_strata"]
    assert sev["strata_with_delta_below_zero"] >= 2 and sev["descriptive_pattern_met"]
    # the same controls on pure-noise EEG show no gain
    for base in ("A", "C"):
        assert not null.J["controls"]["by_scheme"]["loso"][base]["negative_control_shuffled_labels"]["n_reps_with_spurious_gain_95"]


def test_sedative_excluded_subset_removes_exactly_the_sedated(sig):
    a = rsf.build_parser().parse_args(argv_for(sig.paths, sig.d / "x"))
    A = rsf.assemble(a)
    n_sed = int(A.covariates["sedated"].sum())
    assert 0 < n_sed < len(A.frame)
    se = sig.J["controls"]["by_scheme"]["loso"]["A"]["sedative_excluded"]
    assert se["n_excluded"] == n_sed and se["n_kept"] == len(A.frame) - n_sed


def test_site_leakage_probe_flags_a_site_shifted_feature_block(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=0.0, seed=5, site_shift=2.5)
    J, _md, _ = run_main(paths, tmp_path / "out", "--splits", "loso")
    p = J["controls"]["leakage_probes"]["probes"]["site"]
    assert p["flagged"] and p["auroc"] > 0.7


def test_skip_controls_is_loud(tmp_path):
    paths, *_ = make_inputs(tmp_path, signal=1.0, seed=6, n_per_site=300)
    J, md, _ = run_main(paths, tmp_path / "out", "--splits", "loso", "--skip-controls")
    assert "skipped" in J["controls"] and "NOT run" in J["controls"]["skipped"] and "NOT run" in md


# ------------------------------------------------------------------------------------ circularity audit
def test_circularity_audit_runs_on_held_out_predictions(sig):
    ca = sig.J["circularity_audit"]
    assert set(ca["report_coverage"]) == {"E1", "E2", "E5"}
    assert set(ca["per_model"]) == {f"{b}/{m}" for b in "AC" for m in ("baseline_only", HEADLINE)}
    for mdl, d in ca["per_model"].items():
        for lab in ("E1", "E2", "E5"):
            v = d["per_label"][lab]
            assert isinstance(v["n"], int) and v["n"] > 100
            # the comparator mirrors the silver labels exactly here, so the two agreements are identical
            assert v["agreement_eeg_impression"] == v["agreement_gold"] and v["difference"] == 0 and not v["leak"]
        assert d["labels_flagged"] == []


def test_circularity_audit_flags_a_model_that_agrees_more_with_the_report(tmp_path):
    """Report flags that follow the TRUE labels while the silver labels carry 8% noise: held-out predictions agree more with
    the report than with the silver label (the audit's rule), and the audit says so."""
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=3)
    (tmp_path / "store").mkdir()
    J, md, _ = run_main(paths, tmp_path / "out", "--splits", "loso", "--skip-controls", ms=ms, cohort=cohort,
                        rf_mode="true", store_dir=tmp_path / "store")
    assert J["circularity_audit"] == {"note": "skipped with --skip-controls"}
    J2, md2, _ = run_main(paths, tmp_path / "out2", "--splits", "loso", ms=ms, cohort=cohort, rf_mode="true",
                          store_dir=tmp_path / "store")
    ca = J2["circularity_audit"]["per_model"]
    for b in "AC":
        d = ca[f"{b}/{HEADLINE}"]                      # the EEG model reads the planted signal, which follows the true labels
        assert set(d["labels_flagged"]) == {"E1", "E2", "E5"}
        for v in d["per_label"].values():
            assert v["agreement_eeg_impression"] > v["agreement_gold"] and v["difference"] > 0


def test_circularity_is_skipped_without_a_store(null):
    assert "skipped" in null.J["circularity_audit"]["note"]


# ----------------------------------------------------------------------------------------- aggregate only
BAD_TOKENS = re.compile(r"(\b71\d{6}\b|\brec[0-9a-f]{20}\b|sub-|ses-|\d{4}-\d{2}-\d{2}|local_only|\.edf)")


def _ids(sig):
    return {str(p) for p in sig.cohort["person_id"]} | set(rsf.recording_ids(sig.cohort))


def test_outputs_are_aggregate_only(sig):
    out = sig.d / "out"
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "report.md"]
    for text in (sig.md, sig.stdout, (out / "report.json").read_text()):
        assert not BAD_TOKENS.search(text), BAD_TOKENS.search(text)
        assert_aggregate_only(text, _ids(sig))
    assert_aggregate_only(sig.J, _ids(sig))
    # no per-patient vectors: every list in the JSON is short (labels, strata, replicate summaries), never n-long
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            yield len(o)
            for v in o:
                yield from walk(v)
    assert max(walk(sig.J)) < 50


def test_every_count_is_suppressed_or_at_least_11(sig):
    not_people = {"n_boot", "n_valid", "n_reps", "n_strata", "n_favorable", "n_eeg_features", "n_features_in", "n_features_out",
                  "n_reps_with_spurious_gain_95", "strata_with_delta_below_zero"}

    def counts(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from counts(v, k)
        elif isinstance(o, int) and not isinstance(o, bool) and re.match(r"^n($|_)", key) and key not in not_people:
            yield key, o
    seen = list(counts(sig.J))
    assert len(seen) > 50
    for k, v in seen:
        assert v >= 11, (k, v)


def test_banner_on_every_surface(sig):
    assert sig.md.splitlines()[0] == f"# {BANNER}"
    assert sig.J["banner"] == BANNER
    assert sig.stdout.splitlines()[0] == BANNER
    assert "not a test of preregistered hypotheses" in sig.md


def test_sites_are_pseudonymised(sig):
    text = sig.md + json.dumps(sig.J) + sig.stdout
    assert "S0001" not in text and "S0002" not in text
    assert "site_1" in text and "site_2" in text


def test_output_and_input_paths_are_guarded(tmp_path, sig):
    with pytest.raises(SystemExit):
        rsf.main(argv_for(sig.paths, tmp_path / "local_only" / "out"))
    bad = tmp_path / "cohort.csv"
    sig.cohort.to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        rsf.main(argv_for({**sig.paths, "cohort": bad}, tmp_path / "out"))


# --------------------------------------------------------------------------------------------- wiring
def test_model_data_roles_for_each_scheme(sig):
    a = rsf.build_parser().parse_args(argv_for(sig.paths, sig.d / "x"))
    A = rsf.assemble(a)
    lo = rsf.build_model_data(A, "loso", 0.2, 0.2, 0)
    assert set(np.unique(lo.gold_role)) == {"dev", "eval"} and np.array_equal(lo.y_gold, lo.y_silver)
    for s in np.unique(A.sites):
        r = lo.gold_role[A.sites == s]
        assert abs((r == "dev").mean() - 0.2) < 0.01
    tp = rsf.build_model_data(A, "temporal", 0.2, 0.2, 0)
    from sortinghat.metrics.splits import late_temporal_holdout
    ts = late_temporal_holdout(A.sites, A.times, 0.2)
    assert not np.isin(np.flatnonzero(tp.gold_role == "dev"), ts.test_idx).any()   # no dev row in the test region
    assert (tp.gold_role[ts.test_idx] == "eval").all()
    assert np.array_equal(lo.baseline.columns, A.baseline.columns) and tp.eeg.columns.str.startswith(("qeeg.", "conn.")).all()


def test_prediction_tap_reproduces_the_ladder_delta(sig):
    from sortinghat.models import DEFAULT_RUNGS, LadderConfig, run_ladder
    a = rsf.build_parser().parse_args(argv_for(sig.paths, sig.d / "x"))
    A = rsf.assemble(a)
    md = rsf.build_model_data(A, "loso", 0.2, 0.2, 0)
    rungs = [r for r in DEFAULT_RUNGS if r.name in ("prior", "qeeg")]
    cfg = LadderConfig(n_boot=50, include_e7=A.include_e7, per_label=False)
    with rsf.PredictionTap(md, {"A": A.baseline_sets["A"]}) as tap:
        res = run_ladder(md, {"A": A.baseline_sets["A"]}, rungs, cfg, "loso")
    ev = (md.gold_role == "eval") & tap.tested["A"]
    prim = np.array([l in A.primary for l in A.label_names])
    m = md.m_gold & prim[None, :]
    d = per_patient_delta(md.y_gold[ev], tap.P["A"]["qeeg"][ev], tap.P["A"]["__baseline__"][ev], m[ev])
    assert np.nanmean(d) == pytest.approx(res.get("qeeg", "A").delta, abs=1e-12)
    assert np.array_equal(np.flatnonzero(ev), res.eval_idx)
    assert tap.tested["A"].all()                                   # LOSO scores every row exactly once
    from sortinghat.models import ladder
    assert ladder.fit_predict_fold.__name__ == "fit_predict_fold"  # the patch is undone after the with-block


def test_severity_strata_put_missing_severity_in_its_own_group_not_the_top():
    n = 240
    rng = np.random.default_rng(0)
    sites = np.repeat(["a", "b"], n // 2)
    sev = rng.uniform(0, 12, n)
    sev[:40] = np.nan
    d = rng.normal(-0.05, 0.1, n)
    res = SimpleNamespace(get=lambda rung, b: SimpleNamespace(d=d), eval_idx=np.arange(n), eval_sites=sites)
    md = SimpleNamespace(covariates={"severity": sev})
    out = rsf.severity_strata(res, md, "A", "combined", 100, 0, 30)
    assert out["n_with_unknown_severity"] == 40 and set(out["strata"]) == {"low", "mid", "high"}
    assert sum(v["n"] for v in out["strata"].values()) == n - 40
    hi = out["strata"]["high"]
    assert hi["n"] == int((sev >= 9.5).sum())


def test_label_with_too_few_cases_at_a_site_is_not_analysed(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=7, n_per_site=300)
    sil = pd.read_csv(paths["silver"], dtype=str)
    sil["E7"] = np.where(np.isin(np.arange(len(sil)), np.arange(100, 105)), "True", "False")   # 5 positives overall
    sil.to_csv(paths["silver"], index=False)
    a = rsf.build_parser().parse_args(argv_for(paths, tmp_path / "x"))
    A = rsf.assemble(a)
    assert "E7" not in A.label_names and "fewer than 11" in A.label_report["E7"]["excluded_reason"]
    assert not A.include_e7 and set(A.primary) == {"E1", "E2", "E4a", "E5", "E6"}


def test_e7_with_enough_cases_but_under_100_is_exploratory_only(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=8, n_per_site=300)
    sil = pd.read_csv(paths["silver"], dtype=str)
    pos = np.zeros(len(sil), bool)
    pos[np.r_[100:140, 400:440]] = True                                        # 40 per site: >= 11 each, 80 < 100
    sil["E7"] = np.where(pos, "True", "False")
    sil.to_csv(paths["silver"], index=False)
    a = rsf.build_parser().parse_args(argv_for(paths, tmp_path / "x"))
    A = rsf.assemble(a)
    assert "E7" in A.label_names and not A.include_e7 and "E7" not in A.primary
    assert "exploratory" in A.label_report["E7"]["role"]


def test_data_flow_counts_follow_the_inputs(sig):
    st = {s["step"]: s for s in sig.J["data_flow"]["steps"]}
    n_cohort = int(sig.cohort["in_strict"].sum())
    assert st["cohort rows (sites and cohort definition applied)"]["n_total"] == n_cohort
    assert st["primary window passing QC"]["n_total"] < st["with a primary-window EEG feature row"]["n_total"]
    assert sig.J["n_analysed"] == st["with >= 1 assessable analysed label"]["n_total"]
