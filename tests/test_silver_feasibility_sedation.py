"""Sedation-confounding analyses of the silver-feasibility report (SYNTHETIC ONLY): sedation-stratified Delta from the stored
held-out predictions, its small-cell suppression, the difference of Deltas, the baseline-adjustment statement, the sedation
leakage probe, and the report sections. Pure-function tests use made-up arrays; the end-to-end ones run the script on the
synthetic inputs of test_silver_feasibility_inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sortinghat import checkpoint
from sortinghat.metrics.sedation import stratified_delta
from sortinghat.models import leakage_probes
from sortinghat.models.controls import sedation_probe
from sortinghat.models.synthetic import make_synthetic_study
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_silver_feasibility import run_main
from test_silver_feasibility_inputs import argv_for, make_inputs, rsf

HEADLINE = "combined"
LABELS = {"S1": "site_1", "S2": "site_2"}


# ------------------------------------------------------------------------------------------ pure function
def _walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    else:
        yield o


def _planted(n_per_site=300, p_sed=0.3, mu_sed=-0.30, mu_non=-0.10, seed=0):
    rng = np.random.default_rng(seed)
    sites = np.repeat(["S1", "S2"], n_per_site)
    flag = rng.random(len(sites)) < p_sed
    d = np.where(flag, mu_sed, mu_non) + rng.normal(0, 0.2, len(sites))
    return d, flag, sites


def test_stratified_delta_recovers_the_planted_strata_and_difference():
    d, flag, sites = _planted()
    r = stratified_delta(d, flag, sites, LABELS, n_boot=600, seed=1)
    sed, non = r["strata"]["sedated"], r["strata"]["non_sedated"]
    assert sed["estimable"] and non["estimable"]
    assert sed["delta"] == pytest.approx(d[flag].mean()) and non["delta"] == pytest.approx(d[~flag].mean())
    assert sed["n"] == int(flag.sum()) and non["n"] == int((~flag).sum())
    assert sed["ci_99_within_site"]["lo"] < sed["delta"] < sed["ci_99_within_site"]["hi"]
    assert sed["ci_99_within_site"]["lo"] < sed["ci_95_within_site"]["lo"]                  # the 99% interval is the wider one
    assert -0.45 < sed["delta"] < -0.15 and -0.2 < non["delta"] < 0.0
    # pooled Delta is the n-weighted mean of the strata
    assert (sed["n"] * sed["delta"] + non["n"] * non["delta"]) / (sed["n"] + non["n"]) == pytest.approx(r["overall_delta"])
    df = r["difference_sedated_minus_non_sedated"]
    assert df["estimable"] and df["delta"] == pytest.approx(sed["delta"] - non["delta"])
    assert df["ci_99_within_site"]["hi"] < 0                                                 # planted: gain larger when sedated
    assert df["ci_99_within_site"]["lo"] < df["delta"] < df["ci_99_within_site"]["hi"]
    assert r["non_sedated_gain_persists_99"] is True
    assert set(r["per_site"]) == {"site_1", "site_2"}
    assert all(c["sedated"]["delta"] != SUPPRESSED and c["sedated"]["n"] >= 11 for c in r["per_site"].values())


def test_difference_interval_is_paired_by_bootstrap_not_a_difference_of_endpoints():
    # nothing but noise around one common mean: the difference interval straddles 0 and is narrower than the naive
    # endpoint difference of the two 99% stratum intervals (which would ignore that both come from the same resamples)
    d, flag, sites = _planted(mu_sed=-0.2, mu_non=-0.2, seed=5)
    r = stratified_delta(d, flag, sites, LABELS, n_boot=800, seed=2)
    df = r["difference_sedated_minus_non_sedated"]
    assert df["ci_99_within_site"]["lo"] < 0 < df["ci_99_within_site"]["hi"]
    s, n_ = r["strata"]["sedated"]["ci_99_within_site"], r["strata"]["non_sedated"]["ci_99_within_site"]
    naive = (s["hi"] - n_["lo"]) - (s["lo"] - n_["hi"])
    assert df["ci_99_within_site"]["hi"] - df["ci_99_within_site"]["lo"] < naive


def test_null_gain_does_not_persist_in_the_non_sedated_stratum():
    d, flag, sites = _planted(mu_sed=-0.3, mu_non=0.0, seed=7)
    r = stratified_delta(d, flag, sites, LABELS, n_boot=600, seed=3)
    assert r["non_sedated_gain_persists_99"] is False
    assert r["strata"]["non_sedated"]["ci_99_within_site"]["lo"] < 0 < r["strata"]["non_sedated"]["ci_99_within_site"]["hi"]


def test_stratum_under_50_is_not_estimable_but_its_count_is_shown_when_it_is_at_least_11():
    d, flag, sites = _planted(n_per_site=100, p_sed=0.0)
    flag = np.zeros(len(d), bool)
    flag[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]] = True   # 30 sedated
    r = stratified_delta(d, flag, sites, LABELS, n_boot=100, seed=0)
    assert r["strata"]["sedated"] == {"n": 30, "estimable": False, "not_estimable": "fewer than 50 patients in the stratum"}
    assert r["strata"]["non_sedated"]["estimable"] and "ci_99_within_site" in r["strata"]["non_sedated"]   # the big stratum stands alone
    assert r["difference_sedated_minus_non_sedated"]["estimable"] is False
    assert "delta" not in r["difference_sedated_minus_non_sedated"] and r["non_sedated_gain_persists_99"] in (True, False)


def test_counts_under_11_are_suppressed_and_the_complement_is_hidden_too():
    d, flag, sites = _planted(n_per_site=150, p_sed=0.0)
    flag[:8] = True                                                                          # 8 sedated patients in site S1
    r = stratified_delta(d, flag, sites, LABELS, n_boot=100, seed=0)
    assert r["strata"]["sedated"]["n"] == SUPPRESSED and r["strata"]["non_sedated"]["n"] == SUPPRESSED   # no recovery by subtraction
    assert r["strata"]["sedated"]["estimable"] is False and "delta" not in r["strata"]["sedated"]
    assert r["per_site"]["site_1"]["sedated"] == {"n": SUPPRESSED, "delta": SUPPRESSED}
    assert r["per_site"]["site_1"]["non_sedated"]["n"] == SUPPRESSED                        # its own delta is a large cell
    assert r["per_site"]["site_1"]["non_sedated"]["delta"] != SUPPRESSED
    assert r["per_site"]["site_2"]["sedated"]["n"] == SUPPRESSED                            # 0 sedated at S2
    ints = [v for v in _walk(r) if isinstance(v, int) and not isinstance(v, bool)]
    assert 8 not in ints and all(v >= 11 for v in ints if v != 100)                          # the small count appears nowhere (100 = n_boot)


def test_nan_rows_are_dropped_and_shapes_checked():
    d, flag, sites = _planted()
    d2 = d.copy()
    d2[:40] = np.nan
    r = stratified_delta(d2, flag, sites, LABELS, n_boot=50, seed=0)
    assert r["strata"]["sedated"]["n"] + r["strata"]["non_sedated"]["n"] == len(d) - 40
    with pytest.raises(ValueError):
        stratified_delta(d[:-1], flag, sites, LABELS)


# ------------------------------------------------------------------------------------------ sedation probe
def test_sedation_probe_detects_an_encoded_flag_and_is_informational():
    rng = np.random.default_rng(0)
    n = 400
    sites = np.repeat(["S1", "S2"], n // 2)
    sed = rng.random(n) < 0.3
    noise = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"qeeg.f{i}" for i in range(5)])
    enc = noise.assign(**{"qeeg.sed": sed * 1.5 + rng.normal(0, 1, n)})
    p = leakage_probes(enc, sites, sedated=sed, site_labels=LABELS)
    s = p["probes"]["sedation"]
    assert s["informational"] and s["estimable"] and s["auroc"] > 0.8 and s["flagged"]
    assert set(s["per_site"]) == {"site_1", "site_2"} and all(v["auroc"] > 0.7 for v in s["per_site"].values())
    assert s["n_sedated"] == int(sed.sum())
    assert not p["any_flagged"]                                                              # informational: never a control failure
    assert sedation_probe(noise, sed, sites, site_labels=LABELS)["auroc"] < 0.62
    assert "sedation" not in leakage_probes(noise, sites)["probes"]                          # opt-in; old callers unchanged


def test_sedation_probe_needs_11_per_class_and_suppresses_the_count():
    rng = np.random.default_rng(1)
    n = 200
    sed = np.zeros(n, bool)
    sed[:7] = True
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    s = sedation_probe(X, sed, np.repeat(["S1", "S2"], n // 2), site_labels=LABELS)
    assert s["estimable"] is False and np.isnan(s["auroc"]) and s["n_sedated"] == SUPPRESSED
    assert all(not v["estimable"] for v in s["per_site"].values())


# ---------------------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def sed_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("sed_sig")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=3)
    J, md, stdout = run_main(paths, d / "out")
    a = rsf.build_parser().parse_args(argv_for(paths, d / "x"))
    return SimpleNamespace(d=d, paths=paths, cohort=cohort, J=J, md=md, stdout=stdout, A=rsf.assemble(a))


def test_report_has_the_sedation_sections(sed_run):
    md = sed_run.md
    for head in ("## Sedation confounding", "### Is sedation already in the baseline?", "### Sedation-stratified Delta",
                 "### Difference of Deltas", "Sedation x EEG interaction variant"):
        assert head in md, head
    assert md.index("## Mandatory controls") < md.index("## Sedation confounding") < md.index("## Circularity audit")
    assert "not estimable" in md or "| sedated |" in md
    assert "non-sedated" in md and "Delta would be about 0 within the non-sedated stratum" in md
    sb = sed_run.J["sedation"]
    assert sb["headline_rung"] == HEADLINE and set(sb["by_scheme"]) == {"loso", "temporal"}
    assert all(set(v) == {"A", "C"} for v in sb["by_scheme"].values())
    assert sb["sedation_interaction_variant"]["run"] is False


def test_baselines_a_and_c_are_reported_as_containing_sedation_features(sed_run):
    sb = sed_run.J["sedation"]
    for b in ("A", "C"):
        v = sb["baseline_sedation_features"][b]
        assert v["sedation_columns"] >= 2 and v["has_on_t0_class_flags"]
    assert "Baselines A and C already contain sedation features" in sb["baseline_adjustment"]
    assert "Baselines A and C already contain sedation features" in sed_run.md


@pytest.mark.parametrize("split", ["loso", "temporal"])
@pytest.mark.parametrize("base", ["A", "C"])
def test_stratified_delta_is_computed_from_the_headline_predictions(sed_run, split, base):
    J = sed_run.J
    v = J["sedation"]["by_scheme"][split][base]
    head = J["ladder"][split]["rungs"][base][HEADLINE]
    sed, non = v["strata"]["sedated"], v["strata"]["non_sedated"]
    assert non["estimable"]
    assert v["overall_delta"] == pytest.approx(head["delta"], abs=1e-4)                     # the same d_i, split in two
    assert non["delta"] < 0 and non["ci_99_within_site"]["hi"] < 0 and v["non_sedated_gain_persists_99"] is True
    assert set(v["per_site"]) == {"site_1", "site_2"}
    if split == "temporal":
        # ~20% of the rows are scored, so the sedated stratum is a few dozen patients: the < 50 rule must say so, not guess
        assert sed["estimable"] is False and "fewer than 50" in sed["not_estimable"] and isinstance(sed["n"], int) and 11 <= sed["n"] < 50
        assert v["difference_sedated_minus_non_sedated"]["estimable"] is False
        return
    assert sed["estimable"]
    assert (sed["n"] * sed["delta"] + non["n"] * non["delta"]) / (sed["n"] + non["n"]) == pytest.approx(head["delta"], abs=1e-4)
    assert sed["delta"] < 0                                                                  # planted signal is in both strata
    assert sed["ci_99_within_site"]["lo"] <= sed["ci_95_within_site"]["lo"]
    df = v["difference_sedated_minus_non_sedated"]
    assert df["estimable"] and df["delta"] == pytest.approx(sed["delta"] - non["delta"], abs=1e-4)
    assert df["ci_99_within_site"]["lo"] <= df["ci_95_within_site"]["lo"] and df["ci_99_within_site"]["hi"] >= df["ci_95_within_site"]["hi"]
    assert all(isinstance(c[k]["delta"], float) for c in v["per_site"].values() for k in ("sedated", "non_sedated"))


def test_strata_sizes_match_the_sedation_flag(sed_run):
    A = sed_run.A
    n_sed = int(A.covariates["sedated"].sum())
    fl = sed_run.J["sedation"]["flag"]
    assert fl["n_sedated"] == n_sed
    for split in ("loso", "temporal"):
        st = sed_run.J["sedation"]["by_scheme"][split]["A"]["strata"]
        # LOSO scores every non-development row once (d_i is undefined only where no primary label is assessable); the temporal holdout scores
        # only the late test rows of each site
        if split == "loso":
            assert 0.7 * n_sed <= st["sedated"]["n"] <= n_sed                       # the 20% development rows are not scored
            assert 0.7 * (len(A.frame) - n_sed) <= st["non_sedated"]["n"] <= len(A.frame) - n_sed
        else:
            assert 11 <= st["sedated"]["n"] < n_sed


def test_sedation_probe_is_in_the_leakage_table(sed_run):
    p = sed_run.J["controls"]["leakage_probes"]["probes"]
    assert set(p) >= {"site", "duration", "sedation"}
    assert p["sedation"]["informational"] is True and p["sedation"]["n_sedated"] == int(sed_run.A.covariates["sedated"].sum())
    assert 0.0 <= p["sedation"]["auroc"] <= 1.0 and set(p["sedation"]["per_site"]) == {"site_1", "site_2"}
    assert "sedation (informational)" in sed_run.md and "sedation, within site_1" in sed_run.md
    assert not sed_run.J["controls"]["leakage_probes"]["any_flagged"] or p["sedation"]["flagged"] in (True, False)


def test_existing_sedative_excluded_control_is_unchanged(sed_run):
    se = sed_run.J["controls"]["by_scheme"]["loso"]["A"]["sedative_excluded"]
    assert se["delta"] < 0 and "n_excluded" in se and "n_kept" in se
    assert "sedative-excluded Delta" in sed_run.md


def test_sedation_sections_are_aggregate_only_and_counts_suppressed(sed_run):
    ids = {str(p) for p in sed_run.cohort["person_id"]} | set(rsf.recording_ids(sed_run.cohort))
    assert_aggregate_only(sed_run.J["sedation"], ids)
    assert_aggregate_only(sed_run.md, ids)
    assert "S0001" not in json.dumps(sed_run.J["sedation"]) and "S0002" not in json.dumps(sed_run.J["sedation"])

    def counts(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from counts(v, k)
        elif isinstance(o, int) and not isinstance(o, bool) and key.startswith("n_") and key not in ("n_boot",):
            yield key, o
        elif isinstance(o, int) and not isinstance(o, bool) and key == "n":
            yield key, o
    seen = list(counts(sed_run.J["sedation"]))
    assert len(seen) > 10 and all(v >= 11 for _, v in seen)


# --------------------------------------------------------------------------------- checkpoint behaviour
def test_sedation_analysis_adds_no_fits_and_a_rerun_is_all_cache_hits(tmp_path, monkeypatch):
    """The sedation sections read only d_i of the ladder results, so a second run with the same step key refits nothing
    (the stored-unit count does not grow) and reproduces the sections exactly. Also runs with --skip-controls."""
    monkeypatch.setenv(checkpoint.ENV_ROOT, str(tmp_path / "ck"))
    monkeypatch.setenv(checkpoint.ENV_VERSION, "test-code-version")
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=8, n_per_site=300)
    extra = ("--splits", "loso", "--skip-controls")
    J1, md1, _ = run_main(paths, tmp_path / "o1", *extra)

    def units():
        return sum(1 for _ in (tmp_path / "ck").rglob("*.pkl"))
    n1 = units()
    assert n1 > 0 and J1["sedation"]["by_scheme"]["loso"]["A"]["strata"]["sedated"]["estimable"]
    assert J1["controls"].get("skipped")                                                     # probes are controls: skipped with them
    J2, md2, _ = run_main(paths, tmp_path / "o1", *extra)           # --out is part of the step key: same directory, same key
    assert units() == n1                                                                     # nothing new was fitted
    assert J2["sedation"] == J1["sedation"] and "## Sedation confounding" in md2


# ------------------------------------------------------------ sedation probe on the CBraMod / MORGOTH representations
def test_sedation_probe_is_run_for_the_embedding_and_morgoth_findings_when_present(tmp_path):
    from test_silver_feasibility_morgoth import write_fake_parts
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=11, n_per_site=300)
    write_fake_parts(tmp_path / "local_only", ms, cohort)
    J, md, _ = run_main(paths, tmp_path / "out", "--splits", "loso")
    c = J["controls"]
    for key, tag in (("leakage_probes_cbramod_frozen", "frozen CBraMod embedding alone"),
                     ("leakage_probes_morgoth_findings", "MORGOTH findings alone")):
        p = c[key]["probes"]["sedation"]
        assert p["informational"] and 0.0 <= p["auroc"] <= 1.0 and set(p["per_site"]) == {"site_1", "site_2"}
        assert f"sedation (informational) ({tag})" in md and f"sedation, within site_1 ({tag})" in md
    assert c["leakage_probes"]["probes"]["sedation"]["estimable"]
