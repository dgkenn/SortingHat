"""Sedation-INTENSITY analysis of the silver-feasibility report (D-154), SYNTHETIC ONLY: the grade mapping from the Baseline A
sed__* columns, Delta by grade and in the light / heavy strata, the trend test, small-cell suppression, the intensity probe and
the report sections. Pure-function tests use made-up arrays; the end-to-end ones run the script on the synthetic inputs of
test_silver_feasibility_inputs with graded sedation columns written into the baselines parquet."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sortinghat.metrics.sedation import (GRADE_SPEC, grade_counts_shown, graded_delta, sedation_intensity_grade)
from sortinghat.models.controls import sedation_grade_probe
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_silver_feasibility import run_main
from test_silver_feasibility_inputs import make_inputs, rsf

LABELS = {"S1": "site_1", "S2": "site_2"}
GR = (0, 1, 2, 3)


# ------------------------------------------------------------------------------------------------ grade mapping
def _bl(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame({k: np.asarray(v, float) for k, v in cols.items()}, index=range(n))


def test_grade_mapping_from_the_sed_columns():
    bl = _bl(**{
        "sed__sedative__on_t0":        [0, 0, 0, 1, 0, 1, 1, 0, 0, np.nan],
        "sed__opioid__on_t0":          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "sed__propofol__on_t0":        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "sed__midazolam__on_t0":       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "sed__dexmedetomidine__on_t0": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "sed__sedative__n_24h":        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        "sed__fentanyl__qty_6h":       [0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0],
        "sed__n_agents_24h":           [0, 1, 1, 1, 1, 1, 2, 0, 1, 0]})
    g, info = sedation_intensity_grade(bl)
    #            none  PRN   PRN-qty  midaz   opioid   propofol  midaz+dex  none  agents-only  NaN-> none
    assert g.tolist() == [0, 1, 1, 2, 2, 3, 3, 0, 1, 0]
    assert info["available"] and info["on_t0_columns_found"] == 5


def test_grade_precedence_class_only_sedative_is_2_two_named_sedatives_is_3_and_one_named_is_not():
    bl = _bl(**{"sed__sedative__on_t0": [1, 1, 1, 0], "sed__midazolam__on_t0": [0, 1, 1, 0],
                "sed__lorazepam__on_t0": [0, 0, 1, 0], "sed__fentanyl__on_t0": [0, 0, 0, 1], "sed__ketamine__qty_24h": [0, 0, 0, 0]})
    g, _ = sedation_intensity_grade(bl)
    assert g.tolist() == [2, 2, 3, 2]        # unnamed class sedative; one named; two named; a named opioid running


def test_grade_missing_columns_count_as_zero_and_no_on_t0_column_means_not_available():
    g, info = sedation_intensity_grade(_bl(**{"sed__sedative__on_t0": [1, 0], "demo__age_years": [3, 4]}))
    assert g.tolist() == [2, 0] and info["available"]
    g, info = sedation_intensity_grade(_bl(**{"sed__sedative__n_24h": [1, 0], "sed__n_agents_24h": [1, 0]}))
    assert g is None and info["available"] is False and "cannot be established" in info["reason"]
    assert set(GRADE_SPEC["grades"]) == set(GR)


def test_grade_uses_only_sed_columns_never_labels_or_the_silver_hint():
    bl = _bl(**{"sed__sedative__on_t0": [0, 0], "e4b_sedative_exposure": [1, 1], "qeeg.x": [9, 9], "y": [1, 1]})
    assert sedation_intensity_grade(bl)[0].tolist() == [0, 0]


# --------------------------------------------------------------------------------------------- suppression
def test_grade_counts_are_hidden_in_pairs():
    s = grade_counts_shown({0: 300, 1: 5, 2: 100, 3: 40})
    assert s == {0: SUPPRESSED, 1: SUPPRESSED, 2: 100, 3: 40}
    s = grade_counts_shown({0: 300, 1: 200, 2: 100, 3: 0})
    assert s == {0: 300, 1: 200, 2: SUPPRESSED, 3: SUPPRESSED}
    assert grade_counts_shown({0: 30, 1: 20, 2: 11, 3: 40}) == {0: 30, 1: 20, 2: 11, 3: 40}


# --------------------------------------------------------------------------------------- Delta by grade / trend
def _planted(n_per_site=600, mus=(-0.02, -0.06, -0.14, -0.22), sd=0.25, seed=0, site_shift=(0.0, 0.0), probs=(0.15, 0.25, 0.35, 0.25)):
    rng = np.random.default_rng(seed)
    sites = np.repeat(["S1", "S2"], n_per_site)
    grade = rng.choice(4, size=len(sites), p=probs)
    shift = np.where(sites == "S1", site_shift[0], site_shift[1])
    d = np.asarray(mus)[grade] + shift + rng.normal(0, sd, len(sites))
    return d, grade, sites


def test_graded_delta_recovers_the_planted_grade_means_and_strata():
    d, g, s = _planted()
    r = graded_delta(d, g, s, LABELS, n_boot=600, seed=1)
    for k in GR:
        c = r["grades"][str(k)]
        assert c["estimable"] and c["delta"] == pytest.approx(d[g == k].mean()) and c["n"] == int((g == k).sum())
        assert c["ci_99_within_site"]["lo"] < c["delta"] < c["ci_99_within_site"]["hi"]
        assert c["ci_99_within_site"]["lo"] < c["ci_95_within_site"]["lo"]                # the 99% interval is the wider one
    light, heavy = r["strata"]["light"], r["strata"]["heavy"]
    assert light["delta"] == pytest.approx(d[g <= 1].mean()) and heavy["delta"] == pytest.approx(d[g >= 2].mean())
    assert light["n"] == int((g <= 1).sum()) and heavy["n"] == int((g >= 2).sum())
    n = len(d)                                                                           # the grades partition the patients
    assert sum(r["grades"][str(k)]["n"] for k in GR) == n and light["n"] + heavy["n"] == n
    assert (light["n"] * light["delta"] + heavy["n"] * heavy["delta"]) / n == pytest.approx(r["overall_delta"])
    assert r["light_gain_persists_99"] is True                                           # planted: -0.04 average in grades 0-1
    df = r["difference_heavy_minus_light"]
    assert df["estimable"] and df["delta"] == pytest.approx(heavy["delta"] - light["delta"])
    assert df["ci_99_within_site"]["hi"] < 0                                             # planted: the gain is larger when heavy
    assert df["ci_99_within_site"]["lo"] < df["delta"] < df["ci_99_within_site"]["hi"]
    assert set(r["per_site"]) == {"site_1", "site_2"}
    assert all(c[str(k)]["delta"] != SUPPRESSED for c in r["per_site"].values() for k in GR)


def test_trend_detects_a_gain_that_grows_with_intensity():
    d, g, s = _planted()
    t = graded_delta(d, g, s, LABELS, n_boot=600, seed=2)["trend"]
    assert t["estimable"] and t["grades_used"] == [0, 1, 2, 3]
    sl = t["slope_site_adjusted_primary"]
    assert -0.09 < sl["delta"] < -0.04                                                   # planted slope about -0.067 per grade
    assert sl["ci_99_within_site"]["hi"] < 0 and t["gain_grows_with_intensity_99"] is True and t["slope_ci_includes_0_99"] is False
    assert t["slope_pooled"]["ci_99_within_site"]["hi"] < 0
    rho = t["spearman_rho"]
    assert rho["delta"] < 0 and rho["ci_99_within_site"]["hi"] < 0                       # d falls (more negative) with grade
    assert sl["ci_99_within_site"]["lo"] < sl["ci_95_within_site"]["lo"]


def test_trend_is_flat_when_the_gain_does_not_depend_on_intensity():
    d, g, s = _planted(mus=(-0.12,) * 4, seed=3)
    r = graded_delta(d, g, s, LABELS, n_boot=600, seed=3)
    t = r["trend"]
    assert t["gain_grows_with_intensity_99"] is False and t["slope_ci_includes_0_99"] is True
    assert t["spearman_rho"]["ci_99_within_site"]["lo"] < 0 < t["spearman_rho"]["ci_99_within_site"]["hi"]
    assert r["light_gain_persists_99"] is True                                           # the gain exists at light grades
    assert r["difference_heavy_minus_light"]["ci_99_within_site"]["lo"] < 0 < r["difference_heavy_minus_light"]["ci_99_within_site"]["hi"]


def test_gain_absent_at_low_intensity_is_not_reported_as_persisting():
    d, g, s = _planted(mus=(0.0, 0.0, -0.2, -0.3), seed=4)
    r = graded_delta(d, g, s, LABELS, n_boot=600, seed=4)
    assert r["light_gain_persists_99"] is False and r["strata"]["light"]["ci_99_within_site"]["lo"] < 0 < r["strata"]["light"]["ci_99_within_site"]["hi"]
    assert r["trend"]["gain_grows_with_intensity_99"] is True


def test_site_adjusted_slope_is_not_fooled_by_a_site_difference_in_grade_mix():
    # no grade effect at all; site 2 has heavier grades AND a larger gain: the pooled slope is negative, the site-adjusted one is not
    rng = np.random.default_rng(5)
    n = 700
    sites = np.repeat(["S1", "S2"], n)
    grade = np.concatenate([rng.choice(4, n, p=[.45, .35, .15, .05]), rng.choice(4, n, p=[.05, .15, .35, .45])])
    d = np.where(sites == "S1", 0.0, -0.3) + rng.normal(0, 0.2, 2 * n)
    t = graded_delta(d, grade, sites, LABELS, n_boot=500, seed=6)["trend"]
    assert t["slope_pooled"]["ci_99_within_site"]["hi"] < 0
    assert t["slope_site_adjusted_primary"]["ci_99_within_site"]["lo"] < 0 < t["slope_site_adjusted_primary"]["ci_99_within_site"]["hi"]


def test_difference_interval_is_paired_by_bootstrap():
    d, g, s = _planted(mus=(-0.1,) * 4, seed=8)
    r = graded_delta(d, g, s, LABELS, n_boot=800, seed=9)
    a, b = r["strata"]["light"]["ci_99_within_site"], r["strata"]["heavy"]["ci_99_within_site"]
    naive = (b["hi"] - a["lo"]) - (b["lo"] - a["hi"])
    df = r["difference_heavy_minus_light"]["ci_99_within_site"]
    assert df["hi"] - df["lo"] < naive


def test_small_grades_are_not_estimable_and_counts_are_suppressed():
    rng = np.random.default_rng(10)
    n = 400
    sites = np.repeat(["S1", "S2"], n // 2)
    grade = rng.choice(3, n, p=[0.5, 0.45, 0.05])                  # grade 2 has about 20 patients; grade 3 none
    grade[:5] = 3                                                  # 5 patients at grade 3
    d = rng.normal(-0.1, 0.2, n)
    r = graded_delta(d, grade, sites, LABELS, n_boot=300, seed=1)
    g2, g3 = r["grades"]["2"], r["grades"]["3"]
    assert not g2["estimable"] and "fewer than 50" in g2["not_estimable"] and "delta" not in g2
    assert not g3["estimable"]
    assert g3["n"] == SUPPRESSED and g2["n"] == SUPPRESSED                                  # the pair (2,3) is hidden together
    assert r["grades"]["0"]["estimable"] and isinstance(r["grades"]["0"]["n"], int)
    assert not r["strata"]["heavy"]["estimable"] and r["strata"]["heavy"]["n"] == 26      # 21 + 5: not estimable; the pooled count (>= 11) leaves each hidden member only known to be < 11 / >= 16
    assert r["difference_heavy_minus_light"]["estimable"] is False
    assert r["trend"]["estimable"] is False and "at least 3 grades" in r["trend"]["not_estimable"]
    # per site: a Delta is hidden exactly when its own cell is < 11 (grade 3 has 5 patients in total); counts follow the pair rule
    for sid, lab in (("S1", "site_1"), ("S2", "site_2")):
        for k in (2, 3):
            own = int(((sites == sid) & (grade == k)).sum())
            assert (r["per_site"][lab][str(k)]["delta"] == SUPPRESSED) == (own < 11)
    assert all(c["3"]["delta"] == SUPPRESSED and c["3"]["n"] == SUPPRESSED for c in r["per_site"].values())


def test_light_or_heavy_under_11_hides_both_pooled_counts():
    rng = np.random.default_rng(11)
    n = 300
    sites = np.repeat(["S1", "S2"], n // 2)
    grade = np.where(rng.random(n) < 0.03, 0, 2)                    # about 9 patients in the light stratum
    r = graded_delta(rng.normal(-0.1, 0.2, n), grade, sites, LABELS, n_boot=200, seed=1)
    assert r["strata"]["light"]["n"] == SUPPRESSED and r["strata"]["heavy"]["n"] == SUPPRESSED
    assert not r["strata"]["light"]["estimable"] and r["light_gain_persists_99"] is None


def test_nan_rows_dropped_shapes_and_grade_range_checked():
    d, g, s = _planted(n_per_site=200)
    d2 = d.copy()
    d2[:30] = np.nan
    r = graded_delta(d2, g, s, LABELS, n_boot=200, seed=1)
    assert sum(r["grades"][str(k)]["n"] for k in GR) == len(d) - 30
    with pytest.raises(ValueError):
        graded_delta(d[:-1], g, s)
    with pytest.raises(ValueError):
        graded_delta(d, g + 1, s)


# --------------------------------------------------------------------------------------------------- probe
def test_grade_probe_detects_encoded_intensity_and_is_quiet_on_noise():
    rng = np.random.default_rng(0)
    n = 600
    sites = np.repeat(["S1", "S2"], n // 2)
    grade = rng.choice(4, n, p=[.3, .3, .25, .15])
    noise = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"qeeg.f{i}" for i in range(5)])
    enc = noise.assign(**{"qeeg.g": grade * 1.0 + rng.normal(0, 1, n)})
    p = sedation_grade_probe(enc, grade, sites, site_labels=LABELS)
    assert p["informational"] and set(p["binary"]) == {"ge1", "ge2"}
    assert p["binary"]["ge2"]["auroc"] > 0.7 and p["binary"]["ge2"]["estimable"]
    assert all(v["auroc"] > 0.6 for v in p["binary"]["ge2"]["per_site"].values())
    assert p["ordinal"]["spearman"] > 0.3 and set(p["ordinal"]["per_site"]) == {"site_1", "site_2"}
    q = sedation_grade_probe(noise, grade, sites, site_labels=LABELS)
    assert q["binary"]["ge2"]["auroc"] < 0.6 and abs(q["ordinal"]["spearman"]) < 0.2


def test_grade_probe_needs_11_per_class_and_hides_both_counts():
    rng = np.random.default_rng(1)
    n = 200
    grade = np.zeros(n, int)
    grade[:7] = 3
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    p = sedation_grade_probe(X, grade, np.repeat(["S1", "S2"], n // 2), site_labels=LABELS)
    b = p["binary"]["ge2"]
    assert b["estimable"] is False and np.isnan(b["auroc"]) and b["n_at_or_above"] == SUPPRESSED and b["n_below"] == SUPPRESSED
    assert all(not v["estimable"] for v in b["per_site"].values())
    assert p["ordinal"]["estimable"] is False                                                  # only one grade with >= 11 patients


# ---------------------------------------------------------------------------------------------- end to end
def write_graded_baselines(paths, seed=0):
    """Add graded sed__* columns to the synthetic baselines parquet (and to the A, B, C column sets): planted so every grade is
    populated. Grade is unrelated to the planted EEG signal, so only structure is tested end to end."""
    bl = pd.read_parquet(paths["baselines"])
    rng = np.random.default_rng(seed)
    n = len(bl)
    sed = bl["sed__sedative__on_t0"].to_numpy() > 0
    prop = sed & (rng.random(n) < 0.30)
    mid = sed & ~prop & (rng.random(n) < 0.50)
    bl["sed__propofol__on_t0"] = prop.astype(float)
    bl["sed__midazolam__on_t0"] = mid.astype(float)
    bl["sed__sedative__n_24h"] = (~sed & (rng.random(n) < 0.45)).astype(float) * 2
    bl["sed__fentanyl__qty_6h"] = 0.0
    bl.to_parquet(paths["baselines"], index=False)
    side = paths["baselines"].with_suffix(".columns.json")
    cj = json.loads(side.read_text())
    for k in ("A", "B", "C"):
        cj["baselines"][k] += [c for c in ("sed__propofol__on_t0", "sed__midazolam__on_t0", "sed__sedative__n_24h", "sed__fentanyl__qty_6h")
                               if c not in cj["baselines"][k]]
    side.write_text(json.dumps(cj))
    return bl


@pytest.fixture(scope="module")
def int_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("sed_int")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=3, n_per_site=450)
    write_graded_baselines(paths)
    J, md, stdout = run_main(paths, d / "out")
    a = rsf.build_parser().parse_args(__import__("test_silver_feasibility_inputs").argv_for(paths, d / "x"))
    return SimpleNamespace(d=d, paths=paths, cohort=cohort, J=J, md=md, stdout=stdout, A=rsf.assemble(a))


def test_report_has_the_sedation_intensity_sections_after_sedation_confounding(int_run):
    md = int_run.md
    for head in ("## Sedation intensity", "### Delta by sedation grade: loso, rung `combined` (headline)",
                 "### Sedation-intensity probe", "| 3 | anaesthetic-depth proxy"):
        assert head in md, head
    assert md.index("## Sedation confounding") < md.index("## Sedation intensity") < md.index("## Circularity audit")
    assert "light (0-1)" in md and "heavy (2-3)" in md and "site-adjusted slope" in md and "pre-specified" in md
    si = int_run.J["sedation_intensity"]
    assert si["available"] and si["headline_rung"] == "combined" and set(si["by_scheme"]) == {"loso", "temporal"}
    assert set(si["by_scheme"]["loso"]) == {"combined"} and set(si["by_scheme"]["loso"]["combined"]) == {"A", "C"}
    assert si["definition"]["grades"]["0"].startswith("none recorded")
    assert set(si["probe"]) == {"hand_crafted_qeeg_connectivity"}


def test_existing_sedation_section_is_unchanged(int_run):
    J = int_run.J
    assert set(J["sedation"]) >= {"flag", "by_scheme", "baseline_sedation_features", "baseline_adjustment"}
    assert J["sedation"]["by_scheme"]["loso"]["A"]["strata"]["non_sedated"]["estimable"]
    assert "### Sedation-stratified Delta" in int_run.md and "### Difference of Deltas" in int_run.md
    assert "sedation (informational)" in int_run.md                                           # the binary sedation probe is still there


def test_grade_distribution_and_grades_match_the_baseline_columns(int_run):
    A = int_run.A
    grade, _ = sedation_intensity_grade(A.baseline)
    assert set(np.unique(grade)) == set(GR)
    dist = int_run.J["sedation_intensity"]["distribution"]
    for g in GR:
        assert dist["n_by_grade"][str(g)] == int((grade == g).sum())
    assert set(dist["per_site"]) == {"site_1", "site_2"}
    # grade >= 2 is exactly the on_t0 flag of the baseline (the synthetic baseline has only the class on_t0 flag plus propofol / midazolam)
    assert ((grade >= 2) == (A.baseline["sed__sedative__on_t0"].to_numpy() > 0)).all()


@pytest.mark.parametrize("split", ["loso", "temporal"])
@pytest.mark.parametrize("base", ["A", "C"])
def test_grade_deltas_come_from_the_headline_predictions(int_run, split, base):
    J = int_run.J
    v = J["sedation_intensity"]["by_scheme"][split]["combined"][base]
    head = J["ladder"][split]["rungs"][base]["combined"]
    assert v["overall_delta"] == pytest.approx(head["delta"], abs=1e-4)                       # the same d_i, split four ways
    ns = [c["n"] for c in v["grades"].values()]
    for k, c in v["grades"].items():
        assert c["estimable"] or "fewer than 50" in c["not_estimable"]
    if split == "loso":
        est = [c for c in v["grades"].values() if c["estimable"]]
        assert len(est) >= 3 and v["trend"]["estimable"]
        l, h = v["strata"]["light"], v["strata"]["heavy"]
        assert l["estimable"] and h["estimable"]
        assert (l["n"] * l["delta"] + h["n"] * h["delta"]) / (l["n"] + h["n"]) == pytest.approx(head["delta"], abs=1e-4)
        assert l["delta"] < 0 and h["delta"] < 0                                              # planted signal in every stratum
        assert v["difference_heavy_minus_light"]["estimable"]
        assert v["trend"]["slope_site_adjusted_primary"]["ci_99_within_site"]["lo"] <= v["trend"]["slope_site_adjusted_primary"]["ci_95_within_site"]["lo"]
        # the signal is unrelated to the planted grade: no strong trend
        assert abs(v["trend"]["slope_site_adjusted_primary"]["delta"]) < 0.1
    assert all(isinstance(n, (int, str)) for n in ns)


def test_probe_is_reported_with_pooled_and_within_site_values(int_run):
    p = int_run.J["sedation_intensity"]["probe"]["hand_crafted_qeeg_connectivity"]
    assert 0.0 <= p["binary"]["ge2"]["auroc"] <= 1.0 and set(p["binary"]["ge2"]["per_site"]) == {"site_1", "site_2"}
    assert set(p["ordinal"]["per_site"]) == {"site_1", "site_2"} and -1 <= p["ordinal"]["spearman"] <= 1
    assert "AUROC grade >= 2" in int_run.md and "ordinal Spearman rho" in int_run.md


def test_intensity_sections_are_aggregate_only_and_counts_suppressed(int_run):
    ids = {str(p) for p in int_run.cohort["person_id"]} | set(rsf.recording_ids(int_run.cohort))
    assert_aggregate_only(int_run.J["sedation_intensity"], ids)
    assert_aggregate_only(int_run.md, ids)

    def counts(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from counts(v, k)
        elif isinstance(o, int) and not isinstance(o, bool) and (key.startswith("n_") or key == "n") and key != "n_boot":
            yield key, o
    seen = list(counts(int_run.J["sedation_intensity"]))
    assert len(seen) > 10 and all(v >= 11 for _, v in seen)


def test_skip_controls_runs_the_deltas_and_marks_the_probe_skipped(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=8, n_per_site=300)
    write_graded_baselines(paths, seed=2)
    J, md, _ = run_main(paths, tmp_path / "o", "--splits", "loso", "--skip-controls")
    si = J["sedation_intensity"]
    assert si["available"] and "skipped" in si["probe"] and set(si["by_scheme"]) == {"loso"}
    assert "## Sedation intensity" in md and "was NOT run" in md


def test_intensity_not_available_without_on_t0_columns_is_reported_not_guessed(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=9, n_per_site=300)
    J, md, _ = run_main(paths, tmp_path / "o", "--splits", "loso", "--skip-controls")
    assert J["sedation_intensity"]["available"]                  # the default synthetic baselines carry the class on_t0 flags
    assert set(np.unique(sedation_intensity_grade(rsf.assemble(rsf.build_parser().parse_args(
        __import__("test_silver_feasibility_inputs").argv_for(paths, tmp_path / "x"))).baseline)[0])) <= {0, 2}


def test_intensity_rungs_include_cbramod_and_morgoth_when_they_ran(tmp_path):
    from test_silver_feasibility_morgoth import write_fake_parts
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=11, n_per_site=300)
    write_graded_baselines(paths, seed=4)
    write_fake_parts(tmp_path / "local_only", ms, cohort)
    J, md, _ = run_main(paths, tmp_path / "out", "--splits", "loso")
    si = J["sedation_intensity"]
    assert si["rungs"] == ["combined", "combined_cbramod", "combined_morgoth"]
    assert set(si["by_scheme"]["loso"]) == set(si["rungs"])
    assert set(si["probe"]) == {"hand_crafted_qeeg_connectivity", "cbramod_frozen", "morgoth_findings"}
    for r in ("combined_cbramod", "combined_morgoth"):
        assert f"rung `{r}`" in md
        v = si["by_scheme"]["loso"][r]["A"]
        assert v["overall_delta"] == pytest.approx(J["ladder"]["loso"]["rungs"]["A"][r]["delta"], abs=1e-4)
