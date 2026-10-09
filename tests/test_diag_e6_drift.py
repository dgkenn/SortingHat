"""scripts/diag_e6_drift.py on SYNTHETIC inputs: a planted late-period ascertainment change is visible in the aggregate tables, the
stable comparison label is not, every cell obeys the small-cell rules, and nothing record-level (ids, dates) is emitted."""
import contextlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.labels import extract as ex
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_silver_feasibility_inputs import argv_for, load_script, make_inputs  # noqa: F401  (tests/ on sys.path)

dg = load_script("diag_e6_drift")

BASE = pd.Timestamp("2024-03-01 08:00:00")
N_SITE = 400
SITES = ("S0001", "S0002")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------------------------- fixtures
def planted(seed=0):
    """Two sites x 400 people with t0 increasing by 6 h. Blood-culture data exist almost ONLY in the last 20% of each site: 75% of
    those people have a culture row and 60% of the cultured have antibiotics for 4 days plus a lactate (an E6 positive); before
    that only every 40th person has a (complete, E6-positive) culture work-up. E5 (glucose < 50) is planted at a constant 1 in 3."""
    rng = np.random.default_rng(seed)
    meas, drug, rows = [], [], []
    pid = 5_000_000
    for site in SITES:
        for i in range(N_SITE):
            pid += 1
            t0 = BASE + pd.Timedelta(hours=6 * i) + pd.Timedelta(minutes=int(SITES.index(site)) * 7)
            rows.append({"person_id": pid, "t0": t0, "SiteID": site})

            def lab(h, name, value=np.nan, unit=None, vtext=None):
                meas.append({"person_id": pid, "measurement_datetime": t0 + pd.Timedelta(hours=h), "measurement_source_value": name,
                             "value_as_number": value, "unit_source_value": unit, "measurement_concept_id": 0,
                             "measurement_source_concept_id": 0, "value_source_value": vtext, "value_as_concept_id": 0,
                             "unit_concept_id": 0})

            def workup(sepsis):
                lab(-5, "Blood culture", vtext="No growth")
                if sepsis:
                    lab(-3, "LACTATE, WHOLE BLOOD", 3.4, "mmol/L")
                    for d in range(4):
                        drug.append({"person_id": pid, "drug_exposure_start_datetime": t0 + pd.Timedelta(hours=-4 + 24 * d),
                                     "drug_exposure_end_datetime": t0 + pd.Timedelta(hours=-2 + 24 * d),
                                     "drug_source_value": "VANCOMYCIN 1 G IV", "route_source_value": "IV", "drug_concept_id": 0})
            if i % 3 == 0:
                lab(-2, "GLUCOSE", 30.0, "mg/dL")
            if i >= int(0.8 * N_SITE):
                if rng.random() < 0.75:
                    workup(rng.random() < 0.6)
            elif i % 40 == 0:
                workup(True)
    cols = ex.COLUMNS
    tables = {"omop_measurement": pd.DataFrame(meas, columns=cols["measurement"]),
              "omop_drug_exposure": pd.DataFrame(drug, columns=cols["drug_exposure"]),
              "omop_condition_occurrence": pd.DataFrame(columns=cols["condition_occurrence"]),
              "omop_procedure_occurrence": pd.DataFrame(columns=cols["procedure_occurrence"]),
              "omop_observation": pd.DataFrame(columns=cols["observation"]),
              "omop_visit_occurrence": pd.DataFrame(columns=cols["visit_occurrence"]),
              "omop_concept": pd.DataFrame(columns=cols["concept"])}
    return tables, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def drift():
    tables, cohort = planted()
    res = ex.extract_silver(tables, cohort)
    Y = res.labels[[c for c in ("E1", "E2", "E4a", "E5", "E6", "E7")]].astype("float").reindex([str(p) for p in cohort["person_id"]])
    Y = Y.reset_index(drop=True)
    sites = cohort["SiteID"].to_numpy(object)
    times = cohort["t0"].to_numpy("datetime64[us]")
    tb = dg.build_groupings(sites, times, 5, 0.2)["temporal"][1]
    e6 = (Y["E6"] == 1).to_numpy()
    metrics, quants, _ = dg.data_metrics(tables, cohort, sites, tb, e6, Y["E6"].notna().to_numpy(), Y)
    di = dg.DiagInput(sites, times, Y, {"S0001": "site_1", "S0002": "site_2"}, 0.2, metrics=metrics, quants=quants)
    rep = dg.diagnose(di, 5)
    rep["drift_ratio_test_over_train"] = dg.drift_ratios(rep)
    return rep, cohort, Y


def cell(rep, name, scheme, site, b):
    return rep["tables"][name][scheme][site][b]


# ------------------------------------------------------------------------------------- the planted drift
def test_the_planted_e6_concentration_is_visible_in_the_temporal_bins(drift):
    rep, _c, Y = drift
    assert 60 < int((Y["E6"] == 1).sum()) < 90
    # per site the early positives (8) are below 11, so the row is hidden entirely (complementary suppression of the test cell)
    for site in ("site_1", "site_2"):
        assert cell(rep, "label.E6.positive", "temporal", site, "train")["n"] == SUPPRESSED
        assert cell(rep, "label.E6.positive", "temporal", site, "test")["n"] == SUPPRESSED
    # pooled: both cells are shown (none or both site cells hidden), and the concentration is plain
    tr = cell(rep, "label.E6.positive", "temporal", dg.POOLED, "train")
    te = cell(rep, "label.E6.positive", "temporal", dg.POOLED, "test")
    assert isinstance(tr["share"], float) and tr["share"] < 0.05 and isinstance(te["share"], float) and 0.3 < te["share"] < 0.6
    assert rep["drift_ratio_test_over_train"]["label.E6.positive"][dg.POOLED] > 8
    # quantile bins: the first four per-site bins hold fewer than 11 positives (hidden), the last is shown
    q = rep["tables"]["label.E6.positive"]["quantile"]["site_1"]
    assert isinstance(q["Q5"]["share"], float) and all(q[b]["share"] == SUPPRESSED for b in ("Q1", "Q2", "Q3", "Q4"))


def test_data_availability_explains_it(drift):
    rep, *_ = drift
    name = "avail.blood_culture_drawn.in_window"
    tr, te = (cell(rep, name, "temporal", dg.POOLED, b) for b in ("train", "test"))
    assert isinstance(tr["share"], float) and tr["share"] < 0.05 and isinstance(te["share"], float) and 0.6 < te["share"] < 0.9
    assert rep["drift_ratio_test_over_train"][name][dg.POOLED] > 15
    # the anchor's own culture leg gives the same picture as the raw data availability
    assert cell(rep, "comp.E6.E6_blood_culture", "temporal", dg.POOLED, "test")["n"] == te["n"]
    # the conditional prevalence of E6 among the cultured is the planted 60% in the late bin: ascertainment, not a rate change
    cult, both = te["n"], cell(rep, "comp.E6.E6_rule_after_E7_exclusion", "temporal", dg.POOLED, "test")["n"]
    assert 0.4 < both / cult < 0.8
    # E6 positives all carry the lactate leg
    assert cell(rep, "e6pos.E6_lactate", "temporal", dg.POOLED, "test")["share"] in (1.0, SUPPRESSED)
    # antibiotics follow the cultures; raw table coverage of the measurement table (glucose rows) is flat
    ab = cell(rep, "avail.antimicrobial.in_window", "temporal", dg.POOLED, "test")["share"]
    assert isinstance(ab, float) and ab < te["share"]
    assert "avail.raw_measurement.in_window" in rep["tables"] and "vocab.blood_culture.wording_not_seen_in_train" in rep["tables"]


def test_timing_and_conditional_tables(drift):
    rep, *_ = drift
    q = rep["quantile_tables"]["timing.blood_culture_nearest_hours"]["temporal"][dg.POOLED]
    assert q["test"]["q50"] == pytest.approx(-5.0) and q["train"]["q50"] == pytest.approx(-5.0)   # culture planted at t0 - 5 h
    site = rep["quantile_tables"]["timing.blood_culture_nearest_hours"]["temporal"]["site_1"]["train"]
    assert site["q50"] == SUPPRESSED                                                          # only 8 early cultures at one site: fewer than 11
    a = rep["quantile_tables"]["timing.antimicrobial_first_start_hours"]["temporal"][dg.POOLED]["test"]
    assert a["q50"] == pytest.approx(-4.0)
    c = cell(rep, "label.E6.positive_among_culture_in_window", "quantile", "site_1", "Q5")
    assert isinstance(c["share"], float) and 0.4 < c["share"] < 0.8 and isinstance(c["den"], int)    # planted 60% among the cultured
    # (the temporal cells are hidden: the early cultures, all E6-positive, give a complement of 0 and a site count below 11)
    assert cell(rep, "label.E6.positive_among_culture_in_window", "temporal", "site_1", "train")["den"] == SUPPRESSED
    c = cell(rep, "avail.blood_culture_drawn.in_window_among_extract_window", "quantile", "site_1", "Q5")
    assert c["share"] == SUPPRESSED                                                           # every culture is inside the window: complement 0


def test_the_stable_comparison_label_shows_no_drift(drift):
    rep, *_ = drift
    a = cell(rep, "label.E5.positive", "temporal", dg.POOLED, "train")["share"]
    b = cell(rep, "label.E5.positive", "temporal", dg.POOLED, "test")["share"]
    assert isinstance(a, float) and isinstance(b, float) and abs(a - 1 / 3) < 0.05 and abs(b - 1 / 3) < 0.08
    assert 0.8 < rep["drift_ratio_test_over_train"]["label.E5.positive"][dg.POOLED] < 1.25
    assert set(rep["drift_ratio_test_over_train"]["label.E6.positive"]) == {dg.POOLED}   # per-site shares hidden: no per-site ratio is invented


def test_rederived_labels_agree_with_the_label_file(drift):
    rep, *_ = drift
    for lab in ("E6", "E5"):
        for k in ("csv_positive_not_rederived", "rederived_positive_not_csv"):
            c = cell(rep, f"consistency.{lab}.{k}", "temporal", dg.POOLED, "test")
            assert c["n"] == SUPPRESSED                                              # zero disagreements (< 11)


# ------------------------------------------------------------------------------------ disclosure control
def walk_cells(o):
    if isinstance(o, dict):
        if {"n", "den", "share"} <= set(o):
            yield o
        else:
            for v in o.values():
                yield from walk_cells(v)


def test_every_cell_obeys_the_small_cell_rules(drift):
    rep, *_ = drift
    n = 0
    for cellv in walk_cells(rep["tables"]):
        n += 1
        if cellv["n"] == SUPPRESSED:
            assert cellv["share"] == SUPPRESSED and cellv["why"] in ("low", "high", "linked")
        else:
            assert cellv["n"] >= 11 and cellv["den"] != SUPPRESSED and cellv["den"] - cellv["n"] >= 11   # the complement is hidden too
        assert cellv["den"] == SUPPRESSED or cellv["den"] >= 11
    assert n > 500


def test_pooled_cell_never_reveals_a_lone_hidden_site_cell(drift):
    rep, *_ = drift
    for name, t in rep["tables"].items():
        for scheme in ("temporal", "quantile"):
            for b, pooled in t[scheme][dg.POOLED].items():
                hidden_sites = sum(t[scheme][s][b]["n"] == SUPPRESSED for s in ("site_1", "site_2"))
                assert not (pooled["n"] != SUPPRESSED and hidden_sites == 1), (name, scheme, b)


def test_one_hidden_cell_hides_the_smallest_other_cell_of_its_row():
    assert dg._flags([3, 50, 60], [100, 100, 100]) == ["low", "linked", None]
    assert dg._flags([20, 50, 60], [100, 100, 100]) == [None, None, None]            # nothing hidden: nothing extra hidden
    assert dg._flags([20, 95, 60], [100, 100, 100]) == ["linked", "high", None]      # complement 5 < 11: hidden, and its partner too
    assert dg._flags([3, 5, 60], [100, 100, 100]) == ["low", "low", None]            # two hidden already
    cells = dg._cells([3, 50, 60], [100, 100, 100], dg._flags([3, 50, 60], [100, 100, 100]))
    assert [c["n"] for c in cells] == [SUPPRESSED, SUPPRESSED, 60] and cells[0]["why"] == "low" and "why" not in cells[2]


def test_a_restricted_denominator_is_protected_like_a_count():
    """E6 positives per site x bin used as a denominator: 14 at site_1, 3 at site_2 and 17 pooled would reveal 3 by subtraction."""
    n1, n2 = 200, 100
    sites = np.array(["a"] * n1 + ["b"] * n2, dtype=object)
    times = np.array([np.datetime64("2024-01-01") + np.timedelta64(i, "h") for i in list(range(n1)) + list(range(n2))], dtype="datetime64[us]")
    den = np.zeros(n1 + n2, bool)
    den[:14] = True                                  # site a, early: 14 in the denominator
    den[n1:n1 + 3] = True                            # site b, early: 3
    num = den.copy()
    num[:2] = False
    Y = pd.DataFrame({"E6": np.zeros(n1 + n2)})
    di = dg.DiagInput(sites, times, Y, {"a": "site_1", "b": "site_2"}, 0.2, metrics=[dg.Metric("x.restricted", num, den)])
    rep = dg.diagnose(di, 5)
    t = rep["tables"]["x.restricted"]["quantile"]
    assert t["site_2"]["Q1"]["den"] == SUPPRESSED                                    # 3 < 11
    assert t["site_1"]["Q1"]["den"] == SUPPRESSED or t[dg.POOLED]["Q1"]["den"] == SUPPRESSED   # never both shown with site_2 hidden
    assert t[dg.POOLED]["Q1"]["den"] == SUPPRESSED                                   # exactly one site hidden -> pooled hidden
    for site in ("site_1", "site_2", dg.POOLED):
        assert t[site]["Q1"]["n"] == SUPPRESSED                                      # a hidden denominator hides the numerator too


def test_quantile_bins_are_ordinal_and_reveal_nothing_but_the_index():
    sites = np.array(["a"] * 10 + ["b"] * 5, dtype=object)
    times = np.array([np.datetime64("2024-01-01T00:00:00") + np.timedelta64(int(h), "h") for h in
                      [9, 3, 1, 7, 5, 0, 2, 4, 6, 8, 3, 1, 2, 0, 4]], dtype="datetime64[us]")
    b = dg.quantile_bins(sites, times, 5)
    assert sorted(b[:10]) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] and sorted(b[10:]) == [0, 1, 2, 3, 4]
    assert b[5] == 0 and b[0] == 4                                                 # earliest t0 -> bin 0, latest -> last bin


def test_output_is_aggregate_only(drift):
    rep, cohort, _Y = drift
    known = {str(p) for p in cohort["person_id"]}
    assert_aggregate_only(rep, known)
    text = json.dumps(rep)
    assert not DATE_RE.search(text) and "2024" not in text and "2025" not in text
    assert not any(k in text for k in known)
    assert set(rep["settings"]["bins"]["temporal"]) == {"train", "test"} and rep["settings"]["bins"]["quantile"][0] == "Q1"
    assert "cutoff" not in text.lower()
    # quantile tables never come with n < 11
    for q in rep["quantile_tables"].values():
        for cells in q["temporal"].values():
            for v in cells.values():
                assert all(x == SUPPRESSED or isinstance(x, float) for x in v.values())


# ------------------------------------------------------------------------------------------- the CLI
def concentrate_e6(paths, cohort):
    """Overwrite E6 in the (synthetic) silver file: 1 in 15 positive in the first 80% of each site's t0, 40% positive afterwards."""
    sil = pd.read_csv(paths["silver"], dtype=str)
    rank = cohort.groupby("SiteID")["t0"].rank(method="first") / cohort.groupby("SiteID")["t0"].transform("size")
    late = (rank > 0.8).to_numpy()
    idx = np.arange(len(sil))
    pos = np.where(late, idx % 5 < 2, idx % 15 == 0)
    sil["E6"] = np.where(pos, "True", "False")
    sil.to_csv(paths["silver"], index=False)


@pytest.fixture(scope="module")
def cli_inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("diag_cli")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=5)
    concentrate_e6(paths, cohort)
    return d, paths, cohort


def run_cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = dg.main(argv)
    assert rc == 0
    return buf.getvalue()


def test_cli_without_a_store_uses_the_feasibility_assembly_and_split(cli_inputs):
    d, paths, cohort = cli_inputs
    out = d / "diag" / "e6.json"
    stdout = run_cli(argv_for(paths, d / "o1", "--diag-out", str(out), "--also-n-bins", "10"))
    rep = json.loads(out.read_text())
    rep10 = json.loads((d / "diag" / "e6_n10.json").read_text())
    assert rep10["settings"]["bins"]["quantile"] == [f"Q{i}" for i in range(1, 11)] and rep["settings"]["n_bins"] == 5
    assert stdout.startswith(dg.BANNER) and "label.E6.positive [temporal all_sites]" in stdout
    assert any("no --data / --s3" in n for n in rep["settings"]["notes"])
    assert not any(k.startswith(("avail.", "comp.", "vocab.")) for k in rep["tables"])
    assert {"label.E6.positive", "label.E1.positive", "label.E5.positive", "casemix.sedated"} <= set(rep["tables"])
    assert "casemix.age_years" in rep["quantile_tables"]
    te = cell(rep, "label.E6.positive", "temporal", dg.POOLED, "test")
    tr = cell(rep, "label.E6.positive", "temporal", dg.POOLED, "train")
    assert isinstance(te["share"], float) and te["share"] > 0.25 and isinstance(tr["share"], float) and tr["share"] < 0.12
    known = {str(p) for p in cohort["person_id"]}
    assert_aggregate_only(rep, known)
    assert not any(k in stdout for k in known) and not DATE_RE.search(stdout)


def test_cli_refuses_an_output_under_local_only(cli_inputs):
    d, paths, _c = cli_inputs
    with pytest.raises(SystemExit):
        dg.main(argv_for(paths, d / "o2", "--diag-out", str(d / "local_only" / "x.json")))


def synthetic_store(synth_dir):
    p = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    return p if (p / "manifest.json").exists() and (p / "OMOP").is_dir() else synth_dir


def test_cli_with_the_synthetic_store_reads_the_omop_tables(cli_inputs, synth_dir):
    """data/synthetic (or the same generator output in a temp dir): the OMOP block runs end to end through the store reader and the
    shared cache. The analysed ids of the feasibility inputs are fake, so no OMOP row belongs to them: every data-availability and
    component cell is hidden (< 11), and the label tables are those of the label file."""
    d, paths, cohort = cli_inputs
    out = d / "diag" / "e6_store.json"
    run_cli(argv_for(paths, d / "o3", "--diag-out", str(out), "--data", str(synthetic_store(synth_dir))))
    rep = json.loads(out.read_text())
    names = set(rep["tables"])
    assert {"avail.blood_culture_drawn.in_window", "avail.raw_measurement.in_window", "comp.E6.E6_rule", "comp.E6.E6_blood_culture",
            "comp.E6.E6_qad", "comp.E6.E6_lactate", "comp.E1.E1_rule", "comp.E2.E2_rule", "comp.E5.E5_rule",
            "vocab.blood_culture.route_concept", "consistency.E6.csv_positive_not_rederived",
            "avail.blood_culture_drawn.in_window_among_extract_window", "label.E6.positive_among_culture_in_window"} <= names
    assert {"timing.blood_culture_nearest_hours", "timing.antimicrobial_first_start_hours", "casemix.hours_from_visit_start"} <= set(rep["quantile_tables"])
    assert all(c["n"] == SUPPRESSED for t in ("avail.raw_measurement.in_window", "comp.E6.E6_rule")
               for c in walk_cells(rep["tables"][t]))
    # the label file's E6 positives are, by construction, not re-derived from the (unrelated) OMOP rows
    assert isinstance(cell(rep, "consistency.E6.csv_positive_not_rederived", "temporal", dg.POOLED, "test")["n"], int)
    assert_aggregate_only(rep, {str(p) for p in cohort["person_id"]})
