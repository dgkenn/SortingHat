"""scripts/diag_intended_use_fit.py on SYNTHETIC inputs: a planted ICU-cEEG / ED-like / mixed cohort is characterised correctly, the
OMOP proxies (infusion vs PRN, vasopressor, ventilation) are read from in-memory tables and from a cache directory, every cell obeys the
small-cell rules, and nothing record-level (ids, dates) is emitted."""
import contextlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_silver_feasibility_inputs import argv_for, load_script, make_inputs  # noqa: F401  (tests/ on sys.path)

du = load_script("diag_intended_use_fit")

SITES = ("S0001", "S0002")
T0 = pd.Timestamp("2024-03-01 12:00:00")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
N_SITE = 400
LABELS = {"S0001": "site_1", "S0002": "site_2"}


# ------------------------------------------------------------------------------------------------ planted data
def planted(n_site=N_SITE, tiny_site2_ed=False):
    """Per site (n_site people): 50% ICU cEEG (20 h recording, LTM, propofol at t0, a label-family code, 60 h after the visit start),
    25% ED-like (30 min, Routine, 2 h after the visit start, nothing else), 25% mixed (5 h, inpatient, 30 h, a PRN-opioid quantity only).
    With ``tiny_site2_ed`` the second site has only 5 ED-like people (the rest ICU)."""
    rows, bls = [], []
    pid = 9_000_000
    for si, site in enumerate(SITES):
        for i in range(n_site):
            pid += 1
            kind = "icu" if i % 4 in (0, 1) else ("ed" if i % 4 == 2 else "mixed")
            if tiny_site2_ed and si == 1:
                kind = "ed" if i < 5 else ("icu" if i % 2 else "mixed")
            t0 = T0 + pd.Timedelta(hours=i)
            h = {"icu": 60.0, "ed": 2.0, "mixed": 30.0}[kind]
            rows.append({"SiteID": site, "person_id": pid, "person_id_source": pid, "t0": t0,
                         "encounter_start": t0 - pd.Timedelta(hours=h),
                         "EEGFolder": "ceeg" if kind == "icu" else None,
                         "ServiceName": {"icu": "LTM", "ed": "Routine", "mixed": "Inpatient"}[kind],
                         "duration_s": {"icu": 20 * 3600.0, "ed": 1800.0, "mixed": 5 * 3600.0}[kind],
                         "visit_inpatient_length": kind != "ed", "visit_class": None,
                         "acute_basis": "service" if kind == "icu" else "visit_length", "hours_since_onset": h})
            bl = {"person_id": pid, "demo__age_years": 60.0,
                  "meta__hours_since_encounter_start": h, "meta__label_dx_before_t0": 3.0 if kind == "icu" else 0.0,
                  "sed__propofol__on_t0": 1.0 if kind == "icu" else 0.0, "sed__sedative__on_t0": 1.0 if kind == "icu" else 0.0,
                  "sed__opioid__on_t0": 0.0, "sed__fentanyl__qty_6h": 50.0 if kind == "mixed" else 0.0,
                  "sed__n_agents_24h": 1.0 if kind in ("icu", "mixed") else 0.0,
                  "lab__ph_arterial__value": 7.3 if kind == "icu" else np.nan}
            bls.append(bl)
    return pd.DataFrame(rows), pd.DataFrame(bls).drop(columns=["person_id"])


def feat_of(frame, bl, omop=None, Y=None, sedated=None):
    return du.build_features(frame, bl, sedated, Y, omop)


def tables_of(F, order=("S0001", "S0002")):
    return du.pop_tables(F, list(order), LABELS)


def cell(t, block, name, site):
    return t["tables"][block][name]["cells"][site]


def klass(t, block, name, cls, site):
    return t["classes"][block][name]["classes"][cls][site]


@pytest.fixture(scope="module")
def planted_tables():
    frame, bl = planted()
    F = feat_of(frame, bl)
    return tables_of(F), F


# ------------------------------------------------------------------------------------ the planted characterisation
def test_eeg_type_blocks_show_the_continuous_icu_share(planted_tables):
    t, _ = planted_tables
    cc = cell(t, "eeg_type", "ceeg_task_folder", du.POOLED)
    assert cc["share"] == pytest.approx(0.5) and cc["n"] == 400 and cc["den"] == 800
    assert klass(t, "eeg_type", "duration_class", "gt_12h_continuous", du.POOLED)["share"] == pytest.approx(0.5)
    assert klass(t, "eeg_type", "duration_class", "lt_1h_routine_or_spot", du.POOLED)["share"] == pytest.approx(0.25)
    assert klass(t, "eeg_type", "duration_class", "1_to_12h", "site_1")["share"] == pytest.approx(0.25)
    assert klass(t, "eeg_type", "service_class", "ltm_continuous", du.POOLED)["share"] == pytest.approx(0.5)
    assert klass(t, "eeg_type", "service_class", "routine", du.POOLED)["share"] == pytest.approx(0.25)
    q = t["quantiles"]["eeg_type"]["duration_hours"]["by_site"][du.POOLED]
    assert q["q50"] == pytest.approx(12.5) and q["q10"] == pytest.approx(0.5) and q["q90"] == pytest.approx(20.0)
    assert "min" not in q and "max" not in q and q["n"] == 800


def test_timing_and_care_setting(planted_tables):
    t, F = planted_tables
    assert F.hours_src == "cohort encounter_start"
    assert cell(t, "timing", "t0_within_6h_of_visit_start", du.POOLED)["share"] == pytest.approx(0.25)
    assert cell(t, "timing", "t0_within_24h_of_visit_start", du.POOLED)["share"] == pytest.approx(0.25)
    assert cell(t, "timing", "t0_within_48h_of_visit_start", du.POOLED)["share"] == pytest.approx(0.5)
    assert cell(t, "care_setting", "covering_visit_longer_than_a_day", du.POOLED)["share"] == pytest.approx(0.75)
    assert cell(t, "care_setting", "arterial_gas_before_t0", du.POOLED)["share"] == pytest.approx(0.5)
    assert klass(t, "care_setting", "acute_care_basis", "service", du.POOLED)["share"] == pytest.approx(0.5)
    assert "covering_visit_class" not in t["classes"].get("care_setting", {})            # visit_class is all missing in the plant: no class


def test_sedation_tiers_and_known_dx(planted_tables):
    t, _ = planted_tables
    tier = lambda c: klass(t, "sedation", "baselineA_tiers", c, du.POOLED)["share"]          # noqa: E731
    assert tier("infusion_agent_active_at_t0") == pytest.approx(0.5)
    assert tier("recent_6h_only") == pytest.approx(0.25)
    assert tier("none") == pytest.approx(0.25)
    assert cell(t, "sedation", "baselineA_any_agent_active_at_t0", du.POOLED)["share"] == pytest.approx(0.5)
    assert cell(t, "known_dx", "label_family_icd_before_t0", du.POOLED)["share"] == pytest.approx(0.5)


def test_undifferentiated_variants_nest_as_planted(planted_tables):
    t, F = planted_tables
    u = t["undifferentiated"]
    assert u["U0_current"]["cells"][du.POOLED]["n"] == 200               # the ED-like quarter only
    assert u["U0_current"]["estimable"] is True
    assert u["U1_t0_within_48h"]["cells"][du.POOLED]["n"] == 200         # mixed people are on a PRN opioid (still excluded by sedation)
    # the mixed group (PRN opioid quantity, 30 h) enters when sedation is allowed unless it is an infusion: baseline-only proxy = infusion agent
    assert u["U3_sedation_not_infusion"]["cells"][du.POOLED]["n"] == 400
    assert u["U5_48h_dx_tolerated"]["cells"][du.POOLED]["n"] == 400      # ICU people are at 60 h: out
    assert u["U10_early6_no_dx"]["cells"][du.POOLED]["n"] == 200
    # OMOP-dependent variants are unavailable without OMOP rows and say so
    assert "unavailable" in u["U6_ed_like_proxy"] and "unavailable" in u["U7_ed_like_sedation_free"]
    # U8: t0 within 48 h, no label-family code, no infusion, recording <= 12 h -> the ED-like quarter and the mixed quarter
    assert u["U8_short_recording"]["cells"][du.POOLED]["n"] == 400


# ---------------------------------------------------------------------------------------------- suppression
def test_small_cells_hidden_and_pooled_not_a_subtraction_leak():
    frame, bl = planted(tiny_site2_ed=True)
    F = feat_of(frame, bl)
    t = tables_of(F)
    u0 = t["undifferentiated"]["U0_current"]["cells"]
    assert u0["site_2"]["n"] == SUPPRESSED and u0["site_2"]["share"] == SUPPRESSED               # 5 < 11
    assert u0[du.POOLED]["n"] == SUPPRESSED                                                      # one hidden site cell hides the pooled cell
    assert t["undifferentiated"]["U0_current"]["estimable"] is False                            # a site below 11
    d = t["classes"]["eeg_type"]["duration_class"]["classes"]
    # exclusive classes of site_2 with a tiny cell: the lone hidden class hides the smallest other one as well
    hidden = [c for c, cells in d.items() if cells["site_2"]["n"] == SUPPRESSED]
    assert len(hidden) != 1
    rep = {"populations": {"p": {"n": du.pop_sizes(F, ["S0001", "S0002"], LABELS), **t}}}
    assert_aggregate_only(rep)
    s = json.dumps(rep, default=str)
    assert not DATE_RE.search(s)


def test_count_row_and_cell_rules():
    row = du.count_row({"a": 5, "b": 400}, ["a", "b"], {"a": "site_1", "b": "site_2"})
    assert row == {"site_1": SUPPRESSED, "site_2": 400, du.POOLED: SUPPRESSED}
    row = du.count_row({"a": 20, "b": 400}, ["a", "b"], {"a": "site_1", "b": "site_2"})
    assert row[du.POOLED] == 420
    # a share whose complement is below 11 is hidden (it would reveal the complement)
    c = du.tab_binary(np.r_[np.ones(395, bool), np.zeros(5, bool)], np.ones(400, bool), np.array(["a"] * 400, object), ["a"], {"a": "site_1"})
    assert c["site_1"]["share"] == SUPPRESSED and c[du.POOLED]["share"] == SUPPRESSED
    assert c["site_1"]["n"] == SUPPRESSED and c["site_1"]["share_bound"] == ">=0.975"      # only a bound is shown, never the 395
    q = du.tab_quantiles(np.arange(8, dtype=float), np.array(["a"] * 8, object), ["a"], {"a": "site_1"})
    assert q["site_1"]["q50"] == SUPPRESSED


def test_quantile_refuses_min_max_and_missing_columns_are_omitted():
    frame, bl = planted()
    frame = frame.drop(columns=["ServiceName", "EEGFolder", "visit_inpatient_length", "hours_since_onset", "visit_class", "acute_basis"])
    t = tables_of(feat_of(frame, bl))
    assert "ceeg_task_folder" not in t["tables"].get("eeg_type", {}) and "service_class" not in t["classes"].get("eeg_type", {})
    assert "covering_visit_longer_than_a_day" not in t["tables"].get("care_setting", {})
    assert "hours_from_onset_to_t0" not in t["quantiles"]["timing"] and "duration_class" in t["classes"]["eeg_type"]


# ------------------------------------------------------------------------------------------- OMOP proxies
def omop_tables(cohort):
    """Per person by index class: icu -> propofol infusion + norepinephrine + a ventilator row; mixed -> a fentanyl bolus and a morphine
    PRN dose (date-only); ed -> nothing. Plus decoys (vent text outside the window, a sedative infusion before the window)."""
    drug, meas, proc = [], [], []
    for _i, r in cohort.iterrows():
        t0 = r["t0"]
        icu = r["ServiceName"] == "LTM"
        mixed = r["ServiceName"] == "Inpatient"
        drug.append({"person_id": r["person_id"], "drug_exposure_start_datetime": t0 - pd.Timedelta(hours=120), "drug_exposure_end_datetime": t0 - pd.Timedelta(hours=100),
                     "drug_exposure_start_date": None, "drug_exposure_end_date": None, "drug_source_value": "PROPOFOL 10 MG/ML IV EMULSION"})      # decoy: days before t0
        meas.append({"person_id": r["person_id"], "measurement_datetime": t0 - pd.Timedelta(hours=5), "measurement_date": None,
                     "measurement_source_value": "Immunology (CMV)"})                                                                              # decoy: not a ventilator
        if icu:
            drug.append({"person_id": r["person_id"], "drug_exposure_start_datetime": t0 - pd.Timedelta(hours=30), "drug_exposure_end_datetime": t0 + pd.Timedelta(hours=1),
                         "drug_exposure_start_date": None, "drug_exposure_end_date": None, "drug_source_value": "PROPOFOL 10 MG/ML IV EMULSION"})
            drug.append({"person_id": r["person_id"], "drug_exposure_start_datetime": t0 - pd.Timedelta(hours=3), "drug_exposure_end_datetime": t0 - pd.Timedelta(hours=2),
                         "drug_exposure_start_date": None, "drug_exposure_end_date": None, "drug_source_value": "NOREPINEPHRINE 8 MG IN D5W 250 ML"})
            meas.append({"person_id": r["person_id"], "measurement_datetime": t0 - pd.Timedelta(hours=5), "measurement_date": None,
                         "measurement_source_value": "Ventilation Rate"})
        if mixed:
            drug.append({"person_id": r["person_id"], "drug_exposure_start_datetime": t0 - pd.Timedelta(hours=2), "drug_exposure_end_datetime": t0 - pd.Timedelta(hours=2, minutes=-1),
                         "drug_exposure_start_date": None, "drug_exposure_end_date": None, "drug_source_value": "FENTANYL 50 MCG/ML INJ"})
            drug.append({"person_id": r["person_id"], "drug_exposure_start_datetime": None, "drug_exposure_end_datetime": None,
                         "drug_exposure_start_date": t0.normalize(), "drug_exposure_end_date": t0.normalize(), "drug_source_value": "MORPHINE 2 MG/ML INJ"})
            meas.append({"person_id": r["person_id"], "measurement_datetime": t0 - pd.Timedelta(hours=60), "measurement_date": None,
                         "measurement_source_value": "Ventilator Mode"})                             # decoy: outside the 24 h window
    for _i, r in cohort.iloc[:50].iterrows():                                                       # procedures: intubation code, date-only
        if r["ServiceName"] == "Routine":
            proc.append({"person_id": r["person_id"], "procedure_datetime": None, "procedure_date": r["t0"].normalize(),
                         "procedure_source_value": "0BH17EZ"})
    return {"omop_drug_exposure": pd.DataFrame(drug), "omop_measurement": pd.DataFrame(meas), "omop_procedure_occurrence": pd.DataFrame(proc)}


def dict_readers(tables):
    return {t: (lambda ids, cols, tc, pre, _t=t: du.iter_dict_frames(tables, _t, ids))
            for t in ("drug_exposure", "measurement", "procedure_occurrence")}


@pytest.fixture(scope="module")
def omop_planted():
    frame, bl = planted()
    tables = omop_tables(frame)
    flags, counts = du.omop_proxies(frame, dict_readers(tables))
    return frame, bl, tables, flags, counts


def test_omop_proxies_classify_infusion_prn_and_vent(omop_planted):
    frame, _bl, _t, flags, counts = omop_planted
    icu = (frame["ServiceName"] == "LTM").to_numpy()
    mixed = (frame["ServiceName"] == "Inpatient").to_numpy()
    ed = (frame["ServiceName"] == "Routine").to_numpy()
    assert flags["sed_inf_6h"].to_numpy()[icu].all() and not flags["sed_inf_6h"].to_numpy()[~icu].any()
    assert flags["sed_row_6h"].to_numpy()[mixed].all()                                 # PRN rows are seen ...
    assert not flags["sed_inf_6h"].to_numpy()[mixed].any()                             # ... but are not infusions
    assert flags["opioid_row_6h"].to_numpy()[mixed].all() and not flags["opioid_inf_6h"].to_numpy().any()
    assert flags["sedative_inf_6h"].to_numpy()[icu].all() and not flags["sedative_row_6h"].to_numpy()[mixed].any()
    assert flags["vasopressor_24h"].to_numpy()[icu].all() and not flags["vasopressor_24h"].to_numpy()[~icu].any()
    v = flags["vent_24h"].to_numpy()
    assert v[icu].all()                                                                 # ventilator measurement row in the window
    assert not v[mixed].any()                                                            # the decoy 60 h before t0 is outside it
    first50 = np.arange(len(frame)) < 50
    assert v[ed & first50].all() and not v[ed & ~first50].any()                         # procedure code (date-only) on the date of t0
    assert counts["drug_rows_kept"] > 0


def test_omop_backed_variants_and_infusion_vs_prn_class(omop_planted):
    frame, bl, _t, flags, _c = omop_planted
    F = feat_of(frame, bl, flags)
    t = tables_of(F)
    inf = lambda c: klass(t, "sedation", "omop_infusion_vs_prn_6h", c, du.POOLED)["share"]          # noqa: E731
    assert inf("continuous_infusion") == pytest.approx(0.5) and inf("prn_only") == pytest.approx(0.25) and inf("none") == pytest.approx(0.25)
    assert cell(t, "care_setting", "vasopressor_24h", du.POOLED)["share"] == pytest.approx(0.5)
    u = t["undifferentiated"]
    assert u["U3_sedation_not_infusion"]["cells"][du.POOLED]["n"] == 400                  # OMOP infusion flag replaces the baseline proxy
    assert u["U2_prn_opioid_allowed"]["cells"][du.POOLED]["n"] == 400                    # ED-like + the PRN-opioid group (no sedative, no opioid infusion)
    # the ED-like proxy: 24 h, no dx, no infusion, not ventilated (a few ED-like people have a planted procedure), no pressor, not cEEG
    ed_like = u["U6_ed_like_proxy"]["cells"][du.POOLED]["n"]
    assert 100 <= ed_like <= 200
    assert u["U7_ed_like_sedation_free"]["cells"][du.POOLED]["n"] == ed_like


def test_label_positives_per_variant():
    frame, bl = planted()
    rng = np.random.default_rng(1)
    Y = pd.DataFrame({"E1": np.where(rng.random(len(frame)) < 0.2, 1.0, 0.0), "E6": np.where(frame["ServiceName"] == "Routine", 1.0, 0.0)})
    t = tables_of(feat_of(frame, bl, Y=Y))
    pos = t["undifferentiated"]["U0_current"]["positives"]
    assert pos["E6"][du.POOLED] == 200 and isinstance(pos["E1"][du.POOLED], int) and 20 < pos["E1"][du.POOLED] < 70


# ------------------------------------------------------------------------------------------ the cache reader
def write_cache(root: Path, tables: dict, split=3, drop_last_part_file=False):
    for name, df in tables.items():
        tname = name[len("omop_"):]
        prof = root / tname / "profile0"
        if not len(df):
            continue
        chunks = np.array_split(np.arange(len(df)), split)
        for k, ix in enumerate(chunks):
            part = prof / f"part{k}"
            part.mkdir(parents=True, exist_ok=True)
            sub = df.iloc[ix].reset_index(drop=True)
            pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), part / "rg-000000.parquet")
            (part / "manifest.json").write_text(json.dumps({"n": 1 + (1 if (drop_last_part_file and k == split - 1) else 0)}))


def test_cache_reader_matches_in_memory_tables(tmp_path, omop_planted):
    frame, _bl, tables, flags, _c = omop_planted
    write_cache(tmp_path, tables)
    readers = {t: (lambda ids, cols, tc, pre, _t=t: du.iter_cache_frames(_t, tmp_path, cols, ids, tc, pre))
               for t in ("drug_exposure", "measurement", "procedure_occurrence")}
    got, _ = du.omop_proxies(frame, readers)
    pd.testing.assert_frame_equal(got, flags)
    files, cov = du.cache_files("drug_exposure", tmp_path)
    assert len(files) == 3 and cov == {"parts": 3, "complete_parts": 3}


def test_cache_coverage_counts_incomplete_parts(tmp_path, omop_planted):
    _f, _b, tables, _fl, _c = omop_planted
    write_cache(tmp_path, tables, drop_last_part_file=True)
    _files, cov = du.cache_files("drug_exposure", tmp_path)
    assert cov == {"parts": 3, "complete_parts": 2}
    assert du.cache_files("note", tmp_path) == ([], {"parts": 0, "complete_parts": 0})


# ----------------------------------------------------------------------------------- the CLI end to end
@pytest.fixture(scope="module")
def run_inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("iufit")
    paths, ms, cohort = make_inputs(d, signal=0.0, seed=5, n_per_site=300)
    c = pd.read_csv(paths["cohort"], low_memory=False)
    rng = np.random.default_rng(2)
    c["ServiceName"] = np.where(c["EEGFolder"].notna(), "LTM", np.where(rng.random(len(c)) < 0.5, "Routine", "Inpatient"))
    c["visit_inpatient_length"] = rng.random(len(c)) < 0.6
    c["hours_since_onset"] = rng.uniform(0, 60, len(c))
    c.to_csv(paths["cohort"], index=False)
    return paths, d, c


def test_cli_runs_end_to_end_and_is_aggregate_only(run_inputs, tmp_path, capsys):
    paths, d, c = run_inputs
    out_json, out_md = tmp_path / "iu.json", tmp_path / "iu.md"
    argv = argv_for(paths, tmp_path / "out", "--omop-source", "none", "--diag-out", str(out_json), "--diag-md", str(out_md))
    assert du.main(argv) == 0
    rep = json.loads(out_json.read_text())
    assert set(rep["populations"]) == {"source_table", "source_strict", "source_broad", "analysed_strict", "analysed_broad"}
    P = rep["populations"]["analysed_strict"]
    assert P["tables"]["eeg_type"]["ceeg_task_folder"]["cells"][du.POOLED]["share"] == pytest.approx(0.5, abs=0.1)
    assert "U0_current" in P["undifferentiated"] and "unavailable" in P["undifferentiated"]["U6_ed_like_proxy"]
    assert any("reproduces the feasibility script's undifferentiated subgroup exactly: True" in n for n in rep["settings"]["notes"])
    assert rep["settings"]["omop"]["omop_source"] == "none"
    # the source table is larger than the analysed rows (the strict flag is False for a few, QC failures drop more)
    assert rep["populations"]["source_table"]["n"][du.POOLED] >= P["n"][du.POOLED]
    text = out_json.read_text() + out_md.read_text() + capsys.readouterr().out
    assert not DATE_RE.search(text)
    ids = {str(p) for p in c["person_id"]}
    assert not any(tok in ids for tok in re.findall(r"\d{7,}", text))
    assert_aggregate_only(rep, ids)
    assert "analysed_strict" in out_md.read_text() and "Variant definitions" in out_md.read_text() and "Summary (pooled" in out_md.read_text()
    assert set(rep["summary_pooled"]["U0_current"]) == set(rep["populations"])


def test_cli_refuses_outputs_under_local_only(run_inputs, tmp_path):
    paths, d, _c = run_inputs
    bad = tmp_path / "local_only" / "x.json"
    with pytest.raises(SystemExit):
        du.main(argv_for(paths, tmp_path / "out", "--omop-source", "none", "--diag-out", str(bad), "--diag-md", str(tmp_path / "x.md")))


def test_cli_with_a_cache_dir(run_inputs, tmp_path):
    paths, d, c = run_inputs
    cohort = c.copy()
    cohort["t0"] = pd.to_datetime(cohort["t0"])
    tables = omop_tables(cohort.assign(ServiceName=cohort["ServiceName"]))
    root = tmp_path / "cache"
    write_cache(root, tables)
    out_json = tmp_path / "iu.json"
    argv = argv_for(paths, tmp_path / "out", "--omop-source", "cache", "--omop-cache", str(root), "--diag-out", str(out_json),
                    "--diag-md", str(tmp_path / "iu.md"))
    with contextlib.redirect_stdout(io.StringIO()):
        assert du.main(argv) == 0
    rep = json.loads(out_json.read_text())
    P = rep["populations"]["analysed_broad"]
    assert "omop_infusion_vs_prn_6h" in P["classes"]["sedation"]
    assert P["tables"]["care_setting"]["vasopressor_24h"]["cells"][du.POOLED]["share"] != SUPPRESSED
    assert "coverage" in rep["settings"]["omop"]
    assert "unavailable" not in P["undifferentiated"]["U6_ed_like_proxy"]
