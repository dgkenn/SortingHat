"""scripts/build_baselines.py on SYNTHETIC data: the streamed, row-group-chunked build equals the package's in-memory build,
the as_of gate is untouched, and nothing record-level is printed."""
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from sortinghat import data_io
from sortinghat.baselines import BaselineConfig, build_feature_set
from sortinghat.baselines.events import build_index
from sortinghat.cohort import CohortConfig, StoreSources, build_cohort
from sortinghat.safe_output import assert_aggregate_only
from sortinghat.tableio import load_tables
from test_silver_feasibility_inputs import rsf

bb = rsf.bb
CFG = BaselineConfig()


@pytest.fixture(scope="module")
def cohort_csv(tmp_path_factory, synth_dir):
    res = build_cohort(StoreSources(data_io.open_store(synth_dir)), CohortConfig(study_sites=None))
    lo = tmp_path_factory.mktemp("bl") / "local_only"
    lo.mkdir()
    res.table.to_csv(lo / "cohort_study1.csv", index=False)
    return lo / "cohort_study1.csv"


@pytest.fixture(scope="module")
def index(synth_dir, cohort_csv):
    store = data_io.open_store(synth_dir)
    coh = bb.load_cohort(cohort_csv, None, "table")
    return bb.make_index(coh, bb.sex_lookup(store, coh))


@pytest.fixture(scope="module")
def tables(synth_dir):
    t = load_tables(synth_dir)
    t.pop("imaging")                                  # the runner reads no imaging (D-122)
    return t


class FrameStream:
    """StoreSources look-alike over in-memory tables (for injecting events); chunks are small to exercise concatenation."""

    def __init__(self, tables, store, rows=400):
        self.t, self.s3, self.rows = tables, store, rows

    def iter_rows(self, table, columns, person_ids, min_rows=0):
        d = self.t[table]
        d = d[d["person_id"].isin({int(p) for p in person_ids})]
        keep = [c for c in columns if c in d.columns]
        for i in range(0, len(d), self.rows):
            yield d.iloc[i:i + self.rows][keep].copy()


def same(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns) and list(a.index) == list(b.index)
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=0, atol=1e-9)


# ----------------------------------------------------------------------------------------------- equivalence
def test_streamed_chunked_build_equals_the_in_memory_build(synth_dir, index, tables):
    store = data_io.open_store(synth_dir)
    ref = build_feature_set(tables, CFG, index)
    fs, n_pref = bb.build_matrices(StoreSources(store), index, CFG, None, True, chunk_rows=700)
    fs0, n_all = bb.build_matrices(StoreSources(store), index, CFG, None, False, chunk_rows=700)
    same(fs.X, ref.X)
    same(fs0.X, ref.X)                                # with and without the memory prefilter
    assert n_pref < n_all                             # the prefilter really removes rows (vitals / future events)
    assert fs.diagnostics["n_patients"] == len(index)


def test_the_package_t0_gate_still_decides(synth_dir, index, tables):
    """Events after t0 (any domain, 1 second to 400 days) leave the matrix identical; an event just before t0 changes it."""
    t0 = index.set_index("person_id")["t0"]
    pid = index["person_id"].to_numpy()
    base = build_feature_set(tables, CFG, index).X

    def add_meas(t, name, val, unit, when):
        d = pd.DataFrame({"person_id": pid, "measurement_datetime": [t0[p] + when for p in pid], "measurement_date": pd.NaT,
                          "measurement_time": None, "measurement_source_value": name, "value_as_number": val,
                          "unit_source_value": unit, "measurement_concept_id": 0, "measurement_type_concept_id": 0,
                          "visit_occurrence_id": 0})
        d["measurement_date"] = pd.to_datetime(d["measurement_datetime"]).dt.normalize()
        return {**t, "omop_measurement": pd.concat([t["omop_measurement"], d], ignore_index=True)}

    def add_drug(t, when):
        d = pd.DataFrame({"person_id": pid, "drug_exposure_start_datetime": [t0[p] + when for p in pid],
                          "drug_exposure_end_datetime": pd.NaT, "drug_source_value": "PROPOFOL 10 MG/ML IV EMULSION",
                          "quantity": 500.0, "drug_concept_id": 0, "drug_type_concept_id": 0, "route_source_value": "IV",
                          "drug_exposure_start_date": pd.NaT, "drug_exposure_end_date": pd.NaT, "visit_occurrence_id": 0})
        return {**t, "omop_drug_exposure": pd.concat([t["omop_drug_exposure"], d], ignore_index=True)}

    store = data_io.open_store(synth_dir)
    for when in (pd.Timedelta(seconds=1), pd.Timedelta(hours=30), pd.Timedelta(days=400)):
        inj = add_drug(add_meas(add_meas(tables, "Glasgow Coma Scale Score", 3.0, None, when),
                                "LACTATE, WHOLE BLOOD", 99.0, "mmol/L", when), when)
        fs, _ = bb.build_matrices(FrameStream(inj, store), index, CFG, None, True)
        same(fs.X, base)
    # a GCS charted one minute BEFORE t0 is visible: the gate is not simply deleting everything
    inj = add_meas(tables, "Glasgow Coma Scale Score", 4.0, None, -pd.Timedelta(minutes=1))
    fs, _ = bb.build_matrices(FrameStream(inj, store), index, CFG, None, True)
    g = fs.X["score__gcs__value"]
    assert g.notna().all() and (g == 4.0).mean() > 0.9 and not g.equals(base["score__gcs__value"])   # later charting may win


def test_retired_ids_are_rekeyed_through_the_merge_map(synth_dir, index, tables):
    store = data_io.open_store(synth_dir)
    ref = build_feature_set(tables, CFG, index).X
    victims = index["person_id"].iloc[:25].tolist()
    old = {p: 990_000_000 + i for i, p in enumerate(victims)}
    moved = {}
    for name, d in tables.items():
        if "person_id" in d.columns and name.startswith("omop_"):
            d = d.copy()
            d["person_id"] = d["person_id"].map(lambda p: old.get(int(p), p) if pd.notna(p) else p).astype(d["person_id"].dtype)
        moved[name] = d
    mm = {v: k for k, v in old.items()}
    fs, _ = bb.build_matrices(FrameStream(moved, store), index, CFG, mm, True)
    same(fs.X, ref)
    fs_no, _ = bb.build_matrices(FrameStream(moved, store), index, CFG, None, True)      # without the map those rows are lost
    assert not fs_no.X.loc[victims].equals(ref.loc[victims])


def test_sex_lookup_agrees_with_the_package_index(synth_dir, cohort_csv, tables):
    store = data_io.open_store(synth_dir)
    coh = bb.load_cohort(cohort_csv, None, "table")
    mine = pd.Series(bb.sex_lookup(store, coh).to_numpy(), index=coh["person_id"].to_numpy())
    ref = build_index(tables).set_index("person_id")["sex_male"]
    both = mine.index.intersection(ref.index)
    a, b = mine.loc[both], ref.loc[both]
    ok = a.notna() & b.notna()
    assert ok.sum() > 100 and (a[ok] == b[ok]).all()
    assert mine.notna().mean() > 0.9


def test_prune_events_only_removes_rows():
    from sortinghat.baselines.asof import empty_events
    t0 = pd.Series({1: pd.Timestamp("2024-05-01 12:00")})
    n = 6
    ev = pd.DataFrame({
        "person_id": [1] * n, "domain": ["vital", "vital", "vital", "hx", "hx", "lab"],
        "key": ["hr", "hr", "hr", "arrest", "arrest", "sodium"], "value": 1.0, "quantity": np.nan,
        "t_event": pd.to_datetime(["2024-05-01 11:00:00", "2024-05-01 12:00:01", "2024-04-30 00:00:00",
                                   "2023-01-01 00:00:00", None, "2024-04-29 12:00:00"], format="%Y-%m-%d %H:%M:%S"),
        "t_avail": pd.NaT, "t_end_raw": pd.NaT, "time_basis": "x", "approx": False, "unit": "u", "raw_name": "secret name"})
    out = bb.prune_events(ev, t0, CFG)
    assert out["raw_name"].isna().all() and out["unit"].isna().all()
    # kept: recent vital, old history (no lower bound), history with unknown time (as_of decides), old lab (72 h window is as_of's)
    assert out["key"].tolist() == ["hr", "arrest", "arrest", "sodium"]
    assert len(bb.prune_events(empty_events(), t0, CFG)) == 0


def test_cohort_definition_and_site_filters(cohort_csv):
    allr = bb.load_cohort(cohort_csv, None, "table")
    strict = bb.load_cohort(cohort_csv, None, "strict")
    assert 0 < len(strict) < len(allr) and strict["in_strict"].all()
    one = bb.load_cohort(cohort_csv, ["S0001"], "table")
    assert set(one["SiteID"]) == {"S0001"} and allr["person_id"].is_unique


# ------------------------------------------------------------------------------------------------- the CLI
def run_cli(synth_dir, cohort_csv, out, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bb.main(["--data", str(synth_dir), "--cohort", str(cohort_csv), "--out", str(out), "--chunk-rows", "900", *extra])
    assert rc == 0
    return buf.getvalue()


def test_cli_writes_a_private_matrix_and_prints_only_aggregates(synth_dir, cohort_csv, tmp_path):
    out = cohort_csv.parent / "baselines_AC.parquet"
    summ = tmp_path / "summary.json"
    text = run_cli(synth_dir, cohort_csv, out, "--sites", "all", "--summary", str(summ))
    coh = bb.load_cohort(cohort_csv, None, "table")
    df = pd.read_parquet(out)
    assert oct(os.stat(out).st_mode & 0o777) == "0o600"
    side_path = out.with_name("baselines_AC.columns.json")
    assert oct(os.stat(side_path).st_mode & 0o777) == "0o600"
    side = json.loads(side_path.read_text())["baselines"]
    meta = json.loads(side_path.read_text())["meta"]
    assert set(side) == {"A", "B", "C", "P"} and set(side["A"]) <= set(side["B"]) <= set(side["C"])
    assert set(side["P"]) & set(side["C"]) == {c for c in side["A"] if c.startswith("demo__")}      # P shares only age / sex
    assert set(df.columns) == {"person_id", *side["C"], *side["P"], *meta}
    assert json.loads(side_path.read_text())["encounter_scope"] == "current"
    assert not [c for c in df.columns if c.startswith(("ind__", "img__"))]          # Baseline D dropped; no imaging in C
    assert sorted(df["person_id"]) == sorted(coh["person_id"]) and df["person_id"].is_unique
    assert all(df[c].dtype == "float64" for c in df.columns if c != "person_id")
    # nothing record-level on stdout, and the small site is suppressed
    ids = {str(p) for p in coh["person_id"]} | set(coh["BidsFolder"].astype(str)) | set(coh["SessionID"].astype(str)) - {"1", "2"}
    assert_aggregate_only(text, ids)
    for p in coh["person_id"].head(50):
        assert str(p) not in text
    assert "Baseline A" in text and "Baseline C" in text and "Baseline D dropped" in text
    assert "site_1: n=<11 missing=<11" in text                                        # I0002 has fewer than 11 people
    assert_aggregate_only(json.loads(summ.read_text()), ids)
    assert "peak RSS" in text


def test_cli_default_sites_are_the_two_study_sites(synth_dir, cohort_csv):
    out = cohort_csv.parent / "bl_two.parquet"
    run_cli(synth_dir, cohort_csv, out)
    df = pd.read_parquet(out)
    coh = bb.load_cohort(cohort_csv, ["S0001", "S0002"], "table")
    assert sorted(df["person_id"]) == sorted(coh["person_id"])


def test_cli_refuses_paths_outside_local_only(synth_dir, cohort_csv, tmp_path):
    with pytest.raises(SystemExit):
        bb.main(["--data", str(synth_dir), "--cohort", str(cohort_csv), "--out", str(tmp_path / "x.parquet")])
    bad = tmp_path / "cohort.csv"
    bad.write_text(cohort_csv.read_text())
    with pytest.raises(SystemExit):
        bb.main(["--data", str(synth_dir), "--cohort", str(bad), "--out", str(cohort_csv.parent / "y.parquet")])


def test_missingness_summary_pools_and_suppresses(synth_dir, index, tables):
    store = data_io.open_store(synth_dir)
    fs, _ = bb.build_matrices(StoreSources(store), index, CFG)
    s = bb.missingness_summary(fs, index)
    assert set(s["baselines"]) == {"A", "B", "C", "P"}
    for b, v in s["baselines"].items():
        assert v["n_columns"] >= v["n_value_variables"] > 0
        assert 0 <= v["pooled_missing_rate"] <= 1 or v["pooled_missing_rate"] == "<11"
        assert sum(v["variables_by_missingness"].values()) == v["n_value_variables"]
        for site in v["by_site"].values():
            if site["n"] == "<11":
                assert site["pooled_missing_rate"] == "<11"
    assert s["baselines"]["A"]["n_columns"] < s["baselines"]["B"]["n_columns"] < s["baselines"]["C"]["n_columns"]
    assert s["baselines"]["P"]["n_columns"] > 4


def test_the_analysis_script_reads_what_the_baseline_runner_writes(synth_dir, cohort_csv):
    out = cohort_csv.parent / "bl_handoff.parquet"
    run_cli(synth_dir, cohort_csv, out, "--sites", "all")
    bl, cols = rsf.load_baselines(out)
    assert set(cols) == {"A", "C", "P", "AGE_SEX", "meta"} and set(cols["A"]) < set(cols["C"])
    assert rsf.baseline_scope(out) == "current encounter only"
    assert {"sed__sedative__on_t0", "sed__opioid__on_t0", "score__gcs__value", "demo__age_years"} <= set(cols["A"])
    assert set(cols["C"]) <= set(bl.columns) and bl["person_id"].is_unique
    flag = rsf.sedation_flag(bl.set_index("person_id").reset_index(drop=True), pd.DataFrame(index=range(len(bl))), "baseline")
    assert flag.dtype == bool and 0 < flag.sum() < len(flag)          # the synthetic cohort has t0 sedation exposure


def test_prune_rows_removes_only_rows_after_t0():
    t0 = pd.Series({1: pd.Timestamp("2024-05-01 12:00:00"), 2: pd.Timestamp("2024-05-01 12:00:00")})
    d = pd.DataFrame({
        "person_id": [1, 1, 1, 1, 1, 2, 3],
        "measurement_datetime": pd.to_datetime(["2024-05-01 11:59:59", "2024-05-01 12:00:00", "2024-05-01 12:00:01", None,
                                                None, None, "2024-05-01 01:00:00"], format="%Y-%m-%d %H:%M:%S"),
        "measurement_date": pd.to_datetime([None, None, None, "2024-05-01", "2024-05-02", None, None], format="%Y-%m-%d")})
    out = bb.prune_rows(d, t0, "measurement_datetime", "measurement_date")
    # kept: before, exactly at t0, date on t0's calendar day (as_of decides the hour), no usable time (builder judges);
    # dropped: after t0, a later date, and a person with no t0
    assert out.index.tolist() == [0, 1, 3, 5]


def test_concept_names_route_brand_only_drug_rows(synth_dir, index, tables, monkeypatch):
    """A drug row whose source text names nothing but whose concept is a lexicon sedative (brand name in the concept table) is
    kept by the streaming prefilter and becomes an exposure; without the concept it is invisible."""
    import pyarrow as pa
    store = data_io.open_store(synth_dir)
    t0 = index.set_index("person_id")["t0"]
    p = int(index["person_id"].iloc[3])
    row = pd.DataFrame({"person_id": [p], "drug_exposure_start_datetime": [t0[p] - pd.Timedelta(minutes=30)],
                        "drug_exposure_end_datetime": pd.NaT, "drug_source_value": "MED 12345", "quantity": 4.0,
                        "drug_concept_id": 777001, "drug_type_concept_id": 0, "route_source_value": "IV",
                        "drug_exposure_start_date": pd.NaT, "drug_exposure_end_date": pd.NaT, "visit_occurrence_id": 0})
    inj = {**tables, "omop_drug_exposure": pd.concat([tables["omop_drug_exposure"], row], ignore_index=True)}
    batch = pa.RecordBatch.from_pandas(pd.DataFrame({"concept_id": [777001, 777002],
                                                     "concept_name": ["Precedex 4 mcg/mL injection", "Acetaminophen 325 MG"]}))
    real = data_io.iter_omop_batches
    monkeypatch.setattr(bb.data_io, "iter_omop_batches",
                        lambda table, **kw: iter([batch]) if table == "concept" else real(table, **kw))
    assert bb.drug_concept_names(store)["concept_id"].tolist() == [777001]
    fs, _ = bb.build_matrices(FrameStream(inj, store), index, CFG)
    assert fs.X.loc[p, "sed__dexmedetomidine__on_t0"] == 1.0
    monkeypatch.setattr(bb.data_io, "iter_omop_batches", lambda table, **kw: iter([]) if table == "concept" else real(table, **kw))
    fs2, _ = bb.build_matrices(FrameStream(inj, store), index, CFG)
    assert fs2.X.loc[p, "sed__dexmedetomidine__on_t0"] == 0.0
