"""Scale / memory test: >= 2M visit rows and >= 5M measurement rows, generated in-process straight to parquet in
the HEEDB layout (row group by row group, so generation itself stays small), then the cohort build and the visit
diagnostic run in SUBPROCESSES whose peak RSS must stay under 4 GB. Synthetic data only."""

import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sortinghat import data_io

REPO = Path(__file__).resolve().parents[1]
N_EEG = 60_000                 # adult patients with one EEG (S0001 and I0008 layouts)
N_OTHER = 100_000              # patients with visits/measurements but no EEG (must be filtered out on read)
VISITS_PER = 35                # 60k x 35 = 2.1M visit rows for the EEG patients (+ 0.3M for the others)
MEAS_PER = 90                  # 60k x 90 = 5.4M measurement rows
RG = 250_000
BASE = pd.Timestamp("2020-01-01")
LAB = np.array(["SODIUM", "GLUCOSE", "CREATININE", "LACTATE, WHOLE BLOOD", "WBC", "Heart Rate", "SpO2"])
SCORE = np.array(["Glasgow Coma Scale Score", "FOUR Score", "Eye Opening"])
CONCEPTS = np.array([9201, 9202, 9203, 32037, 0])
SOURCE = np.array(["Inpatient", "Outpatient", "Emergency Room Visit", "Intensive Care", ""], dtype=object)


def _gen_visits(root: Path, rng):
    pids = np.arange(1, N_EEG + N_OTHER + 1) + 50_000_000
    owner = np.repeat(pids[:N_EEG], VISITS_PER)
    owner = np.concatenate([owner, np.repeat(pids[N_EEG:], 3)])
    start_d = rng.uniform(-200, 30, len(owner))
    dur_h = rng.exponential(120, len(owner))
    k = rng.integers(0, 5, len(owner))
    order = rng.permutation(len(owner))
    out = root / "OMOP/Merged/visit_occurrence"
    out.mkdir(parents=True)
    nparts = 4
    for part, idx in enumerate(np.array_split(order, nparts)):
        w = None
        for sub in np.array_split(idx, max(1, len(idx) // RG)):
            st = pd.Series(T0_OF(owner[sub]) + pd.to_timedelta(start_d[sub], unit="D")).astype("datetime64[us]")
            en = (st + pd.to_timedelta(dur_h[sub], unit="h")).where(rng.random(len(sub)) > 0.1, pd.NaT)
            t = pa.table({
                "person_id": pa.array(owner[sub], pa.int64()),
                "visit_occurrence_id": pa.array(sub, pa.int64()),
                "visit_start_datetime": pa.array(st.dt.strftime("%Y-%m-%d %H:%M:%S"), pa.string(), from_pandas=True),
                "visit_end_datetime": pa.array(en.dt.strftime("%Y-%m-%d %H:%M:%S"), pa.string(), from_pandas=True),
                "visit_concept_id": pa.array(CONCEPTS[k[sub]], pa.int64()),
                "visit_source_value": pa.array(SOURCE[k[sub]], pa.string())})
            w = w or pq.ParquetWriter(out / f"part-{part:05d}.parquet", t.schema)
            w.write_table(t)
        w.close()
    return pids


def T0_OF(person_ids):
    """Deterministic EEG start of a patient (so visits are placed around it)."""
    return (BASE + pd.to_timedelta((person_ids % 1000).astype("int64"), unit="D") + pd.Timedelta(hours=12)).to_numpy()


def _gen_measurements(root: Path, rng):
    pids = np.arange(1, N_EEG + 1) + 50_000_000
    owner = np.repeat(pids, MEAS_PER)
    off_h = rng.uniform(-30, 30, len(owner))
    is_score = rng.random(len(owner)) < 0.12
    names = np.where(is_score, SCORE[rng.integers(0, 3, len(owner))], LAB[rng.integers(0, len(LAB), len(owner))])
    val = np.where(is_score, rng.integers(3, 16, len(owner)), rng.normal(10, 4, len(owner)))
    order = rng.permutation(len(owner))
    out = root / "OMOP/Merged/measurement"
    out.mkdir(parents=True)
    for part, idx in enumerate(np.array_split(order, 6)):
        w = None
        for sub in np.array_split(idx, max(1, len(idx) // RG)):
            tt = pd.Series(T0_OF(owner[sub]) + pd.to_timedelta(off_h[sub], unit="h")).astype("datetime64[us]")
            t = pa.table({"person_id": pa.array(owner[sub], pa.int64()),
                          "measurement_datetime": pa.array(tt.dt.strftime("%Y-%m-%d %H:%M:%S"), pa.string(), from_pandas=True),
                          "measurement_source_value": pa.array(names[sub], pa.string()),
                          "value_as_number": pa.array(val[sub], pa.float64())})
            w = w or pq.ParquetWriter(out / f"part-{part:05d}.parquet", t.schema)
            w.write_table(t)
        w.close()


def _gen_eeg_tables(root: Path, rng):
    pids = np.arange(1, N_EEG + 1) + 50_000_000
    half = N_EEG // 2
    for site, ids in (("S0001", pids[:half]), ("I0008", pids[half:])):
        t0 = pd.Series(T0_OF(ids))
        age = rng.uniform(18, 90, len(ids))
        dur = rng.uniform(900, 4000, len(ids))
        sid = (900_000_000 + ids).astype(str)
        md = pd.DataFrame({"SiteID": site, "BDSPPatientID": ids.astype(str), "BidsFolder": [f"sub-{site}{i}" for i in ids],
                           "SessionID": sid, "EEGFolder": "eeg", "DurationInSeconds": dur, "ServiceName": "Routine",
                           "DateOfBirth": pd.NaT, "StartTime": pd.NaT, "EndTime": pd.NaT})
        if site == "I0008":
            md["StartTime"], md["EndTime"] = t0, t0 + pd.to_timedelta(dur, unit="s")
            md["DateOfBirth"] = (t0.dt.normalize() - pd.to_timedelta((age * 365.25).round(), unit="D"))
            md["ServiceName"], md["EEGFolder"] = None, None
        d = root / "EEG/eeg-metadata"
        d.mkdir(parents=True, exist_ok=True)
        data_io.denormalise_site_table("eeg_metadata", site, md).astype(object).to_csv(
            d / f"{site}_eeg_metadata_2026_04_30.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
        if site == "S0001":
            rf = pd.DataFrame({"BDSPPatientID": ids.astype(str), "SessionID": sid, "StartTime(EEG)": t0,
                               "EndTime(EEG)": t0 + pd.to_timedelta(dur, unit="s"), "AgeAtVisit": age,
                               "ServiceName(EEG)": "Routine"})
            r = root / "EEG/HEEDB_Metadata"
            r.mkdir(parents=True, exist_ok=True)
            data_io.denormalise_site_table("reports_findings", site, rf).astype(object).to_csv(
                r / f"{site}_EEG__reports_findings.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")


@pytest.fixture(scope="module")
def big_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("scale_store")
    rng = np.random.default_rng(7)
    _gen_eeg_tables(root, rng)
    _gen_visits(root, rng)
    _gen_measurements(root, rng)
    return root


def _run(script, *args):
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "SORTINGHAT_AGENT_SESSION")}
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], capture_output=True, text=True,
                          check=True, env=env, cwd=REPO)


def test_build_and_diag_stay_under_4gb(big_store, tmp_path):
    n_visits = sum(pq.ParquetFile(p).metadata.num_rows for p in (big_store / "OMOP/Merged/visit_occurrence").glob("*.parquet"))
    n_meas = sum(pq.ParquetFile(p).metadata.num_rows for p in (big_store / "OMOP/Merged/measurement").glob("*.parquet"))
    assert n_visits >= 2_000_000 and n_meas >= 5_000_000
    b = _run("build_cohort.py", "--data", str(big_store), "--out", str(tmp_path / "o"))
    d = _run("diag_cohort.py", "--data", str(big_store), "--out", str(tmp_path / "diag.json"))
    child_peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1 << 20)
    own_peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    print(f"\n[scale] visits={n_visits} measurements={n_meas} peak RSS: subprocess {child_peak_gb:.2f} GB, test {own_peak_gb:.2f} GB")
    assert child_peak_gb < 4.0, f"subprocess peak RSS {child_peak_gb:.2f} GB"
    assert own_peak_gb < 4.0, f"test process peak RSS {own_peak_gb:.2f} GB"
    # the scripts report their own peak, and the run did real work
    assert "peak RSS" in b.stdout and "peak RSS" in d.stderr
    rep = json.loads(d.stdout)
    assert rep["visit_table"]["n_visit_rows_for_adult_candidates"] >= 2_000_000
    t = pd.read_csv(tmp_path / "o" / "local_only" / "cohort_study1.csv", low_memory=False)
    assert len(t) > 1000 and t["person_id"].is_unique


def test_phase0a_audit_stays_under_4gb_on_the_same_store(big_store, tmp_path):
    """The audit reuses the cohort's compact visits, two-stage row-group reads and candidate set."""
    import time
    t = time.time()
    p = _run_module("sortinghat.audit.field_audit", "--data", str(big_store), "--out", str(tmp_path / "audit"))
    took = time.time() - t
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1 << 20)
    print(f"\n[scale] audit {took:.0f} s, peak RSS (children, cumulative max) {peak:.2f} GB")
    assert peak < 4.0, f"peak RSS {peak:.2f} GB"
    rep = json.loads((tmp_path / "audit" / "field_audit.json").read_text())
    assert len(rep["rows"]) == 8 and rep["n_candidates"] != "<11"


def _run_module(module, *args):
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "SORTINGHAT_AGENT_SESSION")}
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True, check=True, env=env, cwd=REPO)
