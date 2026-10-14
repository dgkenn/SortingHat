"""SYNTHETIC fixtures shared by the adjudication tests (every id, value and note below is made up) plus tests of the fixtures' own
contract.  Nothing here touches HEEDB data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

T_BASE = pd.Timestamp("2024-05-01 10:00:00")
PID0 = 91_000_000

EEG_NOTE = ("Neurology consult. The EEG showed generalized slowing with triphasic waves. Plan: continue levetiracetam. "
            "Burst suppression was seen on the monitoring.")
CLEAN_NOTE = "Admitted with confusion after a fall. Sodium 118 on arrival. CT head without hemorrhage. Plan: slow sodium correction."
H_AND_P = "Found down at home. Smelled of alcohol. GCS 8 in the ED.\nEEG ordered for evaluation of unresponsiveness.\nBIS 45 on propofol."


def person_ids(n):
    return [PID0 + i for i in range(n)]


def make_tables(n: int = 12, seed: int = 0, empty_idx=None) -> tuple[dict, pd.DataFrame]:
    """({'omop_<table>': DataFrame}, sample DataFrame) for ``n`` fake patients; patient 0 has rich data; the patients in ``empty_idx``
    (default: the last) have no structured rows and no notes."""
    rng = np.random.default_rng(seed)
    pids = person_ids(n)
    t0 = {p: T_BASE + pd.Timedelta(hours=int(i)) for i, p in enumerate(pids)}
    meas, drug, cond, proc, obs, note = [], [], [], [], [], []

    def at(p, h):
        return t0[p] + pd.Timedelta(hours=float(h))

    empty = {pids[i] for i in (empty_idx if empty_idx is not None else (n - 1,))}
    for i, p in enumerate(pids):
        if p in empty:
            continue
        for h in np.arange(-10, 10.01, 0.5):
            meas.append((p, at(p, h), "Heart Rate", 70 + rng.normal(0, 5), "bpm"))
        for h, v in ((-2, 120.0), (6, 128.0), (400, 130.0), (-100, 119.0), (900, 135.0)):
            meas.append((p, at(p, h), "Sodium", v, "mmol/L"))
        for h, v in ((-1, 8), (3, 9), (30, 13)):
            meas.append((p, at(p, h), "Glasgow Coma Scale Total", v, "score"))
        meas.append((p, at(p, 1), "BIS Index", 40.0, ""))
        meas.append((p, at(p, 1), "Suppression Ratio", 20.0, "%"))
        meas.append((p, at(p, 2), "EEG Background Continuity", np.nan, ""))
        drug.append((p, at(p, -1), at(p, 5), "propofol 10 mg/mL infusion", "IV", 1.0))
        drug.append((p, at(p, 2), at(p, 2), "lorazepam 2 mg/mL injection", "IV", 2.0))
        drug.append((p, at(p, -200), at(p, 10), "insulin glargine", "SC", 20.0))               # running at the window start: kept
        drug.append((p, at(p, -300), at(p, -250), "ceftriaxone", "IV", 1.0))                    # ended long before: dropped
        cond.append((p, at(p, 3), "I63.9"))
        cond.append((p, at(p, 3), "G93.41"))                                                    # nonspecific encephalopathy: blocked
        cond.append((p, at(p, 4), "R94.01"))                                                    # abnormal EEG: blocked
        cond.append((p, at(p, -24 * 30), "J18.9"))                                              # prior history: kept, flagged
        cond.append((p, at(p, -24 * 400), "E11.9"))                                             # older than the history window: dropped
        proc.append((p, at(p, 2), "95816"))                                                     # EEG CPT: blocked
        proc.append((p, at(p, 2), "99291"))
        obs.append((p, at(p, 1), "Neuro exam", "pupils reactive"))
        obs.append((p, at(p, 1), "Neuro monitoring", "EEG showed seizures"))                    # blocked by text
        for j, (title, text, h) in enumerate(((f"Admission H&P", H_AND_P, -2), ("Neurology consult", EEG_NOTE, 5),
                                              ("Progress Note", CLEAN_NOTE, 20), ("EEG report", "Routine EEG. Normal.", 6),
                                              ("Old Note", CLEAN_NOTE, -500))):
            note.append((p, at(p, h), title, "", text))
    t = {
        "omop_measurement": pd.DataFrame(meas, columns=["person_id", "measurement_datetime", "measurement_source_value", "value_as_number", "unit_source_value"]),
        "omop_drug_exposure": pd.DataFrame(drug, columns=["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime", "drug_source_value",
                                                          "route_source_value", "quantity"]),
        "omop_condition_occurrence": pd.DataFrame(cond, columns=["person_id", "condition_start_datetime", "condition_source_value"]),
        "omop_procedure_occurrence": pd.DataFrame(proc, columns=["person_id", "procedure_datetime", "procedure_source_value"]),
        "omop_observation": pd.DataFrame(obs, columns=["person_id", "observation_datetime", "observation_source_value", "value_as_string"]),
        "omop_note": pd.DataFrame(note, columns=["person_id", "note_datetime", "note_title", "note_source_value", "note_text"]),
    }
    roles = ["sample"] * (n - 2) + ["duplicate", "calibration"]
    s = pd.DataFrame({"sample_id": [f"s{i + 1:04d}" for i in range(n)], "person_id": pids, "role": roles,
                      "duplicate_of": [""] * (n - 2) + ["s0001", ""], "site": ["S0001" if i % 2 == 0 else "S0002" for i in range(n)],
                      "t0": [t0[p] for p in pids], "age_years": 40 + np.arange(n), "sex_male": [i % 2 for i in range(n)],
                      "hours_since_visit_start": 5.0 + np.arange(n)})
    s.loc[n - 2, "person_id"] = pids[0]                       # the duplicate repeats patient 0
    s.loc[n - 2, "t0"] = t0[pids[0]]
    return t, s


def write_cache(tables: dict, root: Path, with_note_text: bool = False) -> Path:
    """The tables as an OMOP cache directory (one row group per table; the cache's `note` holds timestamps only)."""
    for key, df in tables.items():
        name = key[len("omop_"):]
        if name == "note" and not with_note_text:
            df = df[["person_id", "note_datetime"]]
        part = Path(root) / name / "profile0" / "part0"
        part.mkdir(parents=True, exist_ok=True)
        (part / "manifest.json").write_text(json.dumps({"n": 1}))
        df.to_parquet(part / "rg-000000.parquet", index=False)
    return Path(root)


def write_store(tables: dict, root: Path) -> Path:
    """The tables in the access-point layout read by data_io.LocalStore (--data)."""
    for key, df in tables.items():
        d = Path(root) / "OMOP" / "Merged" / key[len("omop_"):]
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "part-0.parquet", index=False)
    return Path(root)


@pytest.fixture(scope="module")
def tables_and_sample():
    return make_tables()


def test_fixture_shapes(tables_and_sample):
    t, s = tables_and_sample
    assert set(t) == {"omop_measurement", "omop_drug_exposure", "omop_condition_occurrence", "omop_procedure_occurrence", "omop_observation",
                      "omop_note"}
    assert s["role"].value_counts().to_dict() == {"sample": 10, "duplicate": 1, "calibration": 1}
    assert (s["duplicate_of"] == "s0001").sum() == 1


def test_write_cache_keeps_notes_timestamp_only(tmp_path, tables_and_sample):
    t, _ = tables_and_sample
    root = write_cache(t, tmp_path / "cache")
    cols = pd.read_parquet(next((root / "note").rglob("rg-*.parquet"))).columns.tolist()
    assert cols == ["person_id", "note_datetime"]
