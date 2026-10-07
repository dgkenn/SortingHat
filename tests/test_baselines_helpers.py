"""Shared helpers for the baseline tests (no tests here). Synthetic data only.

* ``mini_tables``: hand-built HEEDB-layout tables for exact-value assertions.
* ``augment_bedside``: adds the domains the generator lacks (vitals, pupils, POC glucose, tox, cultures,
  convulsion observation, arrest/trauma conditions, more drugs) PRE-t0 for a subset of synthetic patients.
* ``post_t0_events``: tables with strictly-post-t0 events (sentinel values) in every domain.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from sortinghat import schema

T0 = pd.Timestamp("2020-01-01 12:00:00")
H = lambda h: pd.Timedelta(hours=h)      # noqa: E731
SENTINELS = (9999.0, 777.0, 1.0e6, 424242.0)


def _df(table: str, rows: list[dict]) -> pd.DataFrame:
    return schema.coerce_types(table, pd.DataFrame(rows, columns=schema.columns(table)))


def mini_tables(pid: int = 1, indication: str = "post-arrest", sex: str = "Male", age: float = 61.0,
                site: str = "S0001", t0: pd.Timestamp = T0) -> dict[str, pd.DataFrame]:
    """One adult ICU patient with one EEG at ``t0``; all other tables empty (add rows with ``add_rows``)."""
    t = {name: _df(name, []) for name in schema.TABLE_NAMES}
    t["eeg_metadata"] = _df("eeg_metadata", [{
        "SiteID": site, "BDSPPatientID": str(pid), "BidsFolder": f"sub-{site}{pid}", "SessionID": f"s{pid}",
        "EEGFolder": "eeg", "DurationInSeconds": 1800.0, "ServiceName": "Routine", "PatientClass": "ICU",
        "ReferralIndication": indication, "SexDSC": sex}])
    t["reports_findings"] = _df("reports_findings", [{
        "BDSPPatientID": str(pid), "SessionID": f"s{pid}", schema.START_EEG: t0,
        schema.END_EEG: t0 + H(0.5), "AgeAtVisit": age, "SexDSC": sex, schema.SERVICE_EEG: "Routine"}])
    return t


def add_rows(tables: dict, table: str, rows: list[dict]) -> dict:
    """Return a copy of ``tables`` with ``rows`` appended to ``table``. Keys outside the schema (e.g.
    ``measurement_result_datetime``) are kept as extra datetime columns."""
    out = dict(tables)
    new = pd.DataFrame(rows)
    cols = schema.columns(table)
    extras = [c for c in new.columns if c not in cols]
    for c in cols:
        if c not in new:
            new[c] = None
    new = new[cols + extras]
    typed = schema.coerce_types(table, new[cols])
    for c in extras:
        typed[c] = schema.parse_datetimes(new[c])
    base = tables[table]
    for c in extras:
        if c not in base:
            base = base.assign(**{c: pd.Series(pd.NaT, index=base.index, dtype="datetime64[us]")})
    out[table] = pd.concat([base, typed], ignore_index=True)
    return out


def meas(pid, name, t, value, unit=None) -> dict:
    return {"person_id": pid, "measurement_datetime": t, "measurement_date": pd.Timestamp(t).normalize(),
            "measurement_source_value": name, "value_as_number": value, "unit_source_value": unit,
            "measurement_concept_id": 0}


def drug(pid, name, start, end=None, qty=1.0) -> dict:
    return {"person_id": pid, "drug_exposure_start_datetime": start, "drug_exposure_end_datetime": end,
            "drug_source_value": name, "quantity": qty, "drug_concept_id": 0}


def img(pid, modality, study, final=None) -> dict:
    return {"person_id": pid, "modality": modality, "study_datetime": study, "report_final_datetime": final}


# --------------------------------------------------------------------------------------- synthetic-scale
def _t0s(tables: dict) -> pd.Series:
    from sortinghat.baselines import build_index
    idx = build_index(tables)
    return idx.set_index("person_id")["t0"]


def augment_bedside(tables: dict, seed: int = 7, frac: float = 0.6) -> dict:
    """PRE-t0 bedside/lab/tox/culture/history rows for a random subset of indexed patients."""
    rng = np.random.default_rng(seed)
    t0 = _t0s(tables)
    new: dict[str, list] = {k: [] for k in ("omop_measurement", "omop_drug_exposure", "omop_condition_occurrence",
                                           "omop_observation", "omop_procedure_occurrence", "imaging")}
    for pid, t in t0.items():
        if rng.random() > frac:
            continue
        pid = int(pid)
        for nm, lo, hi in (("Heart rate", 40, 150), ("Systolic blood pressure", 70, 190), ("Respiratory rate", 8, 30),
                           ("SpO2", 85, 100), ("Temperature", 96, 103), ("Mean arterial pressure", 50, 110)):
            new["omop_measurement"].append(meas(pid, nm, t - H(rng.uniform(0.2, 5)), float(rng.uniform(lo, hi)),
                                                "F" if nm == "Temperature" else None))
        new["omop_measurement"].append(meas(pid, "Pupil size left", t - H(1), float(rng.integers(1, 7))))
        new["omop_measurement"].append(meas(pid, "Pupil size right", t - H(1), float(rng.integers(1, 7))))
        new["omop_measurement"].append(meas(pid, "Pupil reactivity left", t - H(1), float(rng.integers(0, 2))))
        new["omop_measurement"].append(meas(pid, "POC glucose (fingerstick)", t - H(rng.uniform(0.2, 20)),
                                            float(rng.uniform(40, 300)), "mg/dL"))
        new["omop_measurement"].append(meas(pid, "Ethanol level, serum", t - H(rng.uniform(1, 30)), float(rng.uniform(0, 300))))
        new["omop_measurement"].append(meas(pid, "Blood culture", t - H(rng.uniform(30, 60)), float(rng.integers(0, 2))))
        new["omop_measurement"].append(meas(pid, "POTASSIUM", t - H(rng.uniform(1, 40)), float(rng.uniform(3, 6))))
        new["omop_drug_exposure"].append(drug(pid, "KETAMINE 50 MG/ML INJ", t - H(rng.uniform(0.5, 20)),
                                              t - H(rng.uniform(-3, 0.4)) if rng.random() < 0.5 else None, 2.0))
        new["omop_drug_exposure"].append(drug(pid, "MORPHINE SULFATE 4 MG/ML INJ", t - H(rng.uniform(0.5, 30)), None, 4.0))
        new["omop_drug_exposure"].append(drug(pid, "HYDROMORPHONE 1 MG/ML INJ", t - H(rng.uniform(0.5, 30)),
                                              t - H(0.1), 1.0))
        if rng.random() < 0.3:
            new["omop_observation"].append({"person_id": pid, "observation_concept_id": 0,
                                            "observation_datetime": t - H(2), "observation_date": (t - H(2)).normalize(),
                                            "observation_source_value": "Witnessed seizure activity",
                                            "value_as_string": "yes"})
        if rng.random() < 0.2:
            new["omop_condition_occurrence"].append({"person_id": pid, "condition_start_datetime": t - H(10),
                                                     "condition_source_value": "S06.5X0A", "condition_concept_id": 0})
        if rng.random() < 0.2:
            new["omop_procedure_occurrence"].append({"person_id": pid, "procedure_datetime": t - H(20),
                                                     "procedure_date": (t - H(20)).normalize(),
                                                     "procedure_concept_id": 0, "procedure_source_value": "92950"})
    out = dict(tables)
    for k, rows in new.items():
        if rows:
            out[k] = add_rows(out, k, rows)[k]
    return out


def post_t0_events(tables: dict, hours_after: float = 0.25) -> dict:
    """Append strictly-post-t0 sentinel events in EVERY event domain for every indexed patient.

    Nothing added here may change any feature: each event has start/collection/availability > t0.
    """
    t0 = _t0s(tables)
    rows: dict[str, list] = {k: [] for k in ("omop_measurement", "omop_drug_exposure", "omop_condition_occurrence",
                                            "omop_observation", "omop_procedure_occurrence", "imaging")}
    for pid, t in t0.items():
        pid, a = int(pid), t + H(hours_after)
        for nm, v in (("Glasgow Coma Scale Score", 3.0), ("FOUR Score", 0.0), ("RASS", -5.0), ("NESI", 777.0),
                      ("Heart rate", 777.0), ("Systolic blood pressure", 777.0), ("Pupil reactivity left", 0.0),
                      ("Pupil size left", 9.0), ("Pupil size right", 1.0), ("POC glucose (fingerstick)", 9999.0),
                      ("SODIUM", 9999.0), ("LACTATE, WHOLE BLOOD", 9999.0), ("AMMONIA, PLASMA", 9999.0),
                      ("Ethanol level, serum", 9999.0), ("Blood culture", 1.0), ("CSF culture", 1.0),
                      ("CSF WBC count", 9999.0), ("Witnessed seizure", 1.0), ("Cardiac arrest", 1.0)):
            rows["omop_measurement"].append(meas(pid, nm, a + H(1.0), v))      # +1h: also clears lab lag
        for nm in ("PROPOFOL 10 MG/ML IV EMULSION", "FENTANYL 50 MCG/ML INJ", "MIDAZOLAM 1 MG/ML INJ",
                   "KETAMINE 50 MG/ML INJ"):
            rows["omop_drug_exposure"].append(drug(pid, nm, a, a + H(2), 1.0e6))     # admin, entirely after t0
            rows["omop_drug_exposure"].append(drug(pid, nm, a + H(0.5), None, 1.0e6))  # order only, after t0
        rows["omop_condition_occurrence"].append({"person_id": pid, "condition_start_datetime": a,
                                                  "condition_source_value": "I46.9", "condition_concept_id": 0})
        rows["omop_condition_occurrence"].append({"person_id": pid, "condition_start_datetime": a,
                                                  "condition_source_value": "S06.0X0A", "condition_concept_id": 0})
        rows["omop_observation"].append({"person_id": pid, "observation_concept_id": 0, "observation_datetime": a,
                                         "observation_date": a.normalize(),
                                         "observation_source_value": "Witnessed convulsion", "value_as_string": "yes"})
        rows["omop_procedure_occurrence"].append({"person_id": pid, "procedure_datetime": a,
                                                  "procedure_date": a.normalize(), "procedure_concept_id": 0,
                                                  "procedure_source_value": "92950"})
        for mod in ("CT head", "MRI brain", "CTA head"):
            rows["imaging"].append(img(pid, mod, a, a + H(1)))                    # study and final after t0
    out = dict(tables)
    for k, r in rows.items():
        out[k] = add_rows(out, k, r)[k]
    return out


def clone(tables: dict) -> dict:
    return {k: v.copy() for k, v in tables.items()}
