"""Synthetic HEEDB-shaped dataset (numpy/pandas only). Entirely fake.

Deliberately injected defects so the field audit has something to find:
  * StartTime missing for ~2-5% of EEGs (site dependent).
  * MAR (administration) coverage is poor at some sites; orders without admin times.
  * Imaging report finalization time missing (worse at one site).
  * Lab result time missing (~1%) and a few collect > result violations.
  * ~3% of patients have one ancillary table (notes/labs/imaging/meds) carrying an
    extra date offset, i.e. an inconsistent within-patient date shift.
  * Score documentation near the EEG is patchy; one pediatric site.
Every patient also receives a random per-patient date shift (consistent across
tables), mirroring BDSP de-identification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema
from ..tableio import write_tables as _write_tables

SITE_WEIGHTS = {"S0001": 0.40, "S0002": 0.30, "I0003": 0.20, "I0002": 0.10}
PEDIATRIC_SITE = "I0002"
START_MISSING = {"S0001": 0.02, "S0002": 0.05, "I0003": 0.02, "I0002": 0.02}
MAR_COVERAGE = {"S0001": 0.93, "S0002": 0.55, "I0003": 0.85, "I0002": 0.70}
IMG_FINAL_MISSING = {"S0001": 0.04, "S0002": 0.06, "I0003": 0.20, "I0002": 0.08}
SCORE_NEAR = {"S0001": 0.75, "S0002": 0.60, "I0003": 0.35, "I0002": 0.50}
TABLE_SHIFT_RATE = 0.03
LAB_VIOLATION_RATE = 0.004

CLASSES = ["ICU", "Inpatient", "ED", "Outpatient"]
CLASS_P = [0.25, 0.35, 0.10, 0.30]
ACUTE = {"ICU", "Inpatient", "ED"}
INDICATIONS = ["rule out NCSE", "post-arrest", "unexplained AMS", "seizure", "spell", "other"]
MEDS = [("propofol", "sedative"), ("midazolam", "sedative"), ("dexmedetomidine", "sedative"),
        ("fentanyl", "analgesic"), ("lorazepam", "antiseizure"), ("levetiracetam", "antiseizure"),
        ("acetaminophen", "other"), ("heparin", "other")]
LABS = ["lactate", "ammonia", "sodium", "glucose", "creatinine", "bun", "ast", "alt", "wbc"]
MODALITIES = ["CT head", "MRI brain", "CTA head"]
NOTE_TYPES = ["physician", "nursing", "eeg_report", "other"]
SHIFTABLE = ["notes", "labs", "imaging", "medications"]
TIME_COLS = {"notes": ["note_time"], "labs": ["collect_time", "result_time"],
             "imaging": ["study_time", "report_final_time"],
             "medications": ["order_time", "admin_time"]}


def _ts(base: pd.Timestamp, hours: float) -> pd.Timestamp:
    return base + pd.Timedelta(minutes=float(hours) * 60.0)


def generate(n_patients: int = 3000, seed: int = 20260101):
    """Return ``(tables, truth)``. ``truth`` holds in-memory ground truth for tests only."""
    rng = np.random.default_rng(seed)
    sites = rng.choice(list(SITE_WEIGHTS), size=n_patients, p=list(SITE_WEIGHTS.values()))

    eeg, meds, labs, img, scores, notes = [], [], [], [], [], []
    truth = {"table_shift_ids": [], "table_shift_table": {}, "no_start_sessions": 0}
    note_counter = 0

    for i in range(n_patients):
        site = str(sites[i])
        pid = f"SYN{site}{i:06d}"
        # --- per-patient demographics and date shift (consistent across tables)
        if site == PEDIATRIC_SITE and rng.random() < 0.9:
            age = float(rng.uniform(0.1, 17.9))
        else:
            age = float(np.clip(rng.normal(58, 18), 18, 95))
        sex = str(rng.choice(["Female", "Male"]))
        true_anchor = pd.Timestamp("2014-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
        shift = pd.Timedelta(days=int(rng.integers(-1800, 1800)))
        anchor = true_anchor + shift + pd.Timedelta(minutes=int((rng.normal(13, 5) % 24) * 60))

        # --- sessions
        n_sess = int(min(1 + rng.poisson(0.5), 4))
        classes = [str(rng.choice(CLASSES, p=CLASS_P)) for _ in range(n_sess)]
        classes[0] = str(rng.choice(CLASSES, p=[0.30, 0.40, 0.10, 0.20]))
        offsets = [0] + sorted(int(x) for x in rng.integers(1, 400, n_sess - 1))
        starts = [anchor + pd.Timedelta(days=o) for o in offsets]
        first_acute = next((k for k, c in enumerate(classes) if c in ACUTE), None)
        start_missing = [bool(rng.random() < START_MISSING[site]) for _ in range(n_sess)]
        if first_acute is not None and start_missing[first_acute]:
            # keep ancillary anchoring unambiguous: later sessions are non-acute
            for k in range(first_acute + 1, n_sess):
                classes[k] = "Outpatient"
        for k in range(n_sess):
            svc = "Routine" if classes[k] in ("Outpatient", "ED") else str(
                rng.choice(["Routine", "LTM"], p=[0.5, 0.5]))
            dur = float(rng.uniform(1200, 3600)) if svc == "Routine" else float(rng.uniform(12, 72) * 3600)
            st = starts[k]
            miss = start_missing[k]
            eeg.append({
                "SiteID": site, "BDSPPatientID": pid,
                "BidsFolder": f"sub-{pid}", "SessionID": f"ses-{pid}-{k + 1:02d}",
                "CreationTime": pd.NaT if miss else st - pd.Timedelta(minutes=int(rng.integers(0, 6))),
                "StartTime": pd.NaT if miss else st,
                "EndTime": st + pd.Timedelta(seconds=dur),
                "DurationInSecond": round(dur, 1), "ServiceName": svc,
                "AgeAtVisit": round(age, 2), "SexDSC": sex, "PatientClass": classes[k],
                "ReferralIndication": str(rng.choice(INDICATIONS)),
            })
            truth["no_start_sessions"] += int(miss)
        if first_acute is None:
            continue
        t0 = starts[first_acute]

        p_meds, p_labs, p_img, p_scores, p_notes = [], [], [], [], []
        # --- medications: orders always; admin only if patient has MAR feed
        has_mar = rng.random() < MAR_COVERAGE[site]
        for _ in range(int(rng.poisson(2.5))):
            name, cls = MEDS[int(rng.integers(len(MEDS)))]
            order = _ts(t0, rng.uniform(-48, 12))
            admin = order + pd.Timedelta(minutes=float(5 + rng.exponential(40)))
            if not (has_mar and rng.random() < 0.9):
                admin = pd.NaT
            p_meds.append({"BDSPPatientID": pid, "med_name": name, "med_class": cls,
                           "order_time": order, "admin_time": admin})
        # --- labs
        for _ in range(int(2 + rng.poisson(3))):
            col = _ts(t0, rng.uniform(-48, 24))
            lag = pd.Timedelta(minutes=float(15 + rng.exponential(60)))
            res = col + lag
            if rng.random() < LAB_VIOLATION_RATE:
                res = col - lag
            if rng.random() < 0.01:
                res = pd.NaT
            p_labs.append({"BDSPPatientID": pid, "lab_name": str(rng.choice(LABS)),
                           "collect_time": col, "result_time": res})
        # --- imaging
        for _ in range(int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20]))):
            st = _ts(t0, rng.uniform(-72, 24))
            fin = st + pd.Timedelta(minutes=float(20 + rng.exponential(120)))
            if rng.random() < IMG_FINAL_MISSING[site]:
                fin = pd.NaT
            p_img.append({"BDSPPatientID": pid, "modality": str(rng.choice(MODALITIES)),
                          "study_time": st, "report_final_time": fin})
        # --- scores
        if rng.random() < SCORE_NEAR[site]:
            for _ in range(int(rng.integers(1, 4))):
                p_scores.append(_score_row(rng, pid, _ts(t0, rng.uniform(-5.5, 5.5))))
        for _ in range(int(rng.integers(0, 4))):
            sgn = 1 if rng.random() < 0.5 else -1
            p_scores.append(_score_row(rng, pid, _ts(t0, sgn * rng.uniform(12, 72))))
        # --- notes (metadata only)
        if rng.random() > 0.02:
            for _ in range(int(1 + rng.poisson(4))):
                note_counter += 1
                tm = _ts(t0, rng.uniform(-72, 72))
                if rng.random() < 0.02:
                    tm = pd.NaT
                p_notes.append({"BDSPPatientID": pid, "note_id": f"SYNNOTE{note_counter:08d}",
                                "note_type": str(rng.choice(NOTE_TYPES)), "note_time": tm})

        # --- inconsistent date shift injected into one ancillary table
        if rng.random() < TABLE_SHIFT_RATE:
            tbl = str(rng.choice(SHIFTABLE))
            off = pd.Timedelta(days=int(rng.integers(60, 400)) * int(rng.choice([-1, 1])))
            rows = {"notes": p_notes, "labs": p_labs, "imaging": p_img, "medications": p_meds}[tbl]
            for r in rows:
                for c in TIME_COLS[tbl]:
                    if not pd.isna(r[c]):
                        r[c] = r[c] + off
            truth["table_shift_ids"].append(pid)
            truth["table_shift_table"][pid] = tbl
        meds += p_meds; labs += p_labs; img += p_img; scores += p_scores; notes += p_notes

    def df(rows, table):
        d = pd.DataFrame(rows, columns=schema.columns(table))
        for c in schema.datetime_cols(table):
            d[c] = pd.to_datetime(d[c])
        return d

    tables = {"eeg_metadata": df(eeg, "eeg_metadata"), "medications": df(meds, "medications"),
              "labs": df(labs, "labs"), "imaging": df(img, "imaging"),
              "clinical_scores": df(scores, "clinical_scores"), "notes": df(notes, "notes")}
    truth["seed"] = seed
    return tables, truth


def _score_row(rng, pid, tm):
    kind = str(rng.choice(["GCS", "FOUR", "RASS"], p=[0.6, 0.15, 0.25]))
    val = {"GCS": lambda: rng.integers(3, 16), "FOUR": lambda: rng.integers(0, 17),
           "RASS": lambda: rng.integers(-5, 5)}[kind]()
    return {"BDSPPatientID": pid, "score_type": kind, "score_value": float(val), "score_time": tm}


def write_tables(tables, outdir, truth=None, fmt: str = "auto"):
    paths = _write_tables(tables, outdir, fmt)
    manifest = {"synthetic": True, "seed": None if truth is None else truth["seed"],
                "n_patients": int(tables["eeg_metadata"]["BDSPPatientID"].nunique()),
                "n_injected_table_shift_patients": None if truth is None else len(truth["table_shift_ids"]),
                "note": "Entirely fake data. No real patients."}
    (Path(outdir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths
