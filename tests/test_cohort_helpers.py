"""Shared helpers for the cohort tests (no tests here). Synthetic, hand-built HEEDB-layout tables only.

``World`` builds raw tables in the real layout, including the per-site variants: S-sites keep the EEG start in
reports_findings (metadata StartTime blank); I0008/I0009 have the start in eeg_metadata (StartDateTime ->
StartTime), a DateOfBirth instead of AgeAtVisit, no ServiceName and NO reports_findings row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sortinghat import schema

T0 = pd.Timestamp("2021-03-01 12:00:00")
H = lambda h: pd.Timedelta(hours=h)      # noqa: E731
VISIT_CONCEPT = {"ICU": 32037, "Inpatient": 9201, "ED": 9203, "Outpatient": 9202}
GCS = "Glasgow Coma Scale Score"
FOUR = "FOUR Score"


def _df(table: str, rows: list[dict]) -> pd.DataFrame:
    return schema.coerce_types(table, pd.DataFrame(rows, columns=schema.columns(table)))


class World:
    def __init__(self):
        self.meta, self.rf, self.visits, self.meas, self.cond = [], [], [], [], []
        self._sid = 0
        self._vid = 0

    # ---------------------------------------------------------------- eeg sessions
    def eeg(self, pid, site="S0001", t0=T0, age=60.0, dur=1800.0, service="Routine", start_blank=False,
            end_blank=False, sid=None, id_blank=False, age_blank=False, eeg_folder="eeg", bids_blank=False,
            clock_dur=None):
        self._sid += 1
        sid = sid or f"sess{self._sid:05d}"
        md_start = site in ("I0008", "I0009")
        bids = f"sub-{site}{pid}"
        end = t0 + pd.Timedelta(seconds=clock_dur if clock_dur is not None else (dur if dur is not None else 1800.0))
        row = {"SiteID": site, "BDSPPatientID": None if id_blank else str(pid),
               "BidsFolder": None if bids_blank else bids, "SessionID": sid,
               "EEGFolder": None if md_start or site == "I0003" else eeg_folder,
               "DurationInSeconds": dur if dur is not None else np.nan,
               "ServiceName": None if md_start else service,
               "StartTime": (pd.NaT if start_blank else t0) if md_start else pd.NaT,
               "EndTime": (pd.NaT if end_blank else end) if md_start else pd.NaT,
               "DateOfBirth": (t0.normalize() - pd.Timedelta(days=int(round(age * 365.25)))) if md_start and
               not age_blank else pd.NaT}
        self.meta.append(row)
        if not md_start:
            self.rf.append({"BDSPPatientID": str(pid), "SessionID": sid,
                            schema.START_EEG: pd.NaT if start_blank else t0,
                            schema.END_EEG: pd.NaT if end_blank else end,
                            "AgeAtVisit": np.nan if age_blank else age, schema.SERVICE_EEG: service})
        return sid

    # ---------------------------------------------------------------- visits, scores, conditions
    def visit(self, pid, cls="ICU", start=T0 - H(3), end=None, concept=True, text=None):
        self._vid += 1
        self.visits.append({
            "person_id": int(pid), "visit_occurrence_id": self._vid, "visit_start_datetime": start,
            "visit_end_datetime": end if end is not None else pd.NaT,
            "visit_concept_id": VISIT_CONCEPT[cls] if concept else 0, "visit_type_concept_id": 0,
            "visit_source_value": text if text is not None else (None if concept else None)})

    def score(self, pid, name, t, value):
        self.meas.append({"person_id": int(pid), "measurement_datetime": t, "measurement_date": t.normalize(),
                          "measurement_time": t.strftime("%H:%M:%S"), "measurement_source_value": name,
                          "value_as_number": float(value)})

    def cond_row(self, pid, code, t):
        self.cond.append({"person_id": int(pid), "condition_start_datetime": t, "condition_source_value": code,
                          "condition_concept_id": 0})

    def patient(self, pid, *, gcs=8.0, gcs_at=-1.0, visit_cls="ICU", visit_start_h=-3.0, **eeg_kw):
        """The default included patient: ICU visit 3 h before t0 and one GCS at t0 - 1 h (strict)."""
        t0 = eeg_kw.get("t0", T0)
        self.eeg(pid, **eeg_kw)
        if visit_cls is not None:
            self.visit(pid, visit_cls, t0 + H(visit_start_h))
        if gcs is not None:
            self.score(pid, GCS, t0 + H(gcs_at), gcs)

    # ---------------------------------------------------------------- tables
    def tables(self) -> dict[str, pd.DataFrame]:
        t = {name: _df(name, []) for name in schema.TABLE_NAMES}
        t["eeg_metadata"] = _df("eeg_metadata", self.meta)
        t["reports_findings"] = _df("reports_findings", self.rf)
        t["omop_visit_occurrence"] = _df("omop_visit_occurrence", self.visits)
        t["omop_measurement"] = _df("omop_measurement", self.meas)
        t["omop_condition_occurrence"] = _df("omop_condition_occurrence", self.cond)
        return t


def removed_at(result, label_start: str) -> int:
    """Exact number removed by the (unmerged) flow step whose label starts with ``label_start``."""
    steps = result.flow_raw["steps"]
    for prev, cur in zip(steps, steps[1:]):
        if cur["label"].startswith(label_start):
            return sum(prev["remaining"].values()) - sum(cur["remaining"].values())
    raise KeyError(label_start)


def remaining_after(result, label_start: str) -> int:
    for s in result.flow_raw["steps"]:
        if s["label"].startswith(label_start):
            return sum(s["remaining"].values())
    raise KeyError(label_start)
