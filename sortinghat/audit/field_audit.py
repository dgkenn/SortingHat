"""Phase 0a HEEDB field audit (metadata only, no waveforms).

Computes aggregates per plan row, applies the pass criterion, and writes a
markdown + JSON report. Output is aggregate-only with small-cell suppression
(n < 11 -> "<11"). The "20 hand-checked cases" sampling list is written ONLY to
a private local file and is never printed or included in reports.

Tables are read through ``sortinghat.data_io`` in the REAL HEEDB layout
(``docs/heedb_schema_real.md``): per-site eeg-metadata / reports_findings CSVs and
``OMOP/Merged/<table>/*.parquet``. Two sources, same code path:

    # synthetic or a local mirror of the layout
    python -m sortinghat.audit.field_audit --data <dir> --out <dir>
    # the real bucket (HUMAN-RUN ONLY, from your own terminal)
    scripts/heedb_run.sh python -m sortinghat.audit.field_audit --s3 --out <dir>

First thing to run on real data (names only, no values, no rows read):

    scripts/heedb_run.sh python -m sortinghat.audit.field_audit --s3 --dry-run-schema [--out <dir>]
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agent_safety, data_io, schema
from ..safe_output import (SUPPRESSED, safe_print, safe_quantiles, safe_write_json,
                           safe_write_text, suppress_count, suppress_proportion,
                           write_local_only)

ACUTE_CLASSES = {"ICU", "Inpatient", "ED"}
ADULT_AGE = 18
# Row filters. drug_source_value / measurement_source_value are free text in the real OMOP tables (no med_class,
# no score_type column); classes are derived by regex, as the earlier research code did.
SEDATION_RE = re.compile(r"propofol|midazolam|dexmedetomidine|precedex|fentanyl|ketamine|pentobarbital", re.I)
# Score classes. The Phase 0a row is "GCS, FOUR or RASS"; GCS components (eye / motor / verbal) count as GCS. Other
# consciousness scales are matched but reported separately and do NOT count towards the criterion.
SCORE_CLASS_RES = (("GCS", re.compile(r"glasgow|\bgcs\b|eye opening|best motor response|best verbal response", re.I)),
                   ("FOUR", re.compile(r"\bfour score|full outline of unresponsiveness", re.I)),
                   ("RASS", re.compile(r"\brass\b|richmond", re.I)),
                   ("OTHER", re.compile(r"sedation scale|level of consciousness|ramsay|arousal", re.I)))
PRIMARY_SCORE_CLASSES = ("GCS", "FOUR", "RASS")
SCORE_RE = re.compile("|".join(r.pattern for _, r in SCORE_CLASS_RES), re.I)
LAB_RE = re.compile(r"lactate|ammonia|sodium|glucose|creatinine|\bbun\b|\bast\b|\balt\b|\bwbc\b|enolase", re.I)
SCORE_DOMAINS = ("Measurement", "Observation")          # omop_concept.domain_id of score concepts
TYPE_VOCABS = ("Drug Type", "Type Concept")             # omop_concept.vocabulary_id of record-provenance concepts
# drug_type_concept_id semantics (OMOP drug-type / type concepts), from the concept NAME. Checked in this order, so
# "EHR administration record" is administration and "Prescription dispensed in pharmacy" is an order/dispense record.
DRUG_ADMIN_RE = re.compile(r"administ|\bmar\b|infusion|given", re.I)
DRUG_ORDER_RE = re.compile(r"prescri|\border|dispens|medication list|pharmacy", re.I)
MISALIGN_DAYS = 30.0          # median |event - EEG start| above this => table misaligned
MISALIGN_MAX_RATE = 0.05      # automated date-shift check tolerance (patients)
MONOTONIC_MAX_RATE = 0.01     # within-record ordering violations tolerated
HANDCHECK_N = 20

OMOP_TIME = {"omop_drug_exposure": "drug_exposure_start_datetime", "omop_measurement": "measurement_datetime",
             "omop_note": "note_datetime", "imaging": "study_datetime"}
START, END = schema.START_EEG, schema.END_EEG


# ---------------------------------------------------------------- loading (real layout, via data_io)
def person_ids(meta: pd.DataFrame, site: str) -> pd.Series:
    """Integer person_id: ``BDSPPatientID`` when non-blank, else parsed from ``BidsFolder`` (sub-<SITE><PID>)."""
    direct = pd.to_numeric(meta["BDSPPatientID"], errors="coerce") if "BDSPPatientID" in meta else \
        pd.Series(np.nan, index=meta.index)
    from_bids = pd.to_numeric(meta["BidsFolder"].astype("string").str.replace(rf"^sub-{site}", "", regex=True),
                              errors="coerce") if "BidsFolder" in meta else pd.Series(np.nan, index=meta.index)
    return direct.fillna(from_bids).astype("Int64")


def meta_age(meta: pd.DataFrame) -> pd.Series:
    """Age in years from a site's eeg_metadata, whichever of the real variants it is: ``AgeAtVisit`` (S-sites, I0002,
    I0003; largely empty), else ``AgeInDaysAtVisit / 365.25`` (I0003), else ``StartTime - DateOfBirth`` (I0008 and
    I0009 have no AgeAtVisit column; their dates are shifted per patient, assumed consistently with DateOfBirth)."""
    age = pd.to_numeric(meta["AgeAtVisit"], errors="coerce") if "AgeAtVisit" in meta else \
        pd.Series(np.nan, index=meta.index, dtype=float)
    if "AgeInDaysAtVisit" in meta:
        age = age.fillna(pd.to_numeric(meta["AgeInDaysAtVisit"], errors="coerce") / 365.25)
    if {"DateOfBirth", "StartTime"} <= set(meta):
        age = age.fillna((meta["StartTime"] - meta["DateOfBirth"]).dt.total_seconds() / (365.25 * 86400.0))
    return age


VISIT_CLASS = {32037: "ICU", 9201: "Inpatient", 9203: "ED", 9202: "Outpatient"}      # standard OMOP visit concepts
_VISIT_TEXT = ((re.compile(r"\bicu\b|intensive|critical", re.I), "ICU"),
               (re.compile(r"emerg|\bed\b|\ber\b", re.I), "ED"),
               (re.compile(r"outpat|ambul|clinic|office", re.I), "Outpatient"),
               (re.compile(r"inpat|admit|hospital", re.I), "Inpatient"))


def visit_class(concept_id, source_value) -> str | None:
    """Care setting of a visit: from ``visit_concept_id`` when it is a known OMOP concept (it can be zero-filled),
    else by keyword from ``visit_source_value``; ``None`` when neither says."""
    if pd.notna(concept_id) and int(concept_id) in VISIT_CLASS:
        return VISIT_CLASS[int(concept_id)]
    if isinstance(source_value, str):
        for pat, label in _VISIT_TEXT:
            if pat.search(source_value):
                return label
    return None


def derive_patient_class(eeg: pd.DataFrame, visits: pd.DataFrame) -> pd.Series:
    """``PatientClass`` for each EEG session, derived from ``omop_visit_occurrence`` (no real eeg_metadata header has
    PatientClass): the visit of the same person with the latest ``visit_start_datetime`` <= the EEG time
    (``ClassTime``: the start, else reports_findings ``ReportBeginDTS`` / ``ReportEEGDateTime``), kept only if it
    has not ended before then. Sessions with no time at all (I0008/I0009 with a missing start) stay unclassified. Index = ``eeg`` index; unmatched sessions are ``None``."""
    out = pd.Series(None, index=eeg.index, dtype=object)
    need = {"person_id", "visit_start_datetime"}
    if not len(visits) or not need <= set(visits):
        return out
    tcol = "ClassTime" if "ClassTime" in eeg else "StartTime"
    e = eeg[["person_id", tcol]].rename(columns={tcol: "StartTime"}).dropna().assign(_i=lambda d: d.index)
    v = visits.dropna(subset=["person_id", "visit_start_datetime"]).copy()
    if not len(e) or not len(v):
        return out
    pairs = v[["visit_concept_id", "visit_source_value"]].drop_duplicates() if {"visit_concept_id",
                                                                                 "visit_source_value"} <= set(v) else None
    if pairs is None:
        return out
    pairs = pairs.assign(_cls=[visit_class(a, b) for a, b in zip(pairs["visit_concept_id"], pairs["visit_source_value"])])
    v = v.merge(pairs, on=["visit_concept_id", "visit_source_value"], how="left")
    if "visit_end_datetime" not in v:
        v["visit_end_datetime"] = pd.NaT
    e["person_id"], v["person_id"] = e["person_id"].astype("int64"), v["person_id"].astype("int64")
    mrg = pd.merge_asof(e.sort_values("StartTime"), v.sort_values("visit_start_datetime")[
        ["person_id", "visit_start_datetime", "visit_end_datetime", "_cls"]],
        left_on="StartTime", right_on="visit_start_datetime", by="person_id", direction="backward")
    ok = mrg["visit_end_datetime"].isna() | (mrg["visit_end_datetime"] >= mrg["StartTime"])
    out.loc[mrg.loc[ok, "_i"].to_numpy()] = mrg.loc[ok, "_cls"].to_numpy()
    return out


def merge_eeg(meta: pd.DataFrame, findings: pd.DataFrame | None, site: str) -> pd.DataFrame:
    """One site's analytic EEG frame: eeg_metadata joined to reports_findings on (person, SessionID).

    Real start/end are ``StartTime(EEG)`` / ``EndTime(EEG)`` in the findings table (the metadata ones are
    blank); age prefers the findings value, falling back to the sparse metadata value.
    """
    m = pd.DataFrame({"SiteID": site, "person_id": person_ids(meta, site),
                      "SessionID": meta["SessionID"].astype("string")})
    m["AgeAtVisit"] = meta_age(meta)
    m["PatientClass"] = meta["PatientClass"] if "PatientClass" in meta else None       # ASSUMED column (derived on disk)
    m[["StartTime", "EndTime"]] = meta[["StartTime", "EndTime"]] if {"StartTime", "EndTime"} <= set(meta) else pd.NaT
    if findings is not None and len(findings):
        f = pd.DataFrame({"person_id": person_ids(findings, site), "SessionID": findings["SessionID"].astype("string"),
                          "f_start": findings.get(START), "f_end": findings.get(END),
                          "f_age": pd.to_numeric(findings.get("AgeAtVisit"), errors="coerce"),
                          "f_begin": findings["ReportBeginDTS"] if "ReportBeginDTS" in findings else pd.NaT,
                          "f_eegdt": findings["ReportEEGDateTime"] if "ReportEEGDateTime" in findings else pd.NaT})
        f = f.sort_values("f_start", na_position="last").drop_duplicates(["person_id", "SessionID"])
        m = m.merge(f, on=["person_id", "SessionID"], how="left")
        m["AgeAtVisit"] = m["f_age"].fillna(m["AgeAtVisit"])
        m["StartTime"] = m["f_start"].fillna(m["StartTime"])
        m["EndTime"] = m["f_end"].fillna(m["EndTime"])
        # time used ONLY to match a visit when StartTime is missing (so the class of such a session is still known)
        m["ClassTime"] = pd.to_datetime(m["StartTime"]).fillna(pd.to_datetime(m["f_begin"])).fillna(
            pd.to_datetime(m["f_eegdt"])).astype("datetime64[us]")
        m = m.drop(columns=["f_start", "f_end", "f_age", "f_begin", "f_eegdt"])
    if "ClassTime" not in m:
        m["ClassTime"] = m["StartTime"]
    return m


def filter_drugs(d: pd.DataFrame) -> pd.DataFrame:
    if "drug_source_value" not in d:
        return d.iloc[0:0]
    return d[d["drug_source_value"].astype("string").str.contains(SEDATION_RE, na=False)]


def score_class_of(source: pd.Series, concept_ids: pd.Series | None = None,
                   concept_class: dict | None = None) -> pd.Series:
    """Score class (GCS / FOUR / RASS / OTHER, else NA) of rows: from the free-text source value first, else from
    the row's concept id via ``concept_class`` (concept ids can be zero-filled, so the text is primary)."""
    v = source.astype("string")
    out = pd.Series(pd.NA, index=source.index, dtype="object")
    for cls, rx in SCORE_CLASS_RES:
        hit = out.isna() & v.str.contains(rx, na=False)
        out[hit] = cls
    if concept_ids is not None and concept_class:
        ids = pd.to_numeric(concept_ids, errors="coerce")
        mapped = ids.map(lambda x: concept_class.get(int(x)) if pd.notna(x) else None)
        out = out.where(out.notna(), mapped)
    return out


def filter_measurements(d: pd.DataFrame, concept_class: dict | None = None) -> pd.DataFrame:
    """Keep score and lab rows; ``kind`` ('score' / 'lab') and ``score_class`` are derived from the free-text
    ``measurement_source_value`` and, for scores, from ``measurement_concept_id`` via ``concept_class``."""
    if "measurement_source_value" not in d:
        return d.iloc[0:0].assign(kind=pd.Series(dtype=object), score_class=pd.Series(dtype=object))
    cls = score_class_of(d["measurement_source_value"], d.get("measurement_concept_id"), concept_class)
    lab = d["measurement_source_value"].astype("string").str.contains(LAB_RE, na=False)
    keep = cls.notna() | lab
    out = d[keep].copy()
    out["score_class"] = cls[keep]
    out["kind"] = np.where(cls[keep].notna(), "score", "lab")
    return out


def filter_observations(d: pd.DataFrame, concept_class: dict | None = None) -> pd.DataFrame:
    """Score rows of ``omop_observation`` (same classes as ``filter_measurements``)."""
    if "observation_source_value" not in d:
        return d.iloc[0:0].assign(score_class=pd.Series(dtype=object))
    cls = score_class_of(d["observation_source_value"], d.get("observation_concept_id"), concept_class)
    out = d[cls.notna()].copy()
    out["score_class"] = cls[cls.notna()]
    return out


def concept_maps(concept: pd.DataFrame | None) -> tuple[dict, dict]:
    """``(score_class_by_concept_id, name_by_concept_id)`` from a (filtered) omop_concept frame. Score concepts are
    Measurement/Observation-domain concepts whose NAME matches a score class."""
    if concept is None or not len(concept) or not {"concept_id", "concept_name"} <= set(concept):
        return {}, {}
    c = concept.dropna(subset=["concept_id"])
    names = {int(i): n for i, n in zip(c["concept_id"], c["concept_name"])}
    dom = c["domain_id"].isin(SCORE_DOMAINS) if "domain_id" in c else pd.Series(True, index=c.index)
    sc = c[dom]
    cls = score_class_of(sc["concept_name"])
    return {int(i): k for i, k in zip(sc["concept_id"], cls) if isinstance(k, str)}, names


def classify_drug_type(name) -> str:
    """'administration' / 'order' / 'other' from an OMOP drug-type concept NAME (administration is checked first)."""
    if not isinstance(name, str):
        return "unresolved"
    if DRUG_ADMIN_RE.search(name):
        return "administration"
    if DRUG_ORDER_RE.search(name):
        return "order"
    return "other"


def from_raw_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Analytic tables from whole raw tables in the real layout (synthetic scale; same shapes as the
    streaming loader produces)."""
    meta, rf = raw["eeg_metadata"], raw.get("reports_findings")
    parts = []
    for site, g in meta.groupby("SiteID", sort=True):
        sess = set(g["SessionID"].astype(str))
        fg = rf[rf["SessionID"].astype(str).isin(sess)] if rf is not None else None
        parts.append(merge_eeg(g, fg, str(site)))
    eeg = pd.concat(parts, ignore_index=True)
    if not eeg["PatientClass"].notna().any() and "omop_visit_occurrence" in raw:     # real files have no PatientClass
        eeg["PatientClass"] = derive_patient_class(eeg, raw["omop_visit_occurrence"])
    cohort = set(build_candidates(eeg)["person_id"])         # same cohort filter as the streaming loader
    drugs = filter_drugs(raw["omop_drug_exposure"])
    concept = select_concepts(raw.get("omop_concept"), _type_ids(drugs))
    score_map, _ = concept_maps(concept)
    out = {"eeg_metadata": eeg, "omop_drug_exposure": drugs,
           "omop_measurement": filter_measurements(raw["omop_measurement"], score_map),
           "omop_observation": filter_observations(raw.get("omop_observation", pd.DataFrame()), score_map),
           "omop_concept": concept}
    for t in ("omop_note", "imaging"):
        out[t] = raw.get(t, pd.DataFrame())
    return {k: (v if k in ("eeg_metadata", "omop_concept") or "person_id" not in v
                else v[v["person_id"].isin(cohort)]) for k, v in out.items()}


def _type_ids(drugs: pd.DataFrame) -> list[int]:
    if "drug_type_concept_id" not in drugs:
        return []
    ids = pd.to_numeric(drugs["drug_type_concept_id"], errors="coerce").dropna().astype("int64").unique()
    return sorted(int(i) for i in ids if i != 0)


def select_concepts(concept: pd.DataFrame | None, type_ids: list[int]) -> pd.DataFrame:
    """The concepts the audit needs from a whole omop_concept frame: score concepts (Measurement/Observation domain,
    name matches a score class) and the drug-type concepts in ``type_ids``."""
    cols = ["concept_id", "concept_name", "domain_id", "vocabulary_id"]
    if concept is None or not len(concept):
        return pd.DataFrame(columns=cols)
    c = concept[[x for x in cols if x in concept]]
    nm = c["concept_name"].astype("string").str.contains(SCORE_RE, na=False)
    dom = c["domain_id"].isin(SCORE_DOMAINS) if "domain_id" in c else True
    return c[(nm & dom) | c["concept_id"].isin(type_ids)].reset_index(drop=True)


# ---- streaming helpers (Arrow-side filtering: memory scales with the matching rows, not the table)
def _arrow_mask(batch, text_col: str | None, pattern: str | None, id_col: str | None, ids):
    """Boolean Arrow mask: text column matches ``pattern`` (case-insensitive RE2) OR id column is in ``ids``.
    ``None`` when neither criterion can apply to this batch (then no row is kept)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    names = set(batch.schema.names)
    masks = []
    if text_col and pattern and text_col in names:
        masks.append(pc.match_substring_regex(pc.cast(batch.column(text_col), pa.string()), pattern, ignore_case=True))
    if id_col and ids and id_col in names:
        masks.append(pc.is_in(pc.cast(batch.column(id_col), pa.int64(), safe=False),
                              value_set=pa.array(sorted(int(i) for i in ids), type=pa.int64())))
    if not masks:
        return None
    m = masks[0]
    for x in masks[1:]:
        m = pc.or_kleene(m, x)
    return pc.fill_null(m, False)


def _stream(s3, table: str, columns: list[str], cohort, *, text_col=None, pattern=None, id_col=None, ids=None,
            prefix=None, filtered: bool = False) -> pd.DataFrame:
    """Stream one OMOP/parquet table part by part for ``cohort``, optionally keeping only rows matching
    ``pattern`` on ``text_col`` or with ``id_col`` in ``ids`` (applied in Arrow before pandas)."""
    frames = []
    kw = {"prefix": prefix} if prefix else {}
    for batch in data_io.iter_omop_batches(table, person_ids=cohort, columns=columns, s3=s3, **kw):
        if filtered:
            mask = _arrow_mask(batch, text_col, pattern, id_col, ids)
            if mask is None:
                continue
            batch = batch.filter(mask)
            if not batch.num_rows:
                continue
        frames.append(batch.to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_concepts(s3, type_ids: list[int]) -> pd.DataFrame:
    """Vocabulary rows the audit needs, streamed from ``omop_concept`` (column-pruned, filtered in Arrow):
    score concepts by NAME (Measurement/Observation domain) and the given drug-type concept ids. Vocabulary
    metadata only; no patient-level data."""
    import pyarrow as pa
    import pyarrow.compute as pc
    cols = ["concept_id", "concept_name", "domain_id", "vocabulary_id"]
    frames = []
    for b in data_io.iter_omop_batches("concept", columns=cols, s3=s3):
        names = set(b.schema.names)
        if not {"concept_id", "concept_name"} <= names:
            continue
        hit = pc.match_substring_regex(pc.cast(b.column("concept_name"), pa.string()), SCORE_RE.pattern,
                                       ignore_case=True)
        if "domain_id" in names:
            hit = pc.and_kleene(hit, pc.is_in(pc.cast(b.column("domain_id"), pa.string()),
                                              value_set=pa.array(list(SCORE_DOMAINS))))
        if type_ids:
            hit = pc.or_kleene(hit, pc.is_in(pc.cast(b.column("concept_id"), pa.int64(), safe=False),
                                             value_set=pa.array(type_ids, type=pa.int64())))
        b = b.filter(pc.fill_null(hit, False))
        if b.num_rows:
            frames.append(b.to_pandas())
    return schema.coerce_types("omop_concept", pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(
        columns=cols)


# Minimal columns requested per table (data minimisation: nothing the audit does not use is read).
DRUG_COLS = ["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime", "drug_source_value",
             "drug_type_concept_id"]
MEAS_COLS = ["person_id", "measurement_datetime", "measurement_date", "measurement_source_value",
             "measurement_concept_id"]
OBS_COLS = ["person_id", "observation_concept_id", "observation_datetime", "observation_date",
            "observation_source_value"]
NOTE_COLS = ["person_id", "note_datetime", "note_date"]


def load_audit_tables(s3, sites: list[str] | None = None, avail: dict | None = None) -> dict[str, pd.DataFrame]:
    """Read what the audit needs from a store (S3 client or ``LocalStore``) in the real layout.

    EEG tables are read whole (one row per session). OMOP/imaging parquet is streamed part by part,
    column-pruned and filtered to the candidate cohort inside Arrow (and, for drugs / measurements /
    observations, by regex or concept id inside Arrow too), so memory scales with the matching rows.
    """
    sites = sites or data_io.discover_sites(s3)
    parts = []
    for site in sites:
        try:
            meta = data_io.read_site_table("eeg_metadata", site, s3=s3)         # site variant -> canonical names
        except FileNotFoundError:
            continue
        try:
            rf = data_io.read_site_table("reports_findings", site, s3=s3)       # absent at I0008 / I0009
        except FileNotFoundError:
            rf = None
        meta = schema.coerce_types("eeg_metadata", meta)
        rf = None if rf is None else schema.coerce_types("reports_findings", rf)
        parts.append(merge_eeg(meta, rf, site))
    if not parts:
        raise FileNotFoundError("no eeg_metadata CSV found for any site")
    eeg = pd.concat(parts, ignore_index=True)
    if not eeg["PatientClass"].notna().any():
        # No real eeg_metadata header has PatientClass: derive it from visit_occurrence for the adult sessions that
        # have a start time (the acute-care cohort is a subset of these), then apply the acute-care filter.
        e0 = eeg[(eeg["AgeAtVisit"] >= ADULT_AGE) & eeg["ClassTime"].notna() & eeg["person_id"].notna()]
        pids = [int(p) for p in e0["person_id"].unique()]
        vcols = ["person_id", "visit_start_datetime", "visit_end_datetime", "visit_concept_id", "visit_source_value"]
        vb = [b.to_pandas() for b in data_io.iter_omop_batches("visit_occurrence", person_ids=pids, columns=vcols, s3=s3)]
        if vb:
            visits = schema.coerce_types("omop_visit_occurrence", pd.concat(vb, ignore_index=True))
            eeg["PatientClass"] = derive_patient_class(eeg, visits)
    cohort = [int(p) for p in build_candidates(eeg)["person_id"].dropna().unique()]
    out = {"eeg_metadata": eeg}
    result_cols = schema.COLUMN_ALIASES["measurement.result_datetime"]

    d = _stream(s3, "drug_exposure", DRUG_COLS, cohort, text_col="drug_source_value", pattern=SEDATION_RE.pattern,
                filtered=True)
    d = schema.coerce_types("omop_drug_exposure", d) if len(d) else pd.DataFrame(columns=DRUG_COLS)
    out["omop_drug_exposure"] = filter_drugs(d)

    out["omop_concept"] = load_concepts(s3, _type_ids(out["omop_drug_exposure"]))
    score_map, _ = concept_maps(out["omop_concept"])
    score_ids = list(score_map)

    m = _stream(s3, "measurement", MEAS_COLS + result_cols, cohort, text_col="measurement_source_value",
                pattern=SCORE_RE.pattern + "|" + LAB_RE.pattern, id_col="measurement_concept_id", ids=score_ids,
                filtered=True)
    m = schema.coerce_types("omop_measurement", m) if len(m) else pd.DataFrame(columns=MEAS_COLS)
    m = filter_measurements(m, score_map)
    for c in result_cols:
        if c in m:
            m[c] = schema.parse_datetimes(m[c])
    out["omop_measurement"] = m

    o = _stream(s3, "observation", OBS_COLS, cohort, text_col="observation_source_value", pattern=SCORE_RE.pattern,
                id_col="observation_concept_id", ids=score_ids, filtered=True)
    o = schema.coerce_types("omop_observation", o) if len(o) else pd.DataFrame(columns=OBS_COLS)
    out["omop_observation"] = filter_observations(o, score_map)

    n = _stream(s3, "note", NOTE_COLS, cohort)
    out["omop_note"] = schema.coerce_types("omop_note", n) if len(n) else pd.DataFrame(columns=NOTE_COLS)

    spec = data_io.TABLES["imaging"]
    im = _stream(s3, "imaging", schema.columns("imaging"), cohort, prefix=spec.pattern)
    out["imaging"] = schema.coerce_types("imaging", im) if len(im) else pd.DataFrame(columns=schema.columns("imaging"))
    return out


# ---------------------------------------------------------------- helpers
def acute_adult(eeg: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Adult EEG sessions in acute care. PatientClass is in NO real eeg_metadata header; the loaders derive it from
    omop_visit_occurrence (``derive_patient_class``). When the column is absent or empty (no visit matched) the
    acute-care filter cannot be applied and all adult sessions are used (second return value False; the report
    says so)."""
    adult = eeg[eeg["AgeAtVisit"] >= ADULT_AGE]
    have_class = "PatientClass" in eeg and bool(eeg["PatientClass"].notna().any())
    return (adult[adult["PatientClass"].isin(ACUTE_CLASSES)] if have_class else adult), have_class


def build_candidates(eeg: pd.DataFrame) -> pd.DataFrame:
    """First qualifying EEG per adult patient (acute care, start time present)."""
    e, _ = acute_adult(eeg)
    e = e[e["StartTime"].notna() & e["person_id"].notna()]
    e = e.sort_values(["person_id", "StartTime"], kind="stable")
    first = e.groupby("person_id", as_index=False).first()
    return first[["person_id", "SiteID", "StartTime"]].rename(columns={"StartTime": "t0"})


def prop_block(flags: pd.Series, sites: pd.Series) -> tuple[dict, float]:
    """Suppressed overall + per-site proportion block, plus the raw overall proportion."""
    flags = flags.astype(bool).reset_index(drop=True)
    sites = sites.reset_index(drop=True)

    def one(f: pd.Series) -> dict:
        num, den = int(f.sum()), int(len(f))
        prop = suppress_proportion(num, den)
        # if the proportion is suppressed, the pass count is too (else den - num leaks)
        return {"n_pass": SUPPRESSED if prop == SUPPRESSED else num,
                "n_total": suppress_count(den), "proportion": prop}

    by_site = {str(s): one(flags[sites == s]) for s in sorted(sites.unique())}
    raw = float(flags.mean()) if len(flags) else float("nan")
    return {"overall": one(flags), "by_site": by_site}, raw


def _fmt(block: dict) -> str:
    o = block["overall"]
    pct = o["proportion"] if isinstance(o["proportion"], str) else f"{o['proportion'] * 100:.1f}%"
    return f"{pct} ({o['n_pass']}/{o['n_total']})"


def _row(field, needed, criterion, fallback, stop, observed, passed, extra=None):
    r = {"field": field, "needed_for": needed, "pass_criterion": criterion,
         "if_it_fails": fallback, "stop_row": stop, "observed": observed,
         "passed": bool(passed), "status": "PASS" if passed else "FAIL",
         "fallback_triggered": (not passed)}
    if extra:
        r.update(extra)
    return r


# ---------------------------------------------------------------- date shift
def _alignment(cands, tables):
    """Per-candidate: does any ancillary table sit far from the EEG start?"""
    t0 = cands.set_index("person_id")["t0"]
    spec = {"notes": ("omop_note", "note_datetime"),
            "labs": ("omop_measurement", "measurement_datetime"),
            "imaging": ("imaging", "study_datetime"),
            "medications": ("omop_drug_exposure", "drug_exposure_start_datetime")}
    mis = pd.Series(False, index=t0.index)
    has = pd.Series(False, index=t0.index)
    per_table = {}
    for name, (tbl, col) in spec.items():
        df = tables.get(tbl)
        if df is None or col not in df or not len(df):
            per_table[name] = (0, 0)
            continue
        d = df[["person_id", col]].dropna()
        if name == "labs" and "kind" in df:
            d = df.loc[df["kind"] == "lab", ["person_id", col]].dropna()
        d = d[d["person_id"].isin(t0.index)].copy()
        d["off"] = ((d[col] - d["person_id"].map(t0)).dt.total_seconds().abs() / 86400.0)
        med = d.groupby("person_id")["off"].median()
        bad = med > MISALIGN_DAYS
        per_table[name] = (int(bad.sum()), int(len(med)))
        mis.loc[med.index[bad]] = True
        has.loc[med.index] = True
    return mis[has], per_table


def _monotonicity(tables):
    pairs = {
        "eeg_start_before_end": (tables["eeg_metadata"], "StartTime", "EndTime"),
        "drug_start_before_end": (tables.get("omop_drug_exposure"), "drug_exposure_start_datetime",
                                  "drug_exposure_end_datetime"),
        "imaging_study_before_final": (tables.get("imaging"), "study_datetime", "report_final_datetime"),
    }
    out, tot_n, tot_bad = {}, 0, 0
    for k, (df, a, b) in pairs.items():
        if df is None or a not in df or b not in df:
            n = bad = 0
        else:
            ok = df[a].notna() & df[b].notna()
            n = int(ok.sum())
            bad = int((df.loc[ok, b] < df.loc[ok, a]).sum())
        out[k] = {"n_pairs": suppress_count(n), "n_violations": suppress_count(bad),
                  "rate": SUPPRESSED if (bad < 11 or n < 11) else round(bad / n, 4)}
        tot_n += n
        tot_bad += bad
    return out, (tot_bad / tot_n if tot_n else float("nan")), tot_n, tot_bad


# ---------------------------------------------------------------- main audit
def run_audit(tables: dict[str, pd.DataFrame], seed: int = 0, handcheck_n: int = HANDCHECK_N):
    """Return ``(report, handcheck_ids)``. ``handcheck_ids`` are record-level: local file only.

    ``tables`` is the analytic dict from ``load_audit_tables`` / ``from_raw_tables`` (raw real-layout tables,
    recognised by a ``reports_findings`` key, are converted first).
    """
    if "reports_findings" in tables:
        tables = from_raw_tables(tables)
    eeg = tables["eeg_metadata"]
    cands = build_candidates(eeg)
    cand_ids = set(cands["person_id"])
    site_of = cands.set_index("person_id")["SiteID"]
    rows = []
    empty = pd.DataFrame()

    # 1. EEG start present (acute-care adult EEGs)
    acute, have_class = acute_adult(eeg)
    blk, raw = prop_block(acute["StartTime"].notna(), acute["SiteID"])
    note1 = "" if have_class else " [PatientClass absent: all adult EEGs, acute-care filter NOT applied]"
    rows.append(_row("EEG start date and time of day", "t0, every analysis",
                     "Present for >=95% of acute-care EEGs", "Stop; no study", True,
                     blk, raw >= 0.95, {"observed_text": _fmt(blk) + note1,
                                        "acute_care_filter_applied": have_class}))

    # 2. Date-shift consistency (automated part + human hand-check sample)
    mis, per_table = _alignment(cands, tables)
    n_mis, n_has = int(mis.sum()), int(len(mis))
    mono, mono_rate, mono_n, mono_bad = _monotonicity(tables)
    mis_rate = n_mis / n_has if n_has else float("nan")
    mis_blk, _ = prop_block(~mis, site_of.loc[mis.index])
    mis_by_site = {s: suppress_proportion(int((mis & (site_of.loc[mis.index] == s)).sum()),
                                          int((site_of.loc[mis.index] == s).sum()))
                   for s in sorted(site_of.unique())}
    passed = (mis_rate <= MISALIGN_MAX_RATE) and (mono_rate <= MONOTONIC_MAX_RATE)
    rows.append(_row(
        "Consistent within-patient date shift", "All timing logic",
        "Note, lab and EEG times line up on 20 hand-checked cases "
        f"(automated proxy: <= {MISALIGN_MAX_RATE:.0%} candidates with an ancillary table "
        f"median > {MISALIGN_DAYS:.0f} d from EEG start; <= {MONOTONIC_MAX_RATE:.0%} ordering violations)",
        "Stop", True,
        {"aligned": mis_blk, "misaligned_proportion_by_site": mis_by_site,
         "misaligned_n": suppress_count(n_mis), "evaluated_n": suppress_count(n_has),
         "misaligned_proportion": suppress_proportion(n_mis, n_has),
         "tables_flagged": {k: {"n_flagged": suppress_count(a), "n_evaluated": suppress_count(b)}
                            for k, (a, b) in per_table.items()},
         "monotonicity": mono},
        passed,
        {"observed_text": (f"misaligned {suppress_proportion(n_mis, n_has)} of "
                           f"{suppress_count(n_has)} candidates; ordering violations "
                           f"{SUPPRESSED if mono_bad < 11 else format(mono_bad / mono_n, '.2%')} of {suppress_count(mono_n)} pairs"
                           if mono_n else "n/a"),
         "human_check_pending": True,
         "human_check_note": (f"{handcheck_n}-case sampling list written to a local file only; "
                              "a human must confirm note/lab/EEG times line up. Automated checks "
                              "cannot detect a shift applied identically to every table.")}))

    # 3. Medication administration times (not just orders). The discriminating column is drug_type_concept_id (record
    # provenance); its meaning comes from the omop_concept NAME (classify_drug_type). When no type id resolves to a
    # name (zero-filled ids, concept table without them) the fallback proxy is a non-empty drug_exposure_end_datetime.
    # Denominator: candidates with >=1 sedation-class exposure in [t0-48h, t0+1h].
    m = tables.get("omop_drug_exposure", empty)
    _, cnames = concept_maps(tables.get("omop_concept"))
    med_detail: dict = {}
    if len(m) and "drug_exposure_start_datetime" in m:
        m = m[m["person_id"].isin(cand_ids)].copy()
        m["t0"] = m["person_id"].map(cands.set_index("person_id")["t0"])
        dt_h = (m["drug_exposure_start_datetime"] - m["t0"]).dt.total_seconds() / 3600
        m = m[(dt_h >= -48) & (dt_h <= 1)]
        tid = pd.to_numeric(m["drug_type_concept_id"], errors="coerce") if "drug_type_concept_id" in m else \
            pd.Series(np.nan, index=m.index)
        m["_cat"] = [classify_drug_type(cnames.get(int(x))) if pd.notna(x) and int(x) != 0 else "unresolved"
                     for x in tid]
        resolved = m["_cat"].isin(["administration", "order", "other"])
        use_concept = bool(resolved.any())
        if use_concept:
            m["_adm"] = m["_cat"] == "administration"
            method = "drug_type_concept_id (concept-name semantics)"
        else:
            m["_adm"] = m["drug_exposure_end_datetime"].notna() if "drug_exposure_end_datetime" in m else False
            method = "FALLBACK PROXY: non-empty drug_exposure_end_datetime (no drug_type_concept_id resolved to a name)"
        per_pt = m.groupby("person_id")["_adm"].any()
        tally = m["_cat"].value_counts()
        name_tally = (m.assign(cname=tid.map(lambda x: cnames.get(int(x)) if pd.notna(x) and int(x) != 0 else None))
                      .dropna(subset=["cname"]).groupby(["cname", "_cat"]).size().reset_index(name="n"))
        med_detail = {
            "method": method, "drug_type_semantics_resolved": use_concept,
            "sedation_rows_in_window_by_category": {k: suppress_count(int(tally.get(k, 0)))
                                                   for k in ("administration", "order", "other", "unresolved")},
            "drug_type_concepts": [{"concept_name": ("<id-like name>" if data_io.looks_id_like(str(nm)) else str(nm)),
                                    "category": cat, "n_rows": suppress_count(int(n))}
                                   for nm, cat, n in zip(name_tally["cname"], name_tally["_cat"], name_tally["n"])][:50],
            "start_datetime_is_administration_time": (
                "yes where the record type is administration" if use_concept else "UNKNOWN (no type semantics)")}
        all_flag = cands["person_id"].map(per_pt).fillna(False).astype(bool)
        all_blk, _ = prop_block(all_flag, cands["SiteID"])
        med_detail["administration_among_all_candidates"] = all_blk["overall"]
    else:
        per_pt = pd.Series(dtype=bool)
        method = "no sedation-class drug_exposure rows for candidates"
        med_detail = {"method": method}
    blk, raw = prop_block(per_pt, site_of.reindex(per_pt.index))
    rows.append(_row("Medication administration times (not just orders)", "Sedation baseline, E4a/E4b",
                     "Administration times for >=80% of candidates (operationalised: among candidates with a "
                     "sedation-class drug_exposure in the 48 h before t0, share with an ADMINISTRATION-type record "
                     "per drug_type_concept_id; fallback proxy: non-empty drug_exposure_end_datetime)",
                     "Use orders; label 1B \"approximate\"", False,
                     blk, raw >= 0.80, {"observed_text": _fmt(blk) + f" [method: {med_detail.get('method', '')}]",
                                        "details": med_detail}))

    # 4. Lab result time: the real measurement table has NO result-time column (only measurement_datetime /
    # measurement_date); pass only if a candidate alias column exists and is filled.
    meas = tables.get("omop_measurement", empty)
    lb = meas[meas["kind"] == "lab"] if len(meas) and "kind" in meas else meas
    lb = lb[lb["person_id"].isin(cand_ids)] if len(lb) else lb
    res_col = next((c for c in schema.COLUMN_ALIASES["measurement.result_datetime"] if c in meas.columns), None)
    if res_col and len(lb):
        blk, raw = prop_block(lb[res_col].notna(), lb["person_id"].map(site_of))
        extra = {"observed_text": _fmt(blk), "result_lag_minutes": _lag(lb, res_col)}
    else:
        blk, raw = {"result_time_column_present": False}, float("nan")
        extra = {"observed_text": "no result-time column in measurement (collection time only)",
                 "details": {"result_time_column_aliases_checked": schema.COLUMN_ALIASES["measurement.result_datetime"],
                             "n_candidate_lab_rows": suppress_count(len(lb))}}
    rows.append(_row("Lab result or verification time", "Study 1B",
                     "Result time present (operationalised: >=95% of candidate lab rows; needs a result-time "
                     "column, none is known in OMOP measurement)",
                     "Collection time plus assay lag; label 1B \"approximate\"", False,
                     blk, bool(res_col) and raw >= 0.95, extra))

    # 5. Imaging report finalization time (ASSUMED table and column)
    im = tables.get("imaging", empty)
    if len(im) and "report_final_datetime" in im:
        im = im[im["person_id"].isin(cand_ids)]
        blk, raw = prop_block(im["report_final_datetime"].notna(), im["person_id"].map(site_of))
        txt = _fmt(blk)
    else:
        blk, raw, txt = {"imaging_table_or_column_found": False}, float("nan"), "imaging table/column not found"
    img_sites = sorted(set(schema.IMAGING_REAL_LAYOUT["sites"]) & set(cands["SiteID"]))
    img_cands = int(cands["person_id"].isin(set(im["person_id"]) if len(im) and "person_id" in im else set()).sum())
    rows.append(_row("Imaging report finalization time", "Study 1B, H5",
                     "Present (operationalised: >=95% of candidate imaging studies)", "Drop H5", False,
                     blk, raw >= 0.95,
                     {"observed_text": txt + f"; {len(img_sites)} candidate site(s) have an Imaging/<SITE>/ prefix",
                      "details": {"candidate_sites_with_imaging_prefix": len(img_sites),
                                  "imaging_prefix_sites_in_bucket": list(schema.IMAGING_REAL_LAYOUT["sites"]),
                                  "candidates_with_any_imaging_row": suppress_count(img_cands)}}))

    # 6. GCS / FOUR / RASS within +-6 h of t0. Score rows come from omop_measurement and omop_observation: a row is a
    # score when its source text matches a score class or its concept id is a score concept found BY NAME in
    # omop_concept (Measurement/Observation domain). Only GCS, FOUR and RASS count (OTHER scales are reported apart).
    t0_of = cands.set_index("person_id")["t0"]
    parts = []
    if len(meas) and "kind" in meas:
        sm = meas[meas["kind"] == "score"]
        parts.append(pd.DataFrame({"person_id": sm["person_id"], "time": sm["measurement_datetime"],
                                   "cls": sm["score_class"], "src": "measurement"}))
    obs = tables.get("omop_observation", empty)
    if len(obs) and "score_class" in obs:
        parts.append(pd.DataFrame({"person_id": obs["person_id"], "time": obs.get("observation_datetime"),
                                   "cls": obs["score_class"], "src": "observation"}))
    sc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["person_id", "time", "cls", "src"])
    sc = sc[sc["person_id"].isin(cand_ids)].copy()
    n_no_time = int(sc["time"].isna().sum())
    sc["off_h"] = (pd.to_datetime(sc["time"]) - sc["person_id"].map(t0_of)).dt.total_seconds().abs() / 3600
    near_rows = sc[(sc["off_h"] <= 6) & sc["cls"].isin(PRIMARY_SCORE_CLASSES)]
    flags = cands["person_id"].isin(set(near_rows["person_id"]))
    blk, raw = prop_block(flags, cands["SiteID"])

    def _cov(mask) -> dict:
        n = int(cands["person_id"].isin(set(near_rows.loc[mask, "person_id"])).sum())
        return {"n_candidates": suppress_count(n), "proportion": suppress_proportion(n, len(cands))}
    sc_detail = {
        "by_class_within_6h": {c: _cov(near_rows["cls"] == c) for c in PRIMARY_SCORE_CLASSES},
        "by_source_within_6h": {c: _cov(near_rows["src"] == c) for c in ("measurement", "observation")},
        "other_scales_within_6h_not_counted": _cov(sc["cls"].eq("OTHER") & (sc["off_h"] <= 6)),
        "n_score_concepts_found_by_name": {c: sum(1 for v in concept_maps(tables.get("omop_concept"))[0].values() if v == c)
                                           for c in ("GCS", "FOUR", "RASS", "OTHER")},
        "candidate_score_rows_without_datetime": suppress_count(n_no_time),
        "candidate_score_rows": suppress_count(len(sc))}
    rows.append(_row("GCS, FOUR or RASS near EEG", "Inclusion, 1A baseline",
                     "Score within +-6 h for >=50% of candidates (GCS incl. components, FOUR, RASS; measurement + "
                     "observation tables; concepts found by name in omop_concept, plus source-text match)",
                     "BDSP GCS-from-EHR tool; broad cohort only", False,
                     blk, raw >= 0.50, {"observed_text": _fmt(blk), "details": sc_detail}))

    # 7. Site identifier
    counts = cands.groupby("SiteID").size()
    big = counts[counts >= 300]
    site_tbl = {str(s): suppress_count(c) for s, c in counts.items()}
    rows.append(_row("Site identifier", "Leave-one-site-out",
                     ">=3 adult sites with >=300 candidates each", "Grouped split; weaker claim", False,
                     {"candidates_per_site": site_tbl, "n_sites_ge_300": int(len(big)),
                      "n_adult_sites": int(len(counts))},
                     len(big) >= 3,
                     {"observed_text": f"{len(big)} of {len(counts)} adult sites have >=300 candidates"}))

    # 8. Timestamped notes (OMOP note.note_datetime; ASSUMED table)
    nt_all = tables.get("omop_note", empty)
    nt_c = nt_all[nt_all["person_id"].isin(cand_ids)] if len(nt_all) else nt_all
    nt = nt_c[nt_c["note_datetime"].notna()] if len(nt_c) else nt_c
    flags = cands["person_id"].isin(set(nt["person_id"]) if len(nt) else set())
    blk, raw = prop_block(flags, cands["SiteID"])
    n_note_rows = int(len(nt_c))
    note_detail = {
        "candidate_note_rows": suppress_count(n_note_rows),
        "note_rows_with_note_datetime": (suppress_proportion(int(len(nt)), n_note_rows) if n_note_rows else "n/a"),
        "candidates_with_any_note_row": suppress_count(int(cands["person_id"].isin(
            set(nt_c["person_id"]) if len(nt_c) else set()).sum())),
        "note_text_requested": False}
    rows.append(_row("Timestamped notes", "ACI onset, silver labels",
                     "Present (operationalised: >=80% of candidates have >=1 timestamped note, D-101)", "Stop",
                     True, blk, raw >= 0.80, {"observed_text": _fmt(blk), "details": note_detail}))

    # Hand-check sampling list (record-level -> local file only)
    labs_ids = set(lb["person_id"]) if len(lb) else set()
    notes_ids = set(nt["person_id"]) if len(nt) else set()
    elig = sorted((labs_ids & notes_ids) & cand_ids)
    rng = np.random.default_rng(seed)
    k = min(handcheck_n, len(elig))
    ids = [str(elig[j]) for j in sorted(rng.choice(len(elig), size=k, replace=False))] if k else []

    stop_ok = all(r["passed"] for r in rows if r["stop_row"])
    report = {
        "audit": "Phase 0a HEEDB field audit",
        "suppression": f"cells with n < 11 shown as \"{SUPPRESSED}\"",
        "n_candidates": suppress_count(len(cands)),
        "rows": rows,
        "gate_0a_automated_stop_rows_pass": bool(stop_ok),
        "human_hand_check_pending": True,
        "n_handcheck_sampled": suppress_count(len(ids)) if len(ids) else 0,
        "fallbacks_triggered": [r["field"] for r in rows if not r["passed"] and not r["stop_row"]],
        "stop_rows_failed": [r["field"] for r in rows if not r["passed"] and r["stop_row"]],
    }
    return report, ids


def _lag(lb: pd.DataFrame, res_col: str) -> dict:
    d = (lb[res_col] - lb["measurement_datetime"]).dt.total_seconds() / 60
    return safe_quantiles(d[d >= 0].dropna().values)


# ---------------------------------------------------------------- rendering
def _detail_lines(d, indent: int = 0) -> list[str]:
    pad = "  " * indent
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}- {k}:")
                out += _detail_lines(v, indent + 1)
            else:
                out.append(f"{pad}- {k}: {v}")
    elif isinstance(d, list):
        for v in d:
            out.append(f"{pad}- " + (", ".join(f"{k}={x}" for k, x in v.items()) if isinstance(v, dict) else str(v)))
    return out


def to_markdown(report: dict) -> str:
    L = ["# Phase 0a: HEEDB field audit", "",
         f"Aggregate-only. {report['suppression']}. Candidates: {report['n_candidates']}.", "",
         "| Field | Needed for | Pass criterion | Observed | Status | If it fails |",
         "|---|---|---|---|---|---|"]
    for r in report["rows"]:
        stop = " (STOP row)" if r["stop_row"] else ""
        L.append(f"| {r['field']}{stop} | {r['needed_for']} | {r['pass_criterion']} | "
                 f"{r['observed_text'] if 'observed_text' in r else ''} | **{r['status']}** | {r['if_it_fails']} |")
    L += ["", "## Per-site breakdown", "",
          "| Field | Site | Pass / total | Proportion |", "|---|---|---|---|"]
    for r in report["rows"]:
        obs = r["observed"]
        if isinstance(obs, dict) and "by_site" in obs:
            for s, b in obs["by_site"].items():
                L.append(f"| {r['field']} | {s} | {b['n_pass']}/{b['n_total']} | {b['proportion']} |")
        elif isinstance(obs, dict) and "aligned" in obs:
            for s, b in obs["aligned"]["by_site"].items():
                L.append(f"| {r['field']} (aligned) | {s} | {b['n_pass']}/{b['n_total']} | {b['proportion']} |")
        elif isinstance(obs, dict) and "candidates_per_site" in obs:
            for s, n in obs["candidates_per_site"].items():
                L.append(f"| {r['field']} (candidates) | {s} | {n} | |")
    L += ["", "## Row details (aggregate-only)", ""]
    for r in report["rows"]:
        if r.get("details"):
            L.append(f"### {r['field']}")
            L += _detail_lines(r["details"])
            L.append("")
    ds = next(r for r in report["rows"] if r["field"].startswith("Consistent"))
    L += ["", "## Date-shift detail", "",
          "| Check | n pairs | violations | rate |", "|---|---|---|---|"]
    for k, v in ds["observed"]["monotonicity"].items():
        L.append(f"| {k} | {v['n_pairs']} | {v['n_violations']} | {v['rate']} |")
    L += ["", f"Human hand-check: {ds['human_check_note']}", "",
          "## Verdict", "",
          f"- Stop rows (automated): {'all pass' if report['gate_0a_automated_stop_rows_pass'] else 'FAILED: ' + '; '.join(report['stop_rows_failed'])}",
          f"- Fallbacks triggered: {', '.join(report['fallbacks_triggered']) or 'none'}",
          "- Human hand-check of date-shift consistency: PENDING (sampling list in local file only)",
          ""]
    return "\n".join(L)


# ---------------------------------------------------------------- schema dry run (names only)
MAX_UNLISTED_NAMES = 300


def _safe_names(names: list[str]) -> list[str]:
    """Column names safe to print: id-like names are masked, the list is capped (names only, never values)."""
    out = [n if not data_io.looks_id_like(n) else "<id-like column>" for n in names[:MAX_UNLISTED_NAMES]]
    return out + (["<more columns not listed>"] if len(names) > MAX_UNLISTED_NAMES else [])


def probe_schema(s3, sites: list[str] | None = None, *, list_unlisted: bool = False) -> dict:
    """Which expected tables/columns exist, by NAME ONLY (CSV header line / parquet footer; no values, no
    rows). One unit per table (per site for per-site CSV tables). Safe to emit: contains table, site and
    column names only."""
    sites = sites or data_io.discover_sites(s3)
    units = []
    for table in schema.TABLE_NAMES:
        spec = data_io.TABLES[table]
        for site in (sites if spec.kind == "csv_site" else [None]):
            try:
                actual = data_io.table_columns(s3, table, site)
            except Exception as exc:  # noqa: BLE001 - report the class only; never echo an error message
                actual, err = None, type(exc).__name__
            else:
                err = None
            exp = schema.expected_columns(table, site)             # the site's REAL variant (actual header names)
            unit = {"table": table, "site": site, "location": spec.pattern, "found": actual is not None,
                    "n_expected": len(exp)}
            v = schema.variant_for(site) if table == "reports_findings" else None
            if actual is None and v is not None and v.reports_findings is None:
                unit["expected_absent"] = True                     # the variant says this site has no such file
            if err:
                unit["error"] = err
            if actual is not None:
                have = set(actual)
                unit["present"] = [c for c, *_ in exp if c in have]
                unit["missing"] = [{"column": c, "provenance": prov,
                                    "closest_actual_name": (difflib.get_close_matches(c, actual, 1, 0.6) or [None])[0]}
                                   for c, _, prov, _ in exp if c not in have]
                unit["n_unlisted_columns"] = len(have - {c for c, *_ in exp})
                if list_unlisted:                 # names in file order, names only
                    known = {c for c, *_ in exp}
                    unit["unlisted_columns"] = _safe_names([c for c in actual if c not in known])
            units.append(unit)
    def _n(prov):
        return sum(1 for u in units for m in u.get("missing", []) if m["provenance"] == prov)
    return {"audit": "Phase 0a schema dry run (names only)", "sites": sites, "units": units,
            "n_tables_not_found": sum(1 for u in units if not u["found"] and not u.get("expected_absent")),
            "n_missing_confirmed": _n(schema.CONFIRMED), "n_missing_named": _n(schema.NAMED),
            "n_missing_assumed": _n(schema.ASSUMED), "still_unknown": schema.UNKNOWN_FIELDS}


PREFIX_PROBES = (("", 2), ("Imaging/", 2))      # (start prefix, levels to list); root => depth <= 2 overall
NEVER_DESCEND = {"bids"}                       # per-patient BIDS data folders: never list inside


def probe_prefixes(s3, *, bucket: str | None = None) -> list[dict]:
    """Metadata-level prefix names (``Delimiter='/'``) at the access point root (depth <= 2) and under
    ``Imaging/`` (first two levels). NAMES ONLY: no per-patient folder is ever listed (id-like names are masked
    and not descended into, a level with too many prefixes is reported as overflow without names, and
    ``bids`` folders are never entered). Returns one entry per listed prefix."""
    out: list[dict] = []

    def walk(prefix: str, levels: int):
        lvl = data_io.list_level(s3, prefix, bucket=bucket)
        out.append(lvl)
        if levels <= 1 or lvl["prefixes_overflow"]:
            return
        for child in lvl["prefixes"]:
            name = child[len(prefix):].rstrip("/")
            if "<id-like" in child or name.lower() in NEVER_DESCEND:
                continue
            walk(child, levels - 1)

    for start, levels in PREFIX_PROBES:
        walk(start, levels)
    seen, uniq = set(), []
    for e in out:
        if e["prefix"] not in seen:
            seen.add(e["prefix"])
            uniq.append(e)
    return uniq


def prefix_markdown(levels: list[dict]) -> list[str]:
    L = ["", "## Access point prefixes (names only; Delimiter='/', depth <= 2; per-patient folders never listed)", ""]
    for e in levels:
        L.append(f"- `{e['prefix'] or '<root>'}`")
        if e["prefixes_overflow"]:
            L.append("    - <more than %d child prefixes: looks like a population of folders, not listed or descended>"
                     % data_io.MAX_LEVEL_ITEMS)
        for c in e["prefixes"]:
            L.append(f"    - prefix `{c[len(e['prefix']):]}`")
        if e["files_overflow"]:
            L.append("    - <too many files at this level to list>")
        for f in e["files"]:
            L.append(f"    - file `{f[len(e['prefix']):]}`")
        if not (e["prefixes"] or e["files"] or e["prefixes_overflow"] or e["files_overflow"]):
            L.append("    - (empty)")
    return L


def dry_run_markdown(rep: dict) -> str:
    L = ["# Phase 0a schema dry run (names only)", "",
         f"Sites checked: {', '.join(rep['sites']) or 'none'}. Only table and column NAMES are read and shown; "
         "no values, no rows.", "",
         f"- Tables/files not found: {rep['n_tables_not_found']}",
         f"- Missing CONFIRMED columns (schema.py is wrong or the release differs): {rep['n_missing_confirmed']}",
         f"- Missing NAMED columns (may simply not exist): {rep['n_missing_named']}",
         f"- Missing ASSUMED columns (placeholders to remap): {rep['n_missing_assumed']}", "",
         "| Table | Site | Found | Expected | Present | Missing | Unlisted cols |", "|---|---|---|---|---|---|---|"]
    for u in rep["units"]:
        L.append(f"| {u['table']} | {u['site'] or ''} | {'yes' if u['found'] else ('absent (expected)' if u.get('expected_absent') else 'NO')} | {u['n_expected']} | "
                 f"{len(u.get('present', []))} | {len(u.get('missing', []))} | {u.get('n_unlisted_columns', '')} |")
    L += ["", "## Missing columns", ""]
    any_missing = False
    for u in rep["units"]:
        if not u["found"] and u.get("expected_absent"):
            L.append(f"- {u['table']} / {u['site']}: no file at `{u['location']}` (the site variant has none; expected)")
        elif not u["found"]:
            L.append(f"- **{u['table']}**{' / ' + u['site'] if u['site'] else ''}: NOT FOUND at `{u['location']}`")
            any_missing = True
        for m in u.get("missing", []):
            near = f" (closest actual name: `{m['closest_actual_name']}`)" if m["closest_actual_name"] else ""
            L.append(f"- {u['table']}{' / ' + u['site'] if u['site'] else ''}: `{m['column']}` [{m['provenance']}]{near}")
            any_missing = True
    if not any_missing:
        L.append("- none")
    if any("unlisted_columns" in u for u in rep["units"]):
        L += ["", "## Unlisted columns (present in the real header, not in schema.py; names only)", ""]
        for u in rep["units"]:
            if u.get("unlisted_columns"):
                L.append(f"- {u['table']}{' / ' + u['site'] if u['site'] else ''}: "
                         + ", ".join(f"`{c}`" for c in u["unlisted_columns"]))
    if rep.get("prefixes") is not None:
        L += prefix_markdown(rep["prefixes"])
    L += ["", "## Still unknown after the names-only dry run", ""] + [f"- {x}" for x in rep["still_unknown"]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="local directory in the HEEDB layout (synthetic data or a local mirror)")
    src.add_argument("--s3", action="store_true", help="read the real BDSP access point (human-run only)")
    ap.add_argument("--profile", help="AWS profile for --s3 (else HEEDB_AWS_PROFILE / AWS_PROFILE / default chain)")
    ap.add_argument("--sites", nargs="+", help="site codes (default: every site with an eeg-metadata CSV)")
    ap.add_argument("--out", help="output directory (required unless --dry-run-schema)")
    ap.add_argument("--dry-run-schema", action="store_true",
                    help="report which expected tables/columns exist vs missing (names only) and exit")
    ap.add_argument("--list-unlisted", action="store_true",
                    help="with --dry-run-schema: also print the NAMES of columns present but not in schema.py")
    ap.add_argument("--probe-prefixes", action="store_true",
                    help="with --dry-run-schema: also list metadata-level prefix names (Delimiter='/') at the "
                         "access point root and under Imaging/ (names only; per-patient folders never listed)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the hand-check sample")
    ap.add_argument("--handcheck-n", type=int, default=HANDCHECK_N)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if a Stop row fails (dry run: if a CONFIRMED column is missing)")
    a = ap.parse_args(argv)
    if not a.dry_run_schema and not a.out:
        ap.error("--out is required unless --dry-run-schema")
    if (a.list_unlisted or a.probe_prefixes) and not a.dry_run_schema:
        ap.error("--list-unlisted / --probe-prefixes are options of --dry-run-schema")

    if a.data:
        agent_safety.assert_not_restricted_in_agent(a.data)
    s3 = data_io.open_store(a.data, profile=a.profile)       # make_client refuses inside an agent session

    if a.dry_run_schema:
        rep = probe_schema(s3, a.sites, list_unlisted=a.list_unlisted)
        if a.probe_prefixes:
            rep["prefixes"] = probe_prefixes(s3)
        text = dry_run_markdown(rep)
        if a.out:
            safe_write_json(Path(a.out) / "schema_dry_run.json", rep)
            safe_write_text(Path(a.out) / "schema_dry_run.md", text)
        safe_print(text.rstrip())
        return 1 if a.strict and (rep["n_missing_confirmed"] or rep["n_tables_not_found"]) else 0

    stage = "load"
    try:
        tables = load_audit_tables(s3, a.sites)
        stage = "audit"
        report, ids = run_audit(tables, a.seed, a.handcheck_n)
    except Exception as exc:  # noqa: BLE001
        if a.data:                       # local / synthetic: normal traceback for debugging
            raise
        # Real data: a message or traceback can quote a value (a bad cell, a key). Print the stage and the exception
        # CLASS only; reproduce on synthetic data to debug (CLAUDE.md rule 3).
        safe_print(f"field audit FAILED at stage '{stage}': {type(exc).__name__} (message withheld; reproduce on "
                   f"synthetic data); S3 read retries before failing: {sum(data_io.RETRY_COUNTS.values())}"
                   + (" " + str(dict(data_io.RETRY_COUNTS)) if data_io.RETRY_COUNTS else ""))
        return 2
    known = {str(p) for p in tables["eeg_metadata"]["person_id"].dropna().unique()}

    out = Path(a.out)
    safe_write_json(out / "field_audit.json", report, known)
    safe_write_text(out / "field_audit.md", to_markdown(report), known)
    hc = write_local_only(out / "local_only" / "handcheck_sample_ids.csv",
                          "BDSPPatientID\n" + "\n".join(ids) + "\n")

    safe_print("Phase 0a field audit complete (aggregate-only; n<11 suppressed).", known_ids=known)
    for r in report["rows"]:
        safe_print(f"  [{r['status']}] {r['field']}: {r.get('observed_text', '')}", known_ids=known)
    safe_print(f"Stop rows (automated): {'PASS' if report['gate_0a_automated_stop_rows_pass'] else 'FAIL'}; "
               f"fallbacks triggered: {len(report['fallbacks_triggered'])}", known_ids=known)
    safe_print(f"Hand-check sampling list ({len(ids)} cases) written to local file: {hc}", known_ids=known)
    safe_print(f"Reports: {out / 'field_audit.md'}, {out / 'field_audit.json'}", known_ids=known)
    if a.strict and not report["gate_0a_automated_stop_rows_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
