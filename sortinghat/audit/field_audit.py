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
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agent_safety, data_io, schema
from ..cohort import rules as cohort_rules
from ..cohort.config import CohortConfig
from . import alignment as al
from ..cohort.sources import (StoreSources, concat_frames, iter_filtered_batches, remap_ids)
from ..safe_output import (SUPPRESS_BELOW, SUPPRESSED, safe_print, safe_quantiles, safe_write_json,
                           safe_write_text, suppress_count, suppress_proportion,
                           write_local_only)

ACUTE_CLASSES = {"ICU", "Inpatient", "ED"}
ADULT_AGE = 18
COHORT_CFG = CohortConfig()                  # D-111..D-115: the audit's candidates are the cohort's candidates
STUDY_SITES = COHORT_CFG.study_sites         # I0002, I0003, S0001, S0002 (D-113); None = every site
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
# Date-shift gate (D-117): nearest-event alignment, see sortinghat/audit/alignment.py
ALIGN_WINDOW_H = 24.0         # a candidate is aligned when SOME ancillary event is within +-24 h of its EEG start
ALIGN_WIDE_H = 72.0           # reported as well
ALIGN_MIN_SHARE = 0.80        # >= 80% of candidates aligned at every Study 1 site with clinical data
DAY_MODE_MAX_SHARE = 0.05     # no non-zero whole-day gap mode may hold more than 5% of a site's candidates
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
    if not len(visits) or "person_id" not in visits or not ({"visit_start_datetime", "_start"} & set(visits)):
        return out
    tcol = "ClassTime" if "ClassTime" in eeg else "StartTime"
    e = eeg[["person_id", tcol]].rename(columns={tcol: "StartTime"}).dropna().assign(_i=lambda d: d.index)
    v = visits.dropna(subset=["person_id", "_start" if "_start" in visits else "visit_start_datetime"]).copy()
    if not len(e) or not len(v):
        return out
    if "_start" in v:                                  # compact visits (cohort.rules.compact_visits): class precomputed
        v = v.rename(columns={"_start": "visit_start_datetime", "_end": "visit_end_datetime"}).assign(
            _cls=v["_cls"].astype(object))
    else:
        pairs = v[["visit_concept_id", "visit_source_value"]].drop_duplicates() if {
            "visit_concept_id", "visit_source_value"} <= set(v) else None
        if pairs is None:
            return out
        pairs = pairs.assign(_cls=[visit_class(a, b) for a, b in zip(pairs["visit_concept_id"], pairs["visit_source_value"])])
        v = v.merge(pairs, on=["visit_concept_id", "visit_source_value"], how="left")
    if "visit_end_datetime" not in v:
        v["visit_end_datetime"] = pd.NaT
    e["person_id"], v["person_id"] = e["person_id"].astype("int64"), v["person_id"].astype("int64")
    e["StartTime"] = e["StartTime"].astype("datetime64[us]")
    v["visit_start_datetime"] = v["visit_start_datetime"].astype("datetime64[us]")
    v["visit_end_datetime"] = v["visit_end_datetime"].astype("datetime64[us]")
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
    m["ServiceName"] = (meta["ServiceName"].astype("string").str.strip().str.upper().to_numpy(object)
                        if "ServiceName" in meta else None)
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
        if "ServiceName(EEG)" in findings and m["ServiceName"].isna().any():          # S-sites: findings service
            sv = pd.DataFrame({"person_id": person_ids(findings, site), "SessionID": findings["SessionID"].astype("string"),
                               "_svc": findings["ServiceName(EEG)"].astype("string").str.strip().str.upper()}
                              ).drop_duplicates(["person_id", "SessionID"])
            n0 = len(m)
            m = m.merge(sv, on=["person_id", "SessionID"], how="left")
            assert len(m) == n0
            m["ServiceName"] = m["ServiceName"].where(m["ServiceName"].notna(), m["_svc"])
            m = m.drop(columns="_svc")
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


def site_findings(rf: pd.DataFrame | None, meta: pd.DataFrame, site: str) -> pd.DataFrame | None:
    """The rows of a whole-cohort ``reports_findings`` that belong to one site's ``meta``: matched on the pair
    (person_id, SessionID). ``SessionID`` alone is NOT a key (it is a per-patient counter, 1..N, repeated across patients)."""
    if rf is None or not len(rf):
        return rf
    keys = pd.MultiIndex.from_arrays([person_ids(meta, site), meta["SessionID"].astype("string")])
    return rf[pd.MultiIndex.from_arrays([person_ids(rf, site), rf["SessionID"].astype("string")]).isin(keys)]


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
        parts.append(merge_eeg(g, site_findings(rf, g, str(site)), str(site)))
    eeg = pd.concat(parts, ignore_index=True)
    if not eeg["PatientClass"].notna().any() and "omop_visit_occurrence" in raw:     # real files have no PatientClass
        eeg["PatientClass"] = derive_patient_class(eeg, raw["omop_visit_occurrence"])
        eeg["PatientClass"] = eeg["PatientClass"].where(eeg["PatientClass"].notna(),
                                                        cohort_proxy_class(eeg, raw["omop_visit_occurrence"]))
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
    # date-shift check (D-117): nearest-event gaps from the WHOLE raw tables (the filtered ones above keep only
    # sedation / score / lab rows, which would understate how close the nearest event is)
    out["alignment_gaps"], out["alignment_order"] = al.alignment_from_frames(build_candidates(eeg), raw)
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


# ---- streaming loader. Reuses the cohort's readers (cohort.sources): compact visits, two-stage row-group reads with
# the predicate pushed down (only person_id + the predicate column are fetched for row groups without a match), the
# cohort's candidate rule (D-111..D-115), the merge map (D-114), and per-table compaction so the working set stays small.
DRUG_COLS = ["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime", "drug_source_value",
             "drug_type_concept_id"]
MEAS_COLS = ["person_id", "measurement_datetime", "measurement_date", "measurement_source_value",
             "measurement_concept_id"]
OBS_COLS = ["person_id", "observation_concept_id", "observation_datetime", "observation_date",
            "observation_source_value"]
NOTE_COLS = ["person_id", "note_datetime", "note_date"]
SCORE_CATS = ["GCS", "FOUR", "RASS", "OTHER"]


def _dt_us(d: pd.DataFrame, col: str) -> pd.Series:
    return (schema.parse_datetimes(d[col]) if col in d else pd.Series(pd.NaT, index=d.index)).astype("datetime64[us]")


def _compact_drugs(batch, remap) -> pd.DataFrame:
    d = remap_ids(batch.to_pandas(), remap)
    d = filter_drugs(d)
    return pd.DataFrame({
        "person_id": pd.to_numeric(d["person_id"]).to_numpy("int64"),
        "drug_exposure_start_datetime": _dt_us(d, "drug_exposure_start_datetime").to_numpy(),
        "drug_exposure_end_datetime": _dt_us(d, "drug_exposure_end_datetime").to_numpy(),
        "drug_type_concept_id": (pd.to_numeric(d["drug_type_concept_id"], errors="coerce").fillna(0).to_numpy("int64")
                                 if "drug_type_concept_id" in d else np.zeros(len(d), "int64"))})


def _compact_measurements(batch, remap, score_map, result_cols) -> pd.DataFrame:
    d = remap_ids(batch.to_pandas(), remap)
    d = filter_measurements(d, score_map)
    out = {"person_id": pd.to_numeric(d["person_id"]).to_numpy("int64"),
           "measurement_datetime": _dt_us(d, "measurement_datetime").to_numpy(),
           "kind": pd.Categorical(d["kind"].to_numpy(object), categories=["score", "lab"]),
           "score_class": pd.Categorical(d["score_class"].to_numpy(object), categories=SCORE_CATS)}
    for c in result_cols:
        if c in d:
            out[c] = _dt_us(d, c).to_numpy()
    return pd.DataFrame(out)


def _compact_observations(batch, remap, score_map) -> pd.DataFrame:
    d = remap_ids(batch.to_pandas(), remap)
    d = filter_observations(d, score_map)
    return pd.DataFrame({"person_id": pd.to_numeric(d["person_id"]).to_numpy("int64"),
                         "observation_datetime": _dt_us(d, "observation_datetime").to_numpy(),
                         "score_class": pd.Categorical(d["score_class"].to_numpy(object), categories=SCORE_CATS)})


def _compact_notes(batch, remap) -> pd.DataFrame:
    d = remap_ids(batch.to_pandas(), remap)
    return pd.DataFrame({"person_id": pd.to_numeric(d["person_id"]).to_numpy("int64"),
                         "note_datetime": _dt_us(d, "note_datetime").to_numpy()})


def _collect(it, compact, like_cols: list[str]) -> pd.DataFrame:
    """Stream Arrow tables through ``compact`` and concatenate column by column (peak = result + one column)."""
    chunks = []
    for b in it:
        c = compact(b)
        if len(c):
            chunks.append(c)
    if not chunks:
        return pd.DataFrame(columns=like_cols)
    return concat_frames(chunks, chunks[0].iloc[0:0])


def load_concepts(s3, type_ids: list[int] | None = None, all_types: bool = False) -> pd.DataFrame:
    """Vocabulary rows the audit needs, streamed from ``omop_concept`` (column-pruned, filtered in Arrow):
    score concepts by NAME (Measurement/Observation domain) and drug-type concepts (the given ids, or every concept of
    the type vocabularies when ``all_types``). Vocabulary metadata only; no patient-level data."""
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
        if all_types and "vocabulary_id" in names:
            hit = pc.or_kleene(hit, pc.is_in(pc.cast(b.column("vocabulary_id"), pa.string()),
                                             value_set=pa.array(list(TYPE_VOCABS))))
        b = b.filter(pc.fill_null(hit, False))
        if b.num_rows:
            frames.append(b.to_pandas())
    return schema.coerce_types("omop_concept", pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(
        columns=cols)


def load_audit_tables(s3, sites: list[str] | None = None, avail: dict | None = None, workers: int = 3,
                      cfg: CohortConfig | None = None) -> dict[str, pd.DataFrame]:
    """Read what the audit needs from a store (S3 client or ``LocalStore``) in the real layout.

    Candidate set = the cohort's (``cohort_proxy_class`` fills the care setting that ``visit_concept_id`` cannot give,
    ``acute_adult`` applies the Study 1 sites and the OR/EMU exclusion, the patient merge map is applied first).
    Visits come as compact frames; every other OMOP table is read row group by row group with the predicate pushed
    down and reduced to the columns the audit rows use (``_compact_*``), so memory scales with the matching rows of
    the candidates, not with the tables. The big tables are read by ``workers`` threads at once (S3 reads overlap)."""
    from concurrent.futures import ThreadPoolExecutor
    cfg = cfg or COHORT_CFG
    sites = sites or data_io.discover_sites(s3)
    store = StoreSources(s3, sites)
    mm, _ = store.merge_map()
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
    if mm:                                                                       # D-114: merged ids -> surviving id
        eeg["person_id"] = remap_ids(eeg.assign(person_id=eeg["person_id"].astype("Int64")), mm)["person_id"] \
            if eeg["person_id"].notna().all() else eeg["person_id"].map(lambda x: mm.get(int(x), x) if pd.notna(x) else x)
        eeg["person_id"] = eeg["person_id"].astype("Int64")
    rev: dict[int, list[int]] = {}
    for old, new in mm.items():
        rev.setdefault(new, []).append(old)
    if not eeg["PatientClass"].notna().any():
        # No real eeg_metadata header has PatientClass: derive it from visit_occurrence for the adult sessions that
        # have a (class) time (every site: I0008/I0009 have no OMOP rows, so this costs nothing there), then fill unknown classes with the cohort proxy.
        e0 = eeg[(eeg["AgeAtVisit"] >= ADULT_AGE) & eeg["ClassTime"].notna() & eeg["person_id"].notna()]
        pids = sorted({int(p) for p in e0["person_id"].unique()} | {o for p in e0["person_id"].unique() for o in rev.get(int(p), [])})
        bounds = e0.groupby("person_id")["ClassTime"].agg(lo="min", hi="max").astype("datetime64[s]")
        visits = store.visits(pids, bounds=bounds, remap=mm or None, slack_h=cfg.visit_slack_h,
                              dates_only=False)          # exact times for the concept-id class; the proxy reduces to dates
        if len(visits):
            eeg["PatientClass"] = derive_patient_class(eeg, visits)
            eeg["PatientClass"] = eeg["PatientClass"].where(eeg["PatientClass"].notna(), cohort_proxy_class(eeg, visits, cfg))
        del visits
    cohort = [int(p) for p in build_candidates(eeg)["person_id"].dropna().unique()]
    fetch_ids = sorted(set(cohort) | {o for p in cohort for o in rev.get(p, [])})
    out = {"eeg_metadata": eeg}
    result_cols = schema.COLUMN_ALIASES["measurement.result_datetime"]
    remap = mm or None

    def drugs():
        it = iter_filtered_batches(s3, "drug_exposure", fetch_ids, DRUG_COLS, text_col="drug_source_value",
                                   pattern=SEDATION_RE.pattern)
        return _collect(it, lambda b: _compact_drugs(b, remap), DRUG_COLS[:2] + DRUG_COLS[2:3] + ["drug_type_concept_id"])

    def concepts():
        return load_concepts(s3, all_types=True)

    def notes():
        it = iter_filtered_batches(s3, "note", fetch_ids, NOTE_COLS)
        return _collect(it, lambda b: _compact_notes(b, remap), ["person_id", "note_datetime"])

    def imaging():
        spec = data_io.TABLES["imaging"]
        im = []
        for b in iter_filtered_batches(s3, "imaging", fetch_ids, schema.columns("imaging"), prefix=spec.pattern):
            im.append(remap_ids(b.to_pandas(), remap))
        return schema.coerce_types("imaging", pd.concat(im, ignore_index=True)) if im else pd.DataFrame(
            columns=schema.columns("imaging"))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        f_drug, f_concept, f_note, f_img = (pool.submit(f) for f in (drugs, concepts, notes, imaging))
        concept_all = f_concept.result()
        score_map, _ = concept_maps(concept_all)
        score_ids = list(score_map)

        def meas():
            it = iter_filtered_batches(s3, "measurement", fetch_ids, MEAS_COLS + result_cols,
                                       text_col="measurement_source_value", pattern=SCORE_RE.pattern + "|" + LAB_RE.pattern,
                                       id_col="measurement_concept_id", ids=score_ids)
            return _collect(it, lambda b: _compact_measurements(b, remap, score_map, result_cols),
                            ["person_id", "measurement_datetime", "kind", "score_class"])

        def obs():
            it = iter_filtered_batches(s3, "observation", fetch_ids, OBS_COLS, text_col="observation_source_value",
                                       pattern=SCORE_RE.pattern, id_col="observation_concept_id", ids=score_ids)
            return _collect(it, lambda b: _compact_observations(b, remap, score_map),
                            ["person_id", "observation_datetime", "score_class"])
        f_meas, f_obs = pool.submit(meas), pool.submit(obs)
        out["omop_drug_exposure"] = f_drug.result()
        out["omop_measurement"] = f_meas.result()
        out["omop_observation"] = f_obs.result()
        out["omop_note"] = f_note.result()
        out["imaging"] = f_img.result()
    out["omop_concept"] = select_concepts(concept_all, _type_ids(out["omop_drug_exposure"]))
    out["alignment_gaps"], out["alignment_order"] = load_alignment(s3, eeg, fetch_ids, remap, workers)
    return out


def load_alignment(s3, eeg: pd.DataFrame, fetch_ids, remap, workers: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Date-shift check inputs (D-117), STREAMED: for each candidate the signed gap in hours from EEG start to the
    nearest row of visit_occurrence (start / end), measurement, observation, note, drug_exposure and condition_occurrence
    (ALL rows of the candidates, not only the sedation / score rows), plus the covering-visit offset and the counts of
    the secondary ordering checks. Only person_id and the time columns are read; memory is O(candidates). These full
    reads of the big OMOP tables are the slow part of the audit."""
    from concurrent.futures import ThreadPoolExecutor
    cands = build_candidates(eeg)
    cands = cands.assign(person_id=cands["person_id"].astype("int64"))

    def small(table, cols):
        parts = [remap_ids(b.to_pandas(), remap) for b in iter_filtered_batches(s3, table, fetch_ids, cols)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)

    birth, dend = al.life_arrays(cands["person_id"], small("person", al.BIRTH_COLS), small("death", al.DEATH_COLS))
    acc = al.NearestGaps(cands, birth, dend)

    def events(src):
        tbl, dt, dd = al.EVENT_SOURCES[src]
        for b in iter_filtered_batches(s3, tbl, fetch_ids, ["person_id", dt, dd]):
            d = remap_ids(b.to_pandas(), remap)
            acc.add_events(src, d["person_id"].to_numpy("int64"), al.event_times(d, dt, dd))

    def visits():
        for b in iter_filtered_batches(s3, "visit_occurrence", fetch_ids, al.VISIT_COLS):
            d = remap_ids(b.to_pandas(), remap)
            acc.add_visits(d["person_id"].to_numpy("int64"), al.event_times(d, "visit_start_datetime", "visit_start_date"),
                           al.event_times(d, "visit_end_datetime", "visit_end_date"))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(visits)] + [pool.submit(events, src) for src in al.EVENT_SOURCES]
        for f in futs:
            f.result()
    return acc.frame(), acc.order_frame()


# ---------------------------------------------------------------- helpers
def cohort_proxy_class(eeg: pd.DataFrame, visits: pd.DataFrame, cfg: CohortConfig | None = None) -> pd.Series:
    """Care setting of each session by the COHORT's acute-care proxy (D-111, D-112): the visit covering the session's
    ``ClassTime`` date ([start date - 24 h, end date + 24 h]); 'Inpatient' when that visit is acute by the cohort rule
    (classified acute, else inpatient-length or an acute ServiceName), 'Outpatient' when a visit covers it but is not
    acute, ``None`` when no visit covers it. ``visits`` is raw ``omop_visit_occurrence`` or compact visits. This fills
    ``PatientClass`` where the concept-id class is unknown (it is 0 on every real visit)."""
    cfg = cfg or COHORT_CFG
    out = pd.Series(None, index=eeg.index, dtype=object)
    if not len(visits) or "ClassTime" not in eeg:
        return out
    e = pd.DataFrame({"person_id": eeg["person_id"], "t0": eeg["ClassTime"],
                      "ServiceName": eeg["ServiceName"] if "ServiceName" in eeg else None}).dropna(subset=["person_id", "t0"])
    if not len(e):
        return out
    e["person_id"] = e["person_id"].astype("int64")
    m = cohort_rules.match_visits(e, visits, cfg.acute_classes, cfg.visit_chain_gap_h, cfg.visit_slack_h, cfg.open_visit_days,
                                  cfg.date_only_end_of_day, dates_only=cfg.visit_dates_only)
    bc, bl, bs = cohort_rules.acute_parts(m["visit_class"], m["visit_inpatient_length"], e["ServiceName"],
                                          cfg.acute_classes, cfg.service_acute, cfg.use_service_proxy)
    has = m["visit_start"].notna()
    lab = np.where(has & (bc | bl | bs), "Inpatient", np.where(has, "Outpatient", None))
    out.loc[e.index] = lab
    return out


def acute_adult(eeg: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Adult EEG sessions in acute care, at the Study 1 sites (D-113), excluding OR / EMU services (cohort rule).
    PatientClass is in NO real eeg_metadata header; the loaders derive it from omop_visit_occurrence
    (``derive_patient_class``, the concept-id class) and fill the unknown ones with the cohort proxy
    (``cohort_proxy_class``). When the column is absent or empty (no visit matched) the acute-care filter cannot be
    applied and all adult sessions are used (second return value False; the report says so)."""
    adult = eeg[eeg["AgeAtVisit"] >= ADULT_AGE]
    if STUDY_SITES is not None and "SiteID" in adult:
        adult = adult[adult["SiteID"].astype(str).isin(STUDY_SITES)]
    if "ServiceName" in adult:
        adult = adult[~adult["ServiceName"].astype("string").str.upper().isin(list(COHORT_CFG.exclude_services)).fillna(False)]
    have_class = "PatientClass" in eeg and bool(eeg["PatientClass"].notna().any())
    return (adult[adult["PatientClass"].isin(ACUTE_CLASSES)] if have_class else adult), have_class


def build_candidates(eeg: pd.DataFrame) -> pd.DataFrame:
    """First qualifying EEG per adult patient (acute care, start time present): the cohort's candidate set."""
    e, _ = acute_adult(eeg)
    e = e[e["StartTime"].notna() & e["person_id"].notna()]
    e = e.sort_values(["person_id", "StartTime"] + (["SessionID"] if "SessionID" in e else []), kind="stable")
    first = e.groupby("person_id", as_index=False).first()
    return first[["person_id", "SiteID", "StartTime"]].rename(columns={"StartTime": "t0"})


def pass_cell(num: int, den: int) -> dict:
    """One suppressed pass / total / proportion cell.

    Small-cell rules (n < 11 -> "<11") are applied to BOTH sides of the split, so that a hidden cell cannot be
    recovered by subtraction:

    * ``den < 11`` or ``num < 11``: ``n_pass`` and ``proportion`` are "<11" (``n_total`` is "<11" only if den < 11);
    * fewer than 11 FAILURES (``den - num < 11``): the pass count is reported as a lower bound ``">=den-10"`` and the
      proportion as the floored lower bound ``">=0.9998"`` (the failure count is "<11" and stays hidden). The earlier
      code printed "<11" for the pass count here, which read as "almost nobody passes" at a site where nearly all
      records pass (a near-perfect site looked like the worst one, inconsistent with the overall row);
    * otherwise exact.
    """
    num, den = int(num), int(den)
    total = suppress_count(den)
    if den < SUPPRESS_BELOW or num < SUPPRESS_BELOW:
        return {"n_pass": SUPPRESSED, "n_total": total, "proportion": SUPPRESSED}
    if den - num < SUPPRESS_BELOW:
        lo = den - SUPPRESS_BELOW + 1
        return {"n_pass": f">={lo}", "n_total": total, "proportion": f">={math.floor(lo / den * 1e4) / 1e4:.4f}"}
    return {"n_pass": num, "n_total": total, "proportion": round(num / den, 4)}


def prop_block(flags: pd.Series, sites: pd.Series) -> tuple[dict, float]:
    """Suppressed overall + per-site proportion block (``pass_cell``), plus the raw overall proportion.

    Overall and per-site cells are built by the same function from the same boolean flags, so per-site pass counts
    add up to the overall pass count (exactly where unsuppressed; within the suppression bounds elsewhere)."""
    flags = flags.astype(bool).reset_index(drop=True)
    sites = sites.reset_index(drop=True)

    def one(f: pd.Series) -> dict:
        return pass_cell(int(f.sum()), int(len(f)))

    by_site = {str(s): one(flags[sites == s]) for s in sorted(sites.dropna().unique())}
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
def _site_alignment(cands: pd.DataFrame, gaps: pd.DataFrame) -> tuple[dict, pd.Series, dict]:
    """Per-site nearest-event alignment report (aggregate-only) from the per-candidate gap frame.

    Returns ``(by_site detail, within_24h flags per candidate, gate info)``. Whole-day gap = trunc(hours / 24)."""
    g = gaps.reindex(cands["person_id"].to_numpy("int64"))
    near = al.nearest_overall(g).to_numpy()
    sites = cands["SiteID"].astype(str).to_numpy()
    detail, evaluated, failed = {}, [], []
    for s_ in sorted(set(sites)):
        m = sites == s_
        n = int(m.sum())
        gm, nr = g[m], near[m]
        has = ~np.isnan(nr)
        w24 = has & (np.abs(nr) <= ALIGN_WINDOW_H)
        w72 = has & (np.abs(nr) <= ALIGN_WIDE_H)
        days = pd.Series(al.whole_days(pd.Series(nr)))
        vc = days.dropna().astype(int).value_counts().head(5)
        peaks = al.day_peaks(days, n, DAY_MODE_MAX_SHARE)
        cov = gm["cov_off_days"].notna().to_numpy()
        off, ln = gm["cov_off_days"].to_numpy(), gm["cov_len_days"].to_numpy()
        inr = cov & (off >= 0) & (off <= ln)
        has_clinical = bool(has.any())
        share24 = float(w24.sum() / n) if n else float("nan")
        site_pass = has_clinical and share24 >= ALIGN_MIN_SHARE and not peaks
        if has_clinical:
            evaluated.append(s_)
            if not site_pass:
                failed.append(s_)
        detail[s_] = {
            "n_candidates": suppress_count(n),
            "has_clinical_data": has_clinical,
            "candidates_with_any_event": pass_cell(int(has.sum()), n),
            "nearest_event_within_24h": pass_cell(int(w24.sum()), n),
            "nearest_event_within_72h": pass_cell(int(w72.sum()), n),
            "nearest_gap_hours_signed": safe_quantiles(nr[has]),
            "nearest_gap_hours_abs": safe_quantiles(np.abs(nr[has])),
            "per_table_within_24h": {
                src: pass_cell(int((gm[f"gap_h_{src}"].abs() <= ALIGN_WINDOW_H).sum()), n)["proportion"]
                for src in al.GAP_SOURCES},
            "per_table_nearest_gap_hours_q": {
                src: safe_quantiles(gm[f"gap_h_{src}"].dropna().to_numpy()) for src in al.GAP_SOURCES},
            "whole_day_gap_top5": [{"day": int(d), "n": suppress_count(c),
                                    "share": pass_cell(int(c), n)["proportion"]} for d, c in vc.items()],
            "nonzero_day_modes_over_5pct": peaks,
            "shift_detection": {
                "candidates_with_covering_visit": suppress_count(int(cov.sum())),
                "eeg_date_in_visit_date_range": pass_cell(int(inr.sum()), int(cov.sum()))["proportion"],
                "eeg_date_before_visit_start_date": pass_cell(int((cov & (off < 0)).sum()), int(cov.sum()))["proportion"],
                "eeg_date_after_visit_end_date": pass_cell(int((cov & (off > ln)).sum()), int(cov.sum()))["proportion"],
                "eeg_date_minus_visit_start_date_days": safe_quantiles(off[cov])},
            "site_gate": "PASS" if site_pass else ("FAIL" if has_clinical else "not evaluated (no clinical data)"),
        }
    flags = pd.Series(np.isfinite(near) & (np.abs(near) <= ALIGN_WINDOW_H), index=cands.index)
    return detail, flags, {"evaluated": evaluated, "failed": failed}


def _ordering_secondary(tables, gaps: pd.DataFrame, order: pd.DataFrame, cands: pd.DataFrame) -> dict:
    """SECONDARY counts (not the gate). Violations, precisely:

    * ``rows_before_birth``: an ancillary row (visit start, measurement, observation, note, drug exposure, condition)
      timed before the patient's birth (``birth_datetime``, else 1 January of ``year_of_birth``);
    * ``rows_after_death``: a row timed more than 24 h after the death time (``death_datetime``, else the end of
      ``death_date``); a missing death row is not survival, so it is never counted;
    * ``eeg_before_first_visit``: candidates whose EEG start is more than 24 h BEFORE the start of their earliest visit;
    * ``within_record_end_before_start``: end < start within one record (EEG start/end, drug start/end, imaging)."""
    rows = {}
    for src in al.GAP_SOURCES:
        if src == "visit_end" or src not in order.index:
            continue
        n_rows, nb, nd = (int(order.loc[src, c]) for c in ("n_rows", "n_before_birth", "n_after_death"))
        rows[src] = {"n_rows": suppress_count(n_rows), "n_before_birth": suppress_count(nb),
                     "n_after_death_plus_24h": suppress_count(nd)}
    fv = gaps["first_visit_gap_h"].reindex(cands["person_id"].to_numpy("int64"))
    n_v = int(fv.notna().sum())
    n_bad = int((fv > al.FIRST_VISIT_TOL_H).sum())
    mono, _, mono_n, mono_bad = _monotonicity(tables)
    return {"rows_by_source": rows,
            "eeg_before_first_visit": {"n_candidates_with_a_visit": suppress_count(n_v),
                                       "n_violations": suppress_count(n_bad)},
            "within_record_end_before_start": mono,
            "within_record_total": {"n_pairs": suppress_count(mono_n), "n_violations": suppress_count(mono_bad)}}


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

    # 2. Date-shift consistency (D-117): nearest-event alignment + shift detection; human hand-check sample separately.
    gaps = tables.get("alignment_gaps")
    order = tables.get("alignment_order")
    if gaps is None:
        gaps, order = al.alignment_from_frames(cands, tables)
    if order is None:
        order = pd.DataFrame(columns=["n_rows", "n_before_birth", "n_after_death"])
    detail, w24_flags, gate = _site_alignment(cands, gaps)
    blk24, _ = prop_block(w24_flags, cands["SiteID"].astype(str))
    w72 = al.nearest_overall(gaps.reindex(cands["person_id"].to_numpy("int64"))).abs() <= ALIGN_WIDE_H
    blk72, _ = prop_block(pd.Series(w72.to_numpy(), index=cands.index), cands["SiteID"].astype(str))
    secondary = _ordering_secondary(tables, gaps, order, cands)
    passed = bool(gate["evaluated"]) and not gate["failed"]
    modes = {s_: d["nonzero_day_modes_over_5pct"] for s_, d in detail.items() if d["nonzero_day_modes_over_5pct"]}
    site_txt = ", ".join(f"{s_} {d['nearest_event_within_24h']['proportion'] if isinstance(d['nearest_event_within_24h']['proportion'], str) else format(d['nearest_event_within_24h']['proportion'], '.1%')}"
                         for s_, d in detail.items() if d["has_clinical_data"])
    rows.append(_row(
        "Consistent within-patient date shift", "All timing logic",
        "Note, lab and EEG times line up on 20 hand-checked cases (automated: at every Study 1 site with clinical data, "
        f">= {ALIGN_MIN_SHARE:.0%} of candidates have an ancillary event (visit start/end, measurement, observation, "
        f"note, drug exposure, condition) within +-{ALIGN_WINDOW_H:.0f} h of EEG start AND no non-zero whole-day "
        f"nearest-gap mode holds > {DAY_MODE_MAX_SHARE:.0%} of the site's candidates; ordering violations are "
        "reported as a secondary count, not the gate; D-117)",
        "Stop", True,
        {"within_24h": blk24, "within_72h": blk72, "by_site_detail": detail,
         "sites_evaluated": gate["evaluated"], "sites_failed": gate["failed"],
         "nonzero_day_modes_by_site": modes,
         "ordering_violations_secondary": secondary},
        passed,
        {"observed_text": (f"nearest event within +-24 h: {site_txt or 'no site with clinical data'}; "
                           f"non-zero day modes >5%: {modes or 'none'}; "
                           f"sites failing: {gate['failed'] or 'none'}"),
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
        elif isinstance(obs, dict) and "within_24h" in obs:
            for key, lab in (("within_24h", "nearest event within +-24 h"), ("within_72h", "nearest event within +-72 h")):
                for s, b in obs[key]["by_site"].items():
                    L.append(f"| {r['field']} ({lab}) | {s} | {b['n_pass']}/{b['n_total']} | {b['proportion']} |")
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
    L += ["", "## Date-shift detail (nearest-event alignment, D-117)", "",
          f"Sites evaluated: {', '.join(ds['observed']['sites_evaluated']) or 'none'}; failing: "
          f"{', '.join(ds['observed']['sites_failed']) or 'none'}. Whole-day gap = trunc(hours / 24); day 0 is within "
          "+-24 h.", "",
          "| Site | Gate | Candidates | Any event | Within 24 h | Within 72 h | Gap h q10/q25/q50/q75/q90 (signed) | "
          "Top-5 whole-day gaps (day: n) | Non-zero day modes > 5% |", "|---|---|---|---|---|---|---|---|---|"]
    for s_, d in ds["observed"]["by_site_detail"].items():
        q = d["nearest_gap_hours_signed"]
        top = ", ".join(f"{t['day']}: {t['n']}" for t in d["whole_day_gap_top5"]) or "n/a"
        L.append(f"| {s_} | {d['site_gate']} | {d['n_candidates']} | {d['candidates_with_any_event']['proportion']} | "
                 f"{d['nearest_event_within_24h']['proportion']} | {d['nearest_event_within_72h']['proportion']} | "
                 f"{'/'.join(str(q[k]) for k in sorted(q))} | {top} | "
                 f"{d['nonzero_day_modes_over_5pct'] or 'none'} |")
    L += ["", "### Shift detection (candidates with a covering visit: EEG date minus visit start date, days)", "",
          "| Site | With covering visit | EEG date within [start, end] | Before start date | After end date | "
          "q10/q25/q50/q75/q90 |", "|---|---|---|---|---|---|"]
    for s_, d in ds["observed"]["by_site_detail"].items():
        sd = d["shift_detection"]
        q = sd["eeg_date_minus_visit_start_date_days"]
        L.append(f"| {s_} | {sd['candidates_with_covering_visit']} | {sd['eeg_date_in_visit_date_range']} | "
                 f"{sd['eeg_date_before_visit_start_date']} | {sd['eeg_date_after_visit_end_date']} | "
                 f"{'/'.join(str(q[k]) for k in sorted(q))} |")
    sec = ds["observed"]["ordering_violations_secondary"]
    L += ["", "### Ordering violations (SECONDARY counts, not the gate)", "",
          "| Check | n | violations |", "|---|---|---|"]
    for src, v in sec["rows_by_source"].items():
        L.append(f"| {src}: row before birth | {v['n_rows']} | {v['n_before_birth']} |")
        L.append(f"| {src}: row more than 24 h after death | {v['n_rows']} | {v['n_after_death_plus_24h']} |")
    e = sec["eeg_before_first_visit"]
    L.append(f"| EEG start > 24 h before the earliest visit start | {e['n_candidates_with_a_visit']} | {e['n_violations']} |")
    for k, v in sec["within_record_end_before_start"].items():
        L.append(f"| {k} (end before start) | {v['n_pairs']} | {v['n_violations']} |")
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
    ap.add_argument("--workers", type=int, default=3,
                    help="threads reading the big OMOP tables at once (S3 reads overlap; raise only with memory headroom)")
    ap.add_argument("--all-sites", action="store_true",
                    help="do not restrict the candidates to the Study 1 sites I0002, I0003, S0001, S0002 (D-113)")
    ap.add_argument("--max-memory-gb", type=float, default=None,
                    help="guard: RLIMIT_AS in GB (address space; pick generously); prints an aggregate error if exceeded")
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
    global STUDY_SITES
    if a.all_sites:
        STUDY_SITES = None
    from ..cohort.memguard import apply_limit, peak_rss_gb
    apply_limit(a.max_memory_gb)
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
        tables = load_audit_tables(s3, a.sites, workers=a.workers)
        stage = "audit"
        report, ids = run_audit(tables, a.seed, a.handcheck_n)
    except MemoryError:
        safe_print(f"field audit FAILED at stage '{stage}': MemoryError (aggregate only): exceeded the memory limit"
                   f"{f' of {a.max_memory_gb:g} GB' if a.max_memory_gb else ''}; peak RSS {peak_rss_gb():.2f} GB. "
                   "Re-run with a larger --max-memory-gb, fewer --sites or --workers 1.")
        return 3
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
    safe_print(f"Reports: {out / 'field_audit.md'}, {out / 'field_audit.json'}; peak RSS {peak_rss_gb():.2f} GB",
               known_ids=known)
    if a.strict and not report["gate_0a_automated_stop_rows_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
