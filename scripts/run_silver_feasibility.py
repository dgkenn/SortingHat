#!/usr/bin/env python3
"""EXPLORATORY — silver-label feasibility; not a test of preregistered hypotheses.   (D-143, docs/silver_feasibility.md)

Trains AND evaluates Study 1A/1B-style models on structured, EEG-blind SILVER labels (no gold labels exist yet), before
Gate 0, at the project lead's direction. Nothing here is a test of H1-H6.

    # synthetic / local mirror (also what the tests run)
    python3 scripts/run_silver_feasibility.py --data data/synthetic --cohort <local_only>/cohort.csv ...
    # real (streaming, aggregate-only; D-118). --s3 is needed only for the circularity audit (EEG-report comparator)
    scripts/heedb_run.sh python3 scripts/run_silver_feasibility.py --s3

Intended use first (D-145): the FIRST reported block, "Intended use: undifferentiated AMS", asks the question of an unknown EEG in
the ED with little or nothing known: (a) EEG-only vs the prevalence prior, (b) EEG + age/sex vs age/sex, (c) Baseline P
(Presentation) + EEG vs P, each for the whole analysed set and for the undifferentiated subgroup (EEG on encounter day 0-1, no
primary-label-family ICD code recorded before t0, no sedative/opioid exposure in the 6 h before t0). The baselines use the
current encounter only unless build_baselines.py was run with --with-history (then the report says "with history"). EEG-derived
information is never an input or label evidence. The existing Baseline A / C comparisons follow.

Inputs (record-level, all under local_only/, never printed)
    --cohort     out/local_only/cohort_study1.csv       (SiteID, person_id, SessionID, BidsFolder, EEGFolder, t0, in_strict*, gcs_*, four_*, duration_s)
    --features   out/local_only/features/part-*.parquet (extractor output; primary window passing QC; keyed by recording_id)
    --silver     out/local_only/silver/silver_labels.csv (sortinghat.labels.extract main; E1 E2 E4a E5 E6 E7 + e4b_* hints)
    --baselines  out/local_only/baselines_AC.parquet    (scripts/build_baselines.py; Baselines A, C and P as_of-gated at t0; meta columns)
Outputs (aggregate-only, safe_output; n < 11 shown as "<11"):  <out>/report.md  and  <out>/report.json

Design
    labels      E1 E2 E4a E5 E6 primary; E7 primary only with >= 100 positives over >= 2 sites (else exploratory); E3 and E4b
                excluded (D-002/D-003). A label with < 11 positives or negatives at any site is not analysed.
    rungs       prior, qEEG, connectivity, qEEG+connectivity (the headline rung). Dynamics skipped. When --embeddings holds
                parquet parts from scripts/extract_embeddings.py, two more rungs are added: cbramod_frozen (frozen public CBraMod
                embedding alone) and combined_cbramod (qEEG + connectivity + embedding); the headline rung is unchanged. No
                embedding parts: both are skipped silently. Likewise --morgoth (default <features>/../morgoth, parts from
                scripts/extract_morgoth.py / extract_rungs.py) adds morgoth_findings (MORGOTH finding probabilities alone) and
                combined_morgoth (qEEG + connectivity + MORGOTH). The report lists the rungs that ran and those skipped.
    commercial-clean gap  only when cbramod_frozen AND morgoth_findings ran: Delta(cbramod_frozen + baseline) - Delta(morgoth_findings +
                baseline), pooled and per site, paired patient-level bootstrap (same CI levels), plus a MORGOTH-exposure sensitivity
                row (--morgoth-exposure, a local_only flag file from scripts/extract_morgoth.py exposure) that drops the flagged
                patients from the scored rows. Reported after the headline Delta; the intended-use block stays first.
    comparators Baseline A (H1-style) and Baseline C (H2-style): Delta = masked log loss(baseline + EEG) - masked log loss(baseline).
    validation  (a) leave-one-site-out across the two sites, (b) late-calendar temporal holdout within site (D-120).
    labels used the SAME structured silver labels train, tune (20% development rows) and score. No gold exists.
    intervals   99% within-site bootstrap (D-095; evaluation_sample_size.md s8) + the every-site rule, 95% intervals co-reported.
    controls    leakage probes (site, duration, channel count and, informationally, the t0 sedation flag), sedative-excluded subset,
                severity strata, shuffled-label and permuted-EEG negative controls.
    sedation    "Sedation confounding" section: Delta of the headline rung within sedated and non-sedated strata of the SAME held-out
                predictions (no refit), their difference with a bootstrap CI, and whether Baselines A / C already carry sedation features.
    audit       circularity audit: held-out silver-trained predictions' agreement with EEG-report findings vs with silver labels.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))             # repo root, so `sortinghat` imports
sys.path.insert(0, str(HERE))                    # sibling scripts (build_baselines)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import build_baselines as bb  # noqa: E402
from sortinghat import agent_safety, checkpoint, data_io  # noqa: E402
from sortinghat.labels.circularity_audit import run_circularity_audit  # noqa: E402
from sortinghat.labels.extract import eeg_impression_comparator, load_reports_findings  # noqa: E402
from sortinghat.metrics.bootstrap import bootstrap_ci, paired_bootstrap_delta  # noqa: E402
from sortinghat.metrics.calibration import per_label_report  # noqa: E402
from sortinghat.metrics.hypotheses import h3_severity_stratified  # noqa: E402
from sortinghat.metrics.labels import e7_eligible  # noqa: E402
from sortinghat.metrics.sedation import MIN_STRATUM_N as SED_MIN_STRATUM_N, stratified_delta  # noqa: E402
from sortinghat.metrics.splits import late_temporal_holdout  # noqa: E402
from sortinghat.models import (CBRAMOD_FROZEN_RUNG, COMMERCIAL_RUNGS, DEFAULT_RUNGS, MORGOTH_FINDINGS_RUNG,  # noqa: E402
                               LadderConfig, ModelData, RungSpec, delta_concentration_by_site, leakage_probes,
                               load_morgoth_findings, negative_control, run_ladder, sedative_excluded_rerun)
from sortinghat.models import ladder as ladder_mod  # noqa: E402
from sortinghat.models.controls import control_failed  # noqa: E402
from sortinghat.models.report import clean  # noqa: E402
from sortinghat.safe_output import (SUPPRESSED, safe_print, safe_write_json, safe_write_text,  # noqa: E402
                                    suppress_count, suppress_proportion)

BANNER = "EXPLORATORY — silver-label feasibility; not a test of preregistered hypotheses"
CANDIDATE_LABELS = ("E1", "E2", "E4a", "E5", "E6", "E7")
EXCLUDED_BY_RULE = {"E3": "partly EEG-defined; the pipeline's positive control, not a Study 1 label (D-002)",
                    "E4b": "iatrogenic sedation, known to the team at t0 (D-003); used only as a control hint"}
RUNG_NAMES = ("prior", "qeeg", "connectivity", "combined")
EEG_PREFIXES = ("qeeg.", "conn.")
CBRAMOD_PREFIX = "emb.cbramod."
MORGOTH_PREFIX = "morgoth."
EXTRA_RUNG_NAMES = ("cbramod_frozen", "combined_cbramod", "morgoth_findings", "combined_morgoth")
GAP_RUNGS = ("cbramod_frozen", "morgoth_findings")           # the commercial-clean gap pair (Delta of the first minus the second)
# "combined" here is qEEG + connectivity whatever else is loaded (DEFAULT_RUNGS' combined also lists emb./morgoth./dyn. columns)
BASE_RUNGS = [RungSpec(r.name, EEG_PREFIXES) if r.name == "combined" else r for r in DEFAULT_RUNGS if r.name in RUNG_NAMES]
QC_COLS = ("recording_id", "window", "qc_pass", "usable_fraction", "onset_offset_s", "qc_n_missing_or_dead_min")
MIN_SITE_CELL = 11
N_MINIMUM_CHANNELS = 10
ANCHOR_OVERLAP_AUROC = 0.90                      # baseline-only AUROC at/above this: the baseline probably sees the label's anchors
MIN_EVENTS_FOR_SLOPE = 100                       # D-094: slope only for labels with >= 100 events, else O/E only
# Severity strata fixed a priori for this exploratory run (SAP: "cut points fixed before unblinding"; here before any run):
# GCS-equivalent <= 5 (severe), 6-8 (moderate), >= 9. FOUR is mapped linearly to a GCS equivalent (3 + 0.75 * FOUR) only
# where no GCS exists. Stored as severity = 15 - GCS-equivalent, strata cut at 6.5 and 9.5 (low < 6.5 <= mid < 9.5 <= high).
SEVERITY_CUTS = (6.5, 9.5)
# ---- Intended use (D-145)
IU_TITLE = "Intended use: undifferentiated AMS"
IU_COMPARISONS = (("prior", "a. EEG-only vs prevalence prior"), ("agesex", "b. EEG + age/sex vs age/sex"),
                  ("P", "c. P (Presentation) + EEG vs P"))
UNDIFF_MAX_H = 36.0                              # EEG on encounter day 0-1: t0 within 36 h of the encounter start
UNDIFF_SEDATION_H = 6.0                          # no sedative / opioid exposure in the 6 h before t0 (Baseline A sedation features)
IU_MIN_SUBGROUP = 50                             # evaluation rows needed to estimate Delta in the subgroup (and >= 11 per site)
UNKNOWN_SCOPE = "not recorded (baselines built before D-145; prior-encounter data may be included)"
LIMITATIONS = (
    "Silver labels train, tune AND score the models: agreement with the label source is not accuracy against a clinical truth.",
    "Silver-label noise is not independent of the baselines (shared structured data); a baseline that sees the anchor "
    "data inflates the baseline and can hide an EEG increment, a baseline blind to it does the opposite.",
    "Silver labels are structured-data only (D-123): toxic/infectious/metabolic etiologies are under-ascertained and E1 "
    "rests on diagnosis/procedure codes with approximate timing.",
    "With two sites, leave-one-site-out has two folds, the cluster and site-level intervals are degenerate or extremely "
    "wide, and the every-site rule is a two-site rule; the cross-site claim is weak (D-120).",
    "The temporal holdout assumes shifted dates preserve calendar order within a site; if the release shifts dates per "
    "patient the 'late' cases are an arbitrary subset and this scheme is a random split.",
    "Development rows (20%) are drawn from the same silver labels; selection and recalibration therefore tune to silver noise.",
    "Delta compares models fitted with identical heads; a nonzero Delta can reflect EEG features that proxy for the "
    "silver-label ascertainment process (for example sedation, ICU monitoring intensity), see the controls.",
    "No result here tests H1-H6 or moves any gate; it is an early look at whether EEG carries signal worth adjudicating.",
)


# ====================================================================================================== inputs
def require_local_only(p, what: str) -> None:
    bb.require_local_only(Path(p), what)


def opaque_recording_id(key: str) -> str:
    """The extractor's recording id of an EDF key (``scripts/extract_eeg_features.opaque_id``; pinned by a test)."""
    return "rec" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def recording_ids(cohort: pd.DataFrame, recording_map: pd.DataFrame | None = None) -> pd.Series:
    """Extractor recording id per cohort row: from the documented EDF key (``data_io.edf_key_for_row``), unless an explicit
    person_id -> recording_id map is supplied."""
    if recording_map is not None:
        m = recording_map.drop_duplicates("person_id").set_index("person_id")["recording_id"].astype(str)
        return cohort["person_id"].map(m)
    ef = cohort["EEGFolder"] if "EEGFolder" in cohort else pd.Series([None] * len(cohort), index=cohort.index)
    return pd.Series([opaque_recording_id(data_io.edf_key_for_row(s, b, sid, e))
                      for s, b, sid, e in zip(cohort["SiteID"], cohort["BidsFolder"], cohort["SessionID"], ef)],
                     index=cohort.index)


def load_baselines(path) -> tuple[pd.DataFrame, dict]:
    """(matrix with person_id + columns, column sets) written by ``scripts/build_baselines.py``. The sets always hold 'A' and 'C';
    'P' (Baseline P, Presentation), 'AGE_SEX' (A's demographic columns) and 'meta' (subgroup-definition columns, never model
    inputs) are present when the sidecar lists them (a parquet written before D-145 has none of these)."""
    import json
    path = Path(path)
    require_local_only(path, "--baselines")
    df = pd.read_parquet(path)
    side_all = json.loads(path.with_name(path.stem + ".columns.json").read_text())
    side = side_all["baselines"]
    cols = {b: [c for c in side[b] if c in df.columns] for b in ("A", "C")}
    if "P" in side:
        cols["P"] = [c for c in side["P"] if c in df.columns]
    demo = [c for c in cols["A"] if c.startswith("demo__")]
    if demo:
        cols["AGE_SEX"] = demo
    meta = [c for c in side_all.get("meta", []) if c in df.columns]
    if meta:
        cols["meta"] = meta
    df["person_id"] = df["person_id"].astype("int64")
    return df.drop_duplicates("person_id").reset_index(drop=True), cols


def baseline_scope(path) -> str:
    """'current encounter only' | 'with history' | UNKNOWN_SCOPE, from the sidecar of the baselines parquet."""
    import json
    p = Path(path)
    try:
        return str(json.loads(p.with_name(p.stem + ".columns.json").read_text()).get("encounter_scope_label", UNKNOWN_SCOPE))
    except (OSError, ValueError):
        return UNKNOWN_SCOPE


def _tf(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    t = s.astype("string").str.strip().str.lower()
    m = t.isin(["true", "false"]).to_numpy(bool)
    return (t == "true").fillna(False).to_numpy(bool).astype(float), m


def load_silver(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Y, hints) indexed by person_id: Y is 0.0 / 1.0 / NaN (not assessable) per label column present in the file
    (excluded labels are never loaded); hints are the e4b_* booleans. ``case_id`` is str(person_id)."""
    path = Path(path)
    require_local_only(path, "--silver")
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    idc = "case_id" if "case_id" in df.columns else df.columns[0]
    pid = pd.to_numeric(df[idc], errors="coerce")
    df = df[pid.notna()].copy()
    pid = pid[pid.notna()].astype("int64")
    Y, H = {}, {}
    for lab in CANDIDATE_LABELS:
        if lab in df.columns:
            y, m = _tf(df[lab])
            Y[lab] = np.where(m, y, np.nan)
    for c in df.columns:
        if str(c).startswith("e4b_"):
            y, m = _tf(df[c])
            H[c] = (y > 0) & m
    Yd = pd.DataFrame(Y, index=pid.to_numpy()).groupby(level=0).first()
    Hd = pd.DataFrame(H, index=pid.to_numpy()).groupby(level=0).max() if H else pd.DataFrame(index=Yd.index)
    return Yd, Hd


def load_eeg_primary(feat_dir, rec_ids: set[str], batch_rows: int = 20_000) -> tuple[pd.DataFrame, dict]:
    """Primary-window rows of the wanted recordings from ``part-*.parquet``, read batch by batch with the window and
    recording filters applied inside Arrow, so only the wanted rows and the qeeg./conn. columns reach pandas.
    Returns (frame indexed by recording_id, one row per recording, QC-passing row preferred), and aggregate counts."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    feat_dir = Path(feat_dir)
    require_local_only(feat_dir, "--features")
    parts = sorted(feat_dir.glob("part-*.parquet"))
    if not parts:
        raise SystemExit("no part-*.parquet feature files in --features")
    want = pa.array(sorted(rec_ids), type=pa.string())
    frames: list[pd.DataFrame] = []
    feat_cols: list[str] | None = None
    for p in parts:
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names
        if feat_cols is None:
            feat_cols = [c for c in names if str(c).startswith(EEG_PREFIXES)]
        use = [c for c in QC_COLS if c in names] + [c for c in feat_cols if c in names]
        if "recording_id" not in use or "window" not in use:
            continue
        for batch in pf.iter_batches(batch_size=batch_rows, columns=use):
            t = pa.Table.from_batches([batch])
            mask = pc.and_(pc.equal(t["window"].cast(pa.string()), "primary"),
                           pc.is_in(t["recording_id"].cast(pa.string()), value_set=want))
            t = t.filter(pc.fill_null(mask, False))
            if t.num_rows:
                frames.append(t.to_pandas())
    stats = {"n_primary_rows_read": sum(len(f) for f in frames)}
    if not frames:
        return pd.DataFrame(columns=list(QC_COLS[1:]) + (feat_cols or [])), {**stats, "n_recordings": 0, "n_qc_pass": 0}
    df = pd.concat(frames, ignore_index=True)
    df["recording_id"] = df["recording_id"].astype(str)
    df["qc_pass"] = df["qc_pass"].astype(bool)
    df = df.sort_values("qc_pass", ascending=False, kind="stable").drop_duplicates("recording_id")
    df = df.set_index("recording_id")
    for c in feat_cols or []:
        if c not in df:
            df[c] = np.nan
    stats.update(n_recordings=len(df), n_qc_pass=int(df["qc_pass"].sum()))
    return df, stats


def load_cbramod_primary(emb_dir, rec_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Primary-window frozen-CBraMod embeddings (``emb.cbramod.<j>``, one row per recording, indexed by recording_id) from the
    parquet parts written by ``scripts/extract_embeddings.py``. A missing / empty directory returns an empty frame (the
    rung is then skipped); an existing directory outside local_only/ is refused. Counts only."""
    from sortinghat.embed.store import load_primary_embeddings
    d = Path(emb_dir)
    require_local_only(d, "--embeddings")
    if not d.is_dir() or not any(d.glob("part-*.parquet")):
        return pd.DataFrame(), {"n_parts": 0, "n_recordings": 0, "n_emb_ok": 0}
    df, st = load_primary_embeddings(d, rec_ids, "primary", CBRAMOD_PREFIX, require_ok=True)
    return df[[c for c in df.columns if str(c).startswith(CBRAMOD_PREFIX)]], st


def load_morgoth_primary(mg_dir, rec_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Primary-window, QC-passing MORGOTH finding probabilities (``morgoth.*``, one row per recording, indexed by recording_id)
    from the parquet parts written by ``scripts/extract_morgoth.py features`` (``inputs.load_morgoth_findings``). A missing /
    empty directory returns an empty frame (the rung is then skipped); an existing directory outside local_only/ is refused.
    Counts only."""
    d = Path(mg_dir)
    require_local_only(d, "--morgoth")
    parts = sorted(d.glob("part-*.parquet")) if d.is_dir() else []
    if not parts:
        return pd.DataFrame(), {"n_parts": 0, "n_recordings": 0, "n_morgoth_ok": 0}
    ids = sorted(rec_ids)
    df = load_morgoth_findings(d, ids)
    if df is None or not len(ids):
        return pd.DataFrame(), {"n_parts": len(parts), "n_recordings": 0, "n_morgoth_ok": 0}
    df.index = pd.Index(ids, name="recording_id")
    df = df[df.notna().any(axis=1)]
    return df, {"n_parts": len(parts), "n_recordings": int(len(df)), "n_morgoth_ok": int(len(df))}


def load_morgoth_exposure(path) -> tuple[pd.Series, str] | None:
    """Per-patient MORGOTH exposure flag (1.0 exposed / 0.0 not, indexed by person_id; RECORD-LEVEL, memory only) and the basis
    used, from the local_only parquet written by ``sortinghat.morgoth.exposure.write_flags``: ``in_morgoth_train_split`` when the
    lists carried a split column, else ``in_morgoth_lists`` (any list: training, validation or test; the conservative flag).
    None when the file does not exist or holds neither column."""
    p = Path(path)
    require_local_only(p, "--morgoth-exposure")
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    if "person_id" not in df:
        return None
    for col, basis in (("in_morgoth_train_split", "MORGOTH training split"),
                       ("in_morgoth_lists", "any MORGOTH data list (training / validation / test not separable)")):
        if col in df and df[col].notna().any():
            f = df[col].map(lambda v: np.nan if pd.isna(v) else float(bool(v)))
            f.index = pd.to_numeric(df["person_id"], errors="coerce")
            f = f[f.index.notna()]
            f.index = f.index.astype("int64")
            return f.groupby(level=0).max(), basis
    return None


# ============================================================================================== assembly
@dataclass
class Assembly:
    frame: pd.DataFrame                  # analysis rows (RECORD-LEVEL, memory only)
    baseline: pd.DataFrame
    eeg: pd.DataFrame
    y: np.ndarray
    m: np.ndarray
    label_names: tuple
    primary: tuple
    include_e7: bool
    baseline_sets: dict
    covariates: dict
    sites: np.ndarray
    times: np.ndarray
    flow: dict
    label_report: dict
    site_labels: dict
    iu_sets: dict = field(default_factory=dict)          # D-145: 'prior' ([]), 'agesex', 'P' column sets of the intended-use block
    subgroup: np.ndarray | None = None                   # bool (n,): undifferentiated subgroup (RECORD-LEVEL, memory only)
    subgroup_info: dict = field(default_factory=dict)    # aggregate, suppressed
    baseline_scope: str = UNKNOWN_SCOPE
    embedding_info: dict = field(default_factory=dict)   # aggregate, suppressed: frozen-CBraMod coverage (empty = rung skipped)
    morgoth_info: dict = field(default_factory=dict)     # aggregate, suppressed: MORGOTH-findings coverage (empty = rung skipped)
    morgoth_exposure: dict = field(default_factory=dict)  # {"flag": float (n,) 1/0/NaN (RECORD-LEVEL, memory only), "basis": str} or {}


def gcs_equivalent(gcs: pd.Series, four: pd.Series) -> pd.Series:
    """GCS where charted, else a linear GCS equivalent of FOUR (3 + 0.75 * FOUR); NaN when neither exists."""
    g = pd.to_numeric(gcs, errors="coerce")
    f = pd.to_numeric(four, errors="coerce")
    return g.where(g.notna(), 3.0 + 0.75 * f)


def sedation_flag(bl: pd.DataFrame, hints: pd.DataFrame, source: str) -> np.ndarray:
    """t0 sedative/opioid exposure: Baseline A ``sed__{sedative,opioid}__on_t0`` and/or the silver e4b sedative hint."""
    n = len(bl)
    from_bl = np.zeros(n, bool)
    for c in ("sed__sedative__on_t0", "sed__opioid__on_t0"):
        if c in bl:
            from_bl |= (bl[c].fillna(0).to_numpy(float) > 0)
    from_e4b = hints["e4b_sedative_exposure"].reindex(bl.index).fillna(False).to_numpy(bool) \
        if "e4b_sedative_exposure" in hints else np.zeros(n, bool)
    return {"baseline": from_bl, "e4b": from_e4b, "either": from_bl | from_e4b}[source]


def sedation_6h_flag(bl: pd.DataFrame) -> np.ndarray | None:
    """Sedative / opioid exposure in the 6 h before t0 from the Baseline A sedation features: a sedative or opioid running at t0
    (``sed__{sedative,opioid}__on_t0``) or a recorded quantity attributable to the 6 h window (``sed__<agent>__qty_6h`` > 0).
    None when the frame has no such column (the exposure cannot be established). A bolus with no recorded quantity that ended
    before t0 is invisible to these features; that limitation is stated in docs/silver_feasibility.md."""
    cols = [c for c in ("sed__sedative__on_t0", "sed__opioid__on_t0") if c in bl] + \
           [c for c in bl.columns if str(c).startswith("sed__") and str(c).endswith("__qty_6h")]
    if not cols:
        return None
    exposed = np.zeros(len(bl), bool)
    for c in cols:
        exposed |= (bl[c].fillna(0).to_numpy(float) > 0)
    return exposed


def hours_since_encounter(sel: pd.DataFrame, bl: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Hours from the encounter start to t0 and where it came from: the cohort table's ``encounter_start`` column, else the
    baselines parquet's ``meta__hours_since_encounter_start`` (resolved by build_baselines for an older cohort table)."""
    if "encounter_start" in sel and sel["encounter_start"].notna().any():
        h = (pd.to_datetime(sel["t0"]) - pd.to_datetime(sel["encounter_start"])).dt.total_seconds().to_numpy(float) / 3600.0
        return h, "cohort encounter_start"
    if "meta__hours_since_encounter_start" in bl:
        return bl["meta__hours_since_encounter_start"].to_numpy(float), "baselines meta (build_baselines)"
    return np.full(len(sel), np.nan), "unavailable"


def undifferentiated_flag(sel: pd.DataFrame, bl: pd.DataFrame, max_hours: float = UNDIFF_MAX_H) -> tuple[np.ndarray, dict]:
    """Undifferentiated-presentation subgroup, all three required:
      1. EEG on encounter day 0-1: 0 <= t0 - encounter start <= ``max_hours`` (36 h);
      2. no ICD diagnosis code of the primary label families recorded at or before t0 (any encounter; ``meta__label_dx_before_t0``);
      3. no sedative / opioid exposure in the 6 h before t0 (Baseline A sedation features).
    A component that cannot be established counts as not met, so the subgroup never grows through missing data. EEG-derived
    information is not used."""
    n = len(sel)
    hours, hsrc = hours_since_encounter(sel, bl)
    early = np.isfinite(hours) & (hours >= 0) & (hours <= max_hours)
    dx = bl["meta__label_dx_before_t0"].to_numpy(float) if "meta__label_dx_before_t0" in bl else np.full(n, np.nan)
    no_dx = np.isfinite(dx) & (dx == 0)
    sed = sedation_6h_flag(bl)
    no_sed = (~sed) if sed is not None else np.zeros(n, bool)
    flag = early & no_dx & no_sed
    return flag, {"early": early, "no_dx": no_dx, "no_sedation": no_sed,
                  "sources": {"visit_start": hsrc, "label_family_dx": ("baselines meta (build_baselines)" if "meta__label_dx_before_t0" in bl
                                                                           else "unavailable"),
                              "sedation_6h": "Baseline A sedation features" if sed is not None else "unavailable"}}


def _site_counts(sites: pd.Series, mask: np.ndarray) -> dict:
    return {s: int((mask & (sites.to_numpy() == s)).sum()) for s in sorted(sites.unique())}


def assemble(a) -> Assembly:
    sites_arg = None if a.sites == ["all"] else a.sites
    cohort = bb.load_cohort(a.cohort, sites_arg, a.cohort_def)
    rmap = None
    if a.recording_map:
        require_local_only(a.recording_map, "--recording-map")
        rmap = pd.read_csv(a.recording_map, dtype={"recording_id": str})
    cohort["rid"] = recording_ids(cohort, rmap)
    bl, bl_cols = load_baselines(a.baselines)
    Y, hints = load_silver(a.silver)
    eeg, eeg_stats = load_eeg_primary(a.features, set(cohort["rid"].dropna()))
    feat_cols = [c for c in eeg.columns if str(c).startswith(EEG_PREFIXES)]
    emb, emb_stats = load_cbramod_primary(getattr(a, "embeddings", None) or Path(a.features).parent / "embeddings", set(cohort["rid"].dropna()))
    emb_cols = list(emb.columns)
    mg, mg_stats = load_morgoth_primary(getattr(a, "morgoth", None) or Path(a.features).parent / "morgoth", set(cohort["rid"].dropna()))
    mg_cols = list(mg.columns)

    sites_s = cohort["SiteID"]
    has_bl = cohort["person_id"].isin(bl["person_id"]).to_numpy(bool)
    has_eeg = cohort["rid"].isin(eeg.index).to_numpy(bool)
    qc_pass = cohort["rid"].map(eeg["qc_pass"]).fillna(False).astype(bool).to_numpy(bool) if len(eeg) else \
        np.zeros(len(cohort), bool)
    has_silver = cohort["person_id"].isin(Y.index).to_numpy(bool)
    steps = [("cohort rows (sites and cohort definition applied)", np.ones(len(cohort), bool)),
             ("with a Baseline A-C row", has_bl),
             ("with a primary-window EEG feature row", has_bl & has_eeg),
             ("primary window passing QC", has_bl & has_eeg & qc_pass),
             ("with a silver-label row", has_bl & has_eeg & qc_pass & has_silver)]
    keep = steps[-1][1]
    sel = cohort[keep].reset_index(drop=True)
    sel_bl = bl.set_index("person_id").reindex(sel["person_id"]).reset_index(drop=True)
    sel_eeg = eeg.reindex(sel["rid"])[[c for c in ("qc_n_missing_or_dead_min", "usable_fraction") if c in eeg] + feat_cols
                                      ].reset_index(drop=True) if len(eeg) else pd.DataFrame(index=range(len(sel)))
    sel_emb = emb.reindex(sel["rid"])[emb_cols].reset_index(drop=True) if emb_cols else pd.DataFrame(index=range(len(sel)))
    sel_mg = mg.reindex(sel["rid"])[mg_cols].reset_index(drop=True) if mg_cols else pd.DataFrame(index=range(len(sel)))
    Ys = Y.reindex(sel["person_id"]).reset_index(drop=True)
    Hs = hints.reindex(sel["person_id"]).reset_index(drop=True) if len(hints.columns) else pd.DataFrame(index=range(len(sel)))
    ssites = sel["SiteID"].astype(str)
    uniq = sorted(ssites.unique())
    site_labels = {s: f"site_{i + 1}" for i, s in enumerate(sorted(sites_s.unique()))}

    # ---- label inclusion (counts per site; cells < 11 mean the label is not analysable)
    included, excluded, label_report = [], {}, {}
    for lab in CANDIDATE_LABELS:
        if lab not in Ys.columns or Ys[lab].notna().sum() == 0:
            excluded[lab] = "unavailable from structured data (no assessable values)"
            continue
        pos = {s: int(((Ys[lab] == 1) & (ssites == s)).sum()) for s in uniq}
        neg = {s: int(((Ys[lab] == 0) & (ssites == s)).sum()) for s in uniq}
        ok = all(pos[s] >= MIN_SITE_CELL and neg[s] >= MIN_SITE_CELL for s in uniq)
        if ok:
            included.append(lab)
            label_report[lab] = {
                "n_assessable": int(Ys[lab].notna().sum()), "n_positive": int((Ys[lab] == 1).sum()),
                "prevalence": suppress_proportion(int((Ys[lab] == 1).sum()), int(Ys[lab].notna().sum())),
                "per_site": {site_labels[s]: {"n_positive": pos[s], "n_negative": neg[s],
                                              "prevalence": suppress_proportion(pos[s], pos[s] + neg[s])} for s in uniq}}
        else:
            excluded[lab] = f"fewer than {MIN_SITE_CELL} positives or negatives at a site (not analysed)"
            label_report[lab] = {"n_assessable": SUPPRESSED, "n_positive": SUPPRESSED, "prevalence": SUPPRESSED}
    e7_pos = {s: int(((Ys["E7"] == 1) & (ssites == s)).sum()) for s in uniq} if "E7" in Ys.columns else {}
    e7_primary = bool("E7" in included and e7_eligible(e7_pos))
    if "E7" in included:
        label_report["E7"]["role"] = ("primary (>= 100 positives over >= 2 sites)" if e7_primary
                                      else "exploratory (fewer than 100 positives or fewer than 2 sites)")
    primary = tuple(l for l in included if l in ("E1", "E2", "E4a", "E5", "E6")) + (("E7",) if e7_primary else ())
    if not primary:
        raise SystemExit("no primary label is analysable (>= 11 positives and negatives at every site)")
    for l in excluded:
        label_report.setdefault(l, {})["excluded_reason"] = excluded[l]
    for l, why in EXCLUDED_BY_RULE.items():
        label_report[l] = {"excluded_reason": why}
    names = tuple(included)
    y = np.nan_to_num(Ys[list(names)].to_numpy(float), nan=0.0)
    m = Ys[list(names)].notna().to_numpy(bool)
    anyl = m.any(axis=1)
    steps.append(("with >= 1 assessable analysed label", np.isin(np.arange(len(cohort)), np.flatnonzero(keep)[anyl])))

    # ---- rows with no assessable label contribute nothing: dropped
    idx = np.flatnonzero(anyl)
    sel, sel_bl, sel_eeg, y, m = sel.iloc[idx].reset_index(drop=True), sel_bl.iloc[idx].reset_index(drop=True), \
        sel_eeg.iloc[idx].reset_index(drop=True), y[idx], m[idx]
    sel_emb = sel_emb.iloc[idx].reset_index(drop=True)
    sel_mg = sel_mg.iloc[idx].reset_index(drop=True)
    Hs = Hs.iloc[idx].reset_index(drop=True) if len(Hs.columns) else pd.DataFrame(index=range(len(sel)))
    ssites = sel["SiteID"].astype(str)                       # re-derived: the row filter above must reach the site vector too
    flow = {"steps": [{"step": s, "n_total": suppress_count(int(msk.sum())),
                       "per_site": {site_labels.get(k, k): suppress_count(v) for k, v in _site_counts(sites_s, msk).items()}}
                      for s, msk in steps],
            "eeg_extraction": {"n_cohort_with_primary_row": suppress_count(int(has_eeg.sum())),
                               "n_primary_rows_read": suppress_count(eeg_stats["n_primary_rows_read"])}}

    # ---- design matrices and covariates
    all_c = list(dict.fromkeys(bl_cols["C"] + bl_cols.get("P", [])))
    baseline = sel_bl[all_c].astype(float)
    bsets = {"A": list(bl_cols["A"]), "C": list(bl_cols["C"])}
    iu_sets = {"prior": []}
    if bl_cols.get("AGE_SEX"):
        iu_sets["agesex"] = list(bl_cols["AGE_SEX"])
    if bl_cols.get("P"):
        iu_sets["P"] = list(bl_cols["P"])
    flag, parts = undifferentiated_flag(sel, sel_bl, a.undiff_max_hours)
    sub_sites = ssites.to_numpy(object)
    subgroup_info = {
        "definition": {"max_hours_from_visit_start": a.undiff_max_hours, "label_family_dx_before_t0": "none",
                       "sedative_opioid_window_hours": UNDIFF_SEDATION_H, "sources": parts["sources"]},
        "n": suppress_count(int(flag.sum())), "share_of_analysed": suppress_proportion(int(flag.sum()), len(flag)),
        "per_site": {site_labels[s_]: suppress_count(int((flag & (sub_sites == s_)).sum())) for s_ in uniq},
        "components_met": {"visit_day_0_1": suppress_count(int(parts["early"].sum())),
                           "no_label_family_dx_before_t0": suppress_count(int(parts["no_dx"].sum())),
                           "no_sedative_opioid_6h": suppress_count(int(parts["no_sedation"].sum()))}}
    eeg_df = sel_eeg[feat_cols].astype(float)
    embedding_info: dict = {}
    if emb_cols:
        has_emb = sel_emb.notna().any(axis=1).to_numpy(bool)
        if has_emb.sum() >= MIN_SITE_CELL:               # rows without an embedding stay NaN (fold-local imputation + indicator)
            eeg_df = pd.concat([eeg_df, sel_emb.astype(float)], axis=1)
            embedding_info = {"family": "cbramod_frozen", "dim": len(emb_cols),
                              "n_recordings_with_embedding_in_cohort": suppress_count(emb_stats["n_emb_ok"]),
                              "n_analysed_with_embedding": suppress_count(int(has_emb.sum())),
                              "share_analysed_with_embedding": suppress_proportion(int(has_emb.sum()), len(sel_emb)),
                              "per_site_with_embedding": {site_labels.get(s_, s_): suppress_count(int((has_emb & (ssites.to_numpy() == s_)).sum()))
                                                          for s_ in uniq}}
    morgoth_info: dict = {}
    morgoth_exposure: dict = {}
    if mg_cols:
        has_mg = sel_mg.notna().any(axis=1).to_numpy(bool)
        if has_mg.sum() >= MIN_SITE_CELL:                # rows without MORGOTH output stay NaN (fold-local imputation + indicator)
            eeg_df = pd.concat([eeg_df, sel_mg.astype(float)], axis=1)
            morgoth_info = {"family": "morgoth_findings", "dim": len(mg_cols),
                            "n_recordings_with_findings_in_cohort": suppress_count(mg_stats["n_morgoth_ok"]),
                            "n_analysed_with_findings": suppress_count(int(has_mg.sum())),
                            "share_analysed_with_findings": suppress_proportion(int(has_mg.sum()), len(sel_mg)),
                            "per_site_with_findings": {site_labels.get(s_, s_): suppress_count(int((has_mg & (ssites.to_numpy() == s_)).sum()))
                                                       for s_ in uniq}}
            expo_path = getattr(a, "morgoth_exposure", None) or Path(getattr(a, "morgoth", None) or Path(a.features).parent / "morgoth") / "exposure.parquet"
            ex = load_morgoth_exposure(expo_path)
            if ex is not None:
                flags, basis = ex
                morgoth_exposure = {"flag": sel["person_id"].map(flags).to_numpy(float), "basis": basis}
    qmiss = pd.to_numeric(sel_eeg.get("qc_n_missing_or_dead_min"), errors="coerce") if "qc_n_missing_or_dead_min" in sel_eeg \
        else pd.Series(np.nan, index=sel.index)
    gcs_eq = gcs_equivalent(sel.get("gcs_nearest_window", pd.Series(np.nan, index=sel.index)),
                            sel.get("four_nearest_window", pd.Series(np.nan, index=sel.index)))
    cov = {"duration_s": pd.to_numeric(sel.get("duration_s"), errors="coerce").to_numpy(float) if "duration_s" in sel
           else np.full(len(sel), np.nan),
           "n_channels": (N_MINIMUM_CHANNELS - qmiss).to_numpy(float),
           "severity": (15.0 - gcs_eq).to_numpy(float),
           "sedated": sedation_flag(sel_bl, Hs, a.sedation_source)}
    return Assembly(sel, baseline, eeg_df, y, m, names, primary, e7_primary, bsets, cov, ssites.to_numpy(object),
                    sel["t0"].to_numpy("datetime64[us]"), flow, label_report, site_labels, iu_sets, flag, subgroup_info,
                    baseline_scope(a.baselines), embedding_info, morgoth_info, morgoth_exposure)


# ===================================================================================== model data per scheme
def build_model_data(A: Assembly, scheme: str, dev_frac: float, test_fraction: float, seed: int,
                     eval_mask: np.ndarray | None = None) -> ModelData:
    """ModelData with the SILVER labels in both roles. Dev rows (tune + recalibrate inside training folds) are drawn at random
    within site (LOSO) or from the training region of the temporal split; every other row is 'eval' and is scored only when
    it is a test row of the fold. ``eval_mask`` (bool, n) restricts scoring to a subgroup: other non-dev rows get role 'none',
    so they still train the head (silver labels) but are never scored; the dev draw is unchanged."""
    n = len(A.frame)
    rng = np.random.default_rng(seed)
    role = np.full(n, "eval", dtype=object)
    if scheme == "loso":
        pool = {s: np.flatnonzero(A.sites == s) for s in np.unique(A.sites)}
    elif scheme == "temporal":
        ts = late_temporal_holdout(A.sites, A.times, test_fraction)
        pool = {s: np.intersect1d(np.flatnonzero(A.sites == s), ts.train_idx) for s in np.unique(A.sites)}
    else:
        raise ValueError(scheme)
    for ix in pool.values():
        k = int(round(dev_frac * len(ix)))
        role[rng.choice(ix, size=k, replace=False)] = "dev"
    if eval_mask is not None:
        role[(role == "eval") & ~np.asarray(eval_mask, bool)] = "none"
    return ModelData(A.baseline, A.eeg, A.y, A.m, A.y.copy(), A.m.copy(), role, A.sites, A.label_names,
                     A.times, dict(A.covariates))


# ============================================================================== prediction capture and metrics
class PredictionTap:
    """Records every fold's held-out predictions made inside ``run_ladder`` (the ladder keeps only d_i). It wraps
    ``ladder.fit_predict_fold`` for the duration of a ``with`` block and changes nothing about what is fitted."""

    def __init__(self, data: ModelData, baseline_sets: dict):
        self.n, self.K = data.n, data.K
        self.names = {tuple(c): b for b, c in baseline_sets.items()}
        self.P: dict[str, dict[str, np.ndarray]] = {b: {} for b in baseline_sets}
        self.tested = {b: np.zeros(self.n, bool) for b in baseline_sets}
        self._orig = None

    def _wrap(self, data, train_idx, test_idx, baseline_cols, variants, cfg, **kw):
        out = self._orig(data, train_idx, test_idx, baseline_cols, variants, cfg, **kw)
        b = self.names[tuple(baseline_cols)]
        for k, v in out[0].items():
            self.P[b].setdefault(k, np.full((self.n, self.K), np.nan))[np.asarray(test_idx)] = v
        self.tested[b][np.asarray(test_idx)] = True
        return out

    def __enter__(self):
        self._orig = ladder_mod.fit_predict_fold
        self._patch = mock.patch.object(ladder_mod, "fit_predict_fold", self._wrap)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def auc_fast(y, p) -> float:
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _auc_cell(y, p):
    y = np.asarray(y, bool)
    if min(int(y.sum()), int((~y).sum())) < MIN_SITE_CELL:
        return SUPPRESSED
    return round(auc_fast(y, p), 4)


def per_label_auroc(md: ModelData, ev: np.ndarray, P: np.ndarray, site_labels: dict) -> dict:
    out = {}
    for k, lab in enumerate(md.label_names):
        r = ev & md.m_gold[:, k]
        row = {"pooled": _auc_cell(md.y_gold[r, k], P[r, k]), "per_site": {}}
        for s, lbl in site_labels.items():
            rs = r & (md.sites.astype(str) == s)
            row["per_site"][lbl] = _auc_cell(md.y_gold[rs, k], P[rs, k]) if rs.any() else SUPPRESSED
        out[lab] = row
    return out


def delta_auroc(md: ModelData, ev: np.ndarray, P1: np.ndarray, P0: np.ndarray, n_boot: int, seed: int,
                site_labels: dict) -> dict:
    """AUROC(model with EEG) - AUROC(baseline only) per label, paired within-site bootstrap 95% interval."""
    out = {}
    for k, lab in enumerate(md.label_names):
        r = np.flatnonzero(ev & md.m_gold[:, k])
        y = md.y_gold[r, k].astype(bool)
        if min(int(y.sum()), int((~y).sum())) < MIN_SITE_CELL:
            out[lab] = {"delta_auroc": SUPPRESSED}
            continue
        p1, p0, s = P1[r, k], P0[r, k], md.sites[r]
        est = auc_fast(y, p1) - auc_fast(y, p0)
        ci = bootstrap_ci(lambda idx: auc_fast(y[idx], p1[idx]) - auc_fast(y[idx], p0[idx]), est, s, "within_site",
                          n_boot, seed, 0.05)
        out[lab] = {"delta_auroc": round(est, 4), "lo": round(ci.lo, 4), "hi": round(ci.hi, 4)}
    return out


def calibration_block(md: ModelData, ev: np.ndarray, P: np.ndarray) -> dict:
    rep = per_label_report(md.y_gold[ev], P[ev], md.m_gold[ev], md.label_names)
    out = {}
    for lab, r in rep.items():
        big = r["n_events"] >= MIN_EVENTS_FOR_SLOPE
        out[lab] = {"n": suppress_count(r["n"]), "n_events": suppress_count(r["n_events"]),
                    "observed_over_expected": r["oe"], "calibration_in_the_large": r["citl"],
                    "slope": r["slope"] if big else None, "slope_lo": r["slope_lo"] if big else None,
                    "slope_hi": r["slope_hi"] if big else None, "ece": r["ece"], "brier": r["brier"],
                    "note": "slope and its interval reported (>= 100 events)" if big else "O/E and CITL only (< 100 events, D-094)"}
    return out


def discrimination_calibration(md, res, tap, bname, headline, site_labels, n_boot, seed) -> dict:
    ev = (md.gold_role == "eval") & tap.tested[bname]
    P = tap.P[bname]
    models = {"baseline_only": "__baseline__", **{r: r for r in RUNG_NAMES if r in P and r != "prior"}}
    out = {"auroc": {nm: per_label_auroc(md, ev, P[v], site_labels) for nm, v in models.items()},
           "calibration": {"baseline_only": calibration_block(md, ev, P["__baseline__"])}}
    if headline in P:
        out["calibration"][headline] = calibration_block(md, ev, P[headline])
        out["delta_auroc_vs_baseline"] = {headline: delta_auroc(md, ev, P[headline], P["__baseline__"], n_boot, seed,
                                                                site_labels)}
    return out


# ============================================================================================== controls
def severity_strata(res, md, bname, rung, n_boot, seed, min_n) -> dict:
    r = res.get(rung, bname)
    sev = np.asarray(md.covariates["severity"], float)[res.eval_idx]
    known = ~np.isnan(sev)
    names = np.where(~known, "unknown", np.array(["low", "mid", "high"])[np.digitize(np.where(known, sev, 0.0), SEVERITY_CUTS)])
    d = np.where(known, r.d, np.nan)
    h3 = h3_severity_stratified(d, names, res.eval_sites, n_boot=n_boot, seed=seed, min_stratum_n=min_n)
    return {"cut_points_prespecified": True, "cut_points_severity_15_minus_gcs_equivalent": list(SEVERITY_CUTS),
            "n_with_unknown_severity": suppress_count(int((~known).sum())),
            "strata_with_delta_below_zero": h3.n_favorable, "n_strata": h3.n_strata,
            "descriptive_pattern_met": h3.met,
            "strata": {k: {**v, "n": suppress_count(v["n"])} for k, v in h3.strata.items()}}


def _rung_summary(rr) -> dict:
    return {"delta": rr.delta, "ci_99_within_site": {k: rr.ci_primary.get(k) for k in ("lo", "hi")},
            "ci_95_within_site": {k: rr.ci_modes.get("within_site", {}).get(k) for k in ("lo", "hi")},
            "all_sites_favorable": rr.all_sites_favorable}


def run_controls(md, res, split, bname, bcols, headline, cfg, probes, null_reps, seed, site_labels, n_boot_small) -> dict:
    rr = res.get(headline, bname)
    spec = [r for r in BASE_RUNGS if r.name == headline]
    cfg_s = replace(cfg, per_label=False, n_boot=n_boot_small)
    out = {"site_concentration": delta_concentration_by_site(rr, site_labels),
           "site_probe_control_failed": control_failed(probes, rr, site_labels),
           "severity_strata": severity_strata(res, md, bname, headline, n_boot_small, seed, 50)}
    try:
        sed = sedative_excluded_rerun(md, md.covariates["sedated"], spec, cfg_s, {bname: bcols}, split)
        sr = sed["result"].get(headline, bname)
        out["sedative_excluded"] = {"n_excluded": sed["n_excluded"], "n_kept": sed["n_kept"], **_rung_summary(sr),
                                    "pooled_delta_negative": bool(sr.delta < 0)}
    except ValueError as e:                       # a site left with too few rows for the scheme
        out["sedative_excluded"] = {"not_estimable": type(e).__name__}
    nl = []
    for r in range(null_reps):
        try:
            nc = negative_control(md, "labels", spec, cfg_s, {bname: bcols}, split, seed + 1 + r)
            nl.append(next(iter(nc["rungs"].values())))
        except (ValueError, StopIteration):
            continue
    out["negative_control_shuffled_labels"] = {
        "n_reps": len(nl), "deltas": [round(x["delta"], 5) for x in nl],
        "mean_delta": float(np.mean([x["delta"] for x in nl])) if nl else None,
        "n_reps_with_spurious_gain_95": int(sum(x["spurious_gain"] for x in nl)),
        "note": "labels permuted within site x role; features and baseline lose their relation to the labels; Delta ~ 0 expected"}
    try:
        ne = negative_control(md, "eeg", spec, cfg_s, {bname: bcols}, split, seed + 101)
        e = next(iter(ne["rungs"].values()))
        out["negative_control_permuted_eeg"] = {"delta": round(e["delta"], 5), "lo_95": e["lo"], "hi_95": e["hi"],
                                                "spurious_gain_95": e["spurious_gain"]}
    except (ValueError, StopIteration):
        out["negative_control_permuted_eeg"] = {"not_estimable": True}
    return out


# ============================================================================================ circularity
def circularity_block(A: Assembly, md: ModelData, taps_loso: PredictionTap, rf: pd.DataFrame, headline: str,
                      n_boot: int, seed: int) -> dict:
    """Agreement (AUROC) of held-out silver-trained predictions with the EEG-report findings vs with the silver labels
    (``labels.circularity_audit``, rule: leak if agreement(EEG report) > agreement(label source)). Predictions are
    leave-one-site-out, so every case is scored by a model that never trained on its site."""
    coh = A.frame[["person_id", "t0", "SiteID", "SessionID"]].copy()
    comp = eeg_impression_comparator(rf, coh)
    out = {"comparator": "reports_findings flags -> label mapping in configs/anchor_concepts.yaml (PROVISIONAL, needs EEG-clinician sign-off)",
           "metric": "AUROC of the held-out prediction against each reference", "per_model": {}}
    if comp.empty:
        out["note"] = "no label has a mapped EEG-report flag"
        return out
    cid = A.frame["person_id"].astype(str).to_numpy()
    k_of = {l: i for i, l in enumerate(A.label_names)}
    cov = {}
    for lab in sorted(set(comp["label"]) & set(A.label_names)):
        e = comp[comp["label"] == lab].set_index("case_id")["eeg_impr"].reindex(cid).to_numpy(float)
        cov[lab] = {"n_cases_with_matching_report": suppress_count(int(np.isfinite(e).sum())),
                    "share_of_cases": suppress_proportion(int(np.isfinite(e).sum()), len(cid))}
    out["report_coverage"] = cov
    for b in A.baseline_sets:
        tested = taps_loso.tested[b]
        for nm, var in (("baseline_only", "__baseline__"), (headline, headline)):
            P = taps_loso.P[b].get(var)
            if P is None:
                continue
            rows = []
            for lab in cov:
                k = k_of[lab]
                e = comp[comp["label"] == lab].set_index("case_id")["eeg_impr"].reindex(cid).to_numpy(float)
                gold = np.where(A.m[:, k], A.y[:, k], np.nan)
                ok = tested & np.isfinite(e) & np.isfinite(gold) & np.isfinite(P[:, k])
                rows.append(pd.DataFrame({"case_id": cid[ok], "label": lab, "pred": P[ok, k], "eeg_impr": e[ok],
                                          "gold": gold[ok]}))
            df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
                {"case_id": [], "label": [], "pred": [], "eeg_impr": [], "gold": []})
            aud = run_circularity_audit(df, list(cov), metric="auroc", n_boot=n_boot, seed=seed)
            out["per_model"][f"{b}/{nm}"] = {"per_label": clean(aud["per_label"]), "labels_flagged": aud["leaking_labels"]}
    out["reading"] = ("agreement_gold below is agreement with the SILVER label (no gold exists). A flagged label means held-out "
                      "predictions agree more with the EEG report than with the silver label. The baseline-only model never sees "
                      "EEG, so a flag there points at silver-label contamination; a flag only for the EEG model is expected "
                      "if it reads the same waveform findings the report describes and is not by itself a leak.")
    return out


# ============================================================================== intended use (D-145)
def _iu_rung(res, bname: str, rung: str) -> dict:
    """Aggregate Delta block of one comparison (headline rung vs its baseline set); counts suppressed by ``rung_to_aggregate``."""
    from sortinghat.models.report import rung_to_aggregate
    rr = res.get(rung, bname)
    return clean({**rung_to_aggregate(rr, res.site_labels), "n_eval": suppress_count(res.n_eval),
                  "loss_reference": rr.loss_baseline, "loss_with_eeg": rr.loss_model})


def subgroup_estimable(A: Assembly) -> str | None:
    """None when Delta can be estimated in the subgroup, else the reason (sizes are never printed unsuppressed)."""
    if A.subgroup is None or not A.subgroup.any():
        return "the undifferentiated subgroup is empty (a component could not be established or none met it)"
    if int(A.subgroup.sum()) < IU_MIN_SUBGROUP:
        return f"fewer than {IU_MIN_SUBGROUP} patients in the subgroup"
    small = [A.site_labels[s] for s in np.unique(A.sites) if int((A.subgroup & (A.sites == s)).sum()) < MIN_SITE_CELL]
    return f"fewer than {MIN_SITE_CELL} subgroup patients at {', '.join(small)}" if small else None


def run_intended_use(A: Assembly, a, cfg: LadderConfig, headline: str, splits: list[str]) -> dict:
    """The "Intended use: undifferentiated AMS" block: three comparisons, each Delta = masked log loss(set + EEG) - masked log
    loss(set) on the SAME evaluation rows, under both validation schemes, for the whole analysed set and for the
    undifferentiated subgroup (trained on all training-fold rows, scored on subgroup rows only). Headline EEG rung only.

      a. 'prior'  : EEG-only vs the prevalence prior (no baseline columns; the reference is the smoothed training prevalence)
      b. 'agesex' : EEG + age/sex vs age/sex
      c. 'P'      : EEG + Baseline P (Presentation) vs P

    EEG-derived information is never an input or label evidence: the sets hold no EEG-derived column, and the silver labels are
    EEG-blind (E3 and the EEG-report flags are excluded; reports_findings feeds only the circularity audit)."""
    spec = [r for r in BASE_RUNGS if r.name == headline]
    sets = {k: A.iu_sets[k] for k, _ in IU_COMPARISONS if k in A.iu_sets}
    out: dict = {"title": IU_TITLE, "question": "Given an unknown EEG recorded in the ED or early in an admission, with little or no "
                 "history known, what is the cause (E1 structural, E2 hypoxic-ischemic, E4a toxic-antidote, E5 metabolic, E6 "
                 "infectious-septic, E7 autoimmune)?",
                 "baseline_scope": A.baseline_scope, "headline_rung": headline,
                 "comparisons": {k: lab for k, lab in IU_COMPARISONS if k in sets},
                 "comparisons_unavailable": {lab: ("no Baseline P columns in the baselines file (rebuild with build_baselines.py)"
                                                   if k == "P" else "no age/sex columns in the baselines file")
                                             for k, lab in IU_COMPARISONS if k not in sets},
                 "eeg_derived_inputs": "No baseline column and no label is derived from EEG (silver labels are EEG-blind; E3 and EEG-report "
                                       "flags are excluded; reports_findings is used only in the circularity audit)",
                 "subgroup": A.subgroup_info, "populations": {}, "negative_controls": {}}
    not_est = subgroup_estimable(A)
    for split in splits:
        out["populations"][split] = {}
        for pop in ("all_analysed", "undifferentiated"):
            if pop == "undifferentiated" and not_est:
                out["populations"][split][pop] = {"not_estimable": not_est}
                continue
            md = build_model_data(A, split, a.dev_frac, a.temporal_fraction, a.seed,
                                  eval_mask=A.subgroup if pop == "undifferentiated" else None)
            try:
                res = run_ladder(md, sets, spec, cfg, split)
                if pop == "undifferentiated":                          # same rule as the severity strata: < 50 rows, or < 11 at a site
                    small = [lab for s_, lab in res.site_labels.items() if int((res.eval_sites.astype(str) == s_).sum()) < MIN_SITE_CELL]
                    if res.n_eval < IU_MIN_SUBGROUP or small:
                        out["populations"][split][pop] = {"not_estimable": f"fewer than {IU_MIN_SUBGROUP} subgroup rows scored in "
                                                          f"this scheme, or fewer than {MIN_SITE_CELL} at a site"}
                        continue
                out["populations"][split][pop] = {k: _iu_rung(res, k, headline) for k in sets}
            except (ValueError, ZeroDivisionError) as e:               # too few rows at a site for the scheme
                out["populations"][split][pop] = {"not_estimable": type(e).__name__}
    if not a.skip_controls:
        n_small = max(200, min(a.n_boot, 1000))
        cfg_s = replace(cfg, per_label=False, n_boot=n_small)
        for split in splits:
            md = build_model_data(A, split, a.dev_frac, a.temporal_fraction, a.seed)
            out["negative_controls"][split] = {}
            for k, cols in sets.items():
                nl, ne = [], None
                for r in range(a.null_reps):
                    try:
                        nl.append(next(iter(negative_control(md, "labels", spec, cfg_s, {k: cols}, split, a.seed + 1 + r)["rungs"].values())))
                    except (ValueError, StopIteration):
                        continue
                try:
                    ne = next(iter(negative_control(md, "eeg", spec, cfg_s, {k: cols}, split, a.seed + 101)["rungs"].values()))
                except (ValueError, StopIteration):
                    pass
                out["negative_controls"][split][k] = {
                    "shuffled_labels_mean_delta": float(np.mean([x["delta"] for x in nl])) if nl else None,
                    "shuffled_labels_reps_with_spurious_gain_95": int(sum(x["spurious_gain"] for x in nl)),
                    "shuffled_labels_reps": len(nl),
                    "permuted_eeg_delta": None if ne is None else round(ne["delta"], 5),
                    "permuted_eeg_spurious_gain_95": None if ne is None else bool(ne["spurious_gain"])}
    else:
        out["negative_controls"] = {"skipped": "--skip-controls was set: the negative controls of this block were NOT run"}
    return clean(out)


# ============================================================================ commercial-clean gap
GAP_NOTE = ("Gap = Delta(cbramod_frozen + baseline) - Delta(morgoth_findings + baseline) on the same evaluation rows (per-patient "
            "differences d_i, paired patient-level bootstrap within site). Positive = the frozen public CBraMod embedding helps LESS "
            "than the MORGOTH findings. Reported, not tested. MORGOTH is the commercial-clean-lineage comparator; CBraMod is public "
            "research weights.")
EXPOSURE_NOTE = ("MORGOTH exposure accounting: patients whose EEG is in MORGOTH's own data lists (sortinghat.morgoth.exposure) may have "
                 "been seen by the model that produced the findings, which could flatter the MORGOTH rung. The sensitivity row drops "
                 "the flagged patients from the SCORED rows (models are not refitted) and re-estimates both Deltas and the gap.")


def _gap_stats(da: np.ndarray, db: np.ndarray, sites: np.ndarray, n_boot: int, seed: int, alpha99: float, alpha95: float) -> dict:
    """Mean d_a, mean d_b and mean (d_a - d_b) over the rows where both are defined, with the paired within-site bootstrap interval
    of the gap at both levels. Fewer than MIN_SITE_CELL rows: every number is suppressed."""
    ok = ~(np.isnan(da) | np.isnan(db))
    n = int(ok.sum())
    out = {"n": suppress_count(n)}
    if n < MIN_SITE_CELL:
        return {**out, "delta_a": SUPPRESSED, "delta_b": SUPPRESSED, "gap": SUPPRESSED, "ci_99_within_site": SUPPRESSED,
                "ci_95_within_site": SUPPRESSED}
    da, db, sites = da[ok], db[ok], np.asarray(sites)[ok]
    diff = da - db
    out.update(delta_a=float(da.mean()), delta_b=float(db.mean()), gap=float(diff.mean()))
    for key, al in (("ci_99_within_site", alpha99), ("ci_95_within_site", alpha95)):
        try:
            ci = paired_bootstrap_delta(diff, sites, "within_site", n_boot, seed, al)
            out[key] = {"lo": float(ci.lo), "hi": float(ci.hi)}
        except ValueError:
            out[key] = None
    return out


def _gap_by_site(da, db, sites, site_labels, n_boot, seed, alpha99, alpha95) -> dict:
    return {site_labels.get(str(s_), str(s_)): _gap_stats(da[sites == s_], db[sites == s_], sites[sites == s_], n_boot, seed, alpha99, alpha95)
            for s_ in sorted(np.unique(sites))}


def commercial_gap_block(res, A: Assembly, cfg: LadderConfig, baseline: str) -> dict:
    """Commercial-clean gap for one scheme x baseline set; ``{"available": False}`` unless both rungs were fitted."""
    ra, rb = res.get(GAP_RUNGS[0], baseline), res.get(GAP_RUNGS[1], baseline)
    if not (ra.available and rb.available):
        return {"available": False}
    da, db = np.asarray(ra.d, float), np.asarray(rb.d, float)
    sites = np.asarray(res.eval_sites).astype(str)
    args = (cfg.n_boot, cfg.seed, cfg.alpha_primary, cfg.alpha_secondary)
    out = {"available": True, "pooled": _gap_stats(da, db, sites, *args),
           "per_site": _gap_by_site(da, db, sites, res.site_labels, *args)}
    ex = A.morgoth_exposure
    if not ex:
        out["exposure_sensitivity"] = {"not_run": "no MORGOTH exposure flag file found (--morgoth-exposure; written by "
                                                  "scripts/extract_morgoth.py exposure)"}
        return out
    flag = np.asarray(ex["flag"], float)[np.asarray(res.eval_idx)]
    excl = flag == 1.0
    keep = ~excl
    n_exc = int(excl.sum())
    out["exposure_sensitivity"] = {
        "basis": ex["basis"], "n_excluded": suppress_count(n_exc),
        "n_without_flag_kept": suppress_count(int(np.isnan(flag).sum())),
        "n_kept": suppress_count(int(keep.sum())) if n_exc >= MIN_SITE_CELL else SUPPRESSED,   # no complement of a small excluded cell
        "pooled": _gap_stats(da[keep], db[keep], sites[keep], *args),
        "per_site": _gap_by_site(da[keep], db[keep], sites[keep], res.site_labels, *args)}
    return out


# ==================================================================================== sedation confounding
SEDATION_READING = ("If the EEG gain were merely a sedation proxy (EEG detects the sedative effect and sedation correlates with the "
                    "silver labels), Delta would be about 0 within the non-sedated stratum. A non-sedated Delta whose 99% interval "
                    "lies below 0 argues against that; an interval that includes 0 is inconclusive, not exculpatory, because that "
                    "stratum has fewer patients than the whole set. A clearly more negative Delta in the sedated stratum (the "
                    "difference of Deltas) is the other signature of a sedation contribution.")
SEDATION_CAVEATS = ("The flag is exposure recorded at t0 (Baseline A on_t0 flags and/or the silver e4b sedative hint, "
                    "--sedation-source); sedation that left no recorded administration, a bolus that ended before t0 and the depth "
                    "of sedation are invisible to it, so the non-sedated stratum can still hold sedated patients. Models are the "
                    "ones fitted on all training rows; the strata only split the held-out rows, so this differs from the "
                    "sedative-excluded rerun (which also removes sedated patients from training and keeps few patients).")
SEDATION_INTERACTION = {"run": False, "reason": "not run; it needs a refitted model grid for every scheme x baseline x fold (the "
                                                "stratified Delta answers the same question from the stored predictions)"}


def baseline_sedation_columns(baseline_sets: dict) -> dict:
    """Which baseline sets carry sedation features (column counts only): Baseline A is 'severity + sedation' and C contains A."""
    out = {}
    for b, cols in baseline_sets.items():
        sc = [c for c in cols if str(c).startswith("sed__")]
        out[b] = {"sedation_columns": len(sc),
                  "has_on_t0_class_flags": all(c in sc for c in ("sed__sedative__on_t0", "sed__opioid__on_t0")),
                  "has_dose_or_count_columns": any(("__qty_" in c) or ("__n_" in c) for c in sc)}
    return out


def sedation_block(A: Assembly, results: dict, cfg: LadderConfig, headline: str, source: str = "either") -> dict:
    """Sedation confounding (aggregate-only): (1) headline-rung Delta within sedated and non-sedated strata for every scheme x
    baseline, from the SAME held-out per-patient d_i (no refit), with within-site bootstrap CIs and the difference of Deltas;
    (2) whether Baselines A / C already contain sedation features, i.e. whether Delta is already conditional on recorded sedation.
    The sedation leakage probe is in ``controls.leakage_probes``."""
    sed = np.asarray(A.covariates["sedated"], bool)
    n, n_sed = len(sed), int(sed.sum())
    seen = sedation_flag(A.baseline, pd.DataFrame(), "baseline")          # the part of the flag the baseline's own class flags see
    bl = baseline_sedation_columns(A.baseline_sets)
    out: dict = {
        "headline_rung": headline, "flag_source": source,
        "flag": {"n_sedated": suppress_count(n_sed), "prevalence": suppress_proportion(n_sed, n),
                 "per_site": {A.site_labels[s_]: {"n_sedated": suppress_count(int((sed & (A.sites == s_)).sum())),
                                                  "prevalence": suppress_proportion(int((sed & (A.sites == s_)).sum()),
                                                                                    int((A.sites == s_).sum()))}
                              for s_ in sorted(np.unique(A.sites))},
                 "n_flagged_only_by_silver_hint": suppress_count(int((sed & ~seen).sum()))},
        "baseline_sedation_features": bl,
        "baseline_adjustment": ("Baselines " + " and ".join(b for b, v in bl.items() if v["sedation_columns"]) +
                                " already contain sedation features (Baseline A = severity + sedation; C contains A), so Delta is "
                                "already the EEG increment over recorded t0 sedation exposure"
                                if any(v["sedation_columns"] for v in bl.values()) else
                                "NO baseline set contains sedation features in this run: Delta is not adjusted for recorded sedation")
                               + ". What remains is residual confounding: sedation the recorded features miss (see caveats).",
        "sedation_interaction_variant": SEDATION_INTERACTION,
        "min_stratum_n": SED_MIN_STRATUM_N, "reading": SEDATION_READING, "caveats": SEDATION_CAVEATS, "by_scheme": {}}
    for split, res in results.items():
        flag = sed[np.asarray(res.eval_idx)]
        out["by_scheme"][split] = {}
        for b in A.baseline_sets:
            rr = res.get(headline, b)
            out["by_scheme"][split][b] = stratified_delta(
                rr.d, flag, res.eval_sites, res.site_labels, cfg.n_boot, cfg.seed, cfg.alpha_primary, cfg.alpha_secondary)
    return clean(out)


# ==================================================================================================== driver
def run_analysis(A: Assembly, a, store=None) -> dict:
    cfg = LadderConfig(n_boot=a.n_boot, seed=a.seed, include_e7=A.include_e7, temporal_test_fraction=a.temporal_fraction)
    rungs = list(BASE_RUNGS)
    has_emb = any(str(c).startswith(CBRAMOD_PREFIX) for c in A.eeg.columns)
    has_mg = any(str(c).startswith(MORGOTH_PREFIX) for c in A.eeg.columns)
    if has_emb:
        rungs += [CBRAMOD_FROZEN_RUNG, next(r for r in COMMERCIAL_RUNGS if r.name == "combined_cbramod")]
    if has_mg:
        rungs += [MORGOTH_FINDINGS_RUNG, next(r for r in COMMERCIAL_RUNGS if r.name == "combined_morgoth")]
    n_sites = len(np.unique(A.sites))
    splits = [s for s in a.splits if (s == "temporal" or n_sites >= 2)]
    avail_rungs = [r for r in RUNG_NAMES if r != "prior" and any(c.startswith(p) for c in A.eeg.columns
                                                                for p in next(x.prefixes for x in rungs if x.name == r))]
    eeg_only_cols = [c for c in A.eeg.columns if str(c).startswith(EEG_PREFIXES)]
    headline = "combined" if "combined" in avail_rungs else (avail_rungs[0] if avail_rungs else None)
    if headline is None:
        raise SystemExit("no qeeg./conn. feature columns found")
    R: dict = {"banner": BANNER, "headline_rung": headline, "intended_use": None, "ladder": {}, "discrimination_calibration": {},
               "commercial_clean_gap": {"available": False}, "sedation": None, "controls": {}, "circularity_audit": None}
    R["intended_use"] = run_intended_use(A, a, cfg, headline, splits)             # D-145: reported FIRST
    taps: dict = {}
    mds: dict = {}
    results: dict = {}
    for split in splits:
        md = build_model_data(A, split, a.dev_frac, a.temporal_fraction, a.seed)
        with PredictionTap(md, A.baseline_sets) as tap:
            res = run_ladder(md, A.baseline_sets, rungs, cfg, split)
        mds[split], taps[split], results[split] = md, tap, res
        R["ladder"][split] = res.to_aggregate()
        R["discrimination_calibration"][split] = {
            b: discrimination_calibration(md, res, tap, b, headline, res.site_labels, min(a.n_boot, a.auc_boot), a.seed)
            for b in A.baseline_sets}
    if has_emb and has_mg:
        R["commercial_clean_gap"] = {"available": True, "rungs": list(GAP_RUNGS), "note": GAP_NOTE, "exposure_note": EXPOSURE_NOTE,
                                     "by_scheme": {sp: {b: commercial_gap_block(results[sp], A, cfg, b) for b in A.baseline_sets}
                                                   for sp in splits}}
    else:
        R["commercial_clean_gap"] = {"available": False, "reason": "needs both cbramod_frozen and morgoth_findings to have run"}
    R["anchor_overlap_suspected"] = {
        sp: {b: [lab for lab, v in R["discrimination_calibration"][sp][b]["auroc"]["baseline_only"].items()
                 if isinstance(v["pooled"], float) and v["pooled"] >= ANCHOR_OVERLAP_AUROC] for b in A.baseline_sets}
        for sp in splits}
    R["sedation"] = sedation_block(A, results, cfg, headline, a.sedation_source)       # reads the held-out d_i only: no refit, cheap, runs even with --skip-controls
    if a.skip_controls:
        R["controls"] = {"skipped": "--skip-controls was set: the mandatory controls were NOT run; do not interpret Delta"}
        R["circularity_audit"] = {"note": "skipped with --skip-controls"}
        return R
    first = mds[splits[0]]
    dur = A.covariates["duration_s"] if np.isfinite(A.covariates["duration_s"]).any() else None
    nch = A.covariates["n_channels"] if np.isfinite(A.covariates["n_channels"]).any() else None
    sed = A.covariates["sedated"]                    # the sedation probe: how much sedation the representation alone encodes
    probes = leakage_probes(first.eeg[eeg_only_cols], first.sites, dur, nch, seed=a.seed, sedated=sed, site_labels=A.site_labels)
    R["controls"]["leakage_probes"] = clean(probes)
    if has_emb:             # the embeddings can encode site / recording length / sedation too: probe them as their own rung
        emb_cols = [c for c in A.eeg.columns if str(c).startswith(CBRAMOD_PREFIX)]
        R["controls"]["leakage_probes_cbramod_frozen"] = clean(leakage_probes(
            first.eeg[emb_cols], first.sites, dur, nch, seed=a.seed, sedated=sed, site_labels=A.site_labels))
    if has_mg:
        mg_cols = [c for c in A.eeg.columns if str(c).startswith(MORGOTH_PREFIX)]
        R["controls"]["leakage_probes_morgoth_findings"] = clean(leakage_probes(
            first.eeg[mg_cols], first.sites, dur, nch, seed=a.seed, sedated=sed, site_labels=A.site_labels))
    R["controls"]["by_scheme"] = {}
    n_small = max(200, min(a.n_boot, 1000))
    for split in splits:
        R["controls"]["by_scheme"][split] = {
            b: clean(run_controls(mds[split], results[split], split, b, A.baseline_sets[b], headline, cfg, probes,
                                  a.null_reps, a.seed, results[split].site_labels, n_small))
            for b in A.baseline_sets}
    if store is not None and "loso" in taps:
        try:
            rf = load_reports_findings(store, sorted(np.unique(A.sites)))
            R["circularity_audit"] = (circularity_block(A, mds["loso"], taps["loso"], rf, headline, min(a.n_boot, 500), a.seed)
                                      if len(rf) else {"note": "no reports_findings rows for these sites"})
        except FileNotFoundError:
            R["circularity_audit"] = {"note": "reports_findings not found"}
    else:
        R["circularity_audit"] = {"note": "skipped: no --data/--s3 store given (needed to read the EEG-report comparator)"}
    return R


def report_json(A: Assembly, a, R: dict) -> dict:
    return clean({
        "banner": BANNER,
        "intended_use": R.get("intended_use"),                     # D-145: the first reported block
        "decisions": ["D-145 (intended-use block, Baseline P, current-encounter baselines)", "D-143 (this analysis)", "D-120 (two-site schemes)", "D-095 (99% within-site CI + every-site rule)",
                      "D-123 (structured-only silver)", "D-124 (t0 anchors)", "D-107 (Baseline D dropped)",
                      "D-122 (no imaging in C, no NESI in A)"],
        "settings": {"cohort_definition": a.cohort_def, "sites": list(A.site_labels.values()),
                     "n_boot": a.n_boot, "auc_boot": a.auc_boot, "null_reps": a.null_reps, "seed": a.seed,
                     "dev_fraction": a.dev_frac, "temporal_test_fraction": a.temporal_fraction,
                     "sedation_source": a.sedation_source, "baseline_scope": A.baseline_scope,
                     "ci_policy": "99% within-site stratified bootstrap (D-095, evaluation_sample_size.md s8) with the every-site "
                                  "rule; 95% within_site / cluster / two_stage / site_t co-reported (cluster, two_stage and site_t "
                                  "are degenerate or extremely wide with two sites)",
                     "rungs": ["prior", "qeeg", "connectivity", "combined (= qeeg + connectivity, the headline rung)"]
                              + (["cbramod_frozen (frozen public CBraMod embedding alone)", "combined_cbramod (= qeeg + connectivity + embedding)"]
                                 if A.embedding_info else [])
                              + (["morgoth_findings (MORGOTH finding probabilities alone)", "combined_morgoth (= qeeg + connectivity + MORGOTH)"]
                                 if A.morgoth_info else []),
                     "rungs_ran": ["prior", "qeeg", "connectivity", "combined"]
                                  + (list(EXTRA_RUNG_NAMES[:2]) if A.embedding_info else []) + (list(EXTRA_RUNG_NAMES[2:]) if A.morgoth_info else []),
                     "rungs_skipped": (["dynamics"] + ([] if A.embedding_info else list(EXTRA_RUNG_NAMES[:2]))
                                       + ([] if A.morgoth_info else list(EXTRA_RUNG_NAMES[2:]))),
                     "skipped_rungs": ["dynamics"] + ([] if A.morgoth_info else ["morgoth"])
                                      + ([] if A.embedding_info else ["embeddings (CBraMod)"]),
                     "morgoth_exposure_flags": (A.morgoth_exposure["basis"] if A.morgoth_exposure else "none (no flag file)")},
        "data_flow": A.flow, "labels": {"analysed": list(A.label_names), "primary_for_delta": list(A.primary),
                                         "e7_primary": A.include_e7, "detail": A.label_report},
        "n_analysed": suppress_count(len(A.frame)),
        "n_eeg_features": sum(str(c).startswith(EEG_PREFIXES) for c in A.eeg.columns),
        "frozen_cbramod": A.embedding_info or {"skipped": "no embedding parquet parts found (scripts/extract_embeddings.py)"},
        "morgoth_findings": A.morgoth_info or {"skipped": "no MORGOTH findings parquet parts found (scripts/extract_morgoth.py)"},
        "n_baseline_columns": {b: len(c) for b, c in A.baseline_sets.items()},
        **{k: v for k, v in R.items() if k not in ("banner", "intended_use")}, "limitations": list(LIMITATIONS)})


def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def _probe_rows(probes: dict, suffix: str) -> list[str]:
    rows = []
    for nm, p in probes.get("probes", {}).items():
        info = " (informational)" if p.get("informational") else ""
        rows.append(f"| {nm}{info}{suffix} | {_f(p['auroc'])} | {p['flagged']} |")
        for sl, q in p.get("per_site", {}).items():
            rows.append(f"| {nm}, within {sl}{suffix} | {_f(q['auroc'])} | n/a |")
    return rows


def _sedation_markdown(sb: dict | None) -> list[str]:
    """"Sedation confounding" section: stratified Delta, difference of Deltas, baseline adjustment."""
    if not sb:
        return []
    fl = sb["flag"]
    L = ["", "## Sedation confounding", "",
         f"Does the EEG gain reflect sedation? Flag: t0 sedative/opioid exposure (--sedation-source {sb['flag_source']}); sedated patients {fl['n_sedated']} "
         f"(prevalence {fl['prevalence']}); per site " + ", ".join(f"{k}: {v['n_sedated']} ({v['prevalence']})" for k, v in fl["per_site"].items())
         + f"; flagged only by the silver sedative hint (not by the baseline's class flags) {fl['n_flagged_only_by_silver_hint']}.", "",
         "### Is sedation already in the baseline?", "", sb["baseline_adjustment"], "",
         "| Baseline | sedation columns | sedative/opioid on_t0 flags | dose or count columns |", "|---|---|---|---|"]
    for b, v in sb["baseline_sedation_features"].items():
        L.append(f"| {b} | {v['sedation_columns']} | {v['has_on_t0_class_flags']} | {v['has_dose_or_count_columns']} |")
    L += ["", "Sedation x EEG interaction variant: " + sb["sedation_interaction_variant"]["reason"] + ".", "",
          f"### Sedation-stratified Delta (headline rung `{sb['headline_rung']}`; same held-out predictions, no refit)", "",
          sb["reading"], "", sb["caveats"], "",
          f"A stratum under {sb['min_stratum_n']} patients is not estimable; counts under 11 (and the other stratum's count) are shown as <11. "
          "Intervals: within-site patient bootstrap, both strata and their difference from the same resamples.", "",
          "| Scheme | Baseline | Stratum | n | Delta | 99% CI | 95% CI | every site < 0 | per-site Delta (n) |", "|---|---|---|---|---|---|---|---|---|"]
    for split, per in sb["by_scheme"].items():
        for b, v in per.items():
            for nm, st in v["strata"].items():
                ps = ", ".join(f"{s_}: {_f(c[nm]['delta'])} (n={c[nm]['n']})" for s_, c in v["per_site"].items())
                if not st["estimable"]:
                    L.append(f"| {split} | {b} | {nm} | {st['n']} | not estimable: {st['not_estimable']} | | | | {ps} |")
                    continue
                c99, c95 = st.get("ci_99_within_site") or {}, st.get("ci_95_within_site") or {}
                L.append(f"| {split} | {b} | {nm} | {st['n']} | {_f(st['delta'])} | [{_f(c99.get('lo'))}, {_f(c99.get('hi'))}] | "
                         f"[{_f(c95.get('lo'))}, {_f(c95.get('hi'))}] | {st.get('all_sites_favorable')} | {ps} |")
    L += ["", "### Difference of Deltas (sedated minus non-sedated; negative = the gain is larger in sedated patients)", "",
          "| Scheme | Baseline | Difference | 99% CI | 95% CI | non-sedated gain persists (99% CI < 0) |", "|---|---|---|---|---|---|"]
    for split, per in sb["by_scheme"].items():
        for b, v in per.items():
            df = v["difference_sedated_minus_non_sedated"]
            if not df["estimable"]:
                L.append(f"| {split} | {b} | not estimable: {df['not_estimable']} | | | {v['non_sedated_gain_persists_99']} |")
                continue
            c99, c95 = df.get("ci_99_within_site") or {}, df.get("ci_95_within_site") or {}
            L.append(f"| {split} | {b} | {_f(df['delta'])} | [{_f(c99.get('lo'))}, {_f(c99.get('hi'))}] | "
                     f"[{_f(c95.get('lo'))}, {_f(c95.get('hi'))}] | {v['non_sedated_gain_persists_99']} |")
    return L


def _controls_markdown(c: dict) -> list[str]:
    L = ["", "## Mandatory controls", ""]
    if "skipped" in c:
        return L + [c["skipped"]]
    L += ["### Leakage probes (cross-validated AUROC predicting the factor from EEG features alone; flag > 0.70, PLACEHOLDER)", "",
          "| Probe | AUROC | flagged |", "|---|---|---|"]
    L += _probe_rows(c["leakage_probes"], "")
    L += _probe_rows(c.get("leakage_probes_cbramod_frozen", {}), " (frozen CBraMod embedding alone)")
    L += _probe_rows(c.get("leakage_probes_morgoth_findings", {}), " (MORGOTH findings alone)")
    L += ["", "`sedation` is informational (not a pass/fail control; not in the site/duration/channel flags): AUROC of predicting the "
          "t0 sedative/opioid flag from the EEG representation alone, pooled and within each site (the within-site rows cannot ride on "
          "a site shift). See \"Sedation confounding\"."]
    L += ["", "### Site concentration, sedative-excluded subset, negative controls", "",
          "| Scheme | Baseline | top-site share of gain | carried by one site | site probe control failed | sedative-excluded Delta | 99% CI | every site < 0 | n excluded | shuffled-label mean Delta | reps with spurious gain | permuted-EEG Delta |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for split, per in c["by_scheme"].items():
        for b, v in per.items():
            se = v["sedative_excluded"]
            nl = v["negative_control_shuffled_labels"]
            ne = v["negative_control_permuted_eeg"]
            ci = se.get("ci_99_within_site", {})
            L.append(f"| {split} | {b} | {_f(v['site_concentration'].get('top_share'))} | {v['site_concentration']['carried_by_one_site']} | "
                     f"{v['site_probe_control_failed']} | {_f(se.get('delta'))} | [{_f(ci.get('lo'))}, {_f(ci.get('hi'))}] | "
                     f"{se.get('all_sites_favorable', 'n/a')} | {se.get('n_excluded', 'n/a')} | {_f(nl['mean_delta'], 5)} | "
                     f"{nl['n_reps_with_spurious_gain_95']}/{nl['n_reps']} | {_f(ne.get('delta'), 5)} |")
    L += ["", "Severity strata (Delta of the headline rung within GCS-equivalent strata; point estimate < 0 is 'favourable'; "
          "a stratum under 50 patients is not estimable):", "",
          "| Scheme | Baseline | stratum | n | Delta | 95% CI | favourable |", "|---|---|---|---|---|---|---|"]
    for split, per in c["by_scheme"].items():
        for b, v in per.items():
            for s, d in v["severity_strata"]["strata"].items():
                L.append(f"| {split} | {b} | {s} | {d['n']} | {_f(d['delta'])} | [{_f(d['lo'])}, {_f(d['hi'])}] | {d['favorable']} |")
    return L


def _iu_markdown(iu: dict) -> list[str]:
    """The first block of report.md (D-145)."""
    L = [f"## {iu['title']}", "", f"**Question.** {iu['question']}", "",
         f"Baselines in this block use **{iu['baseline_scope']}** events (D-145). "
         f"Top available EEG rung: `{iu['headline_rung']}`. Delta = masked log loss(set + EEG) - masked log loss(set) on the same "
         "evaluation rows; negative = EEG helps. Comparisons: " + "; ".join(iu["comparisons"].values()) + ". "
         "'Pattern' = 99% within-site CI upper bound < 0 and Delta < 0 at every site (descriptive here).", "",
         f"**EEG-derived information is never an input or label evidence.** {iu['eeg_derived_inputs']}.", ""]
    if iu["comparisons_unavailable"]:
        L += ["Not run: " + "; ".join(f"{k} ({v})" for k, v in iu["comparisons_unavailable"].items()) + ".", ""]
    sg = iu["subgroup"]
    d = sg["definition"]
    L += ["### Undifferentiated subgroup", "",
          f"All three: EEG on encounter day 0-1 (t0 within {d['max_hours_from_visit_start']:g} h of the encounter start); no ICD "
          f"diagnosis code of the primary label families recorded at or before t0; no sedative or opioid exposure in the "
          f"{d['sedative_opioid_window_hours']:g} h before t0 (Baseline A sedation features). A component that cannot be established "
          f"counts as not met. Sources: {d['sources']}.", "",
          f"Size: {sg['n']} of the analysed set ({sg['share_of_analysed']}); per site "
          + ", ".join(f"{k}: {v}" for k, v in sg["per_site"].items()) + f". Components met: {sg['components_met']}.", "",
          "### Delta by comparison", "",
          "| Scheme | Population | Comparison | n eval | Reference loss | Delta | 99% CI | 95% CI | per-site Delta | every site < 0 | pattern |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for split, pops in iu["populations"].items():
        for pop, comps in pops.items():
            if "not_estimable" in comps:
                L.append(f"| {split} | {pop} | all | n/a | n/a | not estimable: {comps['not_estimable']} | | | | | |")
                continue
            for k, r in comps.items():
                ci99, ci95 = r.get("ci_primary_within_site", {}), r.get("ci_95", {}).get("within_site", {})
                ps = ", ".join(f"{s_}: {_f(v['delta'])} (n={v['n']})" for s_, v in r.get("per_site", {}).items())
                L.append(f"| {split} | {pop} | {iu['comparisons'][k]} | {r.get('n_eval', '')} | {_f(r.get('loss_reference'))} | "
                         f"{_f(r.get('delta'))} | [{_f(ci99.get('lo'))}, {_f(ci99.get('hi'))}] | "
                         f"[{_f(ci95.get('lo'))}, {_f(ci95.get('hi'))}] | {ps} | {r.get('all_sites_favorable')} | "
                         f"{r.get('h1h2_rule_met')} |")
    L += ["", "Reference = the prevalence prior (a), age/sex only (b), Baseline P only (c), each fitted with the same head, grid and "
          "recalibration as the EEG model. The subgroup rows are scored from models trained on all training-fold rows.", ""]
    nc = iu.get("negative_controls", {})
    if "skipped" in nc:
        L += [nc["skipped"], ""]
    elif nc:
        L += ["### Negative controls of this block (whole analysed set)", "",
              "| Scheme | Comparison | shuffled-label mean Delta | reps with spurious gain | permuted-EEG Delta | permuted-EEG spurious gain |",
              "|---|---|---|---|---|---|"]
        for split, per in nc.items():
            for k, v in per.items():
                L.append(f"| {split} | {iu['comparisons'][k]} | {_f(v['shuffled_labels_mean_delta'], 5)} | "
                         f"{v['shuffled_labels_reps_with_spurious_gain_95']}/{v['shuffled_labels_reps']} | "
                         f"{_f(v['permuted_eeg_delta'], 5)} | {v['permuted_eeg_spurious_gain_95']} |")
        L.append("")
    return L


def _gap_markdown(g: dict | None) -> list[str]:
    """"Commercial-clean gap" section; empty unless both cbramod_frozen and morgoth_findings ran."""
    if not g or not g.get("available"):
        return []
    L = ["", "## Commercial-clean gap: CBraMod (frozen) vs MORGOTH findings", "", g["note"], "",
         "| Scheme | Baseline | Row | n | Delta CBraMod | Delta MORGOTH | gap | 99% CI | 95% CI |", "|---|---|---|---|---|---|---|---|---|"]

    def row(sp, b, label, r):
        c99, c95 = r.get("ci_99_within_site"), r.get("ci_95_within_site")
        fmt = lambda c: c if isinstance(c, str) or c is None else f"[{_f(c.get('lo'))}, {_f(c.get('hi'))}]"  # noqa: E731
        return (f"| {sp} | {b} | {label} | {r.get('n', '')} | {_f(r.get('delta_a'))} | {_f(r.get('delta_b'))} | {_f(r.get('gap'))} | "
                f"{'n/a' if c99 is None else fmt(c99)} | {'n/a' if c95 is None else fmt(c95)} |")
    for sp, per in g["by_scheme"].items():
        for b, blk in per.items():
            if not blk.get("available"):
                continue
            L.append(row(sp, b, "all sites", blk["pooled"]))
            L += [row(sp, b, s_, v) for s_, v in blk["per_site"].items()]
            ex = blk["exposure_sensitivity"]
            if "not_run" in ex:
                continue
            L.append(row(sp, b, "excluding MORGOTH-exposed patients", ex["pooled"]))
            L += [row(sp, b, f"{s_} excluding exposed", v) for s_, v in ex["per_site"].items()]
    L += ["", g["exposure_note"], ""]
    for sp, per in g["by_scheme"].items():
        for b, blk in per.items():
            if not blk.get("available"):
                continue
            ex = blk["exposure_sensitivity"]
            L.append(f"- {sp}/{b}: " + (ex["not_run"] if "not_run" in ex else
                     f"flag basis: {ex['basis']}; excluded {ex['n_excluded']}, kept {ex['n_kept']}, kept without a flag {ex['n_without_flag_kept']}."))
    return L


def render_markdown(J: dict) -> str:
    L = [f"# {J['banner']}", "",
         "Models trained, tuned and scored on structured, EEG-blind silver labels (D-143). Pre-Gate-0. Nothing below tests H1-H6.",
         "", f"Cohort definition `{J['settings']['cohort_definition']}`; sites shown as pseudonyms in lexicographic order; "
         f"bootstrap B={J['settings']['n_boot']}; seed {J['settings']['seed']}; baselines: {J['settings'].get('baseline_scope', '')}.", ""]
    if J.get("intended_use"):
        L += _iu_markdown(J["intended_use"])
    L += ["## Data flow (counts < 11 shown as <11)", "", "| Step | n | " + " | ".join(
             sorted({s for st in J["data_flow"]["steps"] for s in st["per_site"]})) + " |",
         "|---|---|" + "---|" * len({s for st in J["data_flow"]["steps"] for s in st["per_site"]})]
    sites = sorted({s for st in J["data_flow"]["steps"] for s in st["per_site"]})
    for st in J["data_flow"]["steps"]:
        L.append(f"| {st['step']} | {st['n_total']} | " + " | ".join(str(st["per_site"].get(s, "")) for s in sites) + " |")
    L += ["", f"Analysed patients: {J['n_analysed']}; EEG features: {J['n_eeg_features']}; "
          f"baseline columns: {J['n_baseline_columns']}.", ""]
    fc = J.get("frozen_cbramod", {})
    if "dim" in fc:
        L += [f"Frozen CBraMod embedding rung: {fc['dim']} dimensions; analysed patients with an embedding {fc['n_analysed_with_embedding']} "
              f"({fc['share_analysed_with_embedding']}); rows without one are imputed fold-locally with a missingness indicator. "
              "Rungs `cbramod_frozen` and `combined_cbramod` are added to the Delta table; the headline rung is unchanged.", ""]
    mf = J.get("morgoth_findings", {})
    if "dim" in mf:
        L += [f"MORGOTH findings rung: {mf['dim']} features; analysed patients with MORGOTH output {mf['n_analysed_with_findings']} "
              f"({mf['share_analysed_with_findings']}); rows without one are imputed fold-locally with a missingness indicator. "
              "Rungs `morgoth_findings` and `combined_morgoth` are added to the Delta table; the headline rung is unchanged.", ""]
    s_ = J["settings"]
    L += [f"Rungs that ran: {', '.join(s_['rungs_ran'])}. Skipped: {', '.join(s_['rungs_skipped']) or 'none'}.", ""]
    L += ["## Labels", "",
          f"Analysed: {', '.join(J['labels']['analysed'])}. Primary for Delta: {', '.join(J['labels']['primary_for_delta'])}.", "",
          "| Label | n assessable | n positive | prevalence | per-site positives | note |", "|---|---|---|---|---|---|"]
    for lab, d in J["labels"]["detail"].items():
        ps = ", ".join(f"{k}: {v['n_positive']}" for k, v in d.get("per_site", {}).items())
        L.append(f"| {lab} | {d.get('n_assessable', '')} | {d.get('n_positive', '')} | {d.get('prevalence', '')} | {ps} | "
                 f"{d.get('excluded_reason', d.get('role', ''))} |")
    L += ["", "## Headline Delta: masked log loss, baseline + EEG minus baseline (negative = EEG helps)", "",
          f"Top available rung: `{J['headline_rung']}`. 'Pattern' = 99% within-site CI upper bound < 0 and Delta < 0 at every site "
          "(the H1/H2 decision pattern, descriptive here).", "",
          "| Scheme | Baseline | Rung | Delta | 99% CI | per-site Delta | every site < 0 | pattern |", "|---|---|---|---|---|---|---|---|"]
    for split, lad in J["ladder"].items():
        for b, rungs in lad["rungs"].items():
            for rn in ("qeeg", "connectivity", "combined", *EXTRA_RUNG_NAMES):
                r = rungs.get(rn)
                if not r or not r.get("available") or not r.get("has_eeg", True):
                    continue
                ci = r["ci_primary_within_site"]
                ps = ", ".join(f"{k}: {_f(v['delta'])} (n={v['n']})" for k, v in r["per_site"].items())
                L.append(f"| {split} | {b} | {rn} | {_f(r['delta'])} | [{_f(ci.get('lo'))}, {_f(ci.get('hi'))}] | {ps} | "
                         f"{r['all_sites_favorable']} | {r['h1h2_rule_met']} |")
    L += ["", "H1-style = Baseline A; H2-style = Baseline C. Reference losses and the prior rung are in report.json.", "",
          "### 95% intervals, all modes (headline rung)", "", "| Scheme | Baseline | within_site | cluster | two_stage | site_t |", "|---|---|---|---|---|---|"]
    for split, lad in J["ladder"].items():
        for b, rungs in lad["rungs"].items():
            r = rungs.get(J["headline_rung"], {})
            if r.get("available") and r.get("ci_95"):
                row = [f"[{_f(r['ci_95'][m].get('lo'))}, {_f(r['ci_95'][m].get('hi'))}]" if m in r["ci_95"] else "n/a"
                       for m in ("within_site", "cluster", "two_stage", "site_t")]
                L.append(f"| {split} | {b} | " + " | ".join(row) + " |")
    L += _gap_markdown(J.get("commercial_clean_gap"))
    L += ["", "## Per-label Delta (headline rung, 95% within-site CI)", "", "| Scheme | Baseline | Label | n | Delta | 95% CI | AUROC baseline | AUROC +EEG | Delta AUROC [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for split, lad in J["ladder"].items():
        for b, rungs in lad["rungs"].items():
            r = rungs.get(J["headline_rung"], {})
            dc = J["discrimination_calibration"][split][b]
            for lab, v in r.get("per_label", {}).items():
                a0 = dc["auroc"]["baseline_only"][lab]["pooled"]
                a1 = dc["auroc"][J["headline_rung"]][lab]["pooled"]
                da = dc.get("delta_auroc_vs_baseline", {}).get(J["headline_rung"], {}).get(lab, {})
                L.append(f"| {split} | {b} | {lab} | {v['n']} | {_f(v['delta'])} | [{_f(v['lo'])}, {_f(v['hi'])}] | {a0} | {a1} | "
                         f"{_f(da.get('delta_auroc'))} [{_f(da.get('lo'))}, {_f(da.get('hi'))}] |")
    ov = {f"{sp}/{b}": labs for sp, per in J.get("anchor_overlap_suspected", {}).items() for b, labs in per.items() if labs}
    L += ["", ("Baseline-only AUROC >= 0.90 (the baseline probably sees this label's anchor data, so an EEG increment is not "
               "interpretable for it): " + "; ".join(f"{k}: {', '.join(v)}" for k, v in ov.items())) if ov else
          "No label has baseline-only AUROC >= 0.90 (anchor overlap with the baselines is not obvious from discrimination alone)."]
    L += ["", "## Calibration (pooled held-out rows; slope only for labels with >= 100 events, D-094)", "",
          "| Scheme | Baseline | Model | Label | n events | O/E | CITL | slope [95% CI] | ECE | Brier |", "|---|---|---|---|---|---|---|---|---|---|"]
    for split, dcs in J["discrimination_calibration"].items():
        for b, dc in dcs.items():
            for mdl, labs in dc["calibration"].items():
                for lab, v in labs.items():
                    sl = f"{_f(v['slope'], 3)} [{_f(v['slope_lo'], 3)}, {_f(v['slope_hi'], 3)}]" if v["slope"] is not None else "O/E only"
                    L.append(f"| {split} | {b} | {mdl} | {lab} | {v['n_events']} | {_f(v['observed_over_expected'], 3)} | "
                             f"{_f(v['calibration_in_the_large'], 3)} | {sl} | {_f(v['ece'], 3)} | {_f(v['brier'], 3)} |")
    L += _controls_markdown(J["controls"])
    L += _sedation_markdown(J.get("sedation"))
    ca = J["circularity_audit"]
    L += ["", "## Circularity audit (leave-one-site-out predictions)", ""]
    if "per_model" in ca and ca["per_model"]:
        L += [ca["reading"], "", "| Model | Label | n | AUROC vs EEG report | AUROC vs silver label | difference | flagged |", "|---|---|---|---|---|---|---|"]
        for mdl, d in ca["per_model"].items():
            for lab, v in d["per_label"].items():
                L.append(f"| {mdl} | {lab} | {v['n']} | {_f(v['agreement_eeg_impression'])} | {_f(v['agreement_gold'])} | "
                         f"{_f(v['difference'])} | {v['leak']} |")
    else:
        L.append(str(ca.get("note", "not run")))
    L += ["", "## Interpretation limits", ""] + [f"- {x}" for x in J["limitations"]]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data", help="local HEEDB-layout directory (only for the circularity audit's reports_findings)")
    src.add_argument("--s3", action="store_true", help="read reports_findings from the BDSP access point (circularity audit)")
    ap.add_argument("--profile")
    ap.add_argument("--cohort", default="out/local_only/cohort_study1.csv")
    ap.add_argument("--features", default="out/local_only/features")
    ap.add_argument("--silver", default="out/local_only/silver/silver_labels.csv")
    ap.add_argument("--baselines", default="out/local_only/baselines_AC.parquet")
    ap.add_argument("--embeddings", default=None,
                    help="frozen-CBraMod parquet parts from scripts/extract_embeddings.py (default: <features dir>/../embeddings); "
                         "adds the cbramod_frozen and combined_cbramod rungs when present, skipped silently otherwise")
    ap.add_argument("--morgoth", default=None,
                    help="MORGOTH findings parquet parts from scripts/extract_morgoth.py / extract_rungs.py (default: <features dir>/../morgoth); "
                         "adds the morgoth_findings and combined_morgoth rungs (and the commercial-clean gap section when --embeddings "
                         "also ran) when parts are present, skipped silently otherwise")
    ap.add_argument("--morgoth-exposure", default=None,
                    help="local_only exposure-flag parquet from `scripts/extract_morgoth.py exposure` (default: <morgoth dir>/exposure.parquet); "
                         "if present, the gap section adds a sensitivity row excluding the flagged patients (aggregate only)")
    ap.add_argument("--recording-map", help="optional CSV (person_id, recording_id) if the extractor input carried explicit ids")
    ap.add_argument("--out", default="out/silver_feasibility")
    ap.add_argument("--cohort-def", choices=sorted(bb.COHORT_DEFS), default="strict")
    ap.add_argument("--sites", nargs="+", default=list(bb.STUDY_SITES))
    ap.add_argument("--splits", nargs="+", choices=["loso", "temporal"], default=["loso", "temporal"])
    ap.add_argument("--n-boot", type=int, default=4000, help="bootstrap replicates for Delta intervals (D-141 uses 10000 for final analyses)")
    ap.add_argument("--auc-boot", type=int, default=1000)
    ap.add_argument("--null-reps", type=int, default=3, help="shuffled-label negative-control repetitions")
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--temporal-fraction", type=float, default=0.2)
    ap.add_argument("--sedation-source", choices=["baseline", "e4b", "either"], default="either")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--undiff-max-hours", type=float, default=UNDIFF_MAX_H,
                    help="undifferentiated subgroup: EEG within this many hours of the encounter start (default 36, D-145)")
    ap.add_argument("--blas-threads", type=int, default=1,
                    help="BLAS/OpenMP threads during model fitting (1 is several times faster for these small logistic fits "
                         "than oversubscribed defaults; 0 = library default)")
    ap.add_argument("--skip-controls", action="store_true", help="skip the mandatory controls (quick looks only; the report says so)")
    ap.add_argument("--max-memory-gb", type=float, default=None)
    checkpoint.add_arguments(ap)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from sortinghat.cohort.memguard import run_guarded
    return run_guarded(lambda: _run(a), a.max_memory_gb, "silver feasibility")


def _run(a) -> int:
    out = Path(a.out)
    if "local_only" in out.resolve().parts:
        raise SystemExit("--out holds the aggregate report and must NOT be under local_only/")
    store = None
    if a.data:
        agent_safety.assert_not_restricted_in_agent(a.data)
        store = data_io.open_store(a.data)
    elif a.s3:
        store = data_io.open_store(None, profile=a.profile)
    from threadpoolctl import threadpool_limits
    # Restart safety: every model fit (scheme x fold x rung, the controls' refits included) is stored as it finishes under
    # out/local_only/checkpoints/silver_feasibility/<key>/ and replayed after a container restart (ladder.fit_predict_fold).
    # The fit keys digest the actual data, so the feature / embedding directories (whose ledgers change while extractors run)
    # are deliberately not part of the step key; the small input files and the arguments are.
    checkpoint.log_to_stderr()          # stdout of this script is a fixed-format report that starts with the banner
    try:
        cp = checkpoint.open_step("silver_feasibility", a, no_resume=a.no_resume,
                                  inputs=[p for p in (a.cohort, a.silver, a.baselines, a.recording_map) if p])
        A = assemble(a)
        with checkpoint.use(cp), threadpool_limits(limits=a.blas_threads or None):
            R = run_analysis(A, a, store)
    finally:
        checkpoint.log_to_stderr(False)
    J = report_json(A, a, R)
    known = {str(p) for p in A.frame["person_id"]} | set(A.frame["rid"].astype(str))
    safe_write_json(out / "report.json", J, known)
    safe_write_text(out / "report.md", render_markdown(J), known)
    safe_print(BANNER, known_ids=known)
    safe_print(f"{IU_TITLE} (baselines: {A.baseline_scope}); undifferentiated subgroup n={J['intended_use']['subgroup']['n']}", known_ids=known)
    for split, pops in J["intended_use"]["populations"].items():
        for pop, comps in pops.items():
            if "not_estimable" in comps:
                safe_print(f"  {split} {pop}: not estimable ({comps['not_estimable']})", known_ids=known)
                continue
            for k, r in comps.items():
                ci = r.get("ci_primary_within_site", {})
                safe_print(f"  {split} {pop} {J['intended_use']['comparisons'][k]}: Delta={_f(r.get('delta'))} 99% CI "
                           f"[{_f(ci.get('lo'))}, {_f(ci.get('hi'))}] every-site<0={r.get('all_sites_favorable')}", known_ids=known)
    safe_print(f"analysed={J['n_analysed']} labels={','.join(A.label_names)} primary={','.join(A.primary)} rung={R['headline_rung']}",
               known_ids=known)
    for split, lad in J["ladder"].items():
        for b, rungs in lad["rungs"].items():
            r = rungs.get(R["headline_rung"], {})
            if r.get("available"):
                ci = r["ci_primary_within_site"]
                safe_print(f"  {split} baseline {b}: Delta={_f(r['delta'])} 99% CI [{_f(ci.get('lo'))}, {_f(ci.get('hi'))}] "
                           f"every-site<0={r['all_sites_favorable']}", known_ids=known)
    safe_print(f"Report: {out / 'report.md'} and {out / 'report.json'} (aggregate-only)", known_ids=known)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"run_silver_feasibility failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
