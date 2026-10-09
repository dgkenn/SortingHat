#!/usr/bin/env python3
"""Is the analysed Study 1 cohort the right dataset for the intended use?   (aggregate-only; exploratory diagnostic)

Intended use: an unknown EEG from an ED patient with undifferentiated altered mental status / unexplained impaired consciousness,
little history known: what is the cause?  Concern: the analysed cohort may be dominated by ICU continuous-EEG monitoring of
already-diagnosed, sedated patients (spectrum mismatch).  This script characterises, for the strict and the broad cohort
definitions (the rows ``run_silver_feasibility.assemble`` analyses) and, for contrast, for the source population (the cohort
table before the EEG / baseline / silver requirements), per site (pseudonyms) and pooled, aggregate-only:

  eeg_type        recording duration quantiles and class (< 1 h routine / spot, 1-12 h, > 12 h continuous); the cEEG task folder flag;
                  the EEG service class (ServiceName, coarse)
  timing          hours from the visit start to t0 (quantiles; shares within 6 / 24 / 48 h); hours from onset (cohort table)
  care_setting    ServiceName class; covering-visit class and length; acute-care basis; arterial-gas recorded before t0 (ICU proxy);
                  mechanical ventilation, vasopressor and sedative/opioid-infusion proxies from the LOCAL OMOP cache (no new streaming)
  sedation        Baseline A sed__ columns: infusion agent / other agent active at t0 / recent only / none; sedation in the 6 h before t0
                  by class; and, with OMOP rows, continuous infusion vs PRN-only vs none in the 6 h before t0
  known_dx        any primary-label-family ICD code recorded before t0 (baselines meta__label_dx_before_t0)
  undifferentiated the current subgroup (feasibility script definition) and relaxed / stricter variants, with sizes (n, share, per
                  site, estimable?) and the silver-label positives each could contribute

Source population (``source_table``, ``source_strict``, ``source_broad``): the rows of the cohort table (before the
features / baselines / silver requirements); baseline-derived measures use only rows that have a baseline row (the share is reported).
NOTE the cohort table is already an adult acute-care proxy cohort at the study sites (OR / EMU excluded, EEG >= 11 min): it is not
all HEEDB EEGs.

Disclosure control: every count < 11 is shown "<11"; a cell is also hidden when its complement (denominator - count) < 11; a hidden
cell hides the smallest other cell of its exclusive class row (complementary suppression); a pooled cell is hidden when exactly one
site cell would reveal it by subtraction.  Quantiles need n >= 11 and never include min / max.  No id, date or timestamp is emitted
(relative hours only).  Everything goes through ``sortinghat.safe_output``.

    # synthetic / local mirror (what the tests run)
    python3 scripts/diag_intended_use_fit.py --cohort ... --features ... --silver ... --baselines ... --omop-source none
    # real (aggregate-only; D-118); OMOP proxies are read from the local cache only (out/local_only/omop_cache)
    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/diag_intended_use_fit.py --s3 --max-memory-gb 6
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))             # repo root, so `sortinghat` imports
sys.path.insert(0, str(HERE))                    # sibling scripts (run_silver_feasibility, build_baselines)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_baselines as bb  # noqa: E402
import run_silver_feasibility as rsf  # noqa: E402
from sortinghat import data_io, omop_cache  # noqa: E402
from sortinghat.baselines import lexicon as lex  # noqa: E402
from sortinghat.safe_output import (SUPPRESS_BELOW, SUPPRESSED, safe_print, safe_quantiles,  # noqa: E402
                                    safe_write_json, safe_write_text, suppress_count)

BANNER = "EXPLORATORY DIAGNOSTIC — intended-use fit of the Study 1 cohort; aggregate-only; not a test of any hypothesis"
K = SUPPRESS_BELOW
POOLED = "all_sites"
DUR_SHORT_S, DUR_LONG_S = 3600.0, 12 * 3600.0           # < 1 h routine / spot; 1-12 h; > 12 h continuous
TIME_CUTS_H = (6.0, 24.0, 48.0)
SED_WINDOW_H = 6.0                                      # sedation / infusion window before t0 (as the feasibility subgroup)
PROXY_WINDOW_H = 24.0                                   # ventilation / vasopressor look-back before t0
INFUSION_ONLY_AGENTS = ("propofol", "dexmedetomidine")  # given as continuous infusions in practice
IU_MIN_SUBGROUP = rsf.IU_MIN_SUBGROUP                   # rows needed to estimate Delta in a subgroup (and >= 11 per site)
LABELS = ("E1", "E2", "E4a", "E5", "E6", "E7")

# ---- coarse classes of free-text columns (counts of the coarse class only; the text itself is never emitted)
SERVICE_CLASSES = (("icu", re.compile(r"\bN?[MSC]?ICU\b|INTENSIVE|\bCCU\b", re.I)),
                   ("ed", re.compile(r"\bED\b|EMERG", re.I)),
                   ("ltm_continuous", re.compile(r"\bLTM\b|\bCEEG\b|CONTINUOUS|LONG[- ]?TERM", re.I)),
                   ("hospital_ward", re.compile(r"INPATIENT|\bIP\b|HOSPITAL|FLOOR|WARD", re.I)),
                   ("routine", re.compile(r"ROUTINE|OUTPATIENT|AMBULATORY|\bSTAT\b", re.I)))
SERVICE_NAMES = [n for n, _ in SERVICE_CLASSES] + ["other", "not_recorded"]

# ---- OMOP text rules (Python regex over free text; classes only are kept)
INFUSION_TEXT = re.compile(r"infus|\bdrip\b|continuous|premix|\bbag\b|\bpump\b|/\s?(?:hr|hour|min)\b|mcg/kg/min|mg/hr|"
                           r"\bin (?:0\.9|d5|dextrose|sodium chloride|nacl|ns\b)", re.I)
VASOPRESSOR = re.compile(r"norepinephrine|levophed|noradrenaline|vasopressin|vasostrict|phenylephrine|neo-?synephrine|"
                         r"\bdopamine\b|angiotensin ii|giapreza", re.I)
VASO_EXCLUDE = re.compile(r"ophth|otic|topical|cream|ointment|nasal|inhal|nebul|tablet|\btab\b|capsule|\bcap\b|patch|spray|"
                          r"drops?\b|\bgel\b|lidocaine|midodrine|droxidopa|auto-?injector|epipen", re.I)
VENT_MEASUREMENT = re.compile(r"ventilator|ventilation rate|\bvent\b|\bvent[ _]mode|\bpeep\b|tidal vol|\bvt\b|pressure support|\bpsv\b|\bsimv\b|"
                              r"\bprvc\b|set (?:resp|rate)|\bett\b|endotracheal|\btrach(?:eostomy)?\b|mechanical vent|\bintubated\b|"
                              r"airway.*(?:tube|ett)", re.I)       # not CMV (cytomegalovirus), not 'trachomatis'
VENT_PROCEDURE = re.compile(r"5a19[345]5z|0bh1[78]ez|\b9400[2345]\b|\b31500\b|intubat|ventilat", re.I)
DRUG_ANY = re.compile("|".join(rx for _n, _c, rx in lex._DRUG_RULES) + "|" + VASOPRESSOR.pattern, re.I)
# Arrow (RE2) prefilter, a superset of the Python rules above (case-insensitive via utf8_lower of the column)
DRUG_PREFILTER = DRUG_ANY.pattern.lower().replace(r"\b", "")
MEAS_PREFILTER = r"vent|peep|tidal|\bvt\b|pressure support|psv|simv|prvc|cmv|set (resp|rate)|ett|endotracheal|trach|intubat|airway"
PROC_PREFILTER = r"5a19|0bh1|9400|31500|intubat|ventilat"


# ====================================================================================== suppression helpers
def _small(num: int, den: int) -> bool:
    return den < K or num < K or (den - num) < K


def _flags(nums: list, dens: list, force: list | None = None) -> list:
    """Per cell of one exclusive class row: None = shown, else why it is hidden ('low' / 'high' / 'linked'); a lone hidden cell hides
    the smallest remaining shown cell too (complementary suppression)."""
    force = force or [False] * len(nums)
    why = [(("low" if (n < K or d < K) else "high") if _small(n, d) else ("linked" if f else None))
           for n, d, f in zip(nums, dens, force)]
    if sum(w is not None for w in why) == 1:
        shown = [i for i, w in enumerate(why) if w is None]
        if shown:
            why[min(shown, key=lambda i: (nums[i], i))] = "linked"
    return why


def _cell(n: int, d: int, why) -> dict:
    c = {"n": SUPPRESSED if why else int(n), "den": SUPPRESSED if d < K else int(d),
         "share": SUPPRESSED if why else round(n / d, 4)}
    if why:
        c["why"] = why
        if why == "high" and d >= K:                      # complement below 11: only a lower bound on the share is shown
            c["share_bound"] = f">={(d - (K - 1)) / d:.3f}"
    return c


def tab_classes(nums_by_site: dict, dens_by_site: dict, order: list, labels: dict) -> dict:
    """{class index: {site label | all_sites: cell}} for one exclusive class set: per site, then pooled."""
    k = len(next(iter(nums_by_site.values())))
    why_site = {s: _flags(nums_by_site[s], [dens_by_site[s]] * k) for s in order}
    pn = [sum(nums_by_site[s][j] for s in order) for j in range(k)]
    pd_ = sum(dens_by_site[s] for s in order)
    lone = [sum(why_site[s][j] is not None for s in order) == 1 for j in range(k)]
    lone_den = sum(dens_by_site[s] < K for s in order) == 1
    why_p = _flags(pn, [pd_] * k, [a or lone_den for a in lone])
    out = {}
    for j in range(k):
        row = {labels[s]: _cell(nums_by_site[s][j], dens_by_site[s], why_site[s][j]) for s in order}
        row[POOLED] = _cell(pn[j], pd_, why_p[j])
        out[j] = row
    return out


def tab_binary(num: np.ndarray, den: np.ndarray, sites: np.ndarray, order: list, labels: dict) -> dict:
    """{site label | all_sites: cell} for one binary measure (num within den)."""
    num = np.asarray(num, bool) & np.asarray(den, bool)
    den = np.asarray(den, bool)
    ns = {s: [int((num & (sites == s)).sum())] for s in order}
    ds = {s: int((den & (sites == s)).sum()) for s in order}
    return tab_classes(ns, ds, order, labels)[0]


def count_row(counts: dict, order: list, labels: dict) -> dict:
    """Counts per site and pooled, each < 11 shown '<11'; the pooled count is hidden when exactly one site count is hidden."""
    row = {labels[s]: suppress_count(int(counts[s])) for s in order}
    hidden = sum(counts[s] < K for s in order)
    tot = int(sum(counts[s] for s in order))
    row[POOLED] = SUPPRESSED if (tot < K or hidden == 1) else tot
    return row


def tab_quantiles(values: np.ndarray, sites: np.ndarray, order: list, labels: dict) -> dict:
    v = np.asarray(values, float)
    v = np.where(np.isfinite(v), v, np.nan)
    out = {labels[s]: {"n": suppress_count(int(np.isfinite(v[sites == s]).sum())), **safe_quantiles(v[(sites == s) & np.isfinite(v)])}
           for s in order}
    out[POOLED] = {"n": suppress_count(int(np.isfinite(v).sum())), **safe_quantiles(v[np.isfinite(v)])}
    return out


# ==================================================================================================== features
def _col(df: pd.DataFrame, c: str, default=np.nan) -> pd.Series:
    return df[c] if c in df else pd.Series(default, index=df.index)


def service_codes(service: pd.Series) -> np.ndarray:
    """Coarse service class index into SERVICE_NAMES per row (first matching class in SERVICE_CLASSES order; 'other'; 'not_recorded')."""
    s = service.astype("string").str.strip()
    empty = (s.isna() | (s == "")).to_numpy(bool)
    code = np.full(len(s), SERVICE_NAMES.index("other"), dtype=int)
    done = empty.copy()
    code[empty] = SERVICE_NAMES.index("not_recorded")
    for i, (_n, rx) in enumerate(SERVICE_CLASSES):
        hit = s.str.contains(rx, na=False).to_numpy(bool) & ~done
        code[hit] = i
        done |= hit
    return code


_FORBIDDEN_WORDS = (("patient", "pt"), ("encounter", "visit"), ("subject", "subj"), ("session", "sess"), ("accession", "acc"),
                    ("personid", "pers"), ("noteid", "note"), ("mrn", "m_r_n"), ("bids", "b_ids"), ("csn", "c_s_n"))


def clean_label(x: str) -> str:
    """A class label safe for the output guard (``safe_output`` refuses key names that look like identifiers, e.g. 'inpatient')."""
    for bad, ok in _FORBIDDEN_WORDS:
        x = re.sub(bad, ok, x, flags=re.I)
    return x


def dynamic_classes(values: pd.Series, top: int = 6) -> tuple[list, np.ndarray]:
    """(class labels, code per row; -1 = missing): the ``top`` most frequent values (pooled), the rest 'other'. The labels are the
    category values of a coarse structural column (e.g. a visit class), never free text about a person."""
    v = values.astype("string").str.strip().str.lower()
    vc = v.value_counts()
    raw = [str(x) for x in vc.index[:top] if int(vc[x]) >= K and re.fullmatch(r"[a-z0-9_ /\-]{1,24}", str(x))]   # short category tokens only
    keep = [clean_label(x) for x in raw]
    names = keep + (["other"] if (len(vc) > len(keep)) else [])
    code = np.full(len(v), -1, dtype=int)
    isna = v.isna().to_numpy() | (v == "").fillna(True).to_numpy()
    for i, x in enumerate(raw):
        code[(v == x).fillna(False).to_numpy()] = i
    if "other" in names:
        code[~isna & (code < 0)] = len(names) - 1
    return names, code


def _gt0(bl: pd.DataFrame, cols) -> np.ndarray:
    out = np.zeros(len(bl), bool)
    for c in cols:
        if c in bl:
            out |= pd.to_numeric(bl[c], errors="coerce").fillna(0).to_numpy(float) > 0
    return out


def sedation_features(bl: pd.DataFrame) -> dict | None:
    """Baseline A sedation arrays (bool, n); None when the baselines carry no sed__ column."""
    if not any(str(c).startswith("sed__") for c in bl.columns):
        return None
    ing = [i for i in lex.NAMED_INGREDIENTS]
    cls = lex.DRUG_CLASS
    on = {i: _gt0(bl, [f"sed__{i}__on_t0"]) for i in ing}
    q6 = {i: _gt0(bl, [f"sed__{i}__qty_6h"]) for i in ing}
    q24 = {i: _gt0(bl, [f"sed__{i}__qty_24h"]) for i in ing}
    sed_cls_on = _gt0(bl, ["sed__sedative__on_t0"])
    opi_cls_on = _gt0(bl, ["sed__opioid__on_t0"])
    infusion_agent = np.zeros(len(bl), bool)
    for i in INFUSION_ONLY_AGENTS:
        infusion_agent |= on.get(i, False)
    any_on = sed_cls_on | opi_cls_on
    for i in ing:
        any_on |= on[i]
    q6_cols = [c for c in bl.columns if str(c).startswith("sed__") and str(c).endswith("__qty_6h")]
    any_q6 = _gt0(bl, q6_cols)
    any_recent24 = _gt0(bl, [c for c in bl.columns if str(c).startswith("sed__") and (str(c).endswith("__qty_24h") or str(c).endswith("_24h"))])
    sedative_6h = sed_cls_on | np.logical_or.reduce([on[i] | q6[i] for i in ing if cls.get(i) == lex.SEDATIVE] or [np.zeros(len(bl), bool)])
    opioid_6h = opi_cls_on | np.logical_or.reduce([on[i] | q6[i] for i in ing if cls.get(i) == lex.OPIOID] or [np.zeros(len(bl), bool)])
    any_6h = rsf.sedation_6h_flag(bl)
    any_6h = np.asarray(any_6h, bool) if any_6h is not None else (any_on | any_q6)
    return {"any_on_t0": any_on, "infusion_agent_on_t0": infusion_agent, "any_6h": any_6h | sedative_6h | opioid_6h,
            "sedative_6h": sedative_6h, "opioid_6h": opioid_6h, "any_recent24": any_recent24 | any_q6 | any_on,
            "sedative_on_t0": sed_cls_on, "opioid_on_t0": opi_cls_on}


@dataclass
class Feat:
    """Per-row measures of one population (RECORD-LEVEL, memory only)."""
    n: int
    sites: np.ndarray
    has_bl: np.ndarray
    dur: np.ndarray
    ceeg: np.ndarray | None                   # bool; None = EEGFolder column absent
    service: np.ndarray | None                # coarse class code; None = ServiceName column absent
    hours: np.ndarray                         # hours from the visit start to t0 (NaN unknown)
    hours_src: str
    onset_h: np.ndarray | None
    inpatient_len: np.ndarray | None          # float 1 / 0 / NaN
    visit_class: tuple | None
    acute_basis: tuple | None
    dx: np.ndarray                            # label-family ICD codes before t0 (float; NaN unknown)
    sed: dict | None
    abg: np.ndarray | None                    # arterial gas value recorded before t0 (ICU proxy)
    arrest: np.ndarray | None
    omop: dict | None                         # name -> bool (n,) from the OMOP proxies; None = not computed
    sedated_any: np.ndarray                   # the feasibility script's t0 sedative / opioid flag (either source when available)
    Y: pd.DataFrame | None
    notes: list = field(default_factory=list)


def build_features(frame: pd.DataFrame, bl: pd.DataFrame, sedated: np.ndarray | None = None, Y: pd.DataFrame | None = None,
                   omop: pd.DataFrame | None = None) -> Feat:
    """``frame``: cohort rows; ``bl``: baselines aligned row by row (all-NaN rows where a person has none); ``omop``: a frame of
    bool proxy columns aligned to ``frame`` (vent_24h, vasopressor_24h, sed_row_6h, ...) or None."""
    n = len(frame)
    frame = frame.reset_index(drop=True)
    bl = bl.reset_index(drop=True)
    has_bl = bl.drop(columns=["person_id"], errors="ignore").notna().any(axis=1).to_numpy(bool)
    hours, hsrc = rsf.hours_since_encounter(frame, bl)
    if "EEGFolder" in frame:
        ef = frame["EEGFolder"].astype("string").str.strip().str.lower()
        ceeg = ef.str.startswith("ceeg").fillna(False).to_numpy(bool)
    else:
        ceeg = None
    onset = pd.to_numeric(frame["hours_since_onset"], errors="coerce").to_numpy(float) if "hours_since_onset" in frame else None
    inpl = None
    if "visit_inpatient_length" in frame:
        t = frame["visit_inpatient_length"].astype("string").str.strip().str.lower()
        inpl = np.where(t.isin(["true", "1", "1.0"]).fillna(False).to_numpy(), 1.0, np.where(t.isin(["false", "0", "0.0"]).fillna(False).to_numpy(), 0.0, np.nan))
    vc = dynamic_classes(frame["visit_class"]) if "visit_class" in frame else None
    ab = dynamic_classes(frame["acute_basis"]) if "acute_basis" in frame else None
    dx = pd.to_numeric(bl["meta__label_dx_before_t0"], errors="coerce").to_numpy(float) if "meta__label_dx_before_t0" in bl \
        else np.full(n, np.nan)
    sed = sedation_features(bl) if has_bl.any() else None
    gas = [c for c in ("lab__ph_arterial__value", "lab__pao2__value", "lab__paco2__value") if c in bl]
    abg = np.logical_or.reduce([bl[c].notna().to_numpy() for c in gas]) if gas else None
    arrest = (pd.to_numeric(bl["hx__arrest_recent"], errors="coerce").fillna(0).to_numpy(float) > 0) if "hx__arrest_recent" in bl else None
    omop_d = {c: omop[c].to_numpy(bool) for c in omop.columns} if omop is not None and len(omop.columns) else None
    sedated_any = np.asarray(sedated, bool) if sedated is not None else (sed["any_on_t0"] if sed is not None else np.zeros(n, bool))
    return Feat(n, frame["SiteID"].astype(str).to_numpy(object), has_bl, pd.to_numeric(_col(frame, "duration_s"), errors="coerce").to_numpy(float),
                ceeg, service_codes(frame["ServiceName"]) if "ServiceName" in frame else None, np.asarray(hours, float), hsrc, onset, inpl,
                vc and (vc[0], vc[1]), ab and (ab[0], ab[1]), dx, sed, abg, arrest, omop_d, sedated_any, Y)


# ============================================================================================= OMOP proxies
OMOP_FLAGS = ("sed_row_6h", "sed_inf_6h", "opioid_row_6h", "opioid_inf_6h", "sedative_row_6h", "sedative_inf_6h",
              "vasopressor_24h", "vent_24h")


def cache_files(table: str, root: Path) -> tuple[list, dict]:
    """Parquet files of one cached OMOP table (the profile with the most complete parts) and coverage counts. Reads nothing."""
    base = Path(root) / table
    best, best_key = [], (-1, -1)
    cov = {"parts": 0, "complete_parts": 0}
    if not base.is_dir():
        return [], cov
    for prof in sorted(p for p in base.iterdir() if p.is_dir()):
        files, parts, complete = [], 0, 0
        for part in sorted(p for p in prof.iterdir() if p.is_dir()):
            man = part / "manifest.json"
            if not man.exists():
                continue
            import json
            try:
                expect = int(json.loads(man.read_text()).get("n", 0))
            except (OSError, ValueError):
                continue
            pq_files = sorted(part.glob("rg-*.parquet"))
            empties = len(list(part.glob("rg-*.empty")))
            parts += 1
            complete += int(len(pq_files) + empties >= expect)
            files.extend(pq_files)
        if (complete, parts) > best_key:
            best, best_key, cov = files, (complete, parts), {"parts": parts, "complete_parts": complete}
    return best, cov


def iter_cache_frames(table: str, root: Path, columns: list, ids: np.ndarray, text_col: str | None, prefilter: str | None):
    """Cached row groups as pandas frames, filtered inside Arrow to ``ids`` and (optionally) a RE2 regex over ``text_col``."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    files, _cov = cache_files(table, root)
    want = pa.array(np.asarray(ids, dtype="int64"), type=pa.int64())
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            have = set(pf.schema_arrow.names)
            if "person_id" not in have:
                continue
            t = pf.read(columns=[c for c in columns if c in have])
        except Exception:                                    # noqa: BLE001 - unreadable file: skipped (counted nowhere, aggregate-only)
            continue
        if not t.num_rows:
            continue
        m = pc.is_in(t["person_id"].cast(pa.int64()), value_set=want)
        if text_col and prefilter and text_col in t.column_names:
            m = pc.and_(m, pc.fill_null(pc.match_substring_regex(pc.utf8_lower(t[text_col].cast(pa.string())), prefilter), False))
        t = t.filter(pc.fill_null(m, False))
        if t.num_rows:
            yield t.to_pandas()


def iter_dict_frames(tables: dict, table: str, ids: np.ndarray):
    df = tables.get(f"omop_{table}")
    if df is not None and len(df):
        yield df[df["person_id"].isin(set(int(i) for i in ids))]


def _times(df: pd.DataFrame, dt: str, dd: str) -> pd.Series:
    t = pd.to_datetime(df[dt], errors="coerce") if dt in df else pd.Series(pd.NaT, index=df.index)
    if dd in df:
        t = t.where(t.notna(), pd.to_datetime(df[dd], errors="coerce"))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t


def _has_time_of_day(t: pd.Series) -> pd.Series:
    return t.notna() & ((t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0))


def omop_proxies(cohort: pd.DataFrame, readers: dict, remap: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Per cohort row (aligned to ``cohort``): bool columns OMOP_FLAGS from the OMOP rows, chunk by chunk (only small flag arrays are kept).
    ``readers[table](ids, columns, text_col, prefilter)`` yields pandas chunks; ``remap`` retired id -> surviving id.

    sed_* / opioid_* / sedative_*: a sedative/opioid (any / opioid class / sedative class) drug row in the 6 h before t0 (start <= t0 and
    end >= t0 - 6 h; end = start when absent) ``_row_6h``, and the subset that is a continuous infusion (an infusion-only agent, an
    infusion-like text, or an administered interval >= 1 h with a time of day) ``_inf_6h``.  vasopressor_24h: a vasopressor row in
    [t0 - 24 h, t0].  vent_24h: a ventilator / intubation measurement or procedure row in [t0 - 24 h, t0] (a date-only row counts when its
    date is the date of t0 or of t0 - 24 h)."""
    pids = cohort["person_id"].to_numpy("int64")
    t0 = pd.to_datetime(cohort["t0"]).to_numpy("datetime64[ns]")
    pos = pd.Series(np.arange(len(pids)), index=pids)
    pos = pos[~pos.index.duplicated()]
    src = pd.to_numeric(cohort.get("person_id_source", cohort["person_id"]), errors="coerce")
    to_surv = {int(s): int(p) for s, p in zip(src.dropna(), cohort.loc[src.notna(), "person_id"]) if int(s) != int(p)}
    if remap:
        to_surv.update({int(o): int(n) for o, n in remap.items() if int(n) in pos.index})
    ids = np.array(sorted(set(pids.tolist()) | set(to_surv)), dtype="int64")
    out = {k: np.zeros(len(pids), bool) for k in OMOP_FLAGS}
    cov: dict = {}

    surv = pd.Series(to_surv, dtype="int64") if to_surv else pd.Series(dtype="int64")

    def locate(df: pd.DataFrame):
        p = pd.to_numeric(df["person_id"], errors="coerce")
        p = p.map(surv).fillna(p) if len(surv) else p
        ix = p.map(pos)
        ok = ix.notna().to_numpy()
        return ok, ix.fillna(-1).astype(int).to_numpy()

    # ---- drugs: sedatives / opioids / vasopressors
    ncls: dict = {}
    for ch in readers["drug_exposure"](ids, ["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime",
                                            "drug_exposure_start_date", "drug_exposure_end_date", "drug_source_value"],
                                       "drug_source_value", DRUG_PREFILTER):
        ok, ix = locate(ch)
        ch = ch[ok].reset_index(drop=True)
        ix = ix[ok]
        if not len(ch):
            continue
        ts = _times(ch, "drug_exposure_start_datetime", "drug_exposure_start_date")
        te = _times(ch, "drug_exposure_end_datetime", "drug_exposure_end_date")
        te = te.where(te.notna() & (te >= ts), ts)
        t0c = t0[ix]
        start_h = (ts.to_numpy("datetime64[ns]") - t0c) / np.timedelta64(1, "h")
        end_h = (te.to_numpy("datetime64[ns]") - t0c) / np.timedelta64(1, "h")
        keep = (start_h <= 0) & (end_h >= -PROXY_WINDOW_H)
        if not keep.any():
            continue
        ch, ix, start_h, end_h = ch[keep].reset_index(drop=True), ix[keep], start_h[keep], end_h[keep]
        ts, te = ts[keep].reset_index(drop=True), te[keep].reset_index(drop=True)
        codes, uniq = pd.factorize(ch["drug_source_value"].astype("string").fillna(""))
        rows = []
        for u in uniq:
            ings = lex.match_ingredients(u)
            klass = {lex.DRUG_CLASS[i] for i in ings}
            rows.append((lex.SEDATIVE in klass, lex.OPIOID in klass, any(i in INFUSION_ONLY_AGENTS for i in ings),
                         bool(INFUSION_TEXT.search(u)), bool(VASOPRESSOR.search(u)) and not VASO_EXCLUDE.search(u)))
        arr = np.array(rows, dtype=bool).reshape(-1, 5)[codes]
        is_sed, is_opi, inf_agent, inf_text, is_vaso = (arr[:, j] for j in range(5))
        timed = (_has_time_of_day(ts) & _has_time_of_day(te)).to_numpy(bool)
        long_iv = timed & ((end_h - start_h) >= 1.0)
        infusion = inf_agent | inf_text | long_iv
        in6 = end_h >= -SED_WINDOW_H
        for name, sel in (("sed", is_sed | is_opi), ("opioid", is_opi), ("sedative", is_sed)):
            r = sel & in6
            out[f"{name}_row_6h"][ix[r]] = True
            out[f"{name}_inf_6h"][ix[r & infusion]] = True
        out["vasopressor_24h"][ix[is_vaso]] = True
        ncls["drug_rows_kept"] = ncls.get("drug_rows_kept", 0) + len(ch)
    # ---- ventilation: measurements and procedures
    for table, tcols, text_col, rx, pre in (
            ("measurement", ["person_id", "measurement_datetime", "measurement_date", "measurement_source_value"],
             "measurement_source_value", VENT_MEASUREMENT, MEAS_PREFILTER),
            ("procedure_occurrence", ["person_id", "procedure_datetime", "procedure_date", "procedure_source_value"],
             "procedure_source_value", VENT_PROCEDURE, PROC_PREFILTER)):
        dt, dd = tcols[1], tcols[2]
        for ch in readers[table](ids, tcols, text_col, pre):
            ok, ix = locate(ch)
            ch = ch[ok].reset_index(drop=True)
            ix = ix[ok]
            if not len(ch):
                continue
            txt = ch[text_col].astype("string").fillna("")
            hit_u = {u: bool(rx.search(u)) for u in txt.unique()}
            hit = txt.map(hit_u).to_numpy(bool)
            if not hit.any():
                continue
            t = _times(ch, dt, dd)[hit]
            ixh, t0h = ix[hit], t0[ix[hit]]
            h = (t.to_numpy("datetime64[ns]") - t0h) / np.timedelta64(1, "h")
            dated = ~_has_time_of_day(t).to_numpy(bool)
            day_h = (t.dt.normalize().to_numpy("datetime64[ns]") - pd.Series(t0h).dt.normalize().to_numpy("datetime64[ns]")) / np.timedelta64(1, "h")
            inw = np.where(dated, (day_h >= -PROXY_WINDOW_H) & (day_h <= 0), (h >= -PROXY_WINDOW_H) & (h <= 0))
            out["vent_24h"][ixh[inw]] = True
            ncls[f"{table}_vent_rows"] = ncls.get(f"{table}_vent_rows", 0) + int(inw.sum())
    return pd.DataFrame(out), ncls


def make_readers(a, source) -> tuple[dict | None, dict]:
    """(readers per OMOP table, coverage notes); None when the OMOP proxies are switched off or unavailable."""
    mode = a.omop_source
    if mode == "none":
        return None, {"omop_source": "none"}
    tables = ("drug_exposure", "measurement", "procedure_occurrence")
    if mode == "cache":
        root = Path(a.omop_cache) if a.omop_cache else omop_cache.cache_root()
        cov = {t: cache_files(t, root)[1] for t in tables}
        if not any(c["parts"] for c in cov.values()):
            return None, {"omop_source": "cache (empty or absent)"}
        return ({t: (lambda ids, cols, tc, pre, _t=t: iter_cache_frames(_t, root, cols, ids, tc, pre)) for t in tables},
                {"omop_source": "local OMOP cache (no new streaming)", "coverage": cov})
    if isinstance(source, dict):
        return ({t: (lambda ids, cols, tc, pre, _t=t: iter_dict_frames(source, _t, ids)) for t in tables}, {"omop_source": "in-memory tables"})
    if source is None:
        return None, {"omop_source": "store requested but no --data / --s3"}

    def from_store(t):                                      # a local mirror / store (the tests' synthetic layout)
        def gen(ids, cols, tc, pre):
            for _u, batches, _tbl in data_io.iter_omop_units(t, person_ids=[int(i) for i in ids], columns=cols, s3=source):
                for b in batches:
                    yield b.to_pandas()
        return gen
    return {t: from_store(t) for t in tables}, {"omop_source": "store"}


# ====================================================================================== populations
@dataclass
class Pop:
    name: str
    description: str
    feat: Feat


def subgroup_parts(F: Feat) -> dict:
    """Components of the undifferentiated definition and the relaxations, as bool (n,) arrays (a component that cannot be established is
    not met, so a variant never grows through missing data)."""
    h = F.hours
    fin = np.isfinite(h)
    early = {x: fin & (h >= 0) & (h <= x) for x in (6.0, 24.0, 36.0, 48.0)}
    no_dx = np.isfinite(F.dx) & (F.dx == 0)
    sed = F.sed
    o = F.omop
    parts = {"early6": early[6.0], "early24": early[24.0], "early36": early[36.0], "early48": early[48.0], "no_dx": no_dx,
             "any_time": F.has_bl.copy()}
    if sed is not None:
        parts["no_sed_6h"] = F.has_bl & ~sed["any_6h"]
        parts["no_sedative_6h"] = F.has_bl & ~sed["sedative_6h"]
    # continuous infusion: OMOP rows when available, else the Baseline A infusion-only agents active at t0
    if o is not None and "sed_inf_6h" in o:
        parts["no_infusion_6h"] = F.has_bl & ~o["sed_inf_6h"]
        parts["no_opioid_infusion_6h"] = F.has_bl & ~o["opioid_inf_6h"]
    elif sed is not None:
        parts["no_infusion_6h"] = F.has_bl & ~sed["infusion_agent_on_t0"]
        parts["no_opioid_infusion_6h"] = parts["no_infusion_6h"]
    if o is not None and "vent_24h" in o:
        parts["not_ventilated"] = ~o["vent_24h"]
        parts["no_vasopressor"] = ~o["vasopressor_24h"]
    if F.ceeg is not None:
        parts["not_ceeg"] = ~F.ceeg
    if F.dur is not None:
        parts["recording_le_12h"] = np.isfinite(F.dur) & (F.dur <= DUR_LONG_S)
    return parts


# (name, description, required components)
VARIANTS = (
    ("U0_current", "current definition: visit day 0-1 (t0 <= 36 h), no label-family ICD code before t0, no sedative / opioid in the 6 h before t0",
     ("early36", "no_dx", "no_sed_6h")),
    ("U1_t0_within_48h", "as U0 with t0 within 48 h of the visit start", ("early48", "no_dx", "no_sed_6h")),
    ("U2_prn_opioid_allowed", "as U0 but PRN opioid allowed (no sedative-class exposure, no opioid infusion in the 6 h)",
     ("early36", "no_dx", "no_sedative_6h", "no_opioid_infusion_6h")),
    ("U3_sedation_not_infusion", "as U0 but sedation allowed when not a continuous infusion (no infusion of any sedative / opioid in the 6 h)",
     ("early36", "no_dx", "no_infusion_6h")),
    ("U4_48h_sedation_not_infusion", "as U3 with t0 within 48 h", ("early48", "no_dx", "no_infusion_6h")),
    ("U5_48h_dx_tolerated", "t0 within 48 h, no continuous infusion, known label-family diagnosis tolerated", ("early48", "no_infusion_6h")),
    ("U6_ed_like_proxy", "t0 within 24 h, no label-family diagnosis, no infusion, not ventilated, no vasopressor, not a cEEG task",
     ("early24", "no_dx", "no_infusion_6h", "not_ventilated", "no_vasopressor", "not_ceeg")),
    ("U7_ed_like_sedation_free", "t0 within 24 h, no label-family diagnosis, no sedative / opioid, not ventilated, not a cEEG task",
     ("early24", "no_dx", "no_sed_6h", "not_ventilated", "not_ceeg")),
    ("U8_short_recording", "t0 within 48 h, no label-family diagnosis, no infusion, recording <= 12 h",
     ("early48", "no_dx", "no_infusion_6h", "recording_le_12h")),
    ("U9_any_time_no_dx_no_infusion", "any time since the visit start, no label-family diagnosis, no infusion", ("any_time", "no_dx", "no_infusion_6h")),
    ("U10_early6_no_dx", "t0 within 6 h of the visit start, no label-family diagnosis (sedation ignored)", ("early6", "no_dx")),
)


def subgroup_variants(F: Feat) -> dict:
    """{name: (description, mask | None)}; None = a required component is unavailable (no baselines / OMOP / column)."""
    parts = subgroup_parts(F)
    out = {}
    for name, desc, req in VARIANTS:
        out[name] = (desc, None if any(r not in parts for r in req) else np.logical_and.reduce([parts[r] for r in req]))
    return out


# =============================================================================================== tabulation
def pop_tables(F: Feat, order: list, labels: dict) -> dict:
    """All tables of one population."""
    s = F.sites
    ones = np.ones(F.n, bool)
    tb: dict = {}          # binary measures: block -> name -> {definition, cells}
    cl: dict = {}          # exclusive class sets: block -> name -> {definition, classes: {label: cells}}
    qt: dict = {}          # quantile measures: block -> name -> {definition, by_site}

    def add_b(block, name, num, den, note):
        tb.setdefault(block, {})[name] = {"definition": note, "cells": tab_binary(num, den, s, order, labels)}

    def add_c(block, name, names, codes, den, note):
        den = np.asarray(den, bool) & (codes >= 0)
        nums = {x: [int((den & (s == x) & (codes == j)).sum()) for j in range(len(names))] for x in order}
        dens = {x: int((den & (s == x)).sum()) for x in order}
        t = tab_classes(nums, dens, order, labels)
        cl.setdefault(block, {})[name] = {"definition": note, "classes": {names[j]: t[j] for j in range(len(names))}}

    def add_q(block, name, values, note):
        qt.setdefault(block, {})[name] = {"definition": note, "by_site": tab_quantiles(values, s, order, labels)}

    # ---- eeg_type
    dur = F.dur
    dk = np.isfinite(dur)
    add_q("eeg_type", "duration_hours", dur / 3600.0, "recording duration (clock, hours)")
    add_c("eeg_type", "duration_class", ["lt_1h_routine_or_spot", "1_to_12h", "gt_12h_continuous"],
          np.where(dk, np.where(dur < DUR_SHORT_S, 0, np.where(dur <= DUR_LONG_S, 1, 2)), -1), ones,
          "recording duration class; denominator rows with a duration")
    if F.ceeg is not None:
        add_b("eeg_type", "ceeg_task_folder", F.ceeg, ones, "EEG folder names the cEEG task (continuous EEG)")
    if F.service is not None:
        add_c("eeg_type", "service_class", SERVICE_NAMES, F.service, ones, "EEG ServiceName, coarse class")
    # ---- timing
    h = F.hours
    hk = np.isfinite(h)
    add_q("timing", "hours_from_visit_start_to_t0", h, f"hours from the visit start to t0 ({F.hours_src}); NaN = unknown")
    add_b("timing", "t0_before_visit_start", hk & (h < 0), hk, "t0 earlier than the recorded visit start (date-only slack); denominator known")
    for x in TIME_CUTS_H:
        add_b("timing", f"t0_within_{x:g}h_of_visit_start", hk & (h >= 0) & (h <= x), hk, f"0 <= hours from the visit start to t0 <= {x:g}; denominator known")
    if F.onset_h is not None:
        add_q("timing", "hours_from_onset_to_t0", F.onset_h, "hours from the onset proxy to t0 (cohort table); NaN = unknown")
    # ---- care setting
    if F.inpatient_len is not None:
        k = np.isfinite(F.inpatient_len)
        add_b("care_setting", "covering_visit_longer_than_a_day", F.inpatient_len == 1, k, "covering visit is inpatient-length (end date after start date); denominator known")
    if F.visit_class is not None:
        add_b("care_setting", "covering_visit_class_recorded", F.visit_class[1] >= 0, ones, "the covering visit carries a class (ED / ICU / inpatient); mostly unrecorded in HEEDB")
    if F.visit_class is not None and F.visit_class[0]:
        add_c("care_setting", "covering_visit_class", F.visit_class[0], F.visit_class[1], ones, "covering visit class (structural column; mostly unrecorded in HEEDB)")
    if F.acute_basis is not None and F.acute_basis[0]:
        add_c("care_setting", "acute_care_basis", F.acute_basis[0], F.acute_basis[1], ones, "why the EEG counts as acute care in the cohort build")
    if F.abg is not None:
        add_b("care_setting", "arterial_gas_before_t0", F.abg, F.has_bl, "arterial pH / PaO2 / PaCO2 recorded in the visit before t0 (ICU / ventilation proxy); denominator rows with baselines")
    if F.arrest is not None:
        add_b("care_setting", "arrest_recent", F.arrest, F.has_bl, "cardiac-arrest code recent before t0; denominator rows with baselines")
    o = F.omop
    if o is not None:
        add_b("care_setting", "mechanical_ventilation_proxy_24h", o["vent_24h"], ones, "ventilator / intubation measurement or procedure row in the 24 h before t0 (OMOP cache; free-text rules)")
        add_b("care_setting", "vasopressor_24h", o["vasopressor_24h"], ones, "vasopressor drug row in the 24 h before t0 (OMOP cache)")
    # ---- sedation
    sd = F.sed
    if sd is not None:
        hb = F.has_bl
        add_b("sedation", "sedative_or_opioid_at_t0_either_source", F.sedated_any, hb, "t0 sedative / opioid flag of the feasibility script; denominator rows with baselines")
        add_b("sedation", "baselineA_any_agent_active_at_t0", sd["any_on_t0"], hb, "Baseline A: any sedative / opioid active at t0")
        tier = np.where(sd["infusion_agent_on_t0"], 0, np.where(sd["any_on_t0"], 1, np.where(sd["any_6h"], 2, np.where(sd["any_recent24"], 3, 4))))
        add_c("sedation", "baselineA_tiers", ["infusion_agent_active_at_t0", "other_agent_active_at_t0_rate_unknown", "recent_6h_only", "recent_24h_only", "none"],
              np.where(hb, tier, -1), ones, "Baseline A sedation tier (infusion-only agents = propofol / dexmedetomidine)")
        add_b("sedation", "baselineA_sedative_class_6h", sd["sedative_6h"], hb, "sedative-class exposure in the 6 h before t0")
        add_b("sedation", "baselineA_opioid_class_6h", sd["opioid_6h"], hb, "opioid-class exposure in the 6 h before t0")
    if o is not None and "sed_row_6h" in o:
        inf, row = o["sed_inf_6h"], o["sed_row_6h"]
        add_c("sedation", "omop_infusion_vs_prn_6h", ["continuous_infusion", "prn_only", "none"],
              np.where(inf, 0, np.where(row, 1, 2)), ones, "OMOP drug rows in the 6 h before t0: any sedative / opioid infusion vs only non-infusion rows vs none")
        add_b("sedation", "omop_sedative_infusion_6h", o["sedative_inf_6h"], ones, "sedative-class infusion in the 6 h before t0")
        add_b("sedation", "omop_opioid_infusion_6h", o["opioid_inf_6h"], ones, "opioid-class infusion in the 6 h before t0")
    # ---- known diagnosis
    dk_ = np.isfinite(F.dx)
    add_b("known_dx", "label_family_icd_before_t0", dk_ & (F.dx > 0), dk_, "any primary-label-family ICD code recorded at or before t0 (any visit); denominator rows with baselines")
    # ---- undifferentiated variants
    var, var_tab, pos_tab = {}, {}, {}
    for name, (desc, mask) in subgroup_variants(F).items():
        if mask is None:
            var[name] = {"definition": desc, "unavailable": "a required component could not be computed from the inputs"}
            continue
        cells = tab_binary(mask, ones, s, order, labels)
        counts = {x: int((mask & (s == x)).sum()) for x in order}
        total = sum(counts.values())
        entry = {"definition": desc, "cells": cells,
                 "estimable": bool(total >= IU_MIN_SUBGROUP and all(c >= K for c in counts.values()))}
        if F.Y is not None:
            entry["positives"] = {}
            for lab in F.Y.columns:
                yy = (F.Y[lab] == 1).to_numpy()
                entry["positives"][lab] = count_row({x: int((mask & yy & (s == x)).sum()) for x in order}, order, labels)
        var[name] = entry
    return {"tables": tb, "classes": cl, "quantiles": qt, "undifferentiated": var}


def pop_sizes(F: Feat, order: list, labels: dict) -> dict:
    return count_row({x: int((F.sites == x).sum()) for x in order}, order, labels)


# ======================================================================================== report assembly
SUMMARY_ROWS = (("tables", "eeg_type", "ceeg_task_folder"), ("classes", "eeg_type", "duration_class:lt_1h_routine_or_spot"),
                ("classes", "eeg_type", "duration_class:gt_12h_continuous"), ("classes", "eeg_type", "service_class:ltm_continuous"),
                ("tables", "timing", "t0_within_24h_of_visit_start"), ("tables", "timing", "t0_within_48h_of_visit_start"),
                ("tables", "care_setting", "covering_visit_longer_than_a_day"), ("tables", "care_setting", "mechanical_ventilation_proxy_24h"),
                ("tables", "care_setting", "vasopressor_24h"), ("tables", "sedation", "sedative_or_opioid_at_t0_either_source"),
                ("classes", "sedation", "baselineA_tiers:infusion_agent_active_at_t0"), ("classes", "sedation", "omop_infusion_vs_prn_6h:continuous_infusion"),
                ("classes", "sedation", "omop_infusion_vs_prn_6h:prn_only"), ("classes", "sedation", "omop_infusion_vs_prn_6h:none"),
                ("tables", "known_dx", "label_family_icd_before_t0"))


def summary_table(pops: dict) -> dict:
    """{measure: {population: pooled cell}} for the headline measures, plus every undifferentiated variant."""
    out: dict = {}
    for kind, block, nm in SUMMARY_ROWS:
        row = {}
        for pname, P in pops.items():
            if kind == "tables":
                c = P["tables"].get(block, {}).get(nm, {}).get("cells", {}).get(POOLED)
            else:
                base, cls = nm.split(":")
                c = P["classes"].get(block, {}).get(base, {}).get("classes", {}).get(cls, {}).get(POOLED)
            if c is not None:
                row[pname] = c
        if row:
            out[nm] = row
    for v, _d, _r in VARIANTS:
        row = {pn: P["undifferentiated"][v]["cells"][POOLED] for pn, P in pops.items() if "cells" in P["undifferentiated"].get(v, {})}
        if row:
            out[v] = row
    return out


def build_report(pops: list, order: list, labels: dict, settings: dict) -> dict:
    P = {p.name: {"description": p.description, "n": pop_sizes(p.feat, order, labels), "n_with_baselines": count_row(
        {x: int((p.feat.has_bl & (p.feat.sites == x)).sum()) for x in order}, order, labels),
        "sources": {"visit_start": p.feat.hours_src}, **pop_tables(p.feat, order, labels)} for p in pops}
    return {"banner": BANNER, "settings": settings, "summary_pooled": summary_table(P), "populations": P}


def _fmt(c: dict) -> str:
    if c["share"] == SUPPRESSED:
        return f"{SUPPRESSED} (share {c['share_bound']})" if "share_bound" in c else SUPPRESSED
    return f"{100 * c['share']:.1f}% ({c['n']}/{c['den']})"


def _fq(q: dict) -> str:
    if q.get("q50") == SUPPRESSED:
        return SUPPRESSED
    return " / ".join(str(q[k]) for k in ("q10", "q25", "q50", "q75", "q90"))


def render_markdown(rep: dict) -> str:
    sites = rep["settings"]["sites"] + [POOLED]
    L = [f"# {BANNER}", "",
         "Per population, site pseudonyms and pooled. Cells: share (count/denominator); `<11` = suppressed (count < 11, complement < 11, or complementary). "
         "Quantiles are q10 / q25 / q50 / q75 / q90 (n >= 11). Relative hours only; no dates.", ""]
    for note in rep["settings"].get("notes", []):
        L.append(f"- {note}")
    L.append("")
    pn = list(rep["populations"])
    L += ["## Summary (pooled over sites), one column per population", "", "| measure | " + " | ".join(pn) + " |", "|---|" + "---|" * len(pn),
          "| n | " + " | ".join(str(rep["populations"][p]["n"][POOLED]) for p in pn) + " |",
          "| n with a baselines row | " + " | ".join(str(rep["populations"][p]["n_with_baselines"][POOLED]) for p in pn) + " |"]
    for nm, row in rep["summary_pooled"].items():
        L.append(f"| {nm} | " + " | ".join(_fmt(row[p]) if p in row else "n/a" for p in pn) + " |")
    L.append("")
    for name, P in rep["populations"].items():
        L += [f"## {name}", "", f"{P['description']}  n = " + ", ".join(f"{s}: {P['n'][s]}" for s in sites) +
              f"; with a baselines row: " + ", ".join(f"{s}: {P['n_with_baselines'][s]}" for s in sites), ""]
        for block in ("eeg_type", "timing", "care_setting", "sedation", "known_dx"):
            rows = []
            for nm, t in P["tables"].get(block, {}).items():
                rows.append(f"| {nm} | " + " | ".join(_fmt(t["cells"][s]) for s in sites) + " |")
            for nm, t in P["classes"].get(block, {}).items():
                for cn, cells in t["classes"].items():
                    rows.append(f"| {nm}: {cn} | " + " | ".join(_fmt(cells[s]) for s in sites) + " |")
            for nm, t in P["quantiles"].get(block, {}).items():
                rows.append(f"| {nm} (q10/q25/q50/q75/q90) | " + " | ".join(_fq(t["by_site"][s]) for s in sites) + " |")
            if rows:
                L += [f"### {name} / {block}", "", "| measure | " + " | ".join(sites) + " |", "|---|" + "---|" * len(sites), *rows, ""]
        rows = []
        for nm, v in P["undifferentiated"].items():
            if "unavailable" in v:
                rows.append(f"| {nm} | unavailable |" + " |" * (len(sites) - 1))
                continue
            rows.append(f"| {nm} (estimable: {v['estimable']}) | " + " | ".join(_fmt(v["cells"][s]) for s in sites) + " |")
        L += [f"### {name} / undifferentiated variants (share of the population)", "", "| variant | " + " | ".join(sites) + " |", "|---|" + "---|" * len(sites), *rows, ""]
        if any("positives" in v for v in P["undifferentiated"].values()):
            labs = next(iter(v for v in P["undifferentiated"].values() if "positives" in v))["positives"].keys()
            rows = []
            for nm, v in P["undifferentiated"].items():
                if "positives" in v:
                    rows.append(f"| {nm} | " + " | ".join(str(v['positives'][lab][POOLED]) for lab in labs) + " |")
            L += [f"### {name} / silver-label positives in each variant (pooled)", "", "| variant | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs), *rows, ""]
    L += ["## Variant definitions", ""]
    first = next(iter(rep["populations"].values()))
    for nm, v in first["undifferentiated"].items():
        L.append(f"- **{nm}**: {v['definition']}")
    return "\n".join(L) + "\n"


# ============================================================================================ the driver
def load_table_population(a, store) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """(cohort table rows, aligned baselines, aligned silver labels, sedated flag) for the source population."""
    sites_arg = None if a.sites == ["all"] else a.sites
    table = bb.load_cohort(a.cohort, sites_arg, "table")
    bl, _cols = rsf.load_baselines(a.baselines)
    Y, hints = rsf.load_silver(a.silver)
    bl_al = bl.set_index("person_id").reindex(table["person_id"]).reset_index(drop=True)
    Ys = Y.reindex(table["person_id"]).reset_index(drop=True)
    return table, bl_al, Ys, None


def analysed_population(a, cohort_def: str):
    a2 = copy.copy(a)
    a2.cohort_def = cohort_def
    A = rsf.assemble(a2)
    bl, _c = rsf.load_baselines(a.baselines)
    bl_al = bl.set_index("person_id").reindex(A.frame["person_id"]).reset_index(drop=True)
    Y, _h = rsf.load_silver(a.silver)
    Ys = Y.reindex(A.frame["person_id"]).reset_index(drop=True)
    return A, bl_al, Ys


def subset(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return df[np.asarray(mask, bool)].reset_index(drop=True)


def run(a, source=None) -> dict:
    out_json, out_md = Path(a.diag_out), Path(a.diag_md)
    for p in (out_json, out_md):
        if "local_only" in p.resolve().parts:
            raise SystemExit("outputs hold aggregates and must NOT be under local_only/")
    from threadpoolctl import threadpool_limits
    notes: list = []
    with threadpool_limits(limits=a.blas_threads or None):
        table, tbl_bl, tbl_Y, _ = load_table_population(a, source)
        sites_sorted = sorted(table["SiteID"].unique())
        labels = {s: f"site_{i + 1}" for i, s in enumerate(sites_sorted)}
        # ---- OMOP proxies for the whole table population (computed once; every population is a subset)
        omop_df, omop_info = None, {"omop_source": "none"}
        readers, info = make_readers(a, source)
        omop_info = info
        if readers is not None:
            remap = None
            if source is not None and not isinstance(source, dict):
                try:
                    from sortinghat.cohort import StoreSources
                    remap, _st = StoreSources(source).merge_map()
                except Exception:                                    # noqa: BLE001 - optional bookkeeping
                    remap = None
                    notes.append("patient merge map unavailable: OMOP rows keyed by retired ids were not re-keyed")
            else:
                notes.append("no store: OMOP rows keyed by retired ids were not re-keyed (only the cohort's source ids were mapped)")
            omop_df, omop_counts = omop_proxies(table, readers, remap)
            omop_info = {**info, "matched_row_counts": {k: suppress_count(v) for k, v in omop_counts.items()}}
        om_by_pid = omop_df.set_axis(table["person_id"].to_numpy()) if omop_df is not None else None
        pops: list[Pop] = []
        known_ids: set = set()

        def src_pop(name, desc, mask):
            fr, bl = subset(table, mask), subset(tbl_bl, mask)
            om = subset(omop_df, mask) if omop_df is not None else None
            pops.append(Pop(name, desc, build_features(fr, bl, None, subset(tbl_Y, mask), om)))
        allm = np.ones(len(table), bool)
        src_pop("source_table", "all rows of the cohort table at the study sites (adult acute-care proxy; before the EEG-feature / baseline / silver requirements)", allm)
        if "in_strict" in table:
            src_pop("source_strict", "cohort-table rows meeting the strict definition (before the feature / baseline / silver requirements)", table["in_strict"].to_numpy(bool))
        if "in_broad" in table:
            src_pop("source_broad", "cohort-table rows meeting the broad definition (before the feature / baseline / silver requirements)", table["in_broad"].to_numpy(bool))
        for d in ("strict", "broad"):
            A, bl_al, Ys = analysed_population(a, d)
            om = om_by_pid.reindex(A.frame["person_id"].to_numpy()).fillna(False).astype(bool).reset_index(drop=True) if om_by_pid is not None else None
            F = build_features(A.frame, bl_al, np.asarray(A.covariates["sedated"], bool), Ys, om)
            chk = subgroup_variants(F).get("U0_current", (None, None))[1]
            if chk is not None and A.subgroup is not None:
                same = bool(np.array_equal(chk, np.asarray(A.subgroup, bool)))
                notes.append(f"{d}: U0_current reproduces the feasibility script's undifferentiated subgroup exactly: {same}")
            pops.append(Pop(f"analysed_{d}", f"rows analysed by run_silver_feasibility ({d} cohort: baselines, QC-passing primary EEG window, silver labels)", F))
            known_ids |= {str(p) for p in A.frame["person_id"]}
    known = {str(p) for p in table["person_id"]} | known_ids
    settings = {"sites": [labels[s] for s in sites_sorted], "suppression_k": K, "thresholds": {
        "duration_short_h": DUR_SHORT_S / 3600.0, "duration_long_h": DUR_LONG_S / 3600.0, "time_cuts_h": list(TIME_CUTS_H),
        "sedation_window_h": SED_WINDOW_H, "proxy_window_h": PROXY_WINDOW_H, "min_subgroup_rows": IU_MIN_SUBGROUP},
        "omop": omop_info, "notes": notes + [
            "the cohort table is an adult acute-care proxy cohort (covering visit longer than a day OR an acute ServiceName; OR / EMU excluded; EEG >= 11 min); it is not every HEEDB EEG",
            "visit_concept_id is zero throughout HEEDB: ED vs ICU is not directly readable; ServiceName, the cEEG task folder, covering-visit length, arterial gases and the OMOP ventilation / vasopressor / infusion proxies stand in",
            "baseline-derived measures use rows with a baselines row only (see n with a baselines row)",
            "the visit start is date-only (midnight): hours from the visit start to t0 are over-stated by 0-24 h, so 'within 24 h' is about 'same calendar day' and "
            "'within 48 h' about 'calendar day 0-1'"]}
    rep = build_report(pops, [s for s in sites_sorted], labels, settings)
    safe_write_json(out_json, rep, known)
    safe_write_text(out_md, render_markdown(rep), known)
    for line in headline_lines(rep):
        safe_print(line, known_ids=known)
    safe_print(f"Report: {out_json} and {out_md} (aggregate-only)", known_ids=known)
    return rep


def headline_lines(rep: dict) -> list:
    out = [BANNER]
    for name, P in rep["populations"].items():
        g = lambda blk, nm: P["tables"].get(blk, {}).get(nm, {}).get("cells", {}).get(POOLED)       # noqa: E731
        bits = [f"n={P['n'][POOLED]}"]
        for blk, nm in (("eeg_type", "ceeg_task_folder"), ("timing", "t0_within_24h_of_visit_start"), ("sedation", "sedative_or_opioid_at_t0_either_source"),
                        ("known_dx", "label_family_icd_before_t0"), ("care_setting", "mechanical_ventilation_proxy_24h")):
            c = g(blk, nm)
            if c is not None:
                bits.append(f"{nm}={_fmt(c)}")
        u0 = P["undifferentiated"].get("U0_current", {}).get("cells", {}).get(POOLED)
        if u0:
            bits.append(f"U0_current={_fmt(u0)}")
        out.append(f"{name}: " + "; ".join(bits))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = rsf.build_parser()
    ap.description = __doc__
    ap.set_defaults(out="out/diag")
    ap.add_argument("--diag-out", default="out/diag/intended_use_fit.json", help="aggregate JSON (must not be under local_only/)")
    ap.add_argument("--diag-md", default="out/diag/intended_use_fit.md", help="aggregate markdown (must not be under local_only/)")
    ap.add_argument("--omop-source", choices=["cache", "store", "none"], default="cache",
                    help="OMOP rows for the ventilation / vasopressor / infusion proxies: the local cache (default; never streams), the store "
                         "given by --data / --s3, or none (those measures are then omitted)")
    ap.add_argument("--omop-cache", default=None, help="root of the local OMOP cache (default: the pipeline's out/local_only/omop_cache)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from sortinghat.cohort.memguard import run_guarded
    source = None
    if a.data:
        from sortinghat import agent_safety
        agent_safety.assert_not_restricted_in_agent(a.data)
        source = data_io.open_store(a.data)
    elif a.s3 and a.omop_source == "store":
        source = data_io.open_store(None, profile=a.profile)
    elif a.s3:                                              # the store is only needed for the merge map
        try:
            source = data_io.open_store(None, profile=a.profile)
        except Exception:                                    # noqa: BLE001
            source = None
    return run_guarded(lambda: (run(a, source), 0)[1], a.max_memory_gb, "intended-use fit diagnostic")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"diag_intended_use_fit failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
