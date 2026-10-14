"""Presentation-based intended-use strata, the shared OMOP proxies they need, and transport re-weighting (D-155; aggregate-only).

Intended use: an unknown EEG from an ED patient with undifferentiated altered mental status, little history known.  The first
operationalisation (U0, D-145) required "no ICD code of the label families before t0".  The silver labels are built from those
same code families, so U0 is almost label-negative by construction (docs/research/intended_use_fit_2026-10-09.md, finding 6).
The strata below are defined by PRESENTATION ONLY and use no diagnosis code, no silver label and no EEG-derived quantity:

  IU-large  t0 on encounter calendar day 0-1 (the visit start is date-only, so floor(hours / 24) in {0, 1}, i.e. up to ~48 h);
            no sedative / opioid continuous infusion running at t0 (Baseline A ``on_t0`` flags, class or named agent: the
            best available proxy, PRN doses with a recorded stop do not count); no ventilation proxy and no vasopressor proxy in
            the 24 h before t0 (OMOP rows, optional); the recording is <= 12 h OR is not a continuous-EEG task folder.
  IU-small  encounter calendar day 0 (same day as the visit start); every IU-large condition; the recording is not a
            continuous-EEG task folder (PRN sedation allowed).  IU-small is a subset of IU-large.

A condition that cannot be established for a row counts as NOT met (a stratum never grows through missing data).  When the OMOP
proxies are not available the stratum is defined WITHOUT the ventilation / vasopressor conditions and says so (``missing``).
When the visit start, the Baseline A sedation columns (or, for IU-small, the cEEG task folder) are absent, the stratum is
unavailable.  The proxies are free-text rules over the local OMOP cache and are not chart-validated.

Transport (re-weighting toward the IU-large population): a domain classifier P(IU-large | pre-t0 covariates independent of the
EEG and of the diagnosis codes) is fitted by cross-fitting on the analysed rows; non-IU rows get stabilised inverse-odds weights
w = [p / (1 - p)] / [pi / (1 - pi)] (pi = IU share), trimmed at the 1st / 99th percentile of those weights, IU rows weight 1.
The weighted Delta is sum(w d) / sum(w) over the scored rows with a within-site bootstrap that keeps the weights fixed (the
interval does not carry the uncertainty of the weight model).  IU-large is defined by a deterministic rule on the very covariates
the domain model sees, so overlap is the issue: the report gives the effective sample size, the cross-fitted AUC and the
standardised mean differences before and after weighting; a low ESS means the weighted Delta is not to be read.

Nothing here is record-level on output.  Counts go through ``sortinghat.safe_output.suppress_count`` (n < 11 shown "<11"); a count
is also hidden when one per-site cell is hidden; counts are keyed ``n`` / ``assessable`` / ``positive`` (never an ``n_*`` key that
could hold a small integer).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..baselines import lexicon as lex
from ..safe_output import SUPPRESS_BELOW, SUPPRESSED, safe_quantiles, suppress_count
from .bootstrap import percentile_ci, resample_indices

K = SUPPRESS_BELOW
POOLED = "all_sites"
MIN_ROWS = 50                                           # scored rows needed to estimate Delta in a stratum (and >= 11 per site)

DUR_SHORT_S, DUR_LONG_S = 3600.0, 12 * 3600.0           # < 1 h routine / spot; 1-12 h; > 12 h continuous
SED_WINDOW_H = 6.0                                      # sedation / infusion window before t0 (as the feasibility subgroup)
PROXY_WINDOW_H = 24.0                                   # ventilation / vasopressor look-back before t0
INFUSION_ONLY_AGENTS = ("propofol", "dexmedetomidine")  # given as continuous infusions in practice

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



def _col(df: pd.DataFrame, c: str, default=np.nan) -> pd.Series:
    return df[c] if c in df else pd.Series(default, index=df.index)


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
    any_6h = sedation_6h_flag(bl)
    any_6h = np.asarray(any_6h, bool) if any_6h is not None else (any_on | any_q6)
    return {"any_on_t0": any_on, "infusion_agent_on_t0": infusion_agent, "any_6h": any_6h | sedative_6h | opioid_6h,
            "sedative_6h": sedative_6h, "opioid_6h": opioid_6h, "any_recent24": any_recent24 | any_q6 | any_on,
            "sedative_on_t0": sed_cls_on, "opioid_on_t0": opi_cls_on}


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



def make_omop_readers(mode: str, cache_arg=None, source=None) -> tuple[dict | None, dict]:
    """(readers per OMOP table, coverage notes); None when the OMOP proxies are switched off or unavailable.  ``mode``: 'cache' (the
    local OMOP cache; never streams), 'store' (a store or in-memory tables given as ``source``) or 'none'."""
    if mode == "none":
        return None, {"omop_source": "none"}
    tables = ("drug_exposure", "measurement", "procedure_occurrence")
    if mode == "cache":
        from .. import omop_cache
        root = Path(cache_arg) if cache_arg else omop_cache.cache_root()
        cov = {t: cache_files(t, root)[1] for t in tables}
        if not any(c["parts"] for c in cov.values()):
            return None, {"omop_source": "cache (empty or absent)"}
        return ({t: (lambda ids, cols, tc, pre, _t=t: iter_cache_frames(_t, root, cols, ids, tc, pre)) for t in tables},
                {"omop_source": "local OMOP cache (no new streaming)", "coverage": cov})
    if isinstance(source, dict):
        return ({t: (lambda ids, cols, tc, pre, _t=t: iter_dict_frames(source, _t, ids)) for t in tables}, {"omop_source": "in-memory tables"})
    if source is None:
        return None, {"omop_source": "store requested but no --data / --s3"}
    from .. import data_io

    def from_store(t):                                      # a local mirror / store (the tests' synthetic layout)
        def gen(ids, cols, tc, pre):
            for _u, batches, _tbl in data_io.iter_omop_units(t, person_ids=[int(i) for i in ids], columns=cols, s3=source):
                for b in batches:
                    yield b.to_pandas()
        return gen
    return {t: from_store(t) for t in tables}, {"omop_source": "store"}


# ============================================================================================ strata definitions
IU_LARGE, IU_SMALL = "IU_large", "IU_small"
STRATA = (IU_LARGE, IU_SMALL)
DEFINITIONS = {
    IU_LARGE: "t0 on encounter calendar day 0-1 (up to ~48 h from the date-only visit start); no sedative / opioid continuous infusion "
              "running at t0 (Baseline A on_t0 flags); no ventilation proxy and no vasopressor proxy in the 24 h before t0 (OMOP, optional); "
              "recording <= 12 h OR not a continuous-EEG task folder. No condition on diagnosis codes, silver labels or EEG-derived quantities.",
    IU_SMALL: "encounter calendar day 0 (same day as the visit start); every IU-large condition; the recording is not a continuous-EEG task "
              "folder; PRN sedation allowed (only a running infusion at t0 excludes). A subset of IU-large."}
DAY_H = 24.0


def calendar_day(hours) -> np.ndarray:
    """Encounter calendar day of t0 (0 = the day of the visit start) from the hours since the date-only visit start; NaN when unknown
    or negative (t0 before the recorded start)."""
    h = np.asarray(hours, float)
    return np.where(np.isfinite(h) & (h >= 0), np.floor(np.where(np.isfinite(h), h, 0.0) / DAY_H), np.nan)


def ceeg_task_flag(frame: pd.DataFrame) -> np.ndarray | None:
    """Bool (n,): the EEG folder names the continuous-EEG task; None when the EEGFolder column is absent."""
    if "EEGFolder" not in frame:
        return None
    ef = frame["EEGFolder"].astype("string").str.strip().str.lower()
    return ef.str.startswith("ceeg").fillna(False).to_numpy(bool)


def intended_use_strata(hours, sed: dict | None, dur, ceeg, omop: dict | None) -> dict:
    """IU-large / IU-small masks from presentation features only.

    ``hours``: hours from the visit start to t0 (NaN unknown); ``sed``: ``sedation_features`` output (None = no Baseline A sed__
    columns); ``dur``: recording duration in seconds (NaN unknown); ``ceeg``: bool cEEG task folder (None = column absent);
    ``omop``: {'vent_24h', 'vasopressor_24h'} bool arrays or None (the strata are then defined without those two conditions).
    Returns {'masks': {name: bool (n,) | None}, 'components': {name: bool (n,)}, 'applied': {name: [..]}, 'missing': {key: why},
    'unavailable': {name: why}} (record-level arrays stay in memory)."""
    hours = np.asarray(hours, float)
    n = len(hours)
    comp: dict = {}
    missing: dict = {}
    unavailable: dict = {}
    day = calendar_day(hours)
    why: list = []
    if not np.isfinite(day).any():
        why.append("the hours from the visit start to t0 cannot be established (no encounter_start / baselines meta)")
    comp["calendar_day_0_1"] = np.isfinite(day) & (day <= 1)
    comp["calendar_day_0"] = np.isfinite(day) & (day == 0)
    if sed is None:
        why.append("no Baseline A sed__ columns: a running infusion at t0 cannot be established")
        comp["no_infusion_at_t0"] = np.zeros(n, bool)
    else:
        comp["no_infusion_at_t0"] = ~np.asarray(sed["any_on_t0"], bool)
    if omop is not None and "vent_24h" in omop and "vasopressor_24h" in omop:
        comp["no_ventilation_proxy_24h"] = ~np.asarray(omop["vent_24h"], bool)
        comp["no_vasopressor_24h"] = ~np.asarray(omop["vasopressor_24h"], bool)
    else:
        missing["ventilation_vasopressor"] = ("OMOP ventilation / vasopressor proxies unavailable: the strata are defined WITHOUT the "
                                              "no-ventilation and no-vasopressor conditions")
    dur = np.asarray(dur, float)
    short = np.isfinite(dur) & (dur <= DUR_LONG_S)
    has_dur = bool(np.isfinite(dur).any())
    if ceeg is not None:
        not_ceeg = ~np.asarray(ceeg, bool)
        comp["not_ceeg_task_folder"] = not_ceeg
    else:
        not_ceeg = None
    if not_ceeg is None and not has_dur:
        missing["recording_type"] = "neither the cEEG task folder nor the recording duration is available: no recording-type condition applied"
        rec_ok = np.ones(n, bool)
    else:
        rec_ok = (short if has_dur else np.zeros(n, bool)) | (not_ceeg if not_ceeg is not None else np.zeros(n, bool))
        comp["recording_le_12h_or_not_ceeg"] = rec_ok
    if why:
        unavailable["both"] = "; ".join(why)
    proxies = "no_ventilation_proxy_24h" in comp
    common = ["no_infusion_at_t0"] + (["no_ventilation_proxy_24h", "no_vasopressor_24h"] if proxies else [])
    applied = {IU_LARGE: ["calendar_day_0_1"] + common + ([] if "recording_type" in missing else ["recording_le_12h_or_not_ceeg"]),
               IU_SMALL: ["calendar_day_0"] + common + ["not_ceeg_task_folder"]}
    masks: dict = {}
    for nm in STRATA:
        if "both" in unavailable:
            masks[nm] = None
        elif nm == IU_SMALL and not_ceeg is None:
            masks[nm] = None
            unavailable[nm] = "the cEEG task folder (EEGFolder) is not in the cohort table"
        else:
            masks[nm] = np.logical_and.reduce([comp[c] for c in applied[nm]])
    return {"masks": masks, "components": comp, "applied": applied, "missing": missing, "unavailable": unavailable,
            "proxies_in": proxies}


# ====================================================================================== suppressed aggregates
def count_row(counts: dict, labels: dict, totals: dict | None = None) -> dict:
    """{pseudonym | all_sites: count} with every count < 11 shown '<11'.  With ``totals`` (the denominators per site) a count whose
    complement (total - count) is < 11 is hidden too, since the complement would be a small cell.  The pooled count is hidden when it
    is itself small, its complement is, or exactly one site count is hidden (it would reveal it by subtraction)."""
    def small(c, t):
        return int(c) < K or (t is not None and int(t) - int(c) < K)
    hid = {s: small(c, None if totals is None else totals[s]) for s, c in counts.items()}
    row = {labels[s]: (SUPPRESSED if hid[s] else int(c)) for s, c in counts.items()}
    tot = int(sum(counts.values()))
    den = None if totals is None else int(sum(totals.values()))
    row[POOLED] = SUPPRESSED if (small(tot, den) or sum(hid.values()) == 1) else tot
    return row


def mask_counts(mask, sites, order: list, labels: dict, den=None) -> dict:
    """Per-site and pooled count of ``mask`` (within ``den`` when given, else within all rows), suppressed by ``count_row``."""
    mask = np.asarray(mask, bool)
    d = np.ones(len(mask), bool) if den is None else np.asarray(den, bool)
    return count_row({s: int((mask & (sites == s)).sum()) for s in order}, labels,
                     {s: int((d & (sites == s)).sum()) for s in order})


def stratum_summary(res: dict, y: np.ndarray, m: np.ndarray, label_names, sites, order: list, labels: dict) -> dict:
    """Sizes (overall and per component met) and per-label assessable / positive counts of every available stratum (suppressed)."""
    out: dict = {"definitions": DEFINITIONS, "applied_components": res["applied"], "missing": res["missing"],
                 "unavailable": res["unavailable"], "proxies_included": res["proxies_in"], "strata": {}}
    sites = np.asarray(sites)
    n = len(sites)
    out["components_met"] = {c: mask_counts(v, sites, order, labels) for c, v in res["components"].items()}
    for nm in STRATA:
        mask = res["masks"][nm]
        if mask is None:
            out["strata"][nm] = {"available": False, "why": res["unavailable"].get(nm, res["unavailable"].get("both", "unavailable"))}
            continue
        size = mask_counts(mask, sites, order, labels)
        share = (float(mask.sum()) / n) if n and int(mask.sum()) >= K and (n - int(mask.sum())) >= K else SUPPRESSED
        lab = {}
        for j, name in enumerate(label_names):
            a = np.asarray(m[:, j], bool)
            pos = a & (np.asarray(y[:, j]) > 0.5)
            lab[name] = {"assessable": mask_counts(mask & a, sites, order, labels, mask),
                         "positive": mask_counts(mask & pos, sites, order, labels, mask & a)}
        out["strata"][nm] = {"available": True, "size": size, "share_of_analysed": round(share, 4) if share != SUPPRESSED else share,
                             "per_label": lab}
    return out


# ================================================================================================ Delta blocks
def _ci(reps: np.ndarray, alpha: float) -> dict | None:
    lo, hi, nv = percentile_ci(reps, alpha)
    return None if nv == 0 else {"lo": lo, "hi": hi}


def _boot_means(d: np.ndarray, w: np.ndarray | None, sites: np.ndarray, boot: int, seed: int) -> np.ndarray:
    groups = [np.flatnonzero(sites == s) for s in np.unique(sites)]
    rng = np.random.default_rng(seed)
    reps = np.empty(boot)
    for b in range(boot):
        idx = resample_indices(groups, rng, "within_site")
        if w is None:
            reps[b] = d[idx].mean()
        else:
            ww = w[idx]
            reps[b] = float((ww * d[idx]).sum() / ww.sum())
    return reps


def delta_block(d, sites, site_labels: dict, w=None, *, boot: int = 2000, seed: int = 0, alpha_primary: float = 0.01,
                alpha_secondary: float = 0.05, min_rows: int = MIN_ROWS, min_cell: int = K) -> dict:
    """Delta = (weighted) mean of the per-patient differences ``d`` (NaN rows dropped) with a within-site patient bootstrap at 99%
    (primary) and 95%.  Not estimable below ``min_rows`` rows or when a site has fewer than ``min_cell`` rows (every site in
    ``site_labels`` counts).  Weights, when given, are held fixed across resamples.  Counts are suppressed; the pooled count is
    hidden whenever a site count is."""
    d = np.asarray(d, float)
    sites = np.asarray(sites).astype(str)
    w = None if w is None else np.asarray(w, float)
    keep = np.isfinite(d) if w is None else (np.isfinite(d) & np.isfinite(w) & (w >= 0))
    d, sites = d[keep], sites[keep]
    w = None if w is None else w[keep]
    n = int(len(d))
    per_n = {s: int((sites == s).sum()) for s in site_labels}
    hide_pooled = n < K or any(c < K for c in per_n.values())
    out: dict = {"n": SUPPRESSED if hide_pooled else n,
                 "per_site_n": {site_labels[s]: suppress_count(c) for s, c in per_n.items()}}
    if n < min_rows:
        return {**out, "estimable": False, "not_estimable": f"fewer than {min_rows} scored patients"}
    small = [site_labels[s] for s, c in per_n.items() if c < min_cell]
    if small:
        return {**out, "estimable": False, "not_estimable": f"fewer than {min_cell} scored patients at a site"}
    est = float(d.mean()) if w is None else float((w * d).sum() / w.sum())
    reps = _boot_means(d, w, sites, boot, seed)
    per_site = {}
    for s in site_labels:
        msk = sites == s
        per_site[site_labels[s]] = {"n": suppress_count(int(msk.sum())),
                                    "delta": (float(d[msk].mean()) if w is None else float((w[msk] * d[msk]).sum() / w[msk].sum()))
                                    if int(msk.sum()) >= K else SUPPRESSED}
    fav = all(isinstance(v["delta"], float) and v["delta"] < 0 for v in per_site.values())
    c99, c95 = _ci(reps, alpha_primary), _ci(reps, alpha_secondary)
    out.update(estimable=True, delta=est, ci_99_within_site=c99, ci_95_within_site=c95, boot_reps=boot, per_site=per_site,
               all_sites_favorable=bool(fav), pattern_met=bool(fav and c99 is not None and c99["hi"] < 0 and est < 0))
    if w is not None:
        out["ess"] = round(ess(w), 2)
        out["ess_over_scored"] = round(ess(w) / n, 4)
        out["unweighted_delta"] = float(d.mean())
    return out


def ess(w) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0


def strata_delta(units: dict, masks: dict, site_labels: dict, **kw) -> dict:
    """{scheme: {stratum: {set key: delta block}}} from ``units[scheme][key] = (d, eval_idx, eval_sites)``: the existing held-out
    per-patient differences, subset to the stratum rows (no refit)."""
    out: dict = {}
    for split, per in units.items():
        out[split] = {}
        for nm, mask in masks.items():
            if mask is None:
                continue
            out[split][nm] = {}
            for key, (d, idx, ev_sites) in per.items():
                sel = np.asarray(mask, bool)[np.asarray(idx)]
                out[split][nm][key] = delta_block(np.asarray(d)[sel], np.asarray(ev_sites)[sel], site_labels, **kw)
    return out


# ================================================================================================== transport
def domain_design(*, age=None, sex=None, sites, site_labels: dict, hours, sed: dict | None, dur, ceeg, omop: dict | None) -> tuple[np.ndarray, list]:
    """Pre-t0, EEG-independent covariates of the domain model: age, sex (with missing indicators), site, calendar-day timing,
    infusion / PRN-only / none (Baseline A), ventilation and vasopressor proxies (when available) and the recording type.  Diagnosis
    codes and silver labels are not covariates.  Returns (matrix, names); missing values are median-imputed with an indicator."""
    n = len(sites)
    cols: dict = {}

    def add(name, v, impute=True):
        v = np.asarray(v, float)
        bad = ~np.isfinite(v)
        if bad.any():
            if impute:
                v = np.where(bad, np.nanmedian(v) if (~bad).any() else 0.0, v)
            cols[name + "_missing"] = bad.astype(float)
        cols[name] = v

    if age is not None:
        add("age_years", age)
    if sex is not None:
        add("sex_male", sex)
    for s in np.unique(sites)[1:]:
        cols[f"site_{site_labels.get(str(s), 'x')}"] = (np.asarray(sites) == s).astype(float)      # pseudonym, never the real site id
    day = calendar_day(hours)
    h = np.asarray(hours, float)
    add("calendar_day", np.minimum(day, 6.0))
    add("log1p_hours_from_visit_start", np.log1p(np.clip(np.minimum(h, 24.0 * 7), 0, None)))
    if sed is not None:
        on = np.asarray(sed["any_on_t0"], bool)
        recent = np.asarray(sed["any_6h"], bool) & ~on
        cols["infusion_at_t0"] = on.astype(float)
        cols["prn_only_6h"] = recent.astype(float)
    if omop is not None and "vent_24h" in omop:
        cols["ventilation_proxy_24h"] = np.asarray(omop["vent_24h"], float)
        cols["vasopressor_24h"] = np.asarray(omop["vasopressor_24h"], float)
    if ceeg is not None:
        cols["ceeg_task_folder"] = np.asarray(ceeg, float)
    d = np.asarray(dur, float)
    if np.isfinite(d).any():
        ok = np.isfinite(d)
        cols["recording_lt_1h"] = (ok & (d < DUR_SHORT_S)).astype(float)
        cols["recording_gt_12h"] = (ok & (d > DUR_LONG_S)).astype(float)
        cols["recording_unknown"] = (~ok).astype(float)
    names = list(cols)
    X = np.column_stack([cols[c] for c in names]) if names else np.zeros((n, 0))
    keepc = np.flatnonzero(X.std(axis=0) > 0)                  # drop constant columns
    return X[:, keepc], [names[i] for i in keepc]


def _auc(y, p) -> float:
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def crossfit_domain(X: np.ndarray, z: np.ndarray, sites, *, folds: int = 5, seed: int = 0, C: float = 1.0) -> np.ndarray:
    """Out-of-fold P(z = 1 | X): folds stratified by (site, z); each row is predicted by a model that never saw it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z = np.asarray(z, bool).astype(int)
    codes = pd.factorize(np.asarray(sites).astype(str))[0]
    strat = codes * 2 + z
    bc = np.bincount(strat)
    k = int(min(folds, bc[bc > 0].min() if len(strat) else 0))
    if k < 2:
        k = int(min(folds, np.bincount(z).min()))
        strat = z
    if k < 2:
        raise ValueError("too few rows in a domain class for cross-fitting")
    p = np.full(len(z), np.nan)
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, strat):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        mdl.fit(X[tr], z[tr])
        p[te] = mdl.predict_proba(X[te])[:, 1]
    return p


def inverse_odds_weights(p: np.ndarray, z: np.ndarray, trim: tuple = (1.0, 99.0), eps: float = 1e-3) -> dict:
    """Stabilised inverse-odds weights toward the IU rows (z = 1): non-IU rows w = [p / (1 - p)] / [pi / (1 - pi)] with pi the IU
    share of the rows, trimmed at the ``trim`` percentiles of those weights; IU rows 1.  Returns {'w', 'raw', 'trimmed'}."""
    z = np.asarray(z, bool)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    pi = float(z.mean())
    raw = (p / (1 - p)) / (pi / (1 - pi))
    w = np.ones(len(z))
    lo, hi = np.percentile(raw[~z], trim)
    w[~z] = np.clip(raw[~z], lo, hi)
    trimmed = ~z & ((raw < lo) | (raw > hi))
    return {"w": w, "raw": raw, "trimmed": trimmed, "bounds": (float(lo), float(hi))}


def _smd(x: np.ndarray, z: np.ndarray, w: np.ndarray | None = None) -> float:
    a, b = x[z], x[~z]
    wb = np.ones(len(b)) if w is None else w[~z]
    mb = float((wb * b).sum() / wb.sum())
    va, vb = float(a.var()), float((wb * (b - mb) ** 2).sum() / wb.sum())
    sd = np.sqrt((va + vb) / 2.0)
    return float((a.mean() - mb) / sd) if sd > 0 else 0.0


def transport_weights(X: np.ndarray, names: list, z, sites, site_labels: dict, *, folds: int = 5, seed: int = 0,
                      min_rows: int = MIN_ROWS) -> dict:
    """Cross-fitted domain model for IU-large vs the rest, the stabilised trimmed weights, and an aggregate diagnostic block.
    Returns {'info': aggregate dict, 'w': weights (n,) or None}."""
    z = np.asarray(z, bool)
    n1, n0 = int(z.sum()), int((~z).sum())
    info: dict = {"target": IU_LARGE, "covariates": list(names),
                  "weights": "stabilised inverse-odds, w = [p/(1-p)] / [pi/(1-pi)] for non-IU rows, IU rows 1; trimmed at the 1st / 99th percentile",
                  "rows": {"iu": suppress_count(n1), "other": suppress_count(n0)}}
    if n1 < min_rows or n0 < min_rows:
        return {"info": {**info, "estimable": False, "not_estimable": f"fewer than {min_rows} rows in the IU-large or the other class"}, "w": None}
    try:
        p = crossfit_domain(X, z, sites, folds=folds, seed=seed)
    except ValueError as e:
        return {"info": {**info, "estimable": False, "not_estimable": str(e)}, "w": None}
    wt = inverse_odds_weights(p, z)
    w = wt["w"]
    other_w = w[~z]
    ok = np.isfinite(w).all() and (w > 0).all()
    info.update(
        estimable=bool(ok), crossfit_folds=folds, crossfit_auc=round(_auc(z, p), 4), iu_share=round(n1 / (n1 + n0), 4),
        ess_all=round(ess(w), 2), ess_other=round(ess(other_w), 2),
        ess_other_over_rows=round(ess(other_w) / n0, 4),
        trimmed_other=suppress_count(int(wt["trimmed"].sum())),
        trim_percentiles=[1, 99],
        weight_quantiles_other=safe_quantiles(other_w),
        weight_top5pct_share_other=round(float(np.sort(other_w)[::-1][: max(1, int(np.ceil(0.05 * n0)))].sum() / other_w.sum()), 4),
        domain_probability_quantiles={"iu": safe_quantiles(p[z]), "other": safe_quantiles(p[~z])},
        overlap_reading=("IU-large is a deterministic rule on covariates the model sees, so p is near 0 for many other rows: read the ESS, "
                         "the share carried by the top 5% of weights and the balance below before the weighted Delta"))
    bal = {}
    for j, nm in enumerate(names):
        x = X[:, j]
        bal[nm] = {"smd_before": round(_smd(x, z), 3), "smd_after": round(_smd(x, z, w), 3)}
    info["balance_iu_minus_other"] = bal
    info["max_abs_smd_before"] = round(max((abs(v["smd_before"]) for v in bal.values()), default=0.0), 3)
    info["max_abs_smd_after"] = round(max((abs(v["smd_after"]) for v in bal.values()), default=0.0), 3)
    return {"info": info, "w": w if ok else None}


def transport_delta(units: dict, w: np.ndarray, site_labels: dict, **kw) -> dict:
    """{scheme: {set key: weighted delta block}}: the held-out per-patient differences of the scored rows, weighted with the domain
    weights of those rows (no refit)."""
    out: dict = {}
    for split, per in units.items():
        out[split] = {}
        for key, (d, idx, ev_sites) in per.items():
            ww = np.asarray(w, float)[np.asarray(idx)]
            out[split][key] = delta_block(d, ev_sites, site_labels, ww, **kw)
    return out


# ============================================================================================== orchestration
SET_LABELS = {"prior": "a. EEG-only vs prevalence prior", "agesex": "b. EEG + age/sex vs age/sex", "P": "c. P (Presentation) + EEG vs P",
              "baseline_A": "Baseline A (H1-style) + EEG vs A", "baseline_C": "Baseline C (H2-style) + EEG vs C"}
TITLE = "Presentation-based intended-use strata (IU-large, IU-small) and transport weighting"


def presentation_block(*, baseline: pd.DataFrame, frame: pd.DataFrame, hours, hours_src: str, dur, omop: pd.DataFrame | None,
                       omop_info: dict, y: np.ndarray, m: np.ndarray, label_names, sites, site_labels: dict, units: dict,
                       boot: int = 2000, seed: int = 0, alpha_primary: float = 0.01, alpha_secondary: float = 0.05, folds: int = 5) -> dict:
    """The whole aggregate block: stratum sizes and per-label positives, Delta by set within IU-large / IU-small (no refit), and the
    transport re-weighting.  ``units[scheme][set key] = (d, eval_idx, eval_sites)`` are the held-out per-patient differences of the
    headline rung (sets: prior / agesex / P of the intended-use block, baseline_A / baseline_C of the headline ladder)."""
    sites = np.asarray(sites).astype(str)
    order = sorted(site_labels)
    sed = sedation_features(baseline) if len(baseline.columns) else None
    om = {c: omop[c].to_numpy(bool) for c in omop.columns} if omop is not None and len(omop.columns) else None
    ceeg = ceeg_task_flag(frame)
    sites = np.asarray(sites).astype(str)
    res = intended_use_strata(hours, sed, dur, ceeg, om)
    kw = dict(boot=boot, seed=seed, alpha_primary=alpha_primary, alpha_secondary=alpha_secondary)
    out: dict = {"title": TITLE, "sources": {"visit_start": hours_src, "omop": omop_info,
                                             "infusion_at_t0": "Baseline A sed__ on_t0 flags (class or named agent)"},
                 "set_labels": {k: v for k, v in SET_LABELS.items() if any(k in per for per in units.values())},
                 "summary": stratum_summary(res, y, m, label_names, sites, order, site_labels),
                 "delta_reading": ("Delta = masked log loss(set + EEG) - masked log loss(set); negative = EEG helps. Scored from the SAME held-out "
                                   "predictions as the whole-set rows (models trained on all training-fold rows; no refit); within-site patient "
                                   "bootstrap over the stratum rows; a stratum with fewer than "
                                   f"{MIN_ROWS} scored patients or fewer than {K} at a site is not estimable."),
                 "delta": strata_delta(units, res["masks"], site_labels, **kw)}
    lg = res["masks"][IU_LARGE]
    if lg is None:
        out["transport"] = {"estimable": False, "not_estimable": "IU-large is not available"}
        return out
    X, names = domain_design(age=_col(baseline, "demo__age_years").to_numpy(float) if "demo__age_years" in baseline else None,
                             sex=_col(baseline, "demo__sex_male").to_numpy(float) if "demo__sex_male" in baseline else None,
                             sites=sites, site_labels=site_labels, hours=hours, sed=sed, dur=dur, ceeg=ceeg, omop=om)
    tw = transport_weights(X, names, lg, sites, site_labels, folds=folds, seed=seed)
    out["transport"] = {"info": tw["info"], "estimable": tw["w"] is not None}
    if tw["w"] is not None:
        out["transport"]["delta"] = transport_delta(units, tw["w"], site_labels, **kw)
        out["transport"]["reading"] = ("Weighted Delta = sum(w d) / sum(w) over the scored rows (IU-large rows weight 1, other rows the "
                                       "stabilised trimmed inverse-odds weight), within-site bootstrap with the weights held fixed. "
                                       "Read it only with its effective sample size and the balance table; it assumes the EEG increment "
                                       "depends on the rows only through the domain covariates.")
    return out
