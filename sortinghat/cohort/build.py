"""Study 1 cohort builder (plan: "Population and index time"; SAP section 2).

One row per patient. Every step is counted per site in a ``FlowRecorder``; ``build_cohort`` returns the
record-level table and key list (IN MEMORY - write them only with ``cohort.output.write_outputs``) and the
suppressed aggregate flow report.

Order of steps (see ``docs/cohort_spec.md`` for the operational choice behind each, C-nn):

  EEG sessions -> patient id -> start time -> age known -> adult -> visit covering start -> visit setting
  classified -> acute-care setting -> not OR/EMU service -> FIRST QUALIFYING EEG PER PATIENT (sessions become
  patients) -> recording covers minutes 1-11 -> ACI onset proxy exists -> EEG within the widest onset window
  (48 h) -> strict severity (primary window or +-6 h sensitivity) or EHR phenotype.

``in_strict`` / ``in_broad`` are the PRIMARY cohorts (EEG within 24 h of onset); the 6 / 12 / 48 h sensitivity
windows are flags on the same rows. The broad cohort contains the strict cohort ("broad only" = broad and not
strict is the separately reported part). t0 = EEG start (``StartTime(EEG)``); the EEG QC step (minimum channel
set, >= 60% usable) belongs to the streaming extractor and is appended to the flow after it has run.

No printing here; record-level frames are returned, never emitted (CLAUDE.md rule 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import data_io
from ..safe_output import safe_quantiles, suppress_count
from . import rules
from .config import CohortConfig
from . import integrity
from .flow import FlowRecorder, debug_report, flow_report

KEY_LIST_COLUMNS = ["SiteID", "person_id", "SessionID", "BidsFolder", "EEGFolder", "edf_key",
                    "task_token_assumed", "window_start_s", "window_duration_s", "clock_duration_s", "in_strict", "in_strict_pm6",
                    "in_broad"]
INCLUDED = "included"
LATER_SESSION = "Not the patient's first qualifying EEG"
SESS, PAT = "EEG sessions", "patients"


@dataclass
class CohortResult:
    table: pd.DataFrame                 # RECORD-LEVEL: local_only
    keys: pd.DataFrame                  # RECORD-LEVEL: local_only
    fates: pd.Series                    # RECORD-LEVEL, in memory only: person_id -> last step reached / "included"
    flow_raw: dict                      # exact counts, in memory only
    report: dict                        # suppressed, aggregate-only
    config: dict = field(default_factory=dict)
    merge_status: str = "absent"        # patient merge history: applied | absent | unrecognised
    debug: dict = field(default_factory=dict)   # unmerged diagnostic report (NOT for sharing)
    stages: dict = field(default_factory=dict)  # RECORD-LEVEL intermediate frames, in memory only (diagnostics)


def _hours(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds() / 3600.0


def _duration(S: pd.DataFrame, cfg: CohortConfig) -> tuple[pd.Series, pd.Series]:
    """Recording duration in seconds = EndTime - StartTime (the clock; D-115). The metadata duration
    (``DurationInSeconds`` / ``RecordingDuration``) is not used: it disagrees with the clock at I0003. The streaming
    extractor confirms the length from the EDF header (n_records x record duration) and drops windows that do not fit."""
    clock = (S["t_end"] - S["t0"]).dt.total_seconds()
    clock = clock.where(clock > 0)
    return clock, pd.Series(np.where(clock.notna(), "clock", "none"), index=S.index)


def duration_unit_check(S: pd.DataFrame, sites: list[str]) -> dict:
    """INFORMATIONAL (the cohort uses the clock duration, D-115). Aggregate check of the metadata duration: quartiles of metadata duration over the
    EndTime - StartTime clock duration per site (about 1 if the unit really is seconds)."""
    out = {}
    clock = (S["t_end"] - S["t0"]).dt.total_seconds()
    meta = pd.to_numeric(S["duration_raw_s"], errors="coerce")
    ok = (clock > 0) & (meta > 0)
    ratio = meta / clock
    for s in sites:
        q = safe_quantiles(ratio[ok & (S["SiteID"] == s)].to_numpy(float), qs=(0.25, 0.5, 0.75))
        med = q["q50"]
        warn = "" if isinstance(med, str) or 0.8 <= med <= 1.25 else " ; UNIT WARNING: median far from 1"
        out[f"duration_over_clock_{s}"] = f"{q}{warn}"
    return out


class _Run:
    """Working state of one build: the shrinking session/patient frame plus the record-level reasons."""

    def __init__(self, S: pd.DataFrame, flow: FlowRecorder):
        self.S = S.assign(_sess_idx=S.index)
        self.flow = flow
        self.reason = pd.Series("", index=S.index, dtype=object)
        self.stage = pd.Series(-1, index=S.index, dtype=int)
        self.n = 0

    def drop(self, label: str, keep: pd.Series, unit: str) -> None:
        """Exclude the rows where ``keep`` is False; count the step per site; remember the reason per session."""
        keep = keep.fillna(False).astype(bool).reindex(self.S.index, fill_value=False)
        self.n += 1
        gone = self.S.loc[~keep, "_sess_idx"].to_numpy()
        self.reason.loc[gone] = label
        self.stage.loc[gone] = self.n
        self.S = self.S[keep.to_numpy()]
        self.flow.step(label, unit, self.S["SiteID"])


MERGE_NOTICE = {
    "applied": "patient merge history applied before first-EEG selection",
    "absent": "NO patient merge history table found: patient identity = BDSPPatientID as given (D-106)",
    "unrecognised": "patient merge history present but its old/new id columns were not recognised: NOT applied (D-106)"}


def build_cohort(src, cfg: CohortConfig | None = None) -> CohortResult:
    cfg = cfg or CohortConfig()
    S0 = src.sessions().reset_index(drop=True)
    mm, merge_status = src.merge_map() if hasattr(src, "merge_map") else ({}, "absent")
    S0["person_id_source"] = S0["person_id"]
    src_index = src.source_index() if hasattr(src, "source_index") else None
    if src_index is not None:
        integrity.verify_rows(S0, src_index, "sessions loaded from eeg_metadata")
    n_remapped = 0
    if mm and len(S0):
        mapped = S0["person_id"].map(mm)
        n_remapped = int(mapped.notna().sum())
        S0["person_id"] = mapped.where(mapped.notna(), S0["person_id"]).astype("Int64")
    rev: dict[int, list[int]] = {}
    for old, new in mm.items():
        rev.setdefault(new, []).append(old)

    def fetch(fn, pids, **kw):
        """Fetch OMOP rows for the surviving ids AND the ids merged into them; the source re-keys them to the
        surviving id (per chunk when streaming)."""
        ids = {int(p) for p in pids}
        ids |= {o for p in list(ids) for o in rev.get(p, [])}
        return fn(sorted(ids), remap=mm or None, **kw)

    sites = sorted(S0["SiteID"].astype(str).unique())
    flow = FlowRecorder(sites)
    run = _Run(S0, flow)
    flow.start("EEG sessions in HEEDB metadata", SESS, S0["SiteID"])
    flow.checks.update(duration_unit_check(S0, sites))
    flow.checks["merge_history_status"] = MERGE_NOTICE[merge_status] + (
        f"; sessions re-keyed: {suppress_count(n_remapped)}" if merge_status == "applied" else "")
    n_unstamped = S0[S0["person_id"].notna() & S0["t0"].isna()].groupby("person_id").size()

    # ------------------------------------------------------------------ session-level qualification
    flow.excluded_sites = [x for x in sites if cfg.study_sites is not None and x not in cfg.study_sites]
    if cfg.study_sites is not None:
        run.drop("Site is not a Study 1 site (no EHR rows; excluded from labelled analyses)",
                 run.S["SiteID"].astype(str).isin(cfg.study_sites), SESS)
    run.drop("Patient id not resolvable", run.S["person_id"].notna(), SESS)
    run.drop("EEG start time missing", run.S["t0"].notna(), SESS)
    run.drop("Age missing", run.S["age_years"].notna(), SESS)
    run.drop(f"Age < {cfg.adult_age_years:g} y", run.S["age_years"] >= cfg.adult_age_years, SESS)

    S = run.S.copy()
    S["person_id"] = S["person_id"].astype("int64")
    bounds = S.groupby("person_id")["t0"].agg(lo="min", hi="max").astype("datetime64[s]")
    visits = fetch(src.visits, S["person_id"].unique(), bounds=bounds, slack_h=cfg.visit_slack_h,
                   dates_only=cfg.visit_dates_only)
    run.S = S.join(rules.match_visits(S, visits, cfg.acute_classes, cfg.visit_chain_gap_h, cfg.visit_slack_h,
                                      cfg.open_visit_days, cfg.date_only_end_of_day, dates_only=cfg.visit_dates_only))
    del visits
    run.S["ServiceName"] = run.S["ServiceName"].astype("string").str.strip().str.upper()
    run.drop("No visit covering the EEG start", run.S["visit_start"].notna(), SESS)

    # acute-care proxy (D-111): a visit classified by concept id / text (no-op while all ids are 0) decides by its class;
    # otherwise the covering visit is inpatient-length OR ServiceName names an acute setting
    S = run.S
    by_class, by_length, by_service = rules.acute_parts(S["visit_class"], S["visit_inpatient_length"], S["ServiceName"],
                                                        cfg.acute_classes, cfg.service_acute, cfg.use_service_proxy)
    run.drop("Not acute care (visit shorter than a day and no acute ServiceName)", by_class | by_length | by_service, SESS)
    run.S = run.S.assign(acute_basis=np.where(by_class[run.S.index], "visit_class",
                                              np.where(by_length[run.S.index], "visit_length", "service")))
    run.drop("OR or EMU service (not an ACI work-up)",
             ~run.S["ServiceName"].isin([x.upper() for x in cfg.exclude_services]).fillna(False), SESS)

    # ------------------------------------------------------------------ first qualifying EEG per patient
    run.S = run.S.sort_values(["person_id", "t0", "SessionID"], kind="stable")
    run.drop(LATER_SESSION, ~run.S["person_id"].duplicated(keep="first"), PAT)
    stages = {"first_eeg": run.S[["person_id", "SiteID", "t0", "encounter_start", "visit_start", "ServiceName",
                                  "acute_basis", "visit_inpatient_length"]].reset_index(drop=True)}

    # ------------------------------------------------------------------ recording must cover minutes 1-11
    dur, basis = _duration(run.S, cfg)
    run.S = run.S.assign(duration_s=dur, duration_basis=basis)
    run.drop("Recording duration unknown", run.S["duration_s"].notna(), PAT)
    run.drop(f"Recording shorter than {cfg.min_duration_s / 60:g} min", run.S["duration_s"] >= cfg.min_duration_s, PAT)

    # ------------------------------------------------------------------ ACI onset proxy and time since onset
    run.S = run.S.reset_index(drop=True)          # unique, positional index for the row-level joins below
    pad_b, pad_a = (max(cfg.score_before_h, cfg.pm6_window_h), max(cfg.score_after_h, cfg.pm6_window_h))
    t0s, enc = run.S["t0"], run.S["encounter_start"]
    swin = pd.DataFrame({"lo": np.minimum(enc.to_numpy(), (t0s - pd.Timedelta(hours=pad_b)).to_numpy()),
                         "hi": (t0s + pd.Timedelta(hours=pad_a)).to_numpy()}, index=run.S["person_id"].to_numpy()
                        ).astype("datetime64[s]")
    scores = rules.extract_scores(fetch(src.scores, run.S["person_id"].unique(), window=swin))
    on = rules.onset_times(run.S, scores, cfg.onset_rule, cfg.abnormal_gcs_max, cfg.abnormal_four_max)
    run.S = run.S.assign(onset=on["onset"], onset_basis=on["onset_basis"])
    run.S["hours_since_onset"] = _hours(run.S["t0"], run.S["onset"])
    stages["onset"] = run.S[["person_id", "SiteID", "hours_since_onset", "onset_basis"]].reset_index(drop=True)
    run.drop("No ACI onset proxy", run.S["onset"].notna(), PAT)
    run.drop(f"EEG more than {cfg.onset_max_h:g} h after onset", run.S["hours_since_onset"] <= cfg.onset_max_h, PAT)
    if (run.S["hours_since_onset"] < 0).any():                  # impossible by construction (onset <= t0)
        raise AssertionError("onset after t0")

    # ------------------------------------------------------------------ strict severity / broad phenotype
    sev = rules.severity(run.S, scores, cfg.score_before_h, cfg.score_after_h, cfg.gcs_strict_max,
                         cfg.four_strict_max, cfg.score_rule)
    pm6 = rules.severity(run.S, scores, cfg.pm6_window_h, cfg.pm6_window_h, cfg.gcs_strict_max,
                         cfg.four_strict_max, "any")
    run.S = run.S.join(sev.rename(columns={"gcs_min": "gcs_min_window", "four_min": "four_min_window",
                                           "gcs_nearest": "gcs_nearest_window", "four_nearest": "four_nearest_window",
                                           "n_score_obs": "n_score_obs_window", "strict": "severity_strict"}))
    run.S["severity_strict_pm6"] = pm6["strict"]
    cwin = pd.DataFrame({"lo": run.S["encounter_start"].to_numpy(),
                         "hi": (run.S["t0"] + pd.Timedelta(hours=cfg.phenotype_after_h)).to_numpy()},
                        index=run.S["person_id"].to_numpy()).astype("datetime64[s]")
    cond = fetch(src.conditions, run.S["person_id"].unique(), window=cwin)
    run.S["phenotype"] = rules.phenotype(run.S, cond, cfg.phenotype_after_h)
    run.drop("Neither strict severity (GCS/FOUR, primary or +-6 h) nor EHR phenotype",
             run.S["severity_strict"] | run.S["severity_strict_pm6"] | run.S["phenotype"], PAT)

    # ------------------------------------------------------------------ flags and the final table
    S = run.S.copy()
    for h in cfg.onset_windows_h:
        S[f"onset_le_{h:g}h"] = S["hours_since_onset"] <= h
    prim = S[f"onset_le_{cfg.onset_primary_h:g}h"]
    S["in_strict"] = S["severity_strict"] & prim
    S["in_strict_pm6"] = S["severity_strict_pm6"] & prim
    S["in_broad"] = (S["severity_strict"] | S["phenotype"]) & prim
    S["n_unstamped_sessions"] = S["person_id"].map(n_unstamped).fillna(0).astype(int)
    fill = pd.Series([data_io.bids_folder_for(s, p) for s, p in zip(S["SiteID"], S["person_id_source"])], index=S.index)
    S["bids_filled"] = S["BidsFolder"].isna()
    S["BidsFolder"] = S["BidsFolder"].astype(object).where(S["BidsFolder"].notna(), fill)
    run.reason.loc[S["_sess_idx"].to_numpy()] = INCLUDED
    run.stage.loc[S["_sess_idx"].to_numpy()] = 10**6

    cols = ["SiteID", "person_id", "person_id_source", "SessionID", "BidsFolder", "EEGFolder", "t0", "age_years", "ServiceName",
            "bids_filled", "visit_class", "visit_match", "visit_inpatient_length", "acute_basis", "duration_s", "duration_basis", "onset", "onset_basis",
            "hours_since_onset", *[f"onset_le_{h:g}h" for h in cfg.onset_windows_h], "gcs_min_window",
            "four_min_window", "gcs_nearest_window", "four_nearest_window", "n_score_obs_window", "severity_strict",
            "severity_strict_pm6", "phenotype", "in_strict", "in_strict_pm6", "in_broad",
            "n_unstamped_sessions"]
    table = S[cols].sort_values("person_id").reset_index(drop=True)
    for c in ("SiteID", "SessionID", "BidsFolder", "EEGFolder", "ServiceName"):
        table[c] = table[c].astype(object)

    _partitions(flow, table, cfg)
    keys = make_key_list(table, cfg)
    if src_index is not None:
        integrity.verify_rows(table, src_index, "final cohort table", skip_bids=table["bids_filled"])
    integrity.verify_output(table, keys)
    fates = _fates(S0, run.reason, run.stage)
    raw = flow.raw()
    return CohortResult(table, keys, fates, raw, flow_report(raw, cfg.to_dict()), cfg.to_dict(), merge_status,
                        debug_report(raw, cfg.to_dict()), stages)


def _fates(S0: pd.DataFrame, reason: pd.Series, stage: pd.Series) -> pd.Series:
    """person_id -> reason of the session that got furthest (or 'included'). Record-level; in memory only."""
    d = pd.DataFrame({"person_id": S0["person_id"], "reason": reason, "stage": stage}).dropna(subset=["person_id"])
    d["person_id"] = d["person_id"].astype("int64")
    best = d.sort_values(["person_id", "stage"], kind="stable").drop_duplicates("person_id", keep="last")
    return best.set_index("person_id")["reason"]


def _partitions(flow: FlowRecorder, t: pd.DataFrame, cfg: CohortConfig) -> None:
    """Disjoint-and-exhaustive splits of the final table (suppressed with complementary suppression)."""
    site = t["SiteID"]
    p = t[f"onset_le_{cfg.onset_primary_h:g}h"]
    flow.partition("Cohort membership of table rows", {
        "strict, primary window": site[t["in_strict"]],
        "strict_pm6 only, primary window": site[t["in_strict_pm6"] & ~t["in_strict"]],
        "broad only, primary window": site[t["in_broad"] & ~t["in_strict"] & ~t["in_strict_pm6"]],
        "sensitivity windows only": site[~p]})
    flow.partition("Strict definitions, primary onset window", {
        "both definitions": site[t["in_strict"] & t["in_strict_pm6"]],
        "primary only (-6 h to +1 h)": site[t["in_strict"] & ~t["in_strict_pm6"]],
        "strict_pm6 only (+-6 h)": site[~t["in_strict"] & t["in_strict_pm6"]]})
    ws = list(cfg.onset_windows_h)
    bins, lo = {}, -1e-9
    for h in ws:
        bins[f"{lo if lo > 0 else 0:g} to {h:g} h"] = site[(t["hours_since_onset"] > lo) & (t["hours_since_onset"] <= h)]
        lo = h
    flow.partition("Hours from onset proxy to EEG start (table rows)", bins)
    flow.partition("Onset proxy basis (table rows)", {
        "first abnormal score": site[t["onset_basis"] == "abnormal_score"],
        "ED arrival or admission": site[t["onset_basis"] == "visit_start"]})
    flow.partition("Acute-care basis (table rows)", {
        "visit class": site[t["acute_basis"] == "visit_class"], "visit longer than a day": site[t["acute_basis"] == "visit_length"],
        "ServiceName only": site[t["acute_basis"] == "service"]})
    flow.partition("First-EEG order uncertain: an unstamped EEG exists (table rows)", {
        "yes": site[t["n_unstamped_sessions"] > 0], "no": site[t["n_unstamped_sessions"] == 0]})


def make_key_list(t: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Recording key list the streaming extractor reads (RECORD-LEVEL; local_only). One row per table row.

    ``edf_key`` is the BIDS EDF key under the access point (``data_io.bids_edf_key``). ``EEGFolder`` exists only in
    the S0001/S0002 headers, so elsewhere the task token defaults to 'EEG' (``task_token_assumed`` = True: UNVERIFIED
    for continuous-EEG sessions at the I-sites). The window is minutes 1-11 (60-660 s from recording start)."""
    edf = [data_io.edf_key_for_row(s, b, sid, ef)                     # the extractor's convention, one function
           for s, b, sid, ef in zip(t["SiteID"], t["BidsFolder"], t["SessionID"], t["EEGFolder"])]
    k = pd.DataFrame({
        "SiteID": t["SiteID"], "person_id": t["person_id"], "SessionID": t["SessionID"],
        "BidsFolder": t["BidsFolder"], "EEGFolder": t["EEGFolder"], "edf_key": edf,
        "task_token_assumed": t["EEGFolder"].isna(),
        "window_start_s": cfg.window_start_s, "window_duration_s": cfg.window_duration_s,
        "clock_duration_s": t["duration_s"],
        "in_strict": t["in_strict"], "in_strict_pm6": t["in_strict_pm6"], "in_broad": t["in_broad"]})
    return k[KEY_LIST_COLUMNS].reset_index(drop=True)
