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
SCORE_RE = re.compile(r"glasgow|\bgcs\b|rass|richmond|sedation scale|level of consciousness|eye opening|"
                      r"best motor response|best verbal response|ramsay|arousal|\bfour score", re.I)
LAB_RE = re.compile(r"lactate|ammonia|sodium|glucose|creatinine|\bbun\b|\bast\b|\balt\b|\bwbc\b|enolase", re.I)
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


def merge_eeg(meta: pd.DataFrame, findings: pd.DataFrame | None, site: str) -> pd.DataFrame:
    """One site's analytic EEG frame: eeg_metadata joined to reports_findings on (person, SessionID).

    Real start/end are ``StartTime(EEG)`` / ``EndTime(EEG)`` in the findings table (the metadata ones are
    blank); age prefers the findings value, falling back to the sparse metadata value.
    """
    m = pd.DataFrame({"SiteID": site, "person_id": person_ids(meta, site),
                      "SessionID": meta["SessionID"].astype("string")})
    m["AgeAtVisit"] = pd.to_numeric(meta.get("AgeAtVisit"), errors="coerce")
    m["PatientClass"] = meta["PatientClass"] if "PatientClass" in meta else None       # ASSUMED column
    m[["StartTime", "EndTime"]] = meta[["StartTime", "EndTime"]] if {"StartTime", "EndTime"} <= set(meta) else pd.NaT
    if findings is not None and len(findings):
        f = pd.DataFrame({"person_id": person_ids(findings, site), "SessionID": findings["SessionID"].astype("string"),
                          "f_start": findings.get(START), "f_end": findings.get(END),
                          "f_age": pd.to_numeric(findings.get("AgeAtVisit"), errors="coerce")})
        f = f.sort_values("f_start", na_position="last").drop_duplicates(["person_id", "SessionID"])
        m = m.merge(f, on=["person_id", "SessionID"], how="left")
        m["AgeAtVisit"] = m["f_age"].fillna(m["AgeAtVisit"])
        m["StartTime"] = m["f_start"].fillna(m["StartTime"])
        m["EndTime"] = m["f_end"].fillna(m["EndTime"])
        m = m.drop(columns=["f_start", "f_end", "f_age"])
    return m


def filter_drugs(d: pd.DataFrame) -> pd.DataFrame:
    if "drug_source_value" not in d:
        return d.iloc[0:0]
    return d[d["drug_source_value"].astype("string").str.contains(SEDATION_RE, na=False)]


def filter_measurements(d: pd.DataFrame) -> pd.DataFrame:
    """Keep score and lab rows; ``kind`` is derived from the free-text ``measurement_source_value``."""
    if "measurement_source_value" not in d:
        return d.iloc[0:0].assign(kind=pd.Series(dtype=object))
    v = d["measurement_source_value"].astype("string")
    score, lab = v.str.contains(SCORE_RE, na=False), v.str.contains(LAB_RE, na=False)
    out = d[score | lab].copy()
    out["kind"] = np.where(score[score | lab], "score", "lab")
    return out


def from_raw_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Analytic tables from whole raw tables in the real layout (synthetic scale; same shapes as the
    streaming loader produces)."""
    meta, rf = raw["eeg_metadata"], raw.get("reports_findings")
    parts = []
    for site, g in meta.groupby("SiteID", sort=True):
        sess = set(g["SessionID"].astype(str))
        fg = rf[rf["SessionID"].astype(str).isin(sess)] if rf is not None else None
        parts.append(merge_eeg(g, fg, str(site)))
    out = {"eeg_metadata": pd.concat(parts, ignore_index=True),
           "omop_drug_exposure": filter_drugs(raw["omop_drug_exposure"]),
           "omop_measurement": filter_measurements(raw["omop_measurement"])}
    for t in ("omop_note", "imaging"):
        out[t] = raw.get(t, pd.DataFrame())
    return out


def load_audit_tables(s3, sites: list[str] | None = None, avail: dict | None = None) -> dict[str, pd.DataFrame]:
    """Read what the audit needs from a store (S3 client or ``LocalStore``) in the real layout.

    EEG tables are read whole (one row per session). OMOP/imaging parquet is streamed part by part,
    filtered to the candidate cohort inside Arrow and by regex, so memory scales with the cohort.
    """
    sites = sites or data_io.discover_sites(s3)
    parts = []
    for site in sites:
        try:
            meta = data_io.read_csv_table("eeg_metadata", site, s3=s3)
        except FileNotFoundError:
            continue
        try:
            rf = data_io.read_csv_table("reports_findings", site, s3=s3)
        except FileNotFoundError:
            rf = None
        meta = schema.coerce_types("eeg_metadata", meta)
        rf = None if rf is None else schema.coerce_types("reports_findings", rf)
        parts.append(merge_eeg(meta, rf, site))
    if not parts:
        raise FileNotFoundError("no eeg_metadata CSV found for any site")
    eeg = pd.concat(parts, ignore_index=True)
    cohort = [int(p) for p in build_candidates(eeg)["person_id"].dropna().unique()]
    out = {"eeg_metadata": eeg}
    result_cols = [c for a in schema.COLUMN_ALIASES["measurement.result_datetime"] for c in [a]]
    for table in ("omop_drug_exposure", "omop_measurement", "omop_note", "imaging"):
        spec = data_io.TABLES[table]
        cols = schema.columns(table) + (result_cols if table == "omop_measurement" else [])
        kw = {"prefix": spec.pattern} if spec.kind == "parquet_dir" else {}
        tname = table[len("omop_"):] if spec.kind == "omop_parquet" else table
        frames = []
        for batch in data_io.iter_omop_batches(tname, person_ids=cohort, columns=cols, s3=s3, **kw):
            frames.append(batch.to_pandas())
        d = schema.coerce_types(table, pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(
            columns=schema.columns(table))
        if table == "omop_drug_exposure":
            d = filter_drugs(d)
        elif table == "omop_measurement":
            d = filter_measurements(d)
            for c in result_cols:
                if c in d:
                    d[c] = schema.parse_datetimes(d[c])
        out[table] = d
    return out


# ---------------------------------------------------------------- helpers
def build_candidates(eeg: pd.DataFrame) -> pd.DataFrame:
    """First qualifying EEG per adult patient (acute care, start time present)."""
    e = eeg[(eeg["AgeAtVisit"] >= ADULT_AGE) & eeg["PatientClass"].isin(ACUTE_CLASSES)
            & eeg["StartTime"].notna()]
    e = e.sort_values(["BDSPPatientID", "StartTime"], kind="stable")
    first = e.groupby("BDSPPatientID", as_index=False).first()
    return first[["BDSPPatientID", "SiteID", "StartTime"]].rename(columns={"StartTime": "t0"})


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
    t0 = cands.set_index("BDSPPatientID")["t0"]
    spec = {"notes": ["note_time"], "labs": ["collect_time"], "imaging": ["study_time"],
            "medications": ["order_time"]}
    mis = pd.Series(False, index=t0.index)
    has = pd.Series(False, index=t0.index)
    per_table = {}
    for name, cols in spec.items():
        d = tables[name][["BDSPPatientID", cols[0]]].dropna()
        d = d[d["BDSPPatientID"].isin(t0.index)]
        d["off"] = ((d[cols[0]] - d["BDSPPatientID"].map(t0)).dt.total_seconds().abs() / 86400.0)
        med = d.groupby("BDSPPatientID")["off"].median()
        bad = med > MISALIGN_DAYS
        per_table[name] = (int(bad.sum()), int(len(med)))
        mis.loc[med.index[bad]] = True
        has.loc[med.index] = True
    return mis[has], per_table


def _monotonicity(tables):
    pairs = {
        "eeg_start_before_end": (tables["eeg_metadata"], "StartTime", "EndTime"),
        "lab_collect_before_result": (tables["labs"], "collect_time", "result_time"),
        "med_order_before_admin": (tables["medications"], "order_time", "admin_time"),
        "imaging_study_before_final": (tables["imaging"], "study_time", "report_final_time"),
    }
    out, tot_n, tot_bad = {}, 0, 0
    for k, (df, a, b) in pairs.items():
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
    """Return ``(report, handcheck_ids)``. ``handcheck_ids`` are record-level: local file only."""
    eeg = tables["eeg_metadata"]
    cands = build_candidates(eeg)
    cand_ids = set(cands["BDSPPatientID"])
    site_of = cands.set_index("BDSPPatientID")["SiteID"]
    rows = []

    # 1. EEG start present (acute-care adult EEGs)
    acute = eeg[(eeg["AgeAtVisit"] >= ADULT_AGE) & eeg["PatientClass"].isin(ACUTE_CLASSES)]
    blk, raw = prop_block(acute["StartTime"].notna(), acute["SiteID"])
    rows.append(_row("EEG start date and time of day", "t0, every analysis",
                     "Present for >=95% of acute-care EEGs", "Stop; no study", True,
                     blk, raw >= 0.95, {"observed_text": _fmt(blk)}))

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

    # 3. Medication administration times
    m = tables["medications"]
    m = m[m["BDSPPatientID"].isin(cands["BDSPPatientID"]) & m["med_class"].isin(SEDATION_CLASSES)].copy()
    m["t0"] = m["BDSPPatientID"].map(cands.set_index("BDSPPatientID")["t0"])
    dt_h = (m["order_time"] - m["t0"]).dt.total_seconds() / 3600
    m = m[(dt_h >= -48) & (dt_h <= 1)]
    per_pt = m.groupby("BDSPPatientID")["admin_time"].apply(lambda s: s.notna().any())
    blk, raw = prop_block(per_pt, site_of.loc[per_pt.index])
    rows.append(_row("Medication administration times (not just orders)", "Sedation baseline, E4a/E4b",
                     "Administration times for >=80% of candidates (with a sedation-class order "
                     "in the 48 h before t0)", "Use orders; label 1B \"approximate\"", False,
                     blk, raw >= 0.80, {"observed_text": _fmt(blk)}))

    # 4. Lab result time
    lb = tables["labs"]
    lb = lb[lb["BDSPPatientID"].isin(cand_ids)]
    blk, raw = prop_block(lb["result_time"].notna(), lb["BDSPPatientID"].map(site_of))
    rows.append(_row("Lab result or verification time", "Study 1B",
                     "Result time present (operationalised: >=95% of candidate lab rows)",
                     "Collection time plus assay lag; label 1B \"approximate\"", False,
                     blk, raw >= 0.95, {"observed_text": _fmt(blk),
                                        "result_lag_minutes": _lag(lb)}))

    # 5. Imaging report finalization time
    im = tables["imaging"]
    im = im[im["BDSPPatientID"].isin(cand_ids)]
    blk, raw = prop_block(im["report_final_time"].notna(), im["BDSPPatientID"].map(site_of))
    rows.append(_row("Imaging report finalization time", "Study 1B, H5",
                     "Present (operationalised: >=95% of candidate imaging studies)", "Drop H5", False,
                     blk, raw >= 0.95, {"observed_text": _fmt(blk)}))

    # 6. GCS / FOUR / RASS within +-6 h
    sc = tables["clinical_scores"]
    sc = sc[sc["BDSPPatientID"].isin(cand_ids) & sc["score_type"].isin(["GCS", "FOUR", "RASS"])].copy()
    sc["off_h"] = (sc["score_time"] - sc["BDSPPatientID"].map(cands.set_index("BDSPPatientID")["t0"])
                   ).dt.total_seconds().abs() / 3600
    near = set(sc.loc[sc["off_h"] <= 6, "BDSPPatientID"])
    flags = cands["BDSPPatientID"].isin(near)
    blk, raw = prop_block(flags, cands["SiteID"])
    rows.append(_row("GCS, FOUR or RASS near EEG", "Inclusion, 1A baseline",
                     "Score within +-6 h for >=50% of candidates",
                     "BDSP GCS-from-EHR tool; broad cohort only", False,
                     blk, raw >= 0.50, {"observed_text": _fmt(blk)}))

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

    # 8. Timestamped notes
    nt = tables["notes"]
    nt = nt[nt["BDSPPatientID"].isin(cand_ids) & nt["note_time"].notna()]
    flags = cands["BDSPPatientID"].isin(set(nt["BDSPPatientID"]))
    blk, raw = prop_block(flags, cands["SiteID"])
    rows.append(_row("Timestamped notes", "ACI onset, silver labels",
                     "Present (operationalised: >=95% of candidates have >=1 timestamped note)", "Stop",
                     True, blk, raw >= 0.95, {"observed_text": _fmt(blk)}))

    # Hand-check sampling list (record-level -> local file only)
    elig = set(tables["notes"].dropna(subset=["note_time"])["BDSPPatientID"]) & \
        set(tables["labs"].dropna(subset=["collect_time"])["BDSPPatientID"]) & cand_ids
    elig = sorted(elig)
    rng = np.random.default_rng(seed)
    k = min(handcheck_n, len(elig))
    ids = [elig[j] for j in sorted(rng.choice(len(elig), size=k, replace=False))] if k else []

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


def _lag(lb: pd.DataFrame) -> dict:
    d = (lb["result_time"] - lb["collect_time"]).dt.total_seconds() / 60
    return safe_quantiles(d[d >= 0].dropna().values)


# ---------------------------------------------------------------- rendering
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with HEEDB-shaped tables")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0, help="seed for the hand-check sample")
    ap.add_argument("--handcheck-n", type=int, default=HANDCHECK_N)
    ap.add_argument("--strict", action="store_true", help="exit 1 if a Stop row fails")
    a = ap.parse_args(argv)

    agent_safety.assert_not_restricted_in_agent(a.data)
    tables = load_tables(a.data)
    report, ids = run_audit(tables, a.seed, a.handcheck_n)
    known = set(tables["eeg_metadata"]["BDSPPatientID"].astype(str))

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
