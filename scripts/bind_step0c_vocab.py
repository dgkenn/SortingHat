"""BIND imaging-E1 plan, step 0c (docs/bind_e1_analysis_plan.md section 9): values-only vocabulary of the categorical columns.

Lists the DISTINCT VALUE STRINGS (no counts, no linkage to patients, sessions, EEG, dates) of the categorical BIND I0001 findings
columns the label uses (Acute/Chronic, Label_LLAMA_Level1, Label_LLAMA_Level2, Nullfinding, Brain-related, Brain, PresentPrior,
Change, Test, Label_manual if categorical) and of the reports ``Type`` column. No free-text column is read: ``Name``,
``General Name``, ``Location``, ``Magnitude``, ``Report_txt`` are never loaded.

GUARD. A value string is WITHHELD (not printed, not written) when it looks like an id or a date (a run of >= 6 digits, a date
pattern), is longer than 80 characters, or trips ``safe_output`` (identifier-like pattern). The number of withheld strings is
reported. A column with more distinct values than ``--max-distinct`` is not categorical: no value is listed for it.

Plan step 0c also asks how normal studies are represented and how much of ``Label_manual`` is filled. That is a separate,
suppressed-aggregate block (``structure``): shares only, every cell < 11 shown as "<11", no ids.

Run (D-118 launcher): env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/bind_step0c_vocab.py
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sortinghat.labels import bind  # noqa: E402
from sortinghat.safe_output import (AggregateOnlyError, assert_aggregate_only, safe_print, safe_write_json,  # noqa: E402
                                    suppress_count, suppress_proportion)

MAX_LEN = 80
MAX_DISTINCT = 300
_DIGITS = re.compile(r"\d{6,}")
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_SUB_WORD = re.compile(r"\b(sub|ses)-([A-Za-z]+)\b", re.I)
_DATE = re.compile(rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}|\b{_MONTH}\.? \d{{1,2}}\b|\b\d{{1,2}} {_MONTH}\b",
                   re.I)

# categorical findings columns (real header names; all CONFIRMED) and the reports column
FINDINGS_VOCAB_COLUMNS = ["Acute/Chronic", "Label_LLAMA_Level1", "Label_LLAMA_Level2", "Nullfinding", "Brain-related", "Brain",
                          "PresentPrior", "Change", "Test", "Label_manual"]
FREE_TEXT = {"Name", "General Name", "Location", "Magnitude", "Report_txt"}


def withheld_reason(v: str) -> str | None:
    """None if the value string may be printed, else a short reason class."""
    if len(v) > MAX_LEN:
        return "too_long"
    if _DIGITS.search(v):
        return "digits"
    if _DATE.search(v):
        return "date"
    probe = _SUB_WORD.sub(r"\1\2", v)       # "sub-acute" is a clinical word, not a BIDS id (ids carry digits)
    try:
        assert_aggregate_only({probe: 0})
        assert_aggregate_only(probe)
    except AggregateOnlyError:
        return "id_pattern"
    return None


TOKEN_COLUMNS = {"Label_LLAMA_Level1", "Label_LLAMA_Level2", "Label_manual"}     # multi-label cells: comma-separated categories
TYPE_OK = {"Not available", "not available"}


def type_value_ok(v: str) -> bool:
    """Report ``Type`` is a closed list of study names written in capitals. A lower-case fragment is report text that leaked into
    the column through a malformed row (seen in the first 0c run); it is never listed."""
    return v == v.upper() or v in TYPE_OK


def tokens_of(cell: str) -> list[str]:
    return [re.sub(r"\s+", " ", t).strip() for t in cell.split(",") if t.strip()]


def distinct_values(series: pd.Series, max_distinct: int = MAX_DISTINCT, *, tokens: bool = False, strict_type: bool = False) -> dict:
    """Distinct non-blank value strings of one column, guarded. No counts. ``tokens``: split multi-label cells on commas and list
    the distinct category names instead of the distinct cells. ``strict_type``: apply the closed-list rule of ``type_value_ok``."""
    s = series.dropna().astype(str).str.strip()
    blank = bool((s == "").any()) or bool(series.isna().any())
    vals = sorted({t for v in set(s[s != ""]) for t in (tokens_of(v) if tokens else [v])})
    if strict_type:
        n0 = len(vals)
        vals = [v for v in vals if type_value_ok(v)]
        held_free_text = n0 - len(vals)
    else:
        held_free_text = 0
    if len(vals) > max_distinct:
        return {"status": "not_categorical", "distinct_gt": max_distinct, "n_distinct": len(vals),
                "unit": "token" if tokens else "cell", "has_blank": blank, "values": []}
    keep, held = [], held_free_text
    for v in vals:
        if withheld_reason(v) is None:
            keep.append(v)
        else:
            held += 1
    return {"status": "listed", "unit": "token" if tokens else "cell", "has_blank": blank, "values": keep, "n_values": len(keep),
            "withheld": held}


def structure_block(findings: pd.DataFrame, demo: pd.DataFrame | None, reports: pd.DataFrame | None, max_types: int = 40) -> dict:
    """Suppressed shares: how normal studies are represented (sessions with no findings row; Nullfinding-only sessions; report
    Type by presence of a findings row) and the fill rate of Label_manual. Keys are (person, session) in memory only."""
    out: dict = {}
    key = lambda d, p, s: set(zip(d[p].map(bind.norm_id), d[s].map(bind.norm_id)))  # noqa: E731
    f_sessions = key(findings, "BDSPPatientID", "Session_id")
    out["n_findings_sessions"] = suppress_count(len(f_sessions))
    if "Label_manual" in findings:
        filled = findings["Label_manual"].fillna("").astype(str).str.strip() != ""
        out["label_manual_rows_filled"] = suppress_proportion(int(filled.sum()), len(findings))
        fk = findings.assign(_k=list(zip(findings["BDSPPatientID"].map(bind.norm_id), findings["Session_id"].map(bind.norm_id))))
        out["share_label_manual_any_by_study"] = suppress_proportion(int(fk.loc[filled, "_k"].nunique()), fk["_k"].nunique())
    if "Nullfinding" in findings:
        nv = findings["Nullfinding"].fillna("").astype(str).str.strip()
        distinct = sorted(v for v in set(nv) if v and withheld_reason(v) is None)
        if 0 < len(distinct) <= 12:
            k = list(zip(findings["BDSPPatientID"].map(bind.norm_id), findings["Session_id"].map(bind.norm_id)))
            tmp = pd.DataFrame({"k": k, "v": nv.values})
            by = tmp.groupby("k")["v"]
            tot = by.ngroups
            blk: dict = {}
            for v in distinct:
                any_v = by.apply(lambda x, v=v: bool((x == v).any()))
                all_v = by.apply(lambda x, v=v: bool((x == v).all()))
                blk[v] = {"studies_any_row": suppress_proportion(int(any_v.sum()), tot),
                          "studies_all_rows": suppress_proportion(int(all_v.sum()), tot)}
            out["nullfinding_value_by_study"] = blk
    if "Label_LLAMA_Level1" in findings:
        l1 = findings["Label_LLAMA_Level1"].fillna("").astype(str)
        toks = sorted({t for v in set(l1) for t in tokens_of(v) if withheld_reason(t) is None})
        if 0 < len(toks) <= 20:
            tmp = pd.DataFrame({"k": list(zip(findings["BDSPPatientID"].map(bind.norm_id), findings["Session_id"].map(bind.norm_id))),
                                "t": [set(tokens_of(v)) for v in l1]})
            by = tmp.groupby("k")["t"]
            tot = by.ngroups
            out["level1_category_by_study"] = {
                t: {"studies_any_row": suppress_proportion(int(by.apply(lambda x, t=t: any(t in y for y in x)).sum()), tot),
                    "studies_all_rows": suppress_proportion(int(by.apply(lambda x, t=t: all(t in y for y in x)).sum()), tot)}
                for t in toks}
    if demo is not None:
        d_sessions = key(demo, "bdsp_patient_id", "Session_id")
        out["n_demographics_studies"] = suppress_count(len(d_sessions))
        out["share_demographics_studies_with_findings_row"] = suppress_proportion(len(d_sessions & f_sessions), len(d_sessions))
    if reports is not None and "Type" in reports:
        r = reports.assign(_k=list(zip(reports["bdsp_patient_id"].map(bind.norm_id), reports["session_id"].map(bind.norm_id))))
        r["_has"] = [k in f_sessions for k in r["_k"]]
        r["Type"] = r["Type"].fillna("").astype(str).str.strip()
        ok_types = sorted(t for t in set(r["Type"]) if t and type_value_ok(t) and withheld_reason(t) is None)
        out["n_report_studies"] = suppress_count(r["_k"].nunique())
        clean = r["Type"].isin(ok_types)
        out["report_type_clean_share"] = suppress_proportion(int(clean.sum()), len(r))
        if len(ok_types) <= max_types:
            out["report_type_findings_coverage"] = {
                t: {"n_reports": suppress_count(int((r["Type"] == t).sum())),
                    "share_with_findings_row": suppress_proportion(int(r.loc[r["Type"] == t, "_has"].sum()), int((r["Type"] == t).sum()))}
                for t in ok_types}
    return out


# Label_LLAMA_Level2 has far more distinct category names than a short list (first 0c run: > 300 tokens), so it cannot be listed whole.
# Step 0c lists only the Level2 names that contain a lesion word of the plan's six primary classes (+ hydrocephalus), which is what
# the regexes must be checked against. Values only; no counts, no linkage.
LESION_LEXICON = re.compile(r"hemorrh|haemorrh|hematoma|haematoma|infarct|ischem|ischaem|stroke|diffusion|contusion|axonal|traumatic|"
                            r"subdural|epidural|extra-axial|subarachnoid|intraparenchymal|intracerebral|mass effect|midline|herniat|"
                            r"edema|oedema|effacement|hydroceph|ventricul|\btbi\b|\bsah\b|\bich\b|\bsdh\b|\bedh\b|bleed|clot", re.I)
LEXICON_CAP = 400


def lexicon_probe(series: pd.Series, cap: int = LEXICON_CAP) -> dict:
    s = series.dropna().astype(str)
    toks = sorted({t for v in set(s) for t in tokens_of(v)})
    hit = [t for t in toks if LESION_LEXICON.search(t)]
    keep = [t for t in hit if withheld_reason(t) is None]
    out = {"n_distinct_tokens_with_lesion_word": len(hit), "withheld": len(hit) - len(keep)}
    if len(keep) > cap:
        out.update(status="too_many", values=[])
    else:
        out.update(status="listed", values=keep)
    return out


def build_vocab(findings: pd.DataFrame, reports: pd.DataFrame | None, demo: pd.DataFrame | None, *, max_distinct: int = MAX_DISTINCT,
                with_structure: bool = True) -> dict:
    bad = (set(findings.columns) | set(reports.columns if reports is not None else [])) & FREE_TEXT
    if bad:
        raise ValueError(f"free-text column present: {sorted(bad)}")
    out: dict = {"site": "I0001", "findings": {}, "reports": {}}
    withheld = 0
    for col in FINDINGS_VOCAB_COLUMNS:
        if col in findings:
            out["findings"][col] = distinct_values(findings[col], max_distinct, tokens=col in TOKEN_COLUMNS)
            withheld += out["findings"][col].get("withheld", 0)
    if reports is not None and "Type" in reports:
        out["reports"]["Type"] = distinct_values(reports["Type"], max_distinct, strict_type=True)
        withheld += out["reports"]["Type"].get("withheld", 0)
    if "Label_LLAMA_Level2" in findings:
        out["level2_lesion_word_names"] = lexicon_probe(findings["Label_LLAMA_Level2"])
        withheld += out["level2_lesion_word_names"]["withheld"]
    out["withheld_value_strings"] = withheld
    if with_structure:
        out["structure"] = structure_block(findings, demo, reports)
    return out


def _read(s3, kind: str, site: str, cfg, use_cols: list[str]) -> pd.DataFrame:
    from sortinghat import data_io
    key = bind.bind_key(kind, site, cfg)
    header = data_io.csv_header(s3, key)
    use = [c for c in header if c in use_cols]
    assert not (set(use) & cfg.never_load)
    body = data_io.read_object(s3, data_io.access_point(), key)
    return pd.read_csv(io.BytesIO(body), dtype=str, usecols=use, encoding="utf-8-sig", low_memory=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="I0001")
    ap.add_argument("--out", default="out/bind/vocab_I0001.json")
    ap.add_argument("--max-distinct", type=int, default=MAX_DISTINCT)
    ap.add_argument("--no-structure", action="store_true")
    a = ap.parse_args(argv)
    from sortinghat import data_io
    cfg = bind.load_config()
    s3 = data_io.make_client()
    findings = _read(s3, "findings", a.site, cfg, FINDINGS_VOCAB_COLUMNS + ["BDSPPatientID", "Session_id"])
    reports = _read(s3, "reports", a.site, cfg, ["Type", "bdsp_patient_id", "session_id"])
    dcol = cfg.raw["demographics_columns"][a.site]
    demo = _read(s3, "demographics", a.site, cfg, [dcol["person"], dcol["session"]])
    demo = demo.rename(columns={dcol["person"]: "bdsp_patient_id", dcol["session"]: "Session_id"})
    out = build_vocab(findings, reports, demo, max_distinct=a.max_distinct, with_structure=not a.no_structure)
    out["site"] = a.site
    safe_write_json(a.out, out)
    safe_print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as ex:  # class name only (a traceback could carry values)
        print(f"bind_step0c_vocab failed: {type(ex).__name__}", file=sys.stderr)
        raise SystemExit(1)
