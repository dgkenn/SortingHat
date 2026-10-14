#!/usr/bin/env python3
"""Gold adjudication pilot, step 2: EEG-blind rater packets (D-157).   Prints COUNTS ONLY; packets are for HUMAN raters.

For each sampled patient (out/local_only/adjudication/sample.csv) assemble a packet of structured data around t0 (diagnoses, procedures,
medications, scores, vitals, labs, observations; every time relative to t0, default window [-72 h, +30 d]) plus clinical notes with every
EEG / neurophysiology note excluded and every sentence mentioning EEG-derived content removed (sortinghat/adjudication/blinding.py, on top of
sortinghat/labels/eeg_filter.py).  Packets go to out/local_only/adjudication/packets/<case_id>/ (packet.html, structured/*.csv, notes/*.txt,
manifest.json; directories 0700, files 0600) under pseudonymous case ids; the id map, rater assignments and a REDCap record-id list are
separate local_only files.  A packet whose final text still trips the EEG audit or contains the patient's own id is withheld.
NO LLM reads a note here (CLAUDE.md rule 4): nothing is summarised, nothing is printed.

Modes
  --dry-run          AGGREGATE AVAILABILITY ONLY: how many sampled cases have notes / labs / meds / ... in the window.  Reads the local OMOP
                     cache (never streams; the cache holds note TIMESTAMPS only), writes no packet, no note text.  Safe to launch from an agent.
  (default)          full build.  Structured data from --structured-source (cache | store); notes from --notes-source:
                       none   no note text: the structured part is built and every packet says "NOTES PENDING"
                       dir    a directory of local note exports (parquet / csv with person_id, note_datetime, note_title, note_text) on approved compute
                       store  stream the OMOP `note` table for the sampled people only (--s3 or --data): LARGE; a HUMAN launches this
  Notes never leave this process except as the redacted text written into the packet files.

    # dry run (aggregate availability; the only mode run inside an agent session)
    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/adjudication_packets.py --dry-run
    # full build, HUMAN, plain terminal (notes streamed from the access point)
    scripts/heedb_run.sh python3 scripts/adjudication_packets.py --s3 --notes-source store --max-memory-gb 12
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sortinghat import agent_safety  # noqa: E402
from sortinghat.adjudication import aggregates as agg  # noqa: E402
from sortinghat.adjudication import packets as pk  # noqa: E402
from sortinghat.adjudication import sources as src  # noqa: E402
from sortinghat.adjudication.private_io import require_local_only, write_private_csv  # noqa: E402
from sortinghat.safe_output import safe_print, safe_quantiles, safe_write_json, suppress_count  # noqa: E402

BANNER = "GOLD ADJUDICATION PILOT (D-157) - EEG-blind packets; counts only; packets are record-level, local_only, for HUMAN raters"
DEFAULT_CONFIG = HERE.parent / "configs" / "adjudication.yaml"
LOCAL = "out/local_only/adjudication"
SECTION_FLAGS = ("diagnoses", "procedures", "medications", "scores", "vitals", "labs", "observations")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adj-config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--sample", default=f"{LOCAL}/sample.csv", help="record-level sample from adjudication_sample.py (local_only)")
    ap.add_argument("--out-dir", default=f"{LOCAL}/packets", help="packet directory (local_only)")
    ap.add_argument("--id-map-out", default=f"{LOCAL}/id_map.csv")
    ap.add_argument("--assignments-out", default=f"{LOCAL}/assignments.csv")
    ap.add_argument("--redcap-ids-out", default=f"{LOCAL}/redcap_case_ids.csv")
    ap.add_argument("--summary-out", default="out/adjudication/packet_summary.json", help="aggregate JSON (must NOT be under local_only/)")
    ap.add_argument("--dry-run", action="store_true", help="aggregate availability only; writes no packet and no note text; never streams")
    src_g = ap.add_mutually_exclusive_group()
    src_g.add_argument("--data", help="local HEEDB-layout directory (synthetic data / a local mirror)")
    src_g.add_argument("--s3", action="store_true", help="the BDSP access point (needed for --structured-source store / --notes-source store)")
    ap.add_argument("--profile")
    ap.add_argument("--structured-source", choices=["cache", "store"], default="cache",
                    help="cache = the local OMOP cache (default; never streams); store = stream from --s3 / --data")
    ap.add_argument("--omop-cache", default=None, help="root of the local OMOP cache (default out/local_only/omop_cache)")
    ap.add_argument("--notes-source", choices=["none", "cache", "dir", "store"], default=None,
                    help="default: cache (timestamp counts) for --dry-run, none for a build; 'cache' cannot supply text")
    ap.add_argument("--notes-dir", help="directory of note exports for --notes-source dir (parquet / csv)")
    ap.add_argument("--max-memory-gb", type=float, default=None)
    return ap


# ===================================================================================================== inputs
def read_sample(path) -> pd.DataFrame:
    s = pd.read_csv(require_local_only(path, "--sample"), dtype={"sample_id": str, "duplicate_of": str})
    s["duplicate_of"] = s["duplicate_of"].fillna("")
    s["t0"] = pd.to_datetime(s["t0"])
    return s


def make_source(a, mode: str) -> tuple[src.OmopSource, dict | None]:
    store = None
    if a.data:
        agent_safety.assert_not_restricted_in_agent(a.data)
        from sortinghat import data_io
        store = data_io.open_store(a.data)
    elif a.s3:
        from sortinghat import data_io
        store = data_io.open_store(None, profile=a.profile)
    remap = None
    if store is not None and not a.data:
        try:
            from sortinghat.cohort import StoreSources
            remap, _st = StoreSources(store).merge_map()
        except Exception:                                    # noqa: BLE001 - optional bookkeeping
            remap = None
    if mode == "store":
        if store is None:
            raise SystemExit("--structured-source store / --notes-source store needs --s3 or --data")
        return src.OmopSource("store", store=store), remap
    return src.OmopSource("cache", cache_root=a.omop_cache), remap


def collect_structured(source: src.OmopSource, ids, to_surv, t0_map, pc: pk.PacketConfig) -> tuple[dict, list]:
    """{table: {person_id: DataFrame}} of window-trimmed raw rows, and the list of unavailable tables."""
    keeps = {"measurement": pk.window_keep(*pk.TIME_COLS["measurement"], t0_map, pc.lo, pc.hi),
             "condition_occurrence": pk.window_keep(*pk.TIME_COLS["condition_occurrence"], t0_map, pc.hist_lo, pc.hi),
             "procedure_occurrence": pk.window_keep(*pk.TIME_COLS["procedure_occurrence"], t0_map, pc.lo, pc.hi),
             "observation": pk.window_keep(*pk.TIME_COLS["observation"], t0_map, pc.lo, pc.hi),
             "drug_exposure": pk.drug_keep(t0_map, pc.lo, pc.hi)}
    out, missing = {}, []
    for table, keep in keeps.items():
        if not source.available(table):
            missing.append(table)
            out[table] = {}
            continue
        df = src.collect(source, table, src.TABLE_COLUMNS[table], ids, to_surv, keep=keep)
        out[table] = {int(p): g.reset_index(drop=True) for p, g in df.groupby("person_id")} if len(df) else {}
    return out, missing


def ingest_notes(chunk: pd.DataFrame, to_surv, t0_map, pc: pk.PacketConfig, notes: dict, drops: Counter) -> None:
    """Window-trim one chunk of raw notes, EEG-filter it IMMEDIATELY and keep only the filtered text, per surviving person id."""
    if not len(chunk):
        return
    p = pd.to_numeric(chunk["person_id"], errors="coerce").map(to_surv)
    chunk = chunk.assign(person_id=p).dropna(subset=["person_id"])
    chunk["person_id"] = chunk["person_id"].astype("int64")
    chunk = pk.window_keep(*pk.TIME_COLS["note"], t0_map, pc.lo, pc.notes_hi)(chunk)
    for pid, g in chunk.groupby("person_id"):
        notes.setdefault(int(pid), []).extend(pk.prepare_notes(g, pc, drops))


def collect_notes(a, mode: str, source_for_notes, ids, to_surv, t0_map, pc, drops: Counter) -> dict | None:
    """{person_id: [filtered notes]} for 'dir' / 'store'; None when no text source (notes pending)."""
    notes: dict = {}
    if mode == "dir":
        d = Path(a.notes_dir or "")
        if not d.is_dir():
            raise SystemExit("--notes-source dir needs --notes-dir")
        files = sorted(list(d.glob("*.parquet")) + list(d.glob("*.csv")))
        idset = set(int(i) for i in ids)
        for f in files:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            df = df[pd.to_numeric(df["person_id"], errors="coerce").isin(idset)]
            ingest_notes(df, to_surv, t0_map, pc, notes, drops)
        return notes
    if mode == "store":
        for ch in source_for_notes.frames("note", ids, src.NOTE_TEXT_COLUMNS):
            ingest_notes(ch, to_surv, t0_map, pc, notes, drops)
        return notes
    return None


def note_counts_from_cache(source: src.OmopSource, ids, to_surv, t0_map, pc) -> dict | None:
    """{person_id: number of note timestamps in the window} from the cached `note` table (timestamps only); None if absent."""
    if not source.available("note"):
        return None
    df = src.collect(source, "note", src.NOTE_META_COLUMNS, ids, to_surv, keep=pk.window_keep(*pk.TIME_COLS["note"], t0_map, pc.lo, pc.notes_hi))
    return df.groupby("person_id").size().to_dict() if len(df) else {}


# ===================================================================================================== build
def person_row(r) -> dict:
    return {"age_years": r.get("age_years"), "sex_male": r.get("sex_male"), "hours_since_visit_start": r.get("hours_since_visit_start")}


def run(a) -> int:
    cfg = yaml.safe_load(Path(a.adj_config).read_text())
    pc = pk.PacketConfig.from_dict(cfg.get("packets"))
    summary_out = Path(a.summary_out)
    if "local_only" in summary_out.resolve().parts:
        raise SystemExit("--summary-out holds aggregates and must NOT be under local_only/")
    notes_mode = a.notes_source or ("cache" if a.dry_run else "none")
    if a.dry_run and notes_mode == "store":
        raise SystemExit("--dry-run never streams the note table; use --notes-source cache (timestamp counts) or dir")
    if not a.dry_run:
        for p, w in ((a.out_dir, "--out-dir"), (a.id_map_out, "--id-map-out"), (a.assignments_out, "--assignments-out"), (a.redcap_ids_out, "--redcap-ids-out")):
            require_local_only(p, w)
    if a.dry_run and a.structured_source == "store":
        raise SystemExit("--dry-run reads the local cache only (--structured-source cache)")
    sample = read_sample(a.sample)
    uniq = sample.drop_duplicates("person_id")
    ids, to_surv = src.id_lookup(uniq)
    t0_map = uniq.set_index("person_id")["t0"].to_dict()
    source, _remap = make_source(a, a.structured_source)
    if _remap:
        ids, to_surv = src.id_lookup(uniq, _remap)
    raw, missing_tables = collect_structured(source, ids, to_surv, t0_map, pc)
    drops_notes: Counter = Counter()
    notes = None
    note_ts = None
    if notes_mode in ("dir", "store"):
        nsrc = source if notes_mode == "store" and a.structured_source == "store" else make_source(a, "store")[0] if notes_mode == "store" else None
        notes = collect_notes(a, notes_mode, nsrc, ids, to_surv, t0_map, pc, drops_notes)
    elif notes_mode == "cache":
        note_ts = note_counts_from_cache(source if a.structured_source == "cache" else make_source(a, "cache")[0], ids, to_surv, t0_map, pc)
    # ---- build in memory
    case_ids = pk.make_case_ids(len(sample), int(cfg["sampling"]["seed"]))
    sample = sample.sort_values("sample_id").reset_index(drop=True)
    sample["case_id"] = case_ids
    cid_of = dict(zip(sample["sample_id"], sample["case_id"]))
    sample["duplicate_of_case"] = sample["duplicate_of"].map(cid_of).fillna("")
    built = []
    dropped_total: Counter = Counter()
    for _, r in sample.iterrows():
        if a.dry_run and r["role"] != "sample":
            continue
        pid = int(r["person_id"])
        raw_p = {t: raw[t].get(pid) for t in raw}
        notes_p = None if notes is None else notes.get(pid, [])
        p = pk.build_packet(r["case_id"], person_row(r), raw_p, notes_p, pc, notes_source_configured=notes is not None)
        forbidden = [str(pid)] + ([str(int(r["person_id_source"]))] if "person_id_source" in r and pd.notna(r["person_id_source"]) else [])
        p, page = pk.finalize(p, pc, forbidden)
        dropped_total.update(p.dropped)
        built.append((r, p, page))
    # ---- write (full build only)
    if not a.dry_run:
        out_dir = Path(a.out_dir)
        for r, p, page in built:
            if p.withheld is None:
                pk.write_packet(out_dir, p, page, manifest_root=out_dir.parent / "manifests")
        idm = sample[["case_id", "sample_id", "role", "duplicate_of_case", "person_id", "site"]].copy()
        idm["packet_status"] = idm["case_id"].map({p.case_id: (p.withheld or "written") for _, p, _ in built})
        write_private_csv(a.id_map_out, idm)
        write_private_csv(a.assignments_out, pk.make_assignments(idm, tuple(cfg["raters"]), int(cfg["sampling"]["seed"]), int(cfg["batch_size"])))
        write_private_csv(a.redcap_ids_out, pd.DataFrame({"case_id": sorted(idm.loc[idm["packet_status"] == "written", "case_id"])}))
    # ---- aggregates
    sites = sorted(sample["site"].unique())
    labels = {s: f"site_{i + 1}" for i, s in enumerate(sites)}
    rows = []
    for r, p, _ in built:
        if r["role"] != "sample":
            continue
        d = {"site": r["site"], **{f: bool(len(p.sections[f])) for f in SECTION_FLAGS}}
        d["any_structured"] = any(d[f] for f in SECTION_FLAGS)
        d["age_recorded"] = p.header["age_years"] is not None
        d["sex_recorded"] = p.header["sex"] != "not recorded"
        if note_ts is not None:
            d["notes_timestamp_in_window"] = bool(note_ts.get(int(r["person_id"]), 0) > 0)
        if notes is not None:
            d["notes_text_in_window"] = bool(len(p.notes))
        d["_n_labs"] = int(r_names(p.sections["labs"]))
        d["_n_meds"] = int(p.sections["medications"]["drug"].nunique()) if len(p.sections["medications"]) else 0
        d["_n_dx"] = len(p.sections["diagnoses"])
        d["_n_note_ts"] = int(note_ts.get(int(r["person_id"]), 0)) if note_ts is not None else np.nan
        rows.append(d)
    flags = pd.DataFrame(rows)
    cols = [c for c in flags.columns if not c.startswith("_") and c != "site"]
    n_withheld = sum(p.withheld is not None for _, p, _ in built)
    rep = {
        "banner": BANNER, "mode": "dry_run" if a.dry_run else "build",
        "sources": {"structured": a.structured_source, "notes": notes_mode,
                    "notes_text_status": ("not available: the OMOP cache holds note timestamps only, so packets would say NOTES PENDING"
                                          if notes_mode in ("cache", "none") else f"from {notes_mode}"),
                    "tables_unavailable": missing_tables},
        "window": {"before_h": pc.before_h, "after_days": pc.after_days, "history_days": pc.history_days},
        "n_cases": {"pooled": suppress_count(len(flags)), **{labels[s]: suppress_count(int((flags["site"] == s).sum())) for s in sites}},
        "availability": agg.availability_report(flags, "site", labels, cols),
        "items_per_case": {"distinct_lab_names": safe_quantiles(flags["_n_labs"]), "distinct_drugs": safe_quantiles(flags["_n_meds"]),
                           "diagnosis_rows": safe_quantiles(flags["_n_dx"]),
                           **({"note_timestamps_in_window": safe_quantiles(flags["_n_note_ts"].dropna())} if note_ts is not None else {})},
        "blinding": {"rows_or_sentences_removed_by_reason": {k: suppress_count(v) for k, v in sorted((dropped_total + drops_notes).items())},
                     "packets_withheld": suppress_count(n_withheld)},
    }
    known = {str(p) for p in uniq["person_id"]} | set(sample["case_id"]) | set(sample["sample_id"])
    safe_write_json(summary_out, rep, known)
    safe_print(BANNER, known_ids=known)
    safe_print(f"mode={rep['mode']} cases={rep['n_cases']['pooled']} structured={a.structured_source} notes={notes_mode}", known_ids=known)
    safe_print("  notes text: " + rep["sources"]["notes_text_status"], known_ids=known)
    for k, v in rep["availability"].items():
        c = v["all_sites"]
        extra = f", share {c['share_bound']}" if "share_bound" in c else (", almost none" if c.get("why") == "low" else "")
        safe_print(f"  {k}: {c['n']} of {c['den']} ({c['share']}{extra})", known_ids=known)
    safe_print(f"  withheld={rep['blinding']['packets_withheld']}; aggregates: {summary_out}", known_ids=known)
    if not a.dry_run:
        safe_print(f"Packets written under {a.out_dir} (record-level, 0600, not printed); id map {a.id_map_out}", known_ids=known)
    return 0


def r_names(labs: pd.DataFrame) -> int:
    return int(labs["name"].nunique()) if len(labs) else 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from sortinghat.cohort.memguard import run_guarded
    return run_guarded(lambda: run(a), a.max_memory_gb, "adjudication packets")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"adjudication_packets failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
