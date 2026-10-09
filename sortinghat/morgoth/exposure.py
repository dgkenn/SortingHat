"""MORGOTH exposure accounting: which cohort patients appear in MORGOTH's own data lists.

The plan's control ("MORGOTH exposure accounting", SAP section on pretrained-model exposure) needs, per cohort patient, a
flag for "this patient's EEG is in the data MORGOTH was built on". The BDSP release ships the labelled-set lists
(``morgoth1/data/internal_dataset/<TASK>/*list*.xlsx|csv`` and ``datasets_deidentified_list.xlsx``) and a pretraining set
(``morgoth1/data/pretrain/*.mat``, 9,242 files per the project page). Those are the only membership evidence the release
exposes. This module scans such lists for HEEDB patient identifiers and flags cohort patients.

Hard rules (CLAUDE.md): the per-patient flag table is record-level, so it is written only under ``local_only/`` (mode
0600); stdout/JSON carry aggregates only (``summarize``: suppressed counts, per-site shares). The list files are read
into memory in the job and never printed. The column layout of the lists has NOT been inspected by an agent
(they are patient-level files); ``inspect_columns`` prints header NAMES only so a human can confirm the id columns
before the first run, and ``scan_lists`` reports how many ids it parsed so a zero is visible, not silently "no exposure".

Identifier convention (docs/heedb_access.md): BDSPPatientID is globally unique after the merge history, a BIDS folder is
``sub-<SITE><BDSPPatientID>`` with SITE like ``S0001``. A cell is searched for ``[SI]dddd`` + 6 or more digits; a numeric
column whose NAME says patient / person / bdsp id is taken as bare ids. Both reduce to the integer patient id.
Assumptions to confirm (docs/morgoth.md): the lists use BDSP ids (not re-hashed ones), and appearing in ANY list means the
patient's EEG was available to MORGOTH's development (training, validation or test); a split column, if found, gives the
narrower ``in_train_split`` flag.

Master-table rule (confirmed on the real release by header names and aggregate value classes, 2026-10-09): the BDSP workbook
``datasets_deidentified_list.xlsx`` holds a master patient table (one row per HEEDB patient with EEG, ``BDSPPatientID``) whose
``Morgoth`` column is blank for patients MORGOTH never used and ``pretrain`` / ``train`` / ``test`` otherwise; the other sheets
(``bdsp_mrn`` = the integer BDSPPatientID, ``file_name`` = ``sub-<SITE><id>_...`` or ``<SITE><id>_<n>_...``) are per-task subsets with
the same column. A column named ``Morgoth`` therefore acts as the membership column: rows where it is blank are NOT in MORGOTH's
data and are dropped before ids are read, and its value is the split (``pretrain``, ``train``, ``val``, ``test``). Mere presence in
that master table is not exposure. The per-task event lists (``internal_dataset/<TASK>/list*.xlsx|csv``) carry no split: their patients
count as members of unknown split.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..safe_output import suppress_count, suppress_proportion

_SITE_ID_RE = re.compile(r"(?<![A-Za-z0-9])(?:sub-)?[SI]\d{4}(\d{6,})(?!\d)")
_ID_COL_RE = re.compile(r"(bdsp|patient|person|subject|\bpid\b|bids|mrn)", re.I)
_PATHLIKE_COL_RE = re.compile(r"(file|name|path|key|record|segment|event|mat|edf)", re.I)
_SPLIT_COL_RE = re.compile(r"^(split|set|subset|partition|fold|group|train_?test|dataset_?split)$", re.I)
_TRAIN_WORDS = {"train", "training", "tr", "trn"}
_MEMBER_COL = "morgoth"                     # membership + split column of the BDSP master / per-task sheets
_TRAINING_SPLITS = {"train", "pretrain"}    # splits that count as "trained on" (SSL pretraining included)


def _norm_split(v) -> str | None:
    """Normalise a split cell to pretrain / train / val / test / other; None when blank."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    t = str(v).strip().lower()
    if t in ("", "nan", "none"):
        return None
    if t.startswith("pretrain") or t.startswith("pre-train") or t.startswith("pre_train"):
        return "pretrain"
    if t in _TRAIN_WORDS or t.startswith("train"):
        return "train"
    if t.startswith("val") or t == "dev":
        return "val"
    if t.startswith("test"):
        return "test"
    return "other"


@dataclass
class TrainingIndex:
    """Patient ids found in MORGOTH's lists. Record-level: keep in memory / local_only only."""
    ids: set[int] = field(default_factory=set)
    train_ids: set[int] = field(default_factory=set)       # ids whose row says split == train (when a split column exists)
    has_split_info: bool = False
    n_files: int = 0
    n_files_with_ids: int = 0
    n_cells: int = 0
    n_lists_per_id: dict[int, int] = field(default_factory=dict)
    splits: dict[int, set] = field(default_factory=dict)   # id -> splits seen (pretrain / train / val / test / other)
    unsplit_ids: set[int] = field(default_factory=set)     # ids seen in a row with no split information
    sources: dict[str, set[int]] = field(default_factory=dict)   # source label -> ids (label = file name [+ sheet / split])

    @property
    def split_complete(self) -> bool:
        """True when every id has a known split (the narrower train flag is then defensible)."""
        return self.has_split_info and not self.unsplit_ids


def _read_any(path: Path) -> list[pd.DataFrame]:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return list(pd.read_excel(path, sheet_name=None, dtype=object).values())
    if suf == ".parquet":
        return [pd.read_parquet(path)]
    if suf in (".csv", ".tsv", ".txt"):
        return [pd.read_csv(path, sep="\t" if suf == ".tsv" else ",", dtype=object, encoding="utf-8-sig",
                            on_bad_lines="skip")]
    raise ValueError(f"unsupported list format {suf!r}")


def inspect_columns(path: str | Path) -> dict:
    """Header NAMES and the chosen id columns of one list. No values. For the human who confirms the layout first."""
    out = {}
    for i, df in enumerate(_read_any(Path(path))):
        cols = [str(c) for c in df.columns]
        out[f"sheet{i}"] = {"columns": cols, "id_columns": _choose_columns(df),
                            "split_columns": [c for c in cols if _SPLIT_COL_RE.match(c) or c.strip().lower() == _MEMBER_COL]}
    return out


def _choose_columns(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns]
    named = [c for c in cols if _ID_COL_RE.search(c) or _PATHLIKE_COL_RE.search(c)]
    return named or cols            # no recognisable column: scan every column (cells are regex-matched, not trusted)


def ids_from_cells(values: Iterable) -> list[int]:
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out += [int(m.group(1)) for m in _SITE_ID_RE.finditer(str(v))]
    return out


def ids_from_bare_column(values: Iterable) -> list[int]:
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return [int(v) for v in s[(s > 0) & (s == s.round())]]


def scan_frame(df: pd.DataFrame, idx: TrainingIndex, merge_map: Mapping[int, int] | None = None, label: str | None = None) -> int:
    """Add the ids found in one table to ``idx``; returns how many distinct ids this table contributed.

    A ``Morgoth`` column (see module docstring) is the membership column: rows with it blank are dropped, its value is the split.
    Otherwise the first generic split-like column, if any, gives the split; with no split column the ids have unknown split."""
    member = next((c for c in df.columns if str(c).strip().lower() == _MEMBER_COL), None)
    split_cols = [c for c in df.columns if _SPLIT_COL_RE.match(str(c))]
    split_col = member if member is not None else (split_cols[0] if split_cols else None)
    if member is not None:
        df = df[df[member].notna()].reset_index(drop=True)
    cols = [c for c in _choose_columns(df) if c != member]
    row_split = [_norm_split(v) for v in df[split_col].tolist()] if split_col is not None else None
    found: dict[int, set] = {}                   # id -> splits seen (None = unknown)
    for c in cols:
        col = df[c]
        idx.n_cells += int(col.notna().sum())
        bare = _ID_COL_RE.search(c) is not None and not _PATHLIKE_COL_RE.search(c)
        for pos, v in enumerate(col.tolist()):
            got = ids_from_cells([v])
            if not got and bare:
                got = ids_from_bare_column([v])
            if not got:
                continue
            sp = row_split[pos] if row_split is not None else None
            for i in got:
                found.setdefault(i, set()).add(sp)
    if split_col is not None:
        idx.has_split_info = True
    for i, sps in found.items():
        for j in {i, (merge_map or {}).get(i, i)}:
            idx.ids.add(j)
            idx.n_lists_per_id[j] = idx.n_lists_per_id.get(j, 0) + 1
            known = {x for x in sps if x is not None}
            idx.splits.setdefault(j, set()).update(known)
            if None in sps:
                idx.unsplit_ids.add(j)
            if known & _TRAINING_SPLITS:
                idx.train_ids.add(j)
            if label is not None:
                if member is not None:       # one source per split value of the membership column
                    for x in known or {"unsplit"}:
                        idx.sources.setdefault(f"{label}:{x}", set()).add(j)
                else:
                    idx.sources.setdefault(label, set()).add(j)
    return len(found)


def scan_lists(paths: Iterable[str | Path], merge_map: Mapping[int, int] | None = None) -> TrainingIndex:
    """Scan list files (xlsx / csv / tsv / parquet) for HEEDB patient ids."""
    idx = TrainingIndex()
    for p in paths:
        idx.n_files += 1
        dfs = _read_any(Path(p))
        stem = Path(p).name
        n = 0
        for k, df in enumerate(dfs):
            n += scan_frame(df, idx, merge_map, label=stem if len(dfs) == 1 else f"{stem}#sheet{k}")
        idx.n_files_with_ids += int(n > 0)
    return idx


def add_names(idx: TrainingIndex, names: Iterable[str]) -> int:
    """Add ids embedded in object NAMES (e.g. the keys under ``morgoth1/data/pretrain/``). Returns ids added."""
    got = set(ids_from_cells(names))
    before = len(idx.ids)
    idx.ids |= got
    for i in got:
        idx.n_lists_per_id[i] = idx.n_lists_per_id.get(i, 0) + 1
        idx.splits.setdefault(i, set()).add("pretrain")     # the pretrain/ prefix holds the SSL pretraining recordings
    idx.sources.setdefault("pretrain_key_names", set()).update(got)
    return len(idx.ids) - before


def _hit(pid: pd.Series, src: pd.Series, ids) -> np.ndarray:
    return (pid.isin(ids) | src.isin(ids)).to_numpy()


def flag_cohort(cohort: pd.DataFrame, idx: TrainingIndex, id_col: str = "person_id",
                source_col: str | None = "person_id_source", site_col: str = "SiteID") -> pd.DataFrame:
    """Per-patient flags. RECORD-LEVEL: write under ``local_only/`` only. Columns: ``person_id``, ``SiteID`` (if present),
    ``in_morgoth_lists`` (patient id, or its pre-merge id, appears in MORGOTH's data: any pretrain / train / test membership,
    or a per-task list), ``in_morgoth_train_or_pretrain`` (split says train or pretrain; a lower bound when some lists have no
    split), ``in_morgoth_pretrain``, ``in_morgoth_test_only`` (split known, test or val only, never train / pretrain, and in no
    split-less list), ``in_morgoth_train_split`` (the narrower flag, emitted only when EVERY list carries a split; else NaN so a
    reader falls back to ``in_morgoth_lists``) and ``n_morgoth_lists``."""
    pid = pd.to_numeric(cohort[id_col], errors="coerce")
    src = pd.to_numeric(cohort[source_col], errors="coerce") if source_col and source_col in cohort else pid
    hit = _hit(pid, src, idx.ids)
    train = _hit(pid, src, idx.train_ids)
    pre = _hit(pid, src, {i for i, sp in idx.splits.items() if "pretrain" in sp})
    held = {i for i, sp in idx.splits.items() if sp and not (sp & _TRAINING_SPLITS) and i not in idx.unsplit_ids}
    out = pd.DataFrame({"person_id": cohort[id_col].to_numpy()})
    if site_col in cohort:
        out["SiteID"] = cohort[site_col].to_numpy()
    out["in_morgoth_lists"] = hit
    out["in_morgoth_train_or_pretrain"] = train
    out["in_morgoth_pretrain"] = pre
    out["in_morgoth_test_only"] = _hit(pid, src, held)
    out["in_morgoth_train_split"] = train if idx.split_complete else np.nan
    out["n_morgoth_lists"] = [max(idx.n_lists_per_id.get(int(a), 0) if a == a else 0,
                                  idx.n_lists_per_id.get(int(b), 0) if b == b else 0)
                              for a, b in zip(pid, src)]
    return out


def source_counts(cohort: pd.DataFrame, idx: TrainingIndex, id_col: str = "person_id",
                  source_col: str | None = "person_id_source") -> dict:
    """Per source (list file / sheet / split) count of cohort patients present, suppressed. Aggregate-only."""
    pid = pd.to_numeric(cohort[id_col], errors="coerce")
    src = pd.to_numeric(cohort[source_col], errors="coerce") if source_col and source_col in cohort else pid
    return {k: suppress_count(int(_hit(pid, src, ids).sum())) for k, ids in sorted(idx.sources.items())}


def summarize(flags: pd.DataFrame, idx: TrainingIndex) -> dict:
    """Aggregate-only report: suppressed counts and proportions, per site. No ids."""
    n = len(flags)
    k = int(flags["in_morgoth_lists"].sum())
    out = {"lists_scanned": suppress_count(idx.n_files),
           "lists_with_parsed_ids": suppress_count(idx.n_files_with_ids),
           "n_ids_parsed_from_lists": suppress_count(len(idx.ids)),
           "n_cohort": suppress_count(n),
           "n_in_lists": suppress_count(k) if (n - k) >= 11 else "<11",
           "in_lists_proportion": suppress_proportion(k, n),
           "split_column_found": bool(idx.has_split_info), "split_complete": bool(idx.split_complete)}
    for col, key in (("in_morgoth_train_or_pretrain", "train_or_pretrain"), ("in_morgoth_pretrain", "pretrain"),
                     ("in_morgoth_test_only", "test_only")):
        if col in flags:
            kk = int(flags[col].sum())
            out[f"n_{key}"] = suppress_count(kk) if (n - kk) >= 11 else "<11"
            out[f"{key}_proportion"] = suppress_proportion(kk, n)
    if idx.split_complete:
        kt = int(pd.Series(flags["in_morgoth_train_split"]).fillna(False).astype(bool).sum())
        out["in_train_split_proportion"] = suppress_proportion(kt, n)
    if "SiteID" in flags:
        sites = {}
        for i, (s, g) in enumerate(sorted(flags.groupby("SiteID"), key=lambda t: str(t[0]))):
            sites[f"site_{i + 1}"] = {"n": suppress_count(len(g)),
                                      "in_lists_proportion": suppress_proportion(int(g["in_morgoth_lists"].sum()), len(g))}
        out["by_site"] = sites                    # pseudonymised like the ladder report (ordering is by real site code)
    if len(idx.ids) == 0:
        out["warning"] = "no patient ids parsed from the lists: zero exposure here means 'unreadable', not 'none'"
    return out


def write_flags(flags: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    if "local_only" not in p.resolve().parts:
        raise ValueError("the per-patient exposure table must be written under a local_only/ directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(p, index=False)
    os.chmod(p, 0o600)
    return p
