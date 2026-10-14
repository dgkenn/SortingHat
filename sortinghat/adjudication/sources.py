"""Readers for the OMOP tables a rater packet needs.

Modes
  cache   the local OMOP cache (out/local_only/omop_cache; never streams).  The cache holds column-pruned structured tables; its ``note``
          table holds NOTE TIMESTAMPS ONLY (no title, no text), so it can count notes but not supply them.
  store   a store (S3 access point via ``--s3`` or a local HEEDB-layout directory via ``--data``): streams the requested row groups.
  dict    in-memory frames ``{"omop_<table>": DataFrame}`` (the tests' synthetic data).
Every reader yields pandas frames filtered to the requested person ids; nothing is printed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from ..metrics import intended_use as iu

TABLE_COLUMNS = {
    "measurement": ["person_id", "measurement_datetime", "measurement_date", "measurement_time", "measurement_source_value",
                    "value_as_number", "value_source_value", "unit_source_value"],
    "drug_exposure": ["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime", "drug_exposure_start_date",
                      "drug_exposure_end_date", "drug_source_value", "route_source_value", "quantity"],
    "condition_occurrence": ["person_id", "condition_start_datetime", "condition_start_date", "condition_source_value"],
    "procedure_occurrence": ["person_id", "procedure_datetime", "procedure_date", "procedure_source_value"],
    "observation": ["person_id", "observation_datetime", "observation_date", "observation_source_value", "value_as_string"],
}
NOTE_META_COLUMNS = ["person_id", "note_datetime", "note_date"]
NOTE_TEXT_COLUMNS = ["person_id", "note_datetime", "note_date", "note_title", "note_source_value", "note_text"]


class OmopSource:
    def __init__(self, mode: str, *, cache_root: str | Path | None = None, store=None, tables: dict | None = None):
        if mode not in ("cache", "store", "dict"):
            raise ValueError(f"unknown source mode {mode!r}")
        self.mode, self.store, self.tables = mode, store, tables or {}
        if mode == "cache":
            from .. import omop_cache
            self.root = Path(cache_root) if cache_root else omop_cache.cache_root()
        else:
            self.root = None

    def available(self, table: str) -> bool:
        if self.mode == "cache":
            return bool(iu.cache_files(table, self.root)[0])
        if self.mode == "dict":
            return self.tables.get(f"omop_{table}") is not None
        return self.store is not None

    def frames(self, table: str, ids: np.ndarray, columns: Sequence[str], text_col: str | None = None, prefilter: str | None = None
               ) -> Iterator[pd.DataFrame]:
        ids = np.asarray(sorted({int(i) for i in ids}), dtype="int64")
        if self.mode == "cache":
            yield from iu.iter_cache_frames(table, self.root, list(columns), ids, text_col, prefilter)
        elif self.mode == "dict":
            for df in iu.iter_dict_frames(self.tables, table, ids):
                yield df[[c for c in columns if c in df.columns]]
        else:
            from .. import data_io
            for _u, batches, _t in data_io.iter_omop_units(table, person_ids=[int(i) for i in ids], columns=list(columns), s3=self.store):
                for b in batches or []:
                    yield b.to_pandas()


def id_lookup(sample: pd.DataFrame, remap: dict | None = None) -> tuple[np.ndarray, pd.Series]:
    """(ids to request, Series source-or-surviving id -> surviving person_id) for the sampled patients.  The cohort's ``person_id`` is the
    surviving id after the patient-merge map; rows may still be keyed by ``person_id_source`` or another retired id (``remap``)."""
    pid = pd.to_numeric(sample["person_id"], errors="coerce").astype("int64")
    src = pd.to_numeric(sample.get("person_id_source", sample["person_id"]), errors="coerce")
    m = {int(p): int(p) for p in pid}
    for s, p in zip(src, pid):
        if pd.notna(s):
            m[int(s)] = int(p)
    if remap:
        for o, n in remap.items():
            if int(n) in m and int(o) not in m:
                m[int(o)] = int(n)
    ser = pd.Series(m, dtype="int64")
    return np.array(sorted(ser.index), dtype="int64"), ser


def collect(source: OmopSource, table: str, columns: Sequence[str], ids: np.ndarray, to_surv: pd.Series, *, keep=None,
            text_col: str | None = None, prefilter: str | None = None) -> pd.DataFrame:
    """All rows of ``table`` for the requested ids, ``person_id`` replaced by the surviving id.  ``keep(df) -> df`` trims each chunk
    (e.g. to the time window) before it is stored, so the working set stays small."""
    parts = []
    for ch in source.frames(table, ids, columns, text_col, prefilter):
        if not len(ch):
            continue
        p = pd.to_numeric(ch["person_id"], errors="coerce")
        ch = ch.assign(person_id=p.map(to_surv)).dropna(subset=["person_id"])
        ch["person_id"] = ch["person_id"].astype("int64")
        if keep is not None:
            ch = keep(ch)
        if len(ch):
            parts.append(ch.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(columns))
