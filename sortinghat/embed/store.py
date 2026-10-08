"""Local-only parquet store for embedding rows (parts + ledgers), mirroring ``scripts/extract_eeg_features.py``.

Layout of ``--out-dir`` (must be under a ``local_only/`` directory, mode 0700; files 0600):
``part-<tag>-<id>.parquet`` rows keyed by ``recording_id`` x ``window`` (``cbramod.part_columns``), ``ledger-<tag>.csv``
(recording_id, status, reason, bytes_fetched, elapsed_s) and ``embed_meta.json`` (model / config provenance, no patient
information). ``load_primary_embeddings`` is what the silver-feasibility ladder reads.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .cbramod import ATTN_PREFIX, MEAN_PREFIX, EmbedConfig, emb_columns, part_columns

META_NAME = "embed_meta.json"
HEAD_COLUMNS = ("recording_id", "onset_offset_s")


def require_local_only(p: Path, what: str) -> None:
    if "local_only" not in Path(p).resolve().parts:
        raise SystemExit(f"{what} must be under a local_only/ directory (gitignored; CLAUDE.md rule 5)")


def write_part(rows: list[dict], cfg: EmbedConfig, out_dir: Path, tag: str) -> Path:
    cols = [*HEAD_COLUMNS, *part_columns(cfg)]
    df = pd.DataFrame(rows).reindex(columns=cols)
    for c in cols:
        if c.startswith("emb."):
            df[c] = df[c].astype("float32")
    for c in ("usable_fraction", "qc_coverage_fraction", "onset_offset_s", "emb_valid_token_frac", "emb_frac_over_amp"):
        df[c] = df[c].astype("float64")
    for c in ("qc_pass", "emb_ok"):
        df[c] = df[c].fillna(False).astype(bool)
    for c in ("qc_n_missing_or_dead_min", "emb_n_segments", "emb_n_used", "emb_n_absent_channels"):
        df[c] = df[c].astype("int32")
    p = Path(out_dir) / f"part-{tag}-{uuid.uuid4().hex[:10]}.parquet"
    df.to_parquet(p, index=False)
    os.chmod(p, 0o600)
    return p


def write_meta(out_dir: Path, meta: dict) -> None:
    """Record provenance once; refuse to mix configurations in one directory."""
    p = Path(out_dir) / META_NAME
    if p.exists():
        old = json.loads(p.read_text())
        if old != meta:
            raise SystemExit("--out-dir already holds embeddings made with a different model / config (embed_meta.json); "
                             "use a new --out-dir")
        return
    p.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.chmod(p, 0o600)


def load_primary_embeddings(emb_dir, rec_ids: set[str] | None = None, window: str = "primary", family_prefix: str = MEAN_PREFIX,
                            require_ok: bool = True) -> tuple[pd.DataFrame, dict]:
    """One row per recording (indexed by ``recording_id``) with ``emb.cbramod.<j>`` columns and the QC columns, for
    ``window``. Duplicate rows (a crash between part and ledger writes) keep the one with an embedding. Rows without an
    embedding (``emb_ok`` False) are dropped when ``require_ok``. Aggregate counts are returned alongside; the frame is
    RECORD-LEVEL (memory only)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    emb_dir = Path(emb_dir)
    require_local_only(emb_dir, "--embeddings")
    parts = sorted(emb_dir.glob("part-*.parquet"))
    want = None if rec_ids is None else pa.array(sorted(rec_ids), type=pa.string())
    frames = []
    for p in parts:
        t = pq.read_table(p)
        if "recording_id" not in t.column_names or "window" not in t.column_names:
            continue
        keep = pc.equal(t["window"].cast(pa.string()), window)
        if want is not None:
            keep = pc.and_(keep, pc.is_in(t["recording_id"].cast(pa.string()), value_set=want))
        t = t.filter(pc.fill_null(keep, False))
        if t.num_rows:
            frames.append(t.to_pandas())
    cols = emb_columns(family_prefix) if family_prefix in (MEAN_PREFIX, ATTN_PREFIX) else []
    if not frames:
        return pd.DataFrame(columns=["emb_ok", *cols]), {"n_parts": len(parts), "n_recordings": 0, "n_emb_ok": 0}
    df = pd.concat(frames, ignore_index=True)
    df["recording_id"] = df["recording_id"].astype(str)
    df = df.sort_values("emb_ok", ascending=False, kind="stable").drop_duplicates("recording_id").set_index("recording_id")
    stats = {"n_parts": len(parts), "n_recordings": int(len(df)), "n_emb_ok": int(df["emb_ok"].sum())}
    if require_ok:
        df = df[df["emb_ok"].astype(bool)]
    use = [c for c in df.columns if c.startswith(family_prefix) or not c.startswith("emb.")]
    return df[use].copy(), stats
