"""Read the extractor's local_only parquet parts back (one row per recording x window), de-duplicated.

Record-level: the frames returned here are for joins inside a job, never for printing (CLAUDE.md rules 3 and 5).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_PREFIX = "morgoth."


def part_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("part-*.parquet"))
    return [p] if p.exists() else []


def read_findings(path: str | Path, window: str | None = "primary") -> pd.DataFrame | None:
    """All parts under ``path`` (a directory of ``part-*.parquet`` or one parquet file) for ``window``, de-duplicated on
    ``(recording_id, window)`` (a crash between a part write and its ledger write can duplicate a recording). ``None``
    when nothing is there (the ladder then skips the MORGOTH rung)."""
    files = part_files(path)
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["recording_id", "window"], keep="last")
    if window is not None:
        df = df[df["window"] == window]
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith(FEATURE_PREFIX)]
