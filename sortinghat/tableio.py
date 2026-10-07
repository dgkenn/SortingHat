"""Read/write HEEDB-shaped tables (parquet if available, else CSV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_tables(tables: dict[str, pd.DataFrame], outdir: str | Path, fmt: str = "auto") -> list[Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "auto":
        fmt = "parquet" if parquet_available() else "csv"
    paths = []
    for name, df in tables.items():
        p = out / f"{name}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(p, index=False)
        elif fmt == "csv":
            df.to_csv(p, index=False)
        else:
            raise ValueError(f"unknown format {fmt}")
        paths.append(p)
    return paths


def load_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all schema tables from ``data_dir`` (``<name>.parquet`` or ``.csv``)."""
    d = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name in schema.TABLE_NAMES:
        pq, csv = d / f"{name}.parquet", d / f"{name}.csv"
        if pq.exists():
            df = pd.read_parquet(pq)
        elif csv.exists():
            df = pd.read_csv(csv, low_memory=False)
        else:
            raise FileNotFoundError(f"missing table '{name}' in data directory")
        for c in schema.datetime_cols(name):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        missing = [c for c in schema.columns(name) if c not in df.columns]
        if missing:
            raise ValueError(f"table '{name}' lacks expected columns: {missing}")
        tables[name] = df
    return tables
