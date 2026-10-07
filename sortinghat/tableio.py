"""Write and read HEEDB-layout tables on a local directory (synthetic data, or a local mirror).

On disk the layout mirrors the BDSP access point (see ``sortinghat.data_io.TABLES``)::

    EEG/eeg-metadata/<SITE>_eeg_metadata_2026_04_30.csv
    EEG/HEEDB_Metadata/<SITE>_EEG__reports_findings.csv
    EEG/HEEDB_Metadata/{HEEDB_patients,HEEDB_ICD10_for_Neurology,HEEDB_Medication_ATC}.csv
    OMOP/Merged/<table>/part-0000N.parquet
    Imaging/imaging_metadata/part-00000.parquet     (ASSUMED location)

Datetimes are written as text ``YYYY-MM-DD HH:MM:SS`` like the real files. Reading goes through
``sortinghat.data_io`` (``LocalStore`` stands in for the S3 client), so the same code path serves synthetic
and real runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data_io, schema

RELEASE = "2026_04_30"          # release date in the synthetic eeg-metadata file names
OMOP_PARTS = 2                  # synthetic OMOP tables are split in two parts, like the real multi-part tables


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _as_text(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Datetime columns -> text in the real format; other columns unchanged."""
    out = df.copy()
    for c in schema.datetime_cols(table):
        if c in out.columns:
            ts = pd.to_datetime(out[c], errors="coerce")
            txt = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
            us = ts.dt.microsecond.fillna(0).astype(int)
            txt = txt.where(us == 0, txt + "." + us.astype(str).str.zfill(6))     # optional .ffffff
            out[c] = txt.astype(object).where(ts.notna(), None)
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_tables(tables: dict[str, pd.DataFrame], outdir: str | Path, fmt: str = "auto") -> list[Path]:
    """Write ``tables`` (keys = ``schema`` table names) in the real layout. ``fmt`` is accepted for
    backwards compatibility and ignored: CSV vs parquet is dictated by the real layout."""
    if not parquet_available():
        raise RuntimeError("pyarrow is required to write the OMOP/Merged parquet layout (pip install pyarrow)")
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    site_of_session = (tables["eeg_metadata"].drop_duplicates("SessionID")
                       .set_index("SessionID")["SiteID"] if "eeg_metadata" in tables else None)
    for name, raw in tables.items():
        spec = data_io.TABLES[name]
        df = _as_text(name, raw)
        if name == "eeg_metadata":
            for site, g in df.groupby("SiteID", sort=True):
                paths.append(_write_csv(g, root / f"{data_io.EEG_METADATA_PREFIX}{site}_eeg_metadata_{RELEASE}.csv"))
        elif name == "reports_findings":
            sites = df["SessionID"].map(site_of_session)
            for site, g in df.groupby(sites, sort=True):
                paths.append(_write_csv(g, root / spec.pattern.format(site=site)))
        elif spec.kind == "csv_global":
            paths.append(_write_csv(df, root / spec.pattern))
        else:                                       # parquet directory (OMOP table or the assumed Imaging/)
            d = root / (spec.pattern.format(table=name[len("omop_"):]))
            d.mkdir(parents=True, exist_ok=True)
            n_parts = OMOP_PARTS if len(df) > 1 and spec.kind == "omop_parquet" else 1
            for k, idx in enumerate(_split(len(df), n_parts)):
                p = d / f"part-{k:05d}.parquet"
                df.iloc[idx].to_parquet(p, index=False)
                paths.append(p)
    return paths


def _split(n: int, parts: int):
    bounds = [round(i * n / parts) for i in range(parts + 1)]
    return [slice(bounds[i], bounds[i + 1]) for i in range(parts)]


def load_tables(data_dir: str | Path, sites: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every schema table from a local layout through ``data_io`` (whole tables; for tests and
    synthetic-scale data only). Missing tables raise ``FileNotFoundError``; missing columns are left out
    (use the audit's ``--dry-run-schema`` to see them). Dtypes follow the schema."""
    s3 = data_io.LocalStore(data_dir)
    sites = sites or data_io.discover_sites(s3)
    tables: dict[str, pd.DataFrame] = {}
    for name in schema.TABLE_NAMES:
        spec = data_io.TABLES[name]
        if spec.kind == "csv_site":
            frames = []
            for site in sites:
                try:
                    frames.append(data_io.read_csv_table(name, site, s3=s3))
                except FileNotFoundError:
                    continue
            if not frames:
                raise FileNotFoundError(f"missing table '{name}' in data directory")
            df = pd.concat(frames, ignore_index=True)
        elif spec.kind == "csv_global":
            df = data_io.read_csv_table(name, s3=s3)
        else:
            if spec.kind == "omop_parquet":
                kw: dict = {"columns": schema.columns(name)}
                tname = name[len("omop_"):]
            else:
                kw = {"columns": schema.columns(name), "prefix": spec.pattern}
                tname = name
            batches = [b.to_pandas() for b in data_io.iter_omop_batches(tname, s3=s3, **kw)]
            if not batches:
                raise FileNotFoundError(f"missing table '{name}' in data directory")
            df = pd.concat(batches, ignore_index=True)
        tables[name] = schema.coerce_types(name, df)
    return tables
