"""Adapters that turn externally produced representations into prefixed EEG feature frames.

The ladder groups columns into rungs by name prefix: ``qeeg.``, ``conn.`` (from ``eeg/pipeline.py`` rows),
``morgoth.``, ``emb.<family>.`` and ``dyn.``. MORGOTH, CBraMod and dynamics extraction run elsewhere (on approved
compute); here they are PLACEHOLDER interfaces that accept arrays / DataFrames and validate them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(df).reset_index(drop=True)
    cols = [c if str(c).startswith(prefix) else prefix + str(c) for c in df.columns]
    out = df.copy()
    out.columns = cols
    bad = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    if bad:
        raise ValueError(f"{len(bad)} non-numeric columns for prefix {prefix!r}")
    return out


def eeg_frame_from_pipeline_rows(rows: pd.DataFrame, window: str = "primary", id_col: str = "recording_id",
                                 require_qc_pass: bool = True) -> pd.DataFrame:
    """Feature frame (qeeg.* / conn.* columns) from ``sortinghat.eeg.pipeline`` rows for one window.

    One row per recording; row order is the order of ``rows`` after filtering. The id column is dropped, so
    alignment with labels/baseline must be done by the caller before calling this (sort on the same key).
    """
    r = rows[rows["window"] == window]
    if require_qc_pass:
        r = r[r["qc_pass"].astype(bool)]
    feat = [c for c in r.columns if str(c).startswith(("qeeg.", "conn."))]
    return r[feat].reset_index(drop=True)


def morgoth_findings_frame(source) -> pd.DataFrame:
    """PLACEHOLDER loader for MORGOTH findings (per-recording class probabilities / burden estimates).

    Accepts a DataFrame (n x m). Reading MORGOTH output files from restricted storage is deliberately not
    implemented: a human-run job produces the frame.
    """
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        raise NotImplementedError("MORGOTH loader is a placeholder: supply a DataFrame of findings")
    return _prefixed(source, "morgoth.")


def load_morgoth_findings(path, recording_ids, window: str = "primary", require_qc_pass: bool = True):
    """MORGOTH findings (``morgoth.*`` columns) for ``recording_ids`` from the extractor's local_only parquet parts
    (``scripts/extract_morgoth.py features``), one row per id in the given order (NaN where a recording has no row or its
    window failed QC). Returns ``None`` when ``path`` holds no parts, so the caller passes it to ``assemble_eeg_frame``
    (which ignores ``None``) and the ladder skips the MORGOTH rungs cleanly (``available: False``)."""
    from ..morgoth.outputs import feature_columns, read_findings
    df = read_findings(path, window)
    if df is None:
        return None
    if require_qc_pass:
        df = df[df["qc_pass"].astype(bool)]
    cols = feature_columns(df)
    if not cols:
        return None
    out = df.set_index("recording_id")[cols].reindex(list(recording_ids)).reset_index(drop=True)
    if out.isna().all().all():
        return None
    return out.astype(float)


def embedding_frame(array, family: str = "cbramod") -> pd.DataFrame:
    """Frozen embeddings supplied as an (n x d) array (PLACEHOLDER interface for CBraMod / other encoders).

    Columns become ``emb.<family>.<j>``. Normalisation of embeddings is a fold-local preprocessing step.
    """
    a = np.asarray(array, dtype=float)
    if a.ndim != 2:
        raise ValueError("embeddings must be 2-D (n x d)")
    return pd.DataFrame(a, columns=[f"emb.{family}.{j}" for j in range(a.shape[1])])


def dynamics_frame(source) -> pd.DataFrame:
    """PLACEHOLDER for temporal-dynamics features (e.g. state transition / burst statistics) as a DataFrame."""
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        raise NotImplementedError("dynamics loader is a placeholder: supply a DataFrame")
    return _prefixed(source, "dyn.")


def assemble_eeg_frame(*frames: pd.DataFrame) -> pd.DataFrame:
    """Column-concatenate representation frames (same rows, same order) into one EEG frame."""
    frames = [f.reset_index(drop=True) for f in frames if f is not None]
    if not frames:
        raise ValueError("no frames supplied")
    n = len(frames[0])
    if any(len(f) != n for f in frames):
        raise ValueError("all representation frames must have the same number of rows")
    out = pd.concat(frames, axis=1)
    if out.columns.duplicated().any():
        raise ValueError("duplicate EEG column names")
    return out
