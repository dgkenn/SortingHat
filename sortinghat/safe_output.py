"""Aggregate-only reporting helpers with small-cell suppression.

Hard rule: code that may run on restricted BDSP/HEEDB data must never emit
record-level data. Everything that leaves a script (stdout, files) goes through
``assert_aggregate_only`` and uses suppressed counts (n < 11 -> "<11").
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SUPPRESS_BELOW = 11
SUPPRESSED = "<11"


class AggregateOnlyError(ValueError):
    """Raised when something that looks record-level is about to be emitted."""


# --------------------------------------------------------------------------
# Suppression primitives
# --------------------------------------------------------------------------
def suppress_count(n: int | float, k: int = SUPPRESS_BELOW) -> int | str:
    n = int(n)
    return SUPPRESSED if n < k else n


def suppress_proportion(num: int | float, den: int | float, k: int = SUPPRESS_BELOW,
                        ndigits: int = 4) -> float | str:
    """Proportion, suppressed if either cell (num, den-num) or den is < k.

    The complement is checked too so a suppressed numerator cannot be recovered
    by subtraction.
    """
    num, den = int(num), int(den)
    if den < k or num < k or (den - num) < k:
        return SUPPRESSED
    return round(num / den, ndigits)


def safe_quantiles(values: Iterable[float], qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
                   k: int = SUPPRESS_BELOW, ndigits: int = 3) -> dict[str, float | str]:
    """Quantiles of a numeric sample; extremes (min/max) are refused."""
    for q in qs:
        if not 0.0 < q < 1.0:
            raise AggregateOnlyError("min/max (q<=0 or q>=1) are record-level; refused")
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    out: dict[str, float | str] = {}
    if len(s) < k:
        return {f"q{round(q * 100):02d}": SUPPRESSED for q in qs}
    for q in qs:
        out[f"q{round(q * 100):02d}"] = round(float(s.quantile(q)), ndigits)
    return out


def suppress_table(df: pd.DataFrame, count_cols: Sequence[str], k: int = SUPPRESS_BELOW) -> pd.DataFrame:
    """Return a copy with the given count columns suppressed (as strings where <k)."""
    out = df.copy()
    for c in count_cols:
        out[c] = [suppress_count(v, k) for v in out[c]]
        out[c] = out[c].astype(object)
    return out


def group_counts(df: pd.DataFrame, by: str | Sequence[str], k: int = SUPPRESS_BELOW) -> pd.DataFrame:
    """Suppressed group sizes (``n``) by non-identifier columns."""
    by_l = [by] if isinstance(by, str) else list(by)
    sizes = df.groupby(by_l, observed=True).size().reset_index(name="n")
    return suppress_table(sizes, ["n"], k)


# --------------------------------------------------------------------------
# Row-level identifier guard
# --------------------------------------------------------------------------
_NAME_RE = re.compile(
    r"(patient|subject|session|mrn|bids|noteid|encounter|accession|csn|personid|^sub$|^id$)", re.I)
_NAME_EXEMPT_RE = re.compile(r"^(n|num|count)_|_count$|^n$", re.I)
_VALUE_RE = re.compile(
    r"(\bSYN[SI]\d{4}\d{6}\b|\bsub-[A-Za-z0-9]+|\bses-[A-Za-z0-9]+|\b[SI]\d{4}\d{6,}\b)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{6,}")


def _check_name(name: Any) -> None:
    if name is None:
        return
    s = re.sub(r"[_\s]", "", str(name))
    if _NAME_EXEMPT_RE.search(str(name)):
        return
    if _NAME_RE.search(s):
        raise AggregateOnlyError(f"identifier-like column/key name refused: {name!r}")


def _check_text(text: str, known_ids: set[str] | None) -> None:
    if _VALUE_RE.search(text):
        raise AggregateOnlyError("identifier-like value pattern found in output")
    if known_ids:
        for tok in _TOKEN_RE.findall(text):
            if tok in known_ids:
                raise AggregateOnlyError("a known record identifier appears in output")


def _check_frame(df: pd.DataFrame, known_ids: set[str] | None) -> None:
    for c in df.columns:
        _check_name(c)
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            raise AggregateOnlyError(f"row-level timestamp column refused: {c!r}")
    _check_name(df.index.name)
    for n in getattr(df.index, "names", []):
        _check_name(n)
    if pd.api.types.is_datetime64_any_dtype(df.index):
        raise AggregateOnlyError("row-level timestamp index refused")
    for c in df.columns:
        col = df[c]
        if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
            for v in col.astype(str).unique():
                _check_text(v, known_ids)
    if not isinstance(df.index, pd.RangeIndex) and not pd.api.types.is_numeric_dtype(df.index):
        for v in df.index.astype(str).unique():
            _check_text(v, known_ids)


def assert_aggregate_only(obj: Any, known_ids: Iterable[str] | None = None, _depth: int = 0) -> None:
    """Raise ``AggregateOnlyError`` if ``obj`` looks like it carries row-level identifiers.

    Checks DataFrames/Series (identifier-like column names, datetime columns,
    ID-like values), dict keys, strings (ID patterns and optionally a set of
    known IDs). Containers are walked recursively.
    """
    ids = set(known_ids) if known_ids is not None and not isinstance(known_ids, set) else known_ids
    if _depth > 20:
        raise AggregateOnlyError("structure too deep")
    if obj is None or isinstance(obj, (bool, int, float, np.integer, np.floating, np.bool_)):
        return
    if isinstance(obj, str):
        _check_text(obj, ids)
    elif isinstance(obj, pd.DataFrame):
        _check_frame(obj, ids)
    elif isinstance(obj, pd.Series):
        _check_frame(obj.to_frame(name=obj.name if obj.name is not None else "value"), ids)
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _check_name(k)
            _check_text(str(k), ids)
            assert_aggregate_only(v, ids, _depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            assert_aggregate_only(v, ids, _depth + 1)
    else:
        raise AggregateOnlyError(f"unsupported output type {type(obj).__name__}")


# --------------------------------------------------------------------------
# Guarded emitters
# --------------------------------------------------------------------------
def safe_print(*parts: Any, known_ids: Iterable[str] | None = None) -> None:
    text = " ".join(str(p) for p in parts)
    assert_aggregate_only(text, known_ids)
    print(text)


def safe_write_text(path: str | Path, text: str, known_ids: Iterable[str] | None = None) -> Path:
    assert_aggregate_only(text, known_ids)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def safe_write_json(path: str | Path, obj: Any, known_ids: Iterable[str] | None = None) -> Path:
    assert_aggregate_only(obj, known_ids)
    return safe_write_text(path, json.dumps(obj, indent=2, sort_keys=False, default=_json_default),
                           known_ids)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_local_only(path: str | Path, text: str) -> Path:
    """Write human-only content (e.g. a hand-check ID list) to a private local file.

    Deliberately bypasses the aggregate guard; callers must never print or
    return the content. File mode 0600.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p
