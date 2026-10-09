"""Objective-anchor rules for silver positives, driven by ``configs/silver_anchors.yaml``.

Input is a tidy *anchor event table* (one row per measurement/event), already mapped from
OMOP/LOINC to canonical ``item`` keys (``loinc_to_item`` gives the map):

    case_id | item | value | hours_from_t0

``value`` is numeric for labs; for flag items (imaging findings, arrest, ...) use 1.
``hours_from_t0`` = (event time - EEG start) in hours. Thresholds in the YAML were signed off by the
project lead (delegated decision, 2026-10-07) and need co-investigator re-review before protocol lock.

Optional column ``acute`` (bool): when a label sets ``acuity_required: true`` (E1) and the column exists,
only rows with ``acute == True`` count. Lab-threshold leaves may set ``rbc_correct`` (CSF WBC): the value is
reduced by RBC/divisor using the RBC row with the same ``hours_from_t0`` (same tap) when one exists.
"""

from __future__ import annotations

import math
import operator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "silver_anchors.yaml"
_OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
EVENT_COLUMNS = ("case_id", "item", "value", "hours_from_t0")


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_anchor_config(path: str | Path | None = None) -> dict:
    return _load(str(path or CONFIG_PATH))


def loinc_to_item(cfg: dict | None = None) -> dict[str, str]:
    cfg = cfg or load_anchor_config()
    return {code: item for item, spec in cfg["items"].items() for code in spec.get("loinc", [])}


def is_proposed(cfg: dict | None = None) -> bool:
    return "PROPOSED" in str((cfg or load_anchor_config()).get("status", ""))


def is_signed_off(cfg: dict | None = None) -> bool:
    return str((cfg or load_anchor_config()).get("status", "")).startswith("SIGNED OFF")


class _Case:
    """ONE case's events as numpy arrays per item (row order kept): ``hours``, numeric ``value`` and, when the event table has
    an ``acute`` column, its boolean ``acute``. Built once per case so each rule leaf is a few array operations instead of a
    pandas filter (the rules are evaluated for every label, leaf and case)."""

    __slots__ = ("items", "has_acute")
    _EMPTY = (np.empty(0), np.empty(0), np.empty(0, dtype=bool))

    def __init__(self, items: dict, has_acute: bool):
        self.items, self.has_acute = items, has_acute

    def get(self, item: str):
        return self.items.get(item, self._EMPTY)


def _as_arrays(ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Whole-table arrays: hours (float), numeric value (float, NaN when not numeric), acute (bool, or None without the column)."""
    h = pd.to_numeric(ev["hours_from_t0"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(ev["value"], errors="coerce").to_numpy(dtype=float)
    a = None
    if "acute" in ev.columns:
        a = np.array([x is True or (not pd.isna(x) and bool(x)) for x in ev["acute"]], dtype=bool)
    return h, v, a


def _cases_of(ev: pd.DataFrame, case_ids: list | None = None) -> dict:
    """``case_id -> _Case`` for every case in ``ev`` (or only ``case_ids``)."""
    out: dict = {}
    if len(ev) == 0:
        return out
    h, v, a = _as_arrays(ev)
    has_acute = a is not None
    if a is None:
        a = np.zeros(len(ev), dtype=bool)
    wanted = None if case_ids is None else set(case_ids)
    for (cid, item), pos in ev.groupby(["case_id", "item"], sort=False).indices.items():
        if wanted is not None and cid not in wanted:
            continue
        out.setdefault(cid, _Case({}, has_acute)).items[item] = (h[pos], v[pos], a[pos])
    return out


def _leaf_rows(c: _Case, item: str, window: list[float], acuity: bool):
    h, v, a = c.get(item)
    m = (h >= window[0]) & (h <= window[1])
    if acuity and c.has_acute:
        m &= a
    return h[m], v[m]


def _rbc_corrected(sub_h: np.ndarray, sub_v: np.ndarray, c: _Case, spec: dict) -> np.ndarray:
    """Values of the WBC rows minus RBC/divisor where an RBC row with the same ``hours_from_t0`` (same tap) exists."""
    rh, rv, _ = c.get(spec["item"])
    div = spec.get("divisor", 500)
    out = []
    for v, h in zip(sub_v, sub_h):
        if math.isnan(v):
            out.append(v)
            continue
        same = [x for hh, x in zip(rh, rv) if math.isclose(h, hh, abs_tol=1e-6) and not math.isnan(x)]
        out.append(max(v - same[0] / div, 0.0) if same else v)
    return np.array(out, dtype=float)


def _eval(node: dict, c: _Case, fired: list[str], acuity: bool = False) -> bool:
    """Evaluate one rule node for one case's events; append fired leaf ids to ``fired``."""
    if "any_of" in node:
        results = [_eval(ch, c, fired, acuity) for ch in node["any_of"]]    # evaluate all (collect ids)
        return any(results)
    if "all_of" in node:
        sub: list[str] = []
        ok = all([_eval(ch, c, sub, acuity) for ch in node["all_of"]])
        if ok:
            fired.extend(sub)
            if node.get("id"):
                fired.append(node["id"])
        return ok
    sh, sv = _leaf_rows(c, node["item"], node["window_hours"], acuity)
    if node["test"] == "event":
        ok = bool((len(sh) > 0) and (np.isnan(sv) | (sv >= 1)).any())
    elif node["test"] == "lab_threshold":
        vals = _rbc_corrected(sh, sv, c, node["rbc_correct"]) if node.get("rbc_correct") else sv
        vals = vals[~np.isnan(vals)]
        ok = bool(_OPS[node["op"]](vals, node["threshold"]).any()) if len(vals) else False
    else:
        raise ValueError(f"unknown test {node['test']!r}")
    if ok:
        fired.append(node["id"])
    return ok


def _evaluate(c: _Case, labels: list[str], cfg: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for lab in labels:
        node = cfg["labels"][lab]
        fired: list[str] = []
        ok = _eval(node, c, fired, bool(node.get("acuity_required")))
        out[lab] = sorted(set(fired)) if ok else []
    # Cross-label exclusions (E6 "without CNS infection")
    for lab in labels:
        ex = cfg["labels"][lab].get("exclude_if_label")
        if ex and out.get(lab):
            other = out.get(ex)
            if other is None:
                other_fired: list[str] = []
                onode = cfg["labels"][ex]
                other = other_fired if _eval(onode, c, other_fired, bool(onode.get("acuity_required"))) else []
            if other:
                out[lab] = []
    return out


def evaluate_case(events: pd.DataFrame, labels: list[str] | None = None,
                  cfg: dict | None = None) -> dict[str, list[str]]:
    """Anchors fired per label for ONE case's events. Empty list = no objective anchor."""
    cfg = cfg or load_anchor_config()
    labels = labels or list(cfg["labels"])
    c = _Case({}, "acute" in events.columns)
    if len(events):
        h, v, a = _as_arrays(events)
        a = a if a is not None else np.zeros(len(events), dtype=bool)
        for item, pos in events.groupby("item", sort=False).indices.items():
            c.items[item] = (h[pos], v[pos], a[pos])
    return _evaluate(c, labels, cfg)


def silver_anchor_table(events: pd.DataFrame, case_ids: list[str] | None = None,
                        cfg: dict | None = None) -> pd.DataFrame:
    """Wide boolean table: index case_id, one column per label (True = anchored positive).

    Also returns fired anchor ids in ``DataFrame.attrs['fired']`` (case_id -> label -> ids),
    for local audit only (record-level; never print).
    """
    cfg = cfg or load_anchor_config()
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"anchor event table missing columns: {missing}")
    labs = list(cfg["labels"])
    ids = list(case_ids) if case_ids is not None else sorted(pd.unique(events["case_id"].dropna()))
    cases = _cases_of(events, ids)
    blank = _Case({}, "acute" in events.columns)
    rows, fired_all = {}, {}
    for cid in ids:
        res = _evaluate(cases.get(cid, blank), labs, cfg)
        fired_all[cid] = res
        rows[cid] = {l: bool(res[l]) for l in labs}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=labs)
    df.index.name = "case_id"
    df.attrs["fired"] = fired_all
    return df
