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


def _in_window(h: pd.Series, window: list[float]) -> pd.Series:
    return (h >= window[0]) & (h <= window[1])


def _events(ev: pd.DataFrame, item: str, window: list[float], acuity: bool = False) -> pd.DataFrame:
    m = (ev["item"] == item) & _in_window(ev["hours_from_t0"], window)
    if acuity and "acute" in ev.columns:
        m &= ev["acute"].map(lambda v: v is True or (not pd.isna(v) and bool(v)))
    return ev[m]


def _rbc_corrected(sub: pd.DataFrame, ev: pd.DataFrame, spec: dict) -> pd.Series:
    """Numeric values of ``sub`` (WBC rows) minus RBC/divisor where a same-timestamp RBC row exists."""
    vals = pd.to_numeric(sub["value"], errors="coerce")
    rbc = ev[ev["item"] == spec["item"]]
    out = []
    for v, h in zip(vals, sub["hours_from_t0"]):
        if pd.isna(v):
            out.append(v)
            continue
        same = pd.to_numeric(rbc.loc[[math.isclose(h, x, abs_tol=1e-6) for x in rbc["hours_from_t0"]], "value"],
                             errors="coerce").dropna()
        out.append(max(v - same.iloc[0] / spec.get("divisor", 500), 0.0) if len(same) else v)
    return pd.Series(out, index=sub.index, dtype=float)


def _eval(node: dict, ev: pd.DataFrame, fired: list[str], acuity: bool = False) -> bool:
    """Evaluate one rule node for one case's events; append fired leaf ids to ``fired``."""
    if "any_of" in node:
        results = [_eval(c, ev, fired, acuity) for c in node["any_of"]]    # evaluate all (collect ids)
        return any(results)
    if "all_of" in node:
        sub: list[str] = []
        ok = all([_eval(c, ev, sub, acuity) for c in node["all_of"]])
        if ok:
            fired.extend(sub)
            if node.get("id"):
                fired.append(node["id"])
        return ok
    sub = _events(ev, node["item"], node["window_hours"], acuity)
    if node["test"] == "event":
        vals = pd.to_numeric(sub["value"], errors="coerce")
        ok = bool((len(sub) > 0) and ((vals.isna()) | (vals >= 1)).any())
    elif node["test"] == "lab_threshold":
        if node.get("rbc_correct"):
            vals = _rbc_corrected(sub, ev, node["rbc_correct"]).dropna()
        else:
            vals = pd.to_numeric(sub["value"], errors="coerce").dropna()
        ok = bool(_OPS[node["op"]](vals, node["threshold"]).any()) if len(vals) else False
    else:
        raise ValueError(f"unknown test {node['test']!r}")
    if ok:
        fired.append(node["id"])
    return ok


def evaluate_case(events: pd.DataFrame, labels: list[str] | None = None,
                  cfg: dict | None = None) -> dict[str, list[str]]:
    """Anchors fired per label for ONE case's events. Empty list = no objective anchor."""
    cfg = cfg or load_anchor_config()
    labels = labels or list(cfg["labels"])
    out: dict[str, list[str]] = {}
    for lab in labels:
        node = cfg["labels"][lab]
        fired: list[str] = []
        ok = _eval(node, events, fired, bool(node.get("acuity_required")))
        out[lab] = sorted(set(fired)) if ok else []
    # Cross-label exclusions (E6 "without CNS infection")
    for lab in labels:
        ex = cfg["labels"][lab].get("exclude_if_label")
        if ex and out.get(lab):
            other = out.get(ex)
            if other is None:
                other_fired: list[str] = []
                onode = cfg["labels"][ex]
                other = other_fired if _eval(onode, events, other_fired, bool(onode.get("acuity_required"))) else []
            if other:
                out[lab] = []
    return out


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
    groups = {k: g for k, g in events.groupby("case_id")}
    ids = list(case_ids) if case_ids is not None else list(groups)
    empty = events.iloc[0:0]
    rows, fired_all = {}, {}
    for cid in ids:
        res = evaluate_case(groups.get(cid, empty), labs, cfg)
        fired_all[cid] = res
        rows[cid] = {l: bool(res[l]) for l in labs}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=labs)
    df.index.name = "case_id"
    df.attrs["fired"] = fired_all
    return df
