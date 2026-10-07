"""Aggregate-only serialisation of ladder and control results (through ``sortinghat.safe_output``).

Per-patient vectors (``RungResult.d``) never leave memory. Sites appear only as pseudonyms (site_1, ...), counts
under 11 are shown as "<11", and a site-level Delta computed on fewer than 11 patients is suppressed too.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..safe_output import SUPPRESS_BELOW, SUPPRESSED, safe_write_json, suppress_count


def clean(obj, nd: int = 5):
    """Recursively convert numpy types / NaN into plain JSON-safe values (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, nd) for v in obj]
    if isinstance(obj, np.ndarray):
        raise ValueError("arrays are record-level; summarise before reporting")
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else round(f, nd)
    return obj


def _ci(d: dict) -> dict:
    return {k: d.get(k) for k in ("estimate", "lo", "hi", "alpha", "n_boot") if k in d} if d else {}


def _supp_delta(entry: dict) -> dict:
    n = int(entry.get("n", 0))
    return {"n": suppress_count(n), "delta": SUPPRESSED if n < SUPPRESS_BELOW else entry.get("delta")}


def rung_to_aggregate(r, site_labels: dict) -> dict:
    if not r.available:
        return {"available": False}
    if not r.has_eeg:
        return {"available": True, "has_eeg": False, "delta": 0.0, "note": "no EEG columns; identical to baseline"}
    out = {
        "available": True, "has_eeg": True, "n_eeg_features": r.n_eeg_features,
        "delta": r.delta, "loss_model": r.loss_model, "loss_baseline": r.loss_baseline,
        "ci_primary_within_site": _ci(r.ci_primary),
        "ci_95": {m: _ci(v) for m, v in r.ci_modes.items()},
        "per_site": {site_labels.get(str(s), str(s)): _supp_delta(v) for s, v in r.per_site.items()},
        "all_sites_favorable": r.all_sites_favorable, "site_weighted_delta": r.site_weighted_delta,
        "h1h2_rule_met": r.rule_met,
    }
    if r.per_label:
        out["per_label"] = {k: {**{kk: vv for kk, vv in v.items() if kk != "n"}, "n": suppress_count(v["n"])}
                            for k, v in r.per_label.items()}
    return out


def ladder_to_aggregate(res) -> dict:
    out = {"split": res.split, "label_names": list(res.label_names), "primary_labels": list(res.primary_labels),
           "n_eval": suppress_count(res.n_eval), "reference": clean(res.reference), "folds": clean(res.fold_info),
           "rungs": {}}
    for (b, rung), r in res.rungs.items():
        out["rungs"].setdefault(b, {})[rung] = clean(rung_to_aggregate(r, res.site_labels))
    return out


def write_results(path, ladder, controls: dict | None = None, extra: dict | None = None) -> Path:
    """Write the aggregate-only results JSON (guarded by ``assert_aggregate_only``)."""
    obj = {"ladder": ladder.to_aggregate() if hasattr(ladder, "to_aggregate") else clean(ladder)}
    if controls is not None:
        obj["controls"] = clean(controls)
    if extra is not None:
        obj["extra"] = clean(extra)
    return safe_write_json(path, obj)
