"""Label-level DATA-SOURCE availability: not-assessable (NaN, never 0) where a label's source cannot fire (D-153).

Why. A silver label anchored on a source table that is (almost) empty for part of a site's record period can only be
positive in the rest of it. Coding the empty period as a confident NEGATIVE makes the label drift with the data feed, not
with the patients (E6 and blood cultures: a ~14x temporal observed/expected, docs/research/e6_drift_2026-10-09.md).

The rule (configured per label under ``availability_rules`` in ``configs/silver_anchors.yaml``; E6 today):

1. ``source_present[case]`` = the case has at least one row of ``source_item`` in the label's own anchor window for that
   item (``window_hours`` null = the window of the label's leaf on that item), from the anchor event table.
2. Within each site the cases are ordered by t0 (rank only, ties by case order) and cut into ``round(1 / bin_fraction)``
   bins of equal size (``bin_fraction`` 0.10 = ten bins of ~10% of the site's cases).
3. A bin whose share of ``source_present`` cases is below ``min_share`` is an ABSENT-SOURCE bin; contiguous absent-source
   bins form a region. Every case of such a bin is not assessable for the label, reason ``culture_source_absent``.
4. A site with fewer than ``2 * min_bin_cases`` cases is not evaluated (its eras cannot be characterised); at most
   ``cases // min_bin_cases`` bins are used for a small site. Nothing is changed for such a site.
5. A case whose label is POSITIVE is never turned not-assessable (a positive proves the source existed). With
   ``keep_cases_with_source`` true the negatives that have a source row are kept too (they are validly assessed); the
   default false makes the whole bin not assessable except its positives.

The rule reads only structured data availability (event-row presence and the t0 rank). It never reads an outcome label
(except to protect positives), an EEG feature or an EEG report. Only counts are returned; nothing here is record-level
apart from the boolean mask returned to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REASON_CULTURE_SOURCE_ABSENT = "culture_source_absent"
DEFAULT_RULE = {"enabled": True, "source_item": None, "window_hours": None, "bin_fraction": 0.10, "min_share": 0.10,
                "min_bin_cases": 50, "keep_cases_with_source": False, "reason": REASON_CULTURE_SOURCE_ABSENT}


def leaf_window(anchor_cfg: dict, label: str, item: str) -> list[float]:
    """Window of the first leaf of ``label``'s rule tree that tests ``item``."""
    def walk(n):
        if n.get("item") == item and "window_hours" in n:
            return n["window_hours"]
        for k in ("any_of", "all_of"):
            for c in n.get(k, []):
                w = walk(c)
                if w is not None:
                    return w
        return None
    w = walk(anchor_cfg["labels"][label])
    if w is None:
        raise ValueError(f"label {label} has no anchor leaf on item {item!r}: give availability_rules.{label}.window_hours")
    return [float(w[0]), float(w[1])]


def resolve_rules(anchor_cfg: dict, overrides: dict | None = None) -> dict:
    """label -> complete rule (defaults < YAML < overrides), validated."""
    raw = {lab: dict(r or {}) for lab, r in (anchor_cfg.get("availability_rules") or {}).items()}
    for lab, r in (overrides or {}).items():
        raw.setdefault(lab, {}).update(r or {})
    out = {}
    for lab, r in raw.items():
        rule = {**DEFAULT_RULE, **r}
        if lab not in anchor_cfg["labels"]:
            raise ValueError(f"availability_rules names unknown label {lab!r}")
        if not rule["source_item"]:
            raise ValueError(f"availability_rules.{lab}.source_item is required")
        if not (0 < float(rule["bin_fraction"]) <= 0.5):
            raise ValueError(f"availability_rules.{lab}.bin_fraction must be in (0, 0.5]")
        if not (0 <= float(rule["min_share"]) <= 1):
            raise ValueError(f"availability_rules.{lab}.min_share must be in [0, 1]")
        if int(rule["min_bin_cases"]) < 1:
            raise ValueError(f"availability_rules.{lab}.min_bin_cases must be >= 1")
        rule["window_hours"] = (leaf_window(anchor_cfg, lab, rule["source_item"]) if rule["window_hours"] is None
                                else [float(x) for x in rule["window_hours"]])
        out[lab] = rule
    return out


def source_present(events: pd.DataFrame, case_ids, item: str, window: list[float]) -> np.ndarray:
    """bool per case: at least one ``item`` row of the anchor event table with ``hours_from_t0`` inside the closed window."""
    ids = list(case_ids)
    if events is None or not len(events):
        return np.zeros(len(ids), bool)
    h = pd.to_numeric(events["hours_from_t0"], errors="coerce")
    sel = events[(events["item"] == item) & h.between(window[0], window[1])]
    return pd.Index(ids).isin(pd.unique(sel["case_id"])) if len(sel) else np.zeros(len(ids), bool)


def bin_site(n: int, bin_fraction: float, min_bin_cases: int) -> int:
    """Number of equal rank bins used for a site of ``n`` cases (0 = site not evaluated)."""
    nb = min(int(round(1.0 / bin_fraction)), n // int(min_bin_cases))
    return nb if nb >= 2 else 0


def rank_bins(t0: np.ndarray, n_bins: int) -> np.ndarray:
    """Ordinal bin 0..n_bins-1 of the t0 rank (stable: ties keep input order). Rank only; no time value leaves."""
    n = len(t0)
    order = np.argsort(np.asarray(t0).astype("datetime64[us]"), kind="stable")
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    return np.minimum(rank * n_bins // n, n_bins - 1)


@dataclass
class AvailabilityResult:
    labels: pd.DataFrame                                     # labels with the not-assessable cases set to NA
    reason: pd.Series                                        # case_id -> reason ("" = untouched)  (RECORD-LEVEL, in memory)
    info: dict = field(default_factory=dict)                 # label -> aggregate counts (raw, not yet suppressed)


def apply_availability(labels: pd.DataFrame, cases: pd.DataFrame, events: pd.DataFrame, anchor_cfg: dict,
                       overrides: dict | None = None, site_col: str = "SiteID") -> AvailabilityResult:
    """Apply ``availability_rules`` to ``labels`` (case_id-indexed nullable-boolean frame). ``cases`` has case_id, t0, SiteID."""
    labels = labels.copy()
    reason = pd.Series("", index=labels.index, dtype=object)
    info: dict = {}
    c = cases.set_index("case_id").reindex(labels.index)
    for lab, rule in resolve_rules(anchor_cfg, overrides).items():
        if not rule["enabled"] or lab not in labels.columns or labels[lab].isna().all():
            continue                                         # disabled, absent, or already unavailable as a whole
        present = source_present(events, labels.index, rule["source_item"], rule["window_hours"])
        positive = labels[lab].fillna(False).to_numpy(bool)
        flagged = np.zeros(len(labels), bool)                # case lies in an absent-source bin
        sites_info: dict = {}
        for site, ix in c.groupby(site_col, sort=True).indices.items():
            nb = bin_site(len(ix), float(rule["bin_fraction"]), int(rule["min_bin_cases"]))
            if nb == 0:
                sites_info[site] = {"n_cases": len(ix), "evaluated": False, "n_bins": 0, "bins": [], "n_bins_absent": 0}
                continue
            b = rank_bins(c["t0"].to_numpy()[ix], nb)
            bins, absent = [], 0
            for k in range(nb):
                m = b == k
                n_k, n_src = int(m.sum()), int(present[ix][m].sum())
                a = n_src < float(rule["min_share"]) * n_k   # share below the threshold (strict)
                absent += int(a)
                flagged[ix[m]] = a
                bins.append({"n_cases": n_k, "n_with_source": n_src, "absent": bool(a)})
            sites_info[site] = {"n_cases": len(ix), "evaluated": True, "n_bins": nb, "bins": bins, "n_bins_absent": absent}
        na = flagged & ~positive
        if rule["keep_cases_with_source"]:
            na &= ~present
        assert not (na & positive).any(), "availability rule must never make a positive not assessable"
        col = labels[lab].copy()
        col[na] = pd.NA
        labels[lab] = col
        reason[na] = rule["reason"]
        for site, ix in c.groupby(site_col, sort=True).indices.items():
            sites_info[site].update({"n_not_assessable": int(na[ix].sum()), "n_in_absent_bins": int(flagged[ix].sum()),
                                     "n_positive_kept_in_absent_bins": int((flagged & positive)[ix].sum())})
        info[lab] = {"reason": rule["reason"], "source_item": rule["source_item"], "window_hours": rule["window_hours"],
                     "bin_fraction": float(rule["bin_fraction"]), "min_share": float(rule["min_share"]),
                     "min_bin_cases": int(rule["min_bin_cases"]), "keep_cases_with_source": bool(rule["keep_cases_with_source"]),
                     "per_site": sites_info, "n_not_assessable": int(na.sum())}
    return AvailabilityResult(labels, reason, info)


def report_block(info: dict, suppress_group, suppress_count, suppress_proportion) -> dict:
    """Aggregate-only, suppressed form of ``AvailabilityResult.info`` for the silver report. ``suppress_group`` is the
    complementary-suppression helper of the report. Bin contents are reported by ordinal position only (no date, no rank)."""
    out: dict = {}
    for lab, d in info.items():
        sites = sorted(d["per_site"])
        tot_na = d["n_not_assessable"]
        n_by = {s: d["per_site"][s]["n_cases"] for s in sites}
        na_by = {s: d["per_site"][s].get("n_not_assessable", 0) for s in sites}
        sup_na = suppress_group(na_by, tot_na)
        per_site = {}
        for s in sites:
            p = d["per_site"][s]
            blk = {"evaluated": p["evaluated"], "n_bins": p["n_bins"], "n_bins_source_absent": p["n_bins_absent"],
                   "n_not_assessable": sup_na[s],
                   "share_not_assessable": (suppress_proportion(na_by[s], n_by[s]) if sup_na[s] != "<11" else "<11"),
                   "n_positive_kept_in_absent_bins": suppress_count(p.get("n_positive_kept_in_absent_bins", 0)),
                   "bin_share_with_source": [suppress_proportion(b["n_with_source"], b["n_cases"]) for b in p["bins"]]}
            per_site[s] = blk
        out[lab] = {"reason": d["reason"], "source_item": d["source_item"], "window_hours": d["window_hours"],
                    "bin_fraction": d["bin_fraction"], "min_share": d["min_share"], "min_bin_cases": d["min_bin_cases"],
                    "keep_cases_with_source": d["keep_cases_with_source"],
                    "n_not_assessable": suppress_count(tot_na),
                    "per_site": per_site}
    return out
