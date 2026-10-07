"""Phase 0b: label-feasibility pilot (200 blinded cases). Aggregate-only outputs.

Pieces
  draw_pilot_sample   seeded, site-stratified draw (IDs go to a local_only file, never printed)
  label_kappas        per-label Cohen's (binary) and weighted (ordinal) kappa between two reviewers
  minutes_summary     minutes per case (quantiles, n >= 11 only)
  eeg_only_share      share of positive silver labels whose ONLY evidence was EEG-derived
  gate0_pilot_check   Gate 0: kappa >= 0.6 for >= 4 primary families and >= 3 families with >= 10% prevalence
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..safe_output import (SUPPRESSED, SUPPRESS_BELOW, safe_quantiles, suppress_count,
                           suppress_proportion, write_local_only)
from .banned_evidence import EvidenceSource, classify_evidence
from .ontology import GoldState, primary_endpoint_labels
from .stats import cohen_kappa

PILOT_N = 200
GATE_KAPPA = 0.6
GATE_MIN_KAPPA_FAMILIES = 4
GATE_MIN_PREVALENT_FAMILIES = 3
GATE_PREVALENCE = 0.10


# ---------------------------------------------------------------- sampling
def allocate_by_site(site_sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Proportional allocation (largest remainder), >=1 per site when n allows, capped at site size."""
    sites = sorted(site_sizes)
    total = sum(site_sizes.values())
    n = min(n, total)
    alloc = {s: 0 for s in sites}
    if n >= len(sites):
        alloc = {s: min(1, site_sizes[s]) for s in sites}
    remaining = n - sum(alloc.values())
    while remaining > 0:
        room = {s: site_sizes[s] - alloc[s] for s in sites if site_sizes[s] > alloc[s]}
        if not room:
            break
        tot_room = sum(site_sizes[s] for s in room)
        quota = {s: remaining * site_sizes[s] / tot_room for s in room}
        take = {s: min(int(np.floor(q)), room[s]) for s, q in quota.items()}
        left = remaining - sum(take.values())
        order = sorted(room, key=lambda s: (-(quota[s] - np.floor(quota[s])), s))
        for s in order:
            if left <= 0:
                break
            if take[s] < room[s]:
                take[s] += 1; left -= 1
        for s, t in take.items():
            alloc[s] += t
        remaining = n - sum(alloc.values())
    return alloc


def draw_pilot_sample(cohort: pd.DataFrame, n: int = PILOT_N, seed: int = 20261007,
                      id_col: str = "case_id", site_col: str = "site") -> tuple[list[str], dict[str, int | str]]:
    """Seeded site-stratified sample. Returns (case ids, suppressed allocation by site).

    The id list is RECORD-LEVEL: write it with ``save_pilot_ids`` (local_only, mode 0600); do
    not print it. Deterministic for a given cohort and seed.
    """
    rng = np.random.default_rng(seed)
    sizes = cohort.groupby(site_col).size().to_dict()
    alloc = allocate_by_site(sizes, n)
    chosen: list[str] = []
    for site in sorted(alloc):
        ids = np.array(sorted(cohort.loc[cohort[site_col] == site, id_col].astype(str)))
        take = rng.choice(len(ids), size=alloc[site], replace=False)
        chosen.extend(ids[np.sort(take)].tolist())
    return chosen, {s: suppress_count(a) for s, a in alloc.items()}


def save_pilot_ids(ids: Sequence[str], path: str | Path) -> Path:
    return write_local_only(path, "\n".join(ids) + "\n")


# ---------------------------------------------------------------- agreement
def _to_state(x) -> GoldState:
    return GoldState.parse(x)


def label_kappas(ratings: pd.DataFrame, labels: Sequence[str], *, reviewers: tuple[str, str] | None = None,
                 min_n: int = SUPPRESS_BELOW) -> dict[str, dict]:
    """Per-label agreement between two reviewers.

    ``ratings``: case_id, label, reviewer, state (absent/possible/probable/definite/unassessable).
    Reports: n paired; n unassessable (either reviewer; excluded from kappa); linear- and
    quadratic-weighted kappa on the 4-level ordinal scale; unweighted binary kappa for
    positive = probable/definite; prevalence (mean of the two reviewers' positive rates).
    """
    revs = reviewers or tuple(sorted(ratings["reviewer"].unique()))[:2]
    if len(revs) != 2:
        raise ValueError("need exactly two reviewers")
    out: dict[str, dict] = {}
    for lab in labels:
        g = ratings[ratings["label"] == lab]
        wide = g.pivot_table(index="case_id", columns="reviewer", values="state", aggfunc="first")
        wide = wide.dropna(subset=list(revs)) if set(revs) <= set(wide.columns) else wide.iloc[0:0]
        n_pair = len(wide)
        if n_pair < min_n:
            out[lab] = {"n_pairs": SUPPRESSED, "n_unassessable": SUPPRESSED, "kappa_binary": SUPPRESSED,
                        "kappa_weighted_linear": SUPPRESSED, "kappa_weighted_quadratic": SUPPRESSED,
                        "prevalence": SUPPRESSED}
            continue
        sa = [_to_state(x) for x in wide[revs[0]]]; sb = [_to_state(x) for x in wide[revs[1]]]
        keep = [GoldState.UNASSESSABLE not in (x, y) for x, y in zip(sa, sb)]
        n_unass = len(keep) - sum(keep)
        sa = [x for x, k in zip(sa, keep) if k]; sb = [y for y, k in zip(sb, keep) if k]
        oa = [int(s) for s in sa]; ob = [int(s) for s in sb]
        ba = [int(s.is_positive) for s in sa]; bb = [int(s.is_positive) for s in sb]
        prev = (np.mean(ba) + np.mean(bb)) / 2 if len(ba) else float("nan")
        n_pos_any = int(sum(x or y for x, y in zip(ba, bb)))
        def rnd(v): return None if np.isnan(v) else round(float(v), 3)
        out[lab] = {
            "n_pairs": suppress_count(n_pair), "n_unassessable": suppress_count(n_unass),
            "kappa_binary": rnd(cohen_kappa(ba, bb, categories=[0, 1])),
            "kappa_weighted_linear": rnd(cohen_kappa(oa, ob, categories=[0, 1, 2, 3], weights="linear")),
            "kappa_weighted_quadratic": rnd(cohen_kappa(oa, ob, categories=[0, 1, 2, 3], weights="quadratic")),
            "prevalence": rnd(prev) if min(n_pos_any, len(ba) - n_pos_any) >= min_n else SUPPRESSED,
            "n_positive_either": suppress_count(n_pos_any),
        }
    return out


def minutes_summary(minutes: pd.DataFrame, by_reviewer: bool = True) -> dict:
    """``minutes``: case_id, reviewer, minutes. Quantiles only (no min/max); n >= 11."""
    res = {"overall": {"n_reviews": suppress_count(len(minutes)),
                       **safe_quantiles(minutes["minutes"].values)}}
    if by_reviewer:
        res["by_reviewer"] = {}
        for i, (_, g) in enumerate(minutes.groupby("reviewer")):
            res["by_reviewer"][f"reviewer_{i + 1}"] = {"n_reviews": suppress_count(len(g)),
                                                       **safe_quantiles(g["minutes"].values)}
    return res


# ---------------------------------------------------------------- EEG-only evidence share
_BANNED_CATS = {"eeg_derived", "neuro_impression_unfiltered", "nonspecific_dx"}


def eeg_only_share(evidence: pd.DataFrame, labels: Sequence[str]) -> dict[str, dict]:
    """Share of positive *naive* silver labels whose only evidence was EEG-derived.

    ``evidence``: case_id, label, source, plus optional text/code/hours_from_t0/eeg_filtered/has_anchor
    (columns of ``banned_evidence.classify_evidence``). A naive silver positive = any evidence row for
    that case/label (what silver would have been without the circularity rules). Each row is
    classified; reported per label:
      eeg_only            every row is EEG-derived (eeg_report, EEG-mentioning sentence, unfiltered post-t0
                          neurology impression)
      banned_only         every row is banned (the above plus G92/G93.4 and anchorless encephalopathy text)
    """
    out: dict[str, dict] = {}
    for lab in labels:
        g = evidence[evidence["label"] == lab].copy()
        if g.empty:
            out[lab] = {"n_naive_positive": 0, "eeg_only_share": SUPPRESSED, "banned_only_share": SUPPRESSED}
            continue
        def cat(r):
            return classify_evidence(
                r["source"], text=str(r.get("text", "") or ""), code=str(r.get("code", "") or ""),
                hours_from_t0=None if pd.isna(r.get("hours_from_t0", np.nan)) else float(r["hours_from_t0"]),
                eeg_filtered=bool(r.get("eeg_filtered", False)), has_anchor=bool(r.get("has_anchor", False)))
        g["_cat"] = [cat(r) for _, r in g.iterrows()]
        g["_eeg"] = [v.eeg_derived for v in g["_cat"]]
        g["_banned"] = [not v.allowed for v in g["_cat"]]
        per = g.groupby("case_id").agg(all_eeg=("_eeg", "all"), all_banned=("_banned", "all"))
        n = len(per)
        out[lab] = {"n_naive_positive": suppress_count(n),
                    "eeg_only_share": suppress_proportion(int(per["all_eeg"].sum()), n),
                    "banned_only_share": suppress_proportion(int(per["all_banned"].sum()), n)}
    return out


# ---------------------------------------------------------------- gate
def gate0_pilot_check(kappas: Mapping[str, dict], labels: Sequence[str] | None = None,
                      kappa_key: str = "kappa_weighted_linear") -> dict:
    """Gate 0 pilot criteria over the primary families (E7 excluded unless the caller lists it)."""
    labels = list(labels or primary_endpoint_labels())
    ok_k, ok_p = 0, 0
    for l in labels:
        r = kappas.get(l, {})
        k, p = r.get(kappa_key), r.get("prevalence")
        if isinstance(k, (int, float)) and k >= GATE_KAPPA:
            ok_k += 1
        if isinstance(p, (int, float)) and p >= GATE_PREVALENCE:
            ok_p += 1
    return {"families_with_kappa_ge_0.6": ok_k, "families_with_prevalence_ge_10pct": ok_p,
            "kappa_ok": ok_k >= GATE_MIN_KAPPA_FAMILIES, "prevalence_ok": ok_p >= GATE_MIN_PREVALENT_FAMILIES,
            "pass": ok_k >= GATE_MIN_KAPPA_FAMILIES and ok_p >= GATE_MIN_PREVALENT_FAMILIES}


def pilot_report(ratings: pd.DataFrame, minutes: pd.DataFrame, evidence: pd.DataFrame,
                 labels: Sequence[str] | None = None) -> dict:
    labels = list(labels or primary_endpoint_labels())
    k = label_kappas(ratings, labels)
    return {"kappas": k, "minutes_per_case": minutes_summary(minutes),
            "eeg_only_evidence": eeg_only_share(evidence, labels), "gate0": gate0_pilot_check(k, labels)}
