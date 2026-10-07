"""Leave-one-site-out and late-calendar temporal holdout split generators.

One row per patient (first qualifying EEG per patient), so index-level splits are
patient-level splits. Splits are index arrays into the caller's tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

MIN_SITES = 3
MIN_CANDIDATES_PER_SITE = 300


@dataclass(frozen=True)
class Split:
    held_out: str
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class TemporalSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    cutoffs: dict = field(default_factory=dict)  # site -> cutoff time
    test_idx_by_site: dict = field(default_factory=dict)


def check_site_requirements(sites, min_sites: int = MIN_SITES, min_per_site: int = MIN_CANDIDATES_PER_SITE) -> dict:
    """Phase 0a site criterion: >= 3 adult sites with >= 300 candidates each.

    If it fails the plan falls back to a grouped split with a weaker claim.
    """
    sites = np.asarray(sites)
    uniq, counts = np.unique(sites, return_counts=True)
    ok_sites = [str(s) for s, c in zip(uniq, counts) if c >= min_per_site]
    return {
        "n_sites": int(len(uniq)),
        "counts": {str(s): int(c) for s, c in zip(uniq, counts)},
        "sites_meeting_min": ok_sites,
        "passes": len(ok_sites) >= min_sites,
    }


def leave_one_site_out(sites) -> Iterator[Split]:
    """Yield one Split per site: train on all other sites, test on the held-out site."""
    sites = np.asarray(sites)
    uniq = np.unique(sites)
    if len(uniq) < 2:
        raise ValueError("leave-one-site-out needs at least 2 sites")
    all_idx = np.arange(len(sites))
    for s in uniq:
        test = all_idx[sites == s]
        train = all_idx[sites != s]
        yield Split(held_out=str(s), train_idx=train, test_idx=test)


def _as_numeric_time(times) -> np.ndarray:
    t = np.asarray(times)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[s]").astype("int64").astype(float)
    return t.astype(float)


def late_temporal_holdout(sites, times, test_fraction: float = 0.2, embargo: float = 0.0) -> TemporalSplit:
    """Within each site, the latest ``test_fraction`` of cases (by calendar time) is test.

    Training = earlier cases at all sites, minus an optional ``embargo`` (in the units of
    ``times``) before each site's cutoff so near-boundary cases cannot straddle the split.
    Requires a time variable whose ordering is comparable across patients within a site;
    if the release date-shifts per patient this is not available (see SAP, section 8).
    """
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be in (0, 1)")
    sites = np.asarray(sites)
    t = _as_numeric_time(times)
    if t.shape[0] != sites.shape[0]:
        raise ValueError("sites and times must have the same length")
    if np.any(np.isnan(t)):
        raise ValueError("times contains NaN; drop or impute cases before splitting")
    train_mask = np.zeros(len(t), dtype=bool)
    test_mask = np.zeros(len(t), dtype=bool)
    cutoffs, by_site = {}, {}
    for s in np.unique(sites):
        m = sites == s
        cut = np.quantile(t[m], 1.0 - test_fraction)
        te = m & (t >= cut)
        tr = m & (t < cut - embargo)
        train_mask |= tr
        test_mask |= te
        cutoffs[str(s)] = cut
        by_site[str(s)] = np.flatnonzero(te)
    return TemporalSplit(
        train_idx=np.flatnonzero(train_mask),
        test_idx=np.flatnonzero(test_mask),
        cutoffs=cutoffs,
        test_idx_by_site=by_site,
    )
