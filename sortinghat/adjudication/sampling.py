"""Deterministic seeded stratified sample for the gold adjudication pilot (IU-large; D-157).

Frame: one row per eligible patient (the IU-large rows of the analysed broad cohort).  Strata = site x SILVER-LABEL PATTERN.  The
pattern is the first silver-positive label in a FIXED priority order (rarest etiology first, set a priori, not from the data), else
``all_negative`` (no silver-positive label; labels with no assessable value are ignored).  The target share per pattern is configured
(``configs/adjudication.yaml``) and deliberately over-represents the rarer etiologies and the silver all-negative patients; a stratum
smaller than its target is taken whole and the shortfall is shared among the other strata of the site (water-filling).

Weights: w = N_h / n_h (stratum population over stratum sample), so sum(w) over the sample = the IU-large population and a prevalence
estimated from the sample is re-weighted to the IU-large cohort.  A 10% duplicate set (same patient, new pseudonymous id, shown to the
rater later) is added for intra-rater reliability, and a few calibration cases (drawn from patients NOT in the sample) for rater
training.  Everything is a pure function of (frame, seed, config): the same inputs give the same sample.
"""
from __future__ import annotations

import zlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

LABELS = ("E1", "E2", "E4a", "E5", "E6", "E7")
DEFAULT_PRIORITY = ("E7", "E2", "E4a", "E6", "E5", "E1")
ALL_NEGATIVE = "all_negative"
DEFAULT_SHARES = {"E7": 0.10, "E2": 0.12, "E4a": 0.12, "E6": 0.14, "E5": 0.14, "E1": 0.16, ALL_NEGATIVE: 0.22}
ROLE_SAMPLE, ROLE_DUPLICATE, ROLE_CALIBRATION = "sample", "duplicate", "calibration"
SAMPLE_COLUMNS = ["sample_id", "role", "duplicate_of", "site", "pattern", "stratum", "stratum_pop", "stratum_n", "inclusion_prob", "weight"]


def assign_pattern(Y: pd.DataFrame, priority: Sequence[str] = DEFAULT_PRIORITY) -> np.ndarray:
    """Silver-label pattern per row: the first label of ``priority`` with value 1, else ``all_negative``.  ``Y`` holds 1.0 / 0.0 / NaN
    (not assessable) per label column; a label not in ``Y`` is skipped."""
    pat = np.full(len(Y), ALL_NEGATIVE, dtype=object)
    done = np.zeros(len(Y), bool)
    for lab in priority:
        if lab not in Y:
            continue
        pos = (pd.to_numeric(Y[lab], errors="coerce").to_numpy(float) == 1.0) & ~done
        pat[pos] = lab
        done |= pos
    return pat


def waterfill(caps: Mapping[str, int], shares: Mapping[str, float], n: int) -> dict:
    """Integer allocation of ``n`` over cells in proportion to ``shares``, never above ``caps``; a capped cell is filled and its surplus
    is shared among the others in proportion to their shares (largest-remainder rounding, ties by key).  Total = min(n, sum caps)."""
    alloc = {k: 0 for k in caps}
    free = {k for k in caps if caps[k] > 0 and shares.get(k, 0.0) > 0}
    rem = int(min(n, sum(int(caps[k]) for k in free)))
    quota: dict = {}
    while free:
        tot = sum(shares[k] for k in free)
        quota = {k: rem * shares[k] / tot for k in free}
        over = sorted(k for k in free if quota[k] >= caps[k])
        if not over:
            break
        for k in over:
            alloc[k] = int(caps[k])
            rem -= int(caps[k])
            free.discard(k)
    if free and rem > 0:
        base = {k: int(np.floor(quota[k])) for k in free}
        left = rem - sum(base.values())
        order = sorted(free, key=lambda k: (-(quota[k] - base[k]), k))
        while left > 0:
            moved = False
            for k in order:
                if left > 0 and base[k] < caps[k]:
                    base[k] += 1
                    left -= 1
                    moved = True
            if not moved:
                break
        for k, v in base.items():
            alloc[k] = v
    return alloc


def allocate_sites(site_sizes: Mapping[str, int], n: int, mode: str = "proportional") -> dict:
    """Sample size per site: ``proportional`` to the eligible population (>= 1 per site when possible) or ``equal``."""
    if mode == "proportional":
        from ..labels.pilot import allocate_by_site
        return allocate_by_site(dict(site_sizes), n)
    if mode == "equal":
        return waterfill(site_sizes, {s: 1.0 for s in site_sizes}, n)
    raise ValueError(f"unknown site allocation mode {mode!r}")


def kish_ess(w) -> float:
    w = np.asarray(w, float)
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def _rng(seed: int, *key) -> np.random.Generator:
    return np.random.default_rng([int(seed), *[zlib.crc32(str(k).encode()) for k in key]])


def draw_sample(frame: pd.DataFrame, *, n: int = 200, seed: int = 20261014, shares: Mapping[str, float] | None = None,
                site_mode: str = "proportional", dup_frac: float = 0.10, n_calibration: int = 5,
                id_col: str = "person_id", site_col: str = "site", pattern_col: str = "pattern") -> pd.DataFrame:
    """The sample table (RECORD-LEVEL: local_only only).  ``frame``: one row per eligible patient with ``id_col``, ``site_col``,
    ``pattern_col`` (+ any columns to carry along, e.g. t0).  Returns SAMPLE_COLUMNS + the carried columns, rows shuffled, ``sample_id``
    ``s0001``...; duplicates repeat the original's row (``role == duplicate``, ``duplicate_of`` = the original's sample_id, weight 0),
    calibration rows come from patients not in the sample (weight 0)."""
    shares = dict(DEFAULT_SHARES if shares is None else shares)
    f = frame.sort_values(id_col, kind="mergesort").reset_index(drop=True).copy()
    if f[id_col].duplicated().any():
        raise ValueError("the frame must hold one row per patient")
    missing = sorted(set(f[pattern_col]) - set(shares))
    if missing:
        raise ValueError(f"no target share for pattern(s) {missing}")
    f["stratum"] = f[site_col].astype(str) + "|" + f[pattern_col].astype(str)
    site_sizes = f.groupby(site_col).size().to_dict()
    site_n = allocate_sites(site_sizes, n, site_mode)
    chosen: list = []
    for site in sorted(site_sizes):
        sub = f[f[site_col] == site]
        caps = {p: int((sub[pattern_col] == p).sum()) for p in shares}
        alloc = waterfill(caps, shares, site_n[site])
        for p in sorted(alloc):
            k = alloc[p]
            if k <= 0:
                continue
            pool = sub.index[sub[pattern_col] == p].to_numpy()
            pick = _rng(seed, "draw", site, p).choice(len(pool), size=k, replace=False)
            chosen.extend(pool[np.sort(pick)].tolist())
    sel = f.loc[sorted(chosen)].copy()
    pop = f.groupby("stratum").size()
    cnt = sel.groupby("stratum").size()
    sel["stratum_pop"] = sel["stratum"].map(pop).astype(int)
    sel["stratum_n"] = sel["stratum"].map(cnt).astype(int)
    sel["inclusion_prob"] = sel["stratum_n"] / sel["stratum_pop"]
    sel["weight"] = sel["stratum_pop"] / sel["stratum_n"]
    sel["role"] = ROLE_SAMPLE
    sel["duplicate_of"] = ""
    rows = [sel]
    # ---- duplicates (same patient, a second pseudonymous packet)
    n_dup = int(round(dup_frac * len(sel)))
    dup_idx = np.sort(_rng(seed, "dup").choice(len(sel), size=min(n_dup, len(sel)), replace=False)) if n_dup else np.array([], int)
    dup = sel.iloc[dup_idx].copy()
    dup["role"] = ROLE_DUPLICATE
    dup["weight"] = 0.0
    dup["inclusion_prob"] = np.nan
    rows.append(dup)
    # ---- calibration (not in the sample)
    rest = f[~f[id_col].isin(sel[id_col])]
    if n_calibration and len(rest):
        pick = _rng(seed, "calibration").choice(len(rest), size=min(n_calibration, len(rest)), replace=False)
        cal = rest.iloc[np.sort(pick)].copy()
        cal["stratum_pop"], cal["stratum_n"] = cal["stratum"].map(pop).astype(int), 0
        cal["inclusion_prob"], cal["weight"], cal["role"], cal["duplicate_of"] = np.nan, 0.0, ROLE_CALIBRATION, ""
        rows.append(cal)
    out = pd.concat(rows, ignore_index=True)
    out = out.iloc[_rng(seed, "shuffle").permutation(len(out))].reset_index(drop=True)
    out["sample_id"] = [f"s{i + 1:04d}" for i in range(len(out))]
    orig = out[out["role"] == ROLE_SAMPLE].set_index(id_col)["sample_id"]
    is_dup = out["role"] == ROLE_DUPLICATE
    out.loc[is_dup, "duplicate_of"] = out.loc[is_dup, id_col].map(orig)
    carried = [c for c in out.columns if c not in SAMPLE_COLUMNS and c != id_col]
    return out[["sample_id", id_col, *[c for c in SAMPLE_COLUMNS if c != "sample_id"], *[c for c in carried if c not in SAMPLE_COLUMNS]]]


def check_weights(sample: pd.DataFrame, frame: pd.DataFrame, *, site_col: str = "site") -> dict:
    """Design checks: the weights of the sample rows sum to the eligible population per site, every sampled stratum has weight >= 1."""
    s = sample[sample["role"] == ROLE_SAMPLE]
    per_site = s.groupby(site_col)["weight"].sum().round(6).to_dict()
    pop = frame.groupby(site_col).size().to_dict()
    ok = all(abs(per_site.get(k, 0.0) - v) < 1e-6 for k, v in pop.items() if k in per_site)
    return {"weights_sum_to_population": bool(ok), "min_weight_ge_1": bool((s["weight"] >= 1.0 - 1e-9).all())}
