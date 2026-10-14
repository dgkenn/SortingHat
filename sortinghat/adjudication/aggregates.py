"""Small-cell suppression for the aggregate output of the adjudication scripts (stdout and aggregate JSON only)."""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from ..safe_output import SUPPRESS_BELOW, SUPPRESSED

K = SUPPRESS_BELOW


def suppress_row(counts: Mapping[str, int], k: int = K) -> dict:
    """{cell: count or '<11'} for one exclusive set of cells (e.g. the silver patterns of one site).  A count below ``k`` is hidden;
    when exactly one cell is hidden the smallest shown cell is hidden too, so a row total printed next to it cannot recover it."""
    shown = {c: int(n) for c, n in counts.items()}
    hidden = [c for c, n in shown.items() if n < k]
    if len(hidden) == 1:
        rest = [c for c in shown if c not in hidden]
        if rest:
            hidden.append(min(rest, key=lambda c: (shown[c], str(c))))
    return {c: (SUPPRESSED if c in hidden else n) for c, n in shown.items()}


def pooled_row(by_site: Mapping[str, Mapping[str, int]], k: int = K) -> dict:
    """Pooled count per cell over sites; a pooled cell is hidden when it is < k or when any site cell of it is hidden (a shown pooled
    value plus the other shown site would give the hidden one by subtraction)."""
    cells = sorted({c for row in by_site.values() for c in row})
    site_vis = {s: suppress_row({c: row.get(c, 0) for c in cells}, k) for s, row in by_site.items()}
    out = {}
    for c in cells:
        tot = sum(int(row.get(c, 0)) for row in by_site.values())
        any_hidden = any(site_vis[s][c] == SUPPRESSED for s in by_site)
        out[c] = SUPPRESSED if (tot < k or any_hidden) else tot
    return out


def share_cell(n: int, d: int, k: int = K) -> dict:
    """{'n', 'den', 'share'} of a binary measure; n and share are hidden when n < k or the complement d - n < k (or d < k).  A cell hidden
    because the COMPLEMENT is small says so (``why: high``) with a lower bound on the share, so "almost everyone" is distinguishable from
    "almost no one" (``why: low``)."""
    n, d = int(n), int(d)
    hide = d < k or n < k or (d - n) < k
    out = {"n": SUPPRESSED if hide else n, "den": SUPPRESSED if d < k else d, "share": SUPPRESSED if hide else round(n / d, 4)}
    if hide and d >= k:
        out["why"] = "low" if n < k else "high"
        if out["why"] == "high":
            out["share_bound"] = f">={(d - (k - 1)) / d:.3f}"
    return out


def pooled_share_cells(by_site: Mapping[str, tuple], k: int = K) -> dict:
    """{site: share_cell, 'all_sites': share_cell} for (n, d) per site; the pooled cell is hidden when any site cell is hidden."""
    out = {s: share_cell(n, d, k) for s, (n, d) in by_site.items()}
    n = sum(int(v[0]) for v in by_site.values())
    d = sum(int(v[1]) for v in by_site.values())
    pooled = share_cell(n, d, k)
    if pooled["n"] != SUPPRESSED and any(c["n"] == SUPPRESSED for c in out.values()):
        pooled = {"n": SUPPRESSED, "den": pooled["den"], "share": SUPPRESSED, "why": "linked to a hidden site cell"}
    out["all_sites"] = pooled
    return out


def availability_report(flags: "pd.DataFrame", site_col: str, site_labels: Mapping[str, str], columns, k: int = K) -> dict:
    """Per flag column: share of cases with the flag, per site and pooled, small cells suppressed (pooled hidden when a site cell is)."""
    out = {}
    sites = sorted(flags[site_col].unique())
    for c in columns:
        by_site = {site_labels[s]: (int(flags.loc[flags[site_col] == s, c].sum()), int((flags[site_col] == s).sum())) for s in sites}
        out[c] = pooled_share_cells(by_site, k)
    return out
