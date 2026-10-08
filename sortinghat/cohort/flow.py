"""CONSORT-style flow: raw counts during the build, suppressed aggregate report for output.

``FlowRecorder`` (used by the builder) keeps exact counts per site in memory. ``flow_report`` turns them into
something that may leave the process: every printed or written number goes through ``sortinghat.safe_output``
(``suppress_count``; n < 11 shown as "<11"), and the output is checked with ``assert_aggregate_only``.

Small cells in a flow can be recovered by subtraction (remaining_before - remaining_after = excluded), so
suppressing the excluded count alone is not enough. Two protections:

* **Sequential exclusions.** A step whose exclusion count is < 11 is MERGED into the next step (and the last one
  into the previous row), so only exclusion totals >= 11 are ever shown together with exact remaining counts.
  A row's label then lists every merged reason. If the whole flow excludes < 11, the remaining count is withheld.
* **Partitions** (strict / broad-only / outside, time-since-onset bins, ...). A part < 11 is suppressed and, if it is
  the only one, the smallest other part is suppressed too (complementary suppression), so it cannot be derived
  from the total.

* **Sites.** Differencing also works across tables (ALL minus the other sites). Sites whose FINAL count is < 11 are
  pooled with the smallest other site until every group has >= 11 (groups are labelled "S1+S2"), and in a partition
  an "ALL" cell is suppressed when exactly one site group's cell is suppressed. A group is withheld only if even the
  pooled total is < 11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..safe_output import (SUPPRESS_BELOW, SUPPRESSED, assert_aggregate_only, suppress_count)

ALL = "ALL"
WITHHELD = "withheld"


@dataclass
class Step:
    label: str
    unit: str                           # "EEG sessions" | "patients"
    remaining: dict[str, int]           # per site
    start: bool = False                 # first row: nothing is excluded here


@dataclass
class Partition:
    title: str
    parts: dict[str, dict[str, int]]    # part label -> per-site count (parts are disjoint and cover the total)


@dataclass
class FlowRecorder:
    sites: list[str]
    steps: list[Step] = field(default_factory=list)
    partitions: list[Partition] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    excluded_sites: list[str] = field(default_factory=list)       # sites kept out of the study (D-113): own block

    def _by_site(self, site_col) -> dict[str, int]:
        c = site_col.astype(str).value_counts()
        return {s: int(c.get(s, 0)) for s in self.sites}

    def start(self, label: str, unit: str, site_col) -> None:
        self.steps.append(Step(label, unit, self._by_site(site_col), start=True))

    def step(self, label: str, unit: str, site_col_after) -> None:
        self.steps.append(Step(label, unit, self._by_site(site_col_after)))

    def partition(self, title: str, labelled_site_cols: dict) -> None:
        self.partitions.append(Partition(title, {k: self._by_site(v) for k, v in labelled_site_cols.items()}))

    def raw(self) -> dict:
        """Exact counts (in memory only; used by tests and by ``flow_report``)."""
        return {"sites": list(self.sites),
                "steps": [{"label": s.label, "unit": s.unit, "remaining": dict(s.remaining), "start": s.start}
                          for s in self.steps],
                "partitions": [{"title": p.title, "parts": {k: dict(v) for k, v in p.parts.items()}}
                               for p in self.partitions],
                "checks": self.checks, "excluded_sites": list(self.excluded_sites)}


def _site_view(steps: list[Step], members: list[str]) -> list[tuple[str, str, int, bool]]:
    return [(s.label, s.unit, sum(s.remaining.get(m, 0) for m in members), s.start) for s in steps]


def pool_sites(sites: list[str], steps: list[Step], k: int = SUPPRESS_BELOW) -> list[list[str]]:
    """Groups of sites for disclosure control: a site whose final count is < k is pooled with the smallest other
    group until every group has >= k (or only one group is left)."""
    final = {s: (steps[-1].remaining.get(s, 0) if steps else 0) for s in sites}
    fin = lambda g: sum(final[x] for x in g)          # noqa: E731
    groups = [[s] for s in sites]
    while True:
        live = [g for g in groups if fin(g) > 0]          # a site with nothing left (e.g. excluded by design) is its own block
        small = [g for g in live if fin(g) < k]
        if not small or len(live) < 2:
            break
        g = min(small, key=fin)
        groups.remove(g)
        other = min((x for x in live if x is not g), key=fin)
        other.extend(g)
        other.sort()
    return groups


def merged_rows(view: list[tuple[str, str, int, bool]], k: int = SUPPRESS_BELOW) -> list[dict]:
    """Rows of one site's flow with every shown exclusion count >= k (small steps merged, see module doc)."""
    rows: list[dict] = []
    pend_labels: list[str] = []
    pend_excl = 0
    prev_n = None
    prev_unit = None
    for label, unit, n, is_start in view:
        if is_start or prev_n is None:
            rows.append({"step": label, "unit": unit, "n_excluded": None, "n_remaining": n, "merged": [label]})
            prev_n, prev_unit = n, unit
            continue
        if pend_labels and unit != prev_unit and len(rows) > 1:       # never merge across sessions -> patients
            last = rows[-1]
            last["step"] += " + " + " + ".join(pend_labels)
            last["merged"] += pend_labels
            last["n_excluded"] += pend_excl
            last["n_remaining"], last["unit"] = prev_n, prev_unit
            pend_labels, pend_excl = [], 0
        pend_labels.append(label)
        pend_excl += prev_n - n
        prev_n, prev_unit = n, unit
        if pend_excl >= k:
            rows.append({"step": " + ".join(pend_labels), "unit": unit, "n_excluded": pend_excl, "n_remaining": n,
                         "merged": list(pend_labels)})
            pend_labels, pend_excl = [], 0
    if pend_labels:
        if len(rows) > 1:                                     # fold the tail into the previous exclusion row
            last = rows[-1]
            last["step"] = last["step"] + " + " + " + ".join(pend_labels)
            last["merged"] += pend_labels
            last["n_excluded"] += pend_excl
            last["n_remaining"] = prev_n
            last["unit"] = prev_unit
        else:                                                 # nothing >= k excluded anywhere
            rows.append({"step": " + ".join(pend_labels), "unit": prev_unit, "n_excluded": pend_excl,
                         "n_remaining": prev_n, "merged": list(pend_labels), "remaining_withheld": True})
    return rows


def suppress_parts(parts: dict[str, int], k: int = SUPPRESS_BELOW, forced: set[str] | None = None
                   ) -> dict[str, int | str]:
    """Suppress parts < k (and any in ``forced``). Complementary suppression: while exactly one part is hidden, or
    the hidden parts sum to < k (the total would reveal that small sum), also hide the smallest shown part."""
    shown: dict[str, int | str] = {p: (SUPPRESSED if forced and p in forced else suppress_count(n, k))
                                   for p, n in parts.items()}
    while True:
        hidden = [p for p, v in shown.items() if v == SUPPRESSED]
        left = sorted((n, p) for p, n in parts.items() if shown[p] != SUPPRESSED)
        if not hidden or not left:
            break
        if len(hidden) == 1 or sum(parts[p] for p in hidden) < k:
            shown[left[0][1]] = SUPPRESSED
        else:
            break
    return shown


def flow_report(raw: dict, config: dict | None = None, k: int = SUPPRESS_BELOW) -> dict:
    """Aggregate-only report dict (suppressed). ``raw`` is ``FlowRecorder.raw()``."""
    steps = [Step(s["label"], s["unit"], s["remaining"], s["start"]) for s in raw["steps"]]
    report: dict = {"suppression": f"counts < {k} are shown as \"{SUPPRESSED}\"; exclusion steps with fewer than {k} "
                                   "patients/sessions are merged into the next row (or the previous one) so they "
                                   "cannot be recovered by subtraction",
                    "sites": {}, "partitions": {}, "checks": raw.get("checks", {})}
    groups = pool_sites(raw["sites"], steps, k)
    labelled = [("+".join(g), g) for g in groups]
    excl = set(raw.get("excluded_sites", []))
    report["excluded_site_blocks"] = [lab for lab, g in labelled if g and set(g) <= excl]
    if any(len(g) > 1 for g in groups):
        report["pooled_site_groups"] = [lab for lab, g in labelled if len(g) > 1]
    for lab, members in [(ALL, raw["sites"]), *labelled]:
        view = _site_view(steps, members)
        if not view or view[0][2] < k or view[-1][2] < 0:
            report["sites"][lab] = {WITHHELD: f"starting count < {k}"}
            continue
        rows = []
        for r in merged_rows(view, k):
            rows.append({"step": r["step"], "unit": r["unit"],
                         "n_excluded": None if r["n_excluded"] is None else suppress_count(r["n_excluded"], k),
                         "n_remaining": WITHHELD if r.get("remaining_withheld") else suppress_count(r["n_remaining"], k)})
        report["sites"][lab] = {"rows": rows}
    for p in raw["partitions"]:
        parts = list(p["parts"])
        by_group = {lab: {x: sum(p["parts"][x].get(m, 0) for m in g) for x in parts} for lab, g in labelled}
        all_counts = {x: sum(by_group[lab][x] for lab, _ in labelled) for x in parts}
        tbl = {}
        # ALL cell of a part is withheld when exactly one site group's cell for it is < k (it would be ALL - others)
        forced = {x for x in parts if sum(1 for lab, _ in labelled if by_group[lab][x] < k) == 1
                  and all_counts[x] >= k}
        tbl[ALL] = {WITHHELD: f"total < {k}"} if sum(all_counts.values()) < k else suppress_parts(all_counts, k, forced)
        for lab, _ in labelled:
            c = by_group[lab]
            tbl[lab] = {WITHHELD: f"total < {k}"} if sum(c.values()) < k else suppress_parts(c, k)
        report["partitions"][p["title"]] = tbl
    if config is not None:
        report["config"] = config
    assert_aggregate_only(report)
    return report


def debug_report(raw: dict, config: dict | None = None, k: int = SUPPRESS_BELOW) -> dict:
    """UNMERGED report for diagnosing a run: every step on its own row with its exclusion count, suppressing only
    individual counts < k (``<11``). No step merging and no site pooling, so a small exclusion count is bounded by
    "<11" but can be recovered by subtracting two exact remaining counts that are both >= 11. NOT for sharing; the
    shareable report is ``flow_report``."""
    steps = [Step(s["label"], s["unit"], s["remaining"], s["start"]) for s in raw["steps"]]
    report: dict = {"debug": "DIAGNOSTIC, NOT FOR SHARING: every step is listed; only individual counts < "
                             f"{k} are suppressed, so small exclusions are inferable from neighbouring exact counts",
                    "suppression": f"counts < {k} are shown as \"{SUPPRESSED}\"", "sites": {}, "partitions": {},
                    "checks": raw.get("checks", {}),
                    "excluded_site_blocks": [x for x in raw["sites"] if x in set(raw.get("excluded_sites", []))]}
    for lab, members in [(ALL, raw["sites"]), *[(x, [x]) for x in raw["sites"]]]:
        view = _site_view(steps, members)
        if not view or view[0][2] < k:
            report["sites"][lab] = {WITHHELD: f"starting count < {k}"}
            continue
        rows, prev = [], None
        for label, unit, n, is_start in view:
            rows.append({"step": label, "unit": unit,
                         "n_excluded": None if prev is None or is_start else suppress_count(prev - n, k),
                         "n_remaining": suppress_count(n, k)})
            prev = n
        report["sites"][lab] = {"rows": rows}
    for p in raw["partitions"]:
        parts = list(p["parts"])
        tbl = {}
        for lab, members in [(ALL, raw["sites"]), *[(x, [x]) for x in raw["sites"]]]:
            c = {x: sum(p["parts"][x].get(m, 0) for m in members) for x in parts}
            tbl[lab] = {WITHHELD: f"total < {k}"} if sum(c.values()) < k else suppress_parts(c, k)
        report["partitions"][p["title"]] = tbl
    if config is not None:
        report["config"] = config
    assert_aggregate_only(report)
    return report


def _cell(v) -> str:
    return "" if v is None else str(v)


def flow_markdown(report: dict, pending: list[str] | None = None) -> str:
    L = ["# Study 1 cohort flow (CONSORT-style, aggregate only)", "",
         *([f"**{report['debug']}**", ""] if report.get("debug") else []),
         f"Suppression: {report['suppression']}.", "",
         "Units: a row counts the unit of its last step. From \"Not the patient's first qualifying EEG\" on, one "
         "EEG per patient remains, so sessions = patients; that row's Excluded counts later sessions.", ""]
    for site, blk in report["sites"].items():
        tag = " (excluded block: not a Study 1 site, no EHR rows)" if site in report.get("excluded_site_blocks", []) else ""
        L += [f"## {'All sites' if site == ALL else site}{tag}", ""]
        if WITHHELD in blk:
            L += [f"Withheld: {blk[WITHHELD]}.", ""]
            continue
        L += ["| Step | Unit | Excluded | Remaining |", "|---|---|---|---|"]
        for r in blk["rows"]:
            L.append(f"| {r['step']} | {r['unit']} | {_cell(r['n_excluded'])} | {r['n_remaining']} |")
        L.append("")
    for title, tbl in report["partitions"].items():
        L += [f"## {title}", ""]
        labels = next((list(v) for v in tbl.values() if WITHHELD not in v), [])
        L += ["| Site | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for site, v in tbl.items():
            if WITHHELD in v:
                L.append(f"| {'All sites' if site == ALL else site} | " + " | ".join([WITHHELD] * len(labels)) + " |")
            else:
                L.append(f"| {'All sites' if site == ALL else site} | " + " | ".join(str(v[lab]) for lab in labels) + " |")
        L.append("")
    if report.get("checks"):
        L += ["## Checks", ""]
        for name, val in report["checks"].items():
            L.append(f"- {name}: {val}")
        L.append("")
    if pending:
        L += ["## Steps added after this flow", ""] + [f"- {p}" for p in pending] + [""]
    return "\n".join(L)
