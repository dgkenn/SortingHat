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

Known limit (documented in docs/cohort_spec.md): sites are not complementarily suppressed against the "ALL"
table, so a site-level cell that is suppressed could in principle be derived from ALL minus the other sites. A
site whose starting count is < 11 is withheld entirely.
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
                "checks": self.checks}


def _site_view(steps: list[Step], site: str) -> list[tuple[str, str, int, bool]]:
    out = []
    for s in steps:
        n = sum(s.remaining.values()) if site == ALL else s.remaining.get(site, 0)
        out.append((s.label, s.unit, n, s.start))
    return out


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


def suppress_parts(parts: dict[str, int], k: int = SUPPRESS_BELOW) -> dict[str, int | str]:
    """Suppress parts < k; if exactly one part is suppressed also suppress the smallest other part."""
    shown: dict[str, int | str] = {p: suppress_count(n, k) for p, n in parts.items()}
    hidden = [p for p, v in shown.items() if v == SUPPRESSED]
    if len(hidden) == 1 and len(parts) > 1:
        rest = sorted((n, p) for p, n in parts.items() if p != hidden[0])
        shown[rest[0][1]] = SUPPRESSED
    return shown


def flow_report(raw: dict, config: dict | None = None, k: int = SUPPRESS_BELOW) -> dict:
    """Aggregate-only report dict (suppressed). ``raw`` is ``FlowRecorder.raw()``."""
    steps = [Step(s["label"], s["unit"], s["remaining"], s["start"]) for s in raw["steps"]]
    report: dict = {"suppression": f"counts < {k} are shown as \"{SUPPRESSED}\"; exclusion steps with fewer than {k} "
                                   "patients/sessions are merged into the next row (or the previous one) so they "
                                   "cannot be recovered by subtraction",
                    "sites": {}, "partitions": {}, "checks": raw.get("checks", {})}
    for site in [ALL, *raw["sites"]]:
        view = _site_view(steps, site)
        if not view or view[0][2] < k:
            report["sites"][site] = {WITHHELD: f"starting count < {k}"}
            continue
        rows = []
        for r in merged_rows(view, k):
            rows.append({"step": r["step"], "unit": r["unit"],
                         "n_excluded": None if r["n_excluded"] is None else suppress_count(r["n_excluded"], k),
                         "n_remaining": WITHHELD if r.get("remaining_withheld") else suppress_count(r["n_remaining"], k)})
        report["sites"][site] = {"rows": rows}
    for p in raw["partitions"]:
        tbl = {}
        for site in [ALL, *raw["sites"]]:
            counts = {lab: (sum(v.values()) if site == ALL else v.get(site, 0)) for lab, v in p["parts"].items()}
            tbl[site] = {WITHHELD: f"total < {k}"} if sum(counts.values()) < k else suppress_parts(counts, k)
        report["partitions"][p["title"]] = tbl
    if config is not None:
        report["config"] = config
    assert_aggregate_only(report)
    return report


def _cell(v) -> str:
    return "" if v is None else str(v)


def flow_markdown(report: dict, pending: list[str] | None = None) -> str:
    L = ["# Study 1 cohort flow (CONSORT-style, aggregate only)", "",
         f"Suppression: {report['suppression']}.", "",
         "Units: a row counts the unit of its last step. From \"Not the patient's first qualifying EEG\" on, one "
         "EEG per patient remains, so sessions = patients; that row's Excluded counts later sessions.", ""]
    for site, blk in report["sites"].items():
        L += [f"## {'All sites' if site == ALL else site}", ""]
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
