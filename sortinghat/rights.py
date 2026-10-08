"""Data-rights ledger and the production-build refusal rule (commercial track).

Ledger: ``configs/data_rights.yaml``, one entry per dataset or model artefact.

Gate:
* ``research`` passes when every artefact is known and ``permitted_uses.research`` is ``permitted``.
* ``commercial_training``, ``production_build`` and ``regulatory_submission`` pass only when every
  artefact is ``green`` and each use the purpose needs is ``permitted``. Anything else refuses.
* An artefact missing from the ledger is refused for every purpose (fail closed).
* An empty manifest is refused.

Every check is appended to ``docs/rights_checks.log`` (one dated line per check, ids and verdicts
only, no patient data). The log is never truncated or rewritten.

CLI::

    python -m sortinghat.rights check --purpose production_build --manifest build.yaml
    python -m sortinghat.rights check --purpose research --ids heedb_v4_1,bind_v1_0
    python -m sortinghat.rights list

Exit codes for ``check``: 0 allowed, 3 refused, 2 usage or ledger error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "configs" / "data_rights.yaml"
DEFAULT_LOG = REPO_ROOT / "docs" / "rights_checks.log"

USES = ("research", "commercial_training", "derivative_weights", "regulatory_submission", "sublicensing")
COMMERCIAL_USES = USES[1:]
PERMITTED, NOT_PERMITTED, UNKNOWN = "permitted", "not_permitted", "unknown"
USE_VALUES = frozenset({PERMITTED, NOT_PERMITTED, UNKNOWN})
LINEAGES = frozenset({"research_only", "commercial_candidate", "yellow"})
STATUSES = frozenset({"green", "yellow", "red"})
REQUIRED_KEYS = frozenset({
    "kind", "name", "version", "version_hash", "lineage", "licence", "source_url",
    "access", "permitted_uses", "status", "open_questions", "evidence_file",
})

# Purpose -> uses that must be permitted (and status green) for every artefact.
PURPOSE_USES: dict[str, tuple[str, ...]] = {
    "research": ("research",),
    "commercial_training": ("commercial_training",),
    "production_build": ("commercial_training", "derivative_weights", "sublicensing"),
    "regulatory_submission": ("commercial_training", "derivative_weights", "regulatory_submission"),
}
GATED_PURPOSES = frozenset(p for p in PURPOSE_USES if p != "research")

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.:\-]")


class RightsError(Exception):
    """Base class for rights errors."""


class LedgerError(RightsError, ValueError):
    """The ledger is malformed or an entry is inconsistent."""


class UnknownPurposeError(RightsError, ValueError):
    """The purpose is not in PURPOSE_USES."""


class RightsRefused(RightsError):
    """A build or use was refused. ``.decision`` holds the Decision."""

    def __init__(self, decision: "Decision"):
        self.decision = decision
        super().__init__(format_decision(decision))


@dataclass(frozen=True)
class Artefact:
    id: str
    kind: str
    name: str
    version: str
    version_hash: str | None
    lineage: str
    licence: str
    source_url: str
    access: str
    permitted_uses: Mapping[str, str]
    status: str
    open_questions: tuple[str, ...]
    evidence_file: str


@dataclass(frozen=True)
class Decision:
    purpose: str
    allowed: bool
    reasons: Mapping[str, tuple[str, ...]]  # id -> reasons it fails; () means it passes
    checked_at: str

    @property
    def artefacts(self) -> tuple[str, ...]:
        return tuple(self.reasons)


def derive_status(lineage: str, permitted_uses: Mapping[str, str], open_questions: Sequence[str]) -> str:
    """Status implied by the uses and lineage. The ledger value must match this."""
    if lineage == "research_only" or any(v == NOT_PERMITTED for v in permitted_uses.values()):
        return "red"
    if lineage != "yellow" and not open_questions and all(v == PERMITTED for v in permitted_uses.values()):
        return "green"
    return "yellow"


def _validate_entry(aid: str, raw: object, repo_root: Path) -> list[str]:
    problems: list[str] = []
    if not re.fullmatch(r"[a-z0-9_]+", aid):
        problems.append(f"{aid!r}: id must match [a-z0-9_]+")
    if not isinstance(raw, dict):
        return problems + [f"{aid}: entry must be a mapping"]
    missing = REQUIRED_KEYS - raw.keys()
    extra = raw.keys() - REQUIRED_KEYS
    if missing:
        problems.append(f"{aid}: missing keys {sorted(missing)}")
    if extra:
        problems.append(f"{aid}: unknown keys {sorted(extra)}")
    if missing:
        return problems

    if raw["lineage"] not in LINEAGES:
        problems.append(f"{aid}: lineage {raw['lineage']!r} not in {sorted(LINEAGES)}")
    if raw["status"] not in STATUSES:
        problems.append(f"{aid}: status {raw['status']!r} not in {sorted(STATUSES)}")
    if raw["version_hash"] is not None and not isinstance(raw["version_hash"], str):
        problems.append(f"{aid}: version_hash must be a string or null")
    for key in ("kind", "name", "version", "licence", "source_url", "access"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            problems.append(f"{aid}: {key} must be a non-empty string")
    if not isinstance(raw["open_questions"], list) or not all(isinstance(q, str) for q in raw["open_questions"]):
        problems.append(f"{aid}: open_questions must be a list of strings")
        return problems

    uses = raw["permitted_uses"]
    if not isinstance(uses, dict) or set(uses) != set(USES):
        problems.append(f"{aid}: permitted_uses must have exactly the keys {list(USES)}")
        return problems
    bad = {k: v for k, v in uses.items() if v not in USE_VALUES}
    if bad:
        problems.append(f"{aid}: permitted_uses values must be in {sorted(USE_VALUES)}: {bad}")
        return problems

    if raw["lineage"] == "research_only":
        leaked = [u for u in COMMERCIAL_USES if uses[u] != NOT_PERMITTED]
        if leaked:
            problems.append(f"{aid}: research_only lineage must mark {leaked} not_permitted")

    expected = derive_status(raw["lineage"], uses, raw["open_questions"])
    if raw["status"] != expected:
        problems.append(f"{aid}: status is {raw['status']!r} but its uses and lineage imply {expected!r}")

    ev = raw["evidence_file"]
    if not isinstance(ev, str) or Path(ev).is_absolute() or not (repo_root / ev).is_file():
        problems.append(f"{aid}: evidence_file {ev!r} must be a relative path to an existing file")
    return problems


def load_ledger(path: Union[str, Path, None] = None, *, repo_root: Path = REPO_ROOT) -> dict[str, Artefact]:
    """Load and validate the ledger. Raises LedgerError listing every problem found."""
    path = Path(path) if path is not None else LEDGER_PATH
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LedgerError(f"cannot read ledger {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise LedgerError(f"{path}: expected a mapping with schema_version: 1")
    arts = doc.get("artefacts")
    if not isinstance(arts, dict) or not arts:
        raise LedgerError(f"{path}: 'artefacts' must be a non-empty mapping")

    problems: list[str] = []
    for aid, raw in arts.items():
        problems += _validate_entry(str(aid), raw, repo_root)
    if problems:
        raise LedgerError(f"{path}: {len(problems)} problem(s):\n  " + "\n  ".join(problems))

    return {
        aid: Artefact(
            id=aid,
            kind=raw["kind"],
            name=raw["name"],
            version=raw["version"],
            version_hash=raw["version_hash"],
            lineage=raw["lineage"],
            licence=raw["licence"],
            source_url=raw["source_url"],
            access=raw["access"],
            permitted_uses=dict(raw["permitted_uses"]),
            status=raw["status"],
            open_questions=tuple(raw["open_questions"]),
            evidence_file=raw["evidence_file"],
        )
        for aid, raw in arts.items()
    }


def required_uses(purpose: str) -> tuple[str, ...]:
    if purpose not in PURPOSE_USES:
        raise UnknownPurposeError(f"unknown purpose {purpose!r}; expected one of {sorted(PURPOSE_USES)}")
    return PURPOSE_USES[purpose]


def _normalise_ids(artefacts: Iterable[str]) -> list[str]:
    if isinstance(artefacts, str):
        artefacts = [artefacts]
    return list(dict.fromkeys(str(a) for a in artefacts))


def _reasons(ledger: Mapping[str, Artefact], ids: Sequence[str], purpose: str) -> dict[str, tuple[str, ...]]:
    if not ids:
        return {"<manifest>": ("empty manifest: nothing to check (fail closed)",)}
    needed = PURPOSE_USES[purpose]
    out: dict[str, tuple[str, ...]] = {}
    for aid in ids:
        art = ledger.get(aid)
        if art is None:
            out[aid] = ("not in ledger (fail closed)",)
            continue
        reasons: list[str] = []
        if purpose == "research":
            reasons += [f"{u}={art.permitted_uses[u]}" for u in needed if art.permitted_uses[u] != PERMITTED]
        else:
            if art.status != "green":
                reasons.append(f"status={art.status}")
            reasons += [f"{u}={art.permitted_uses[u]}" for u in needed if art.permitted_uses[u] != PERMITTED]
        out[aid] = tuple(reasons)
    return out


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(token: str) -> str:
    return _SAFE_ID.sub("?", token)[:200]


def _append_log(decision: Decision | None, *, purpose: str, source: str, error: str | None,
                log_path: Union[str, Path, None]) -> None:
    path = Path(log_path) if log_path is not None else DEFAULT_LOG
    if decision is None:
        verdict = "ERROR"
        detail = _safe(error or "unknown error")
        checked = _now()
    else:
        verdict = "ALLOWED" if decision.allowed else "REFUSED"
        parts = []
        for aid, rs in decision.reasons.items():
            parts.append(_safe(aid) + ("[" + ";".join(_safe(r) for r in rs) + "]" if rs else ""))
        detail = ",".join(parts) or "-"
        checked = decision.checked_at
    line = "\t".join([checked, f"source={_safe(source)}", f"purpose={_safe(purpose)}",
                      f"verdict={verdict}", f"ids={detail}"]) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:  # append-only: never truncate or rewrite
        fh.write(line)


def check(artefacts: Iterable[str], purpose: str, *, ledger: Mapping[str, Artefact] | None = None,
          log_path: Union[str, Path, None] = None, source: str = "api") -> Decision:
    """Evaluate a manifest against the ledger and log the result. Never raises for a refusal."""
    ids = _normalise_ids(artefacts)
    if purpose not in PURPOSE_USES:
        msg = f"unknown purpose {purpose!r}"
        _append_log(None, purpose=str(purpose), source=source, error=msg, log_path=log_path)
        raise UnknownPurposeError(msg + f"; expected one of {sorted(PURPOSE_USES)}")
    try:
        led = load_ledger() if ledger is None else ledger
    except LedgerError as exc:
        _append_log(None, purpose=purpose, source=source, error=f"ledger error: {exc}", log_path=log_path)
        raise
    reasons = _reasons(led, ids, purpose)
    allowed = not any(reasons.values())
    decision = Decision(purpose=purpose, allowed=allowed, reasons=reasons, checked_at=_now())
    _append_log(decision, purpose=purpose, source=source, error=None, log_path=log_path)
    return decision


def assert_buildable(artefacts: Iterable[str], purpose: str, *, ledger: Mapping[str, Artefact] | None = None,
                     log_path: Union[str, Path, None] = None, source: str = "assert_buildable") -> Decision:
    """Raise RightsRefused unless every artefact is green for ``purpose`` (or research-permitted)."""
    decision = check(artefacts, purpose, ledger=ledger, log_path=log_path, source=source)
    if not decision.allowed:
        raise RightsRefused(decision)
    return decision


def requires_rights(purpose: str, artefacts: Union[Sequence[str], Callable[..., Sequence[str]]], *,
                    ledger: Mapping[str, Artefact] | None = None,
                    log_path: Union[str, Path, None] = None) -> Callable:
    """Decorator: refuse to call the function unless its artefacts pass the purpose gate.

    ``artefacts`` is a list of ids, or a callable that receives the call's arguments and returns ids.
    """
    if purpose not in PURPOSE_USES:
        raise UnknownPurposeError(f"unknown purpose {purpose!r}")

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ids = artefacts(*args, **kwargs) if callable(artefacts) else artefacts
            assert_buildable(ids, purpose, ledger=ledger, log_path=log_path,
                             source=f"decorator:{fn.__qualname__}")
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def format_decision(decision: Decision) -> str:
    verdict = "ALLOWED" if decision.allowed else "REFUSED"
    lines = [f"purpose: {decision.purpose}", f"decision: {verdict}"]
    for aid, rs in decision.reasons.items():
        lines.append(f"  {aid}: " + ("ok" if not rs else "; ".join(rs)))
    return "\n".join(lines)


def _read_manifest(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    doc = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if isinstance(doc, dict):
        doc = doc.get("artefacts")
    if not isinstance(doc, list) or not all(isinstance(x, str) for x in doc):
        raise ValueError("manifest must be a list of artefact ids, or a mapping with an 'artefacts' list")
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sortinghat.rights", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    chk = sub.add_parser("check", help="check a build manifest against the ledger")
    chk.add_argument("--purpose", required=True, choices=sorted(PURPOSE_USES))
    src = chk.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="YAML or JSON list of ids, or mapping with 'artefacts'")
    src.add_argument("--ids", help="comma-separated artefact ids")
    chk.add_argument("--ledger", type=Path, default=None)
    chk.add_argument("--log", type=Path, default=None, help=f"default {DEFAULT_LOG}")

    lst = sub.add_parser("list", help="print the ledger")
    lst.add_argument("--ledger", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "list":
            led = load_ledger(args.ledger)
            for aid, a in led.items():
                print(f"{aid:<26} {a.lineage:<20} {a.status:<7} research={a.permitted_uses['research']:<13}"
                      f" commercial_training={a.permitted_uses['commercial_training']}")
            return 0

        ids = [x.strip() for x in args.ids.split(",")] if args.ids is not None else _read_manifest(args.manifest)
        led = load_ledger(args.ledger)
        decision = check(ids, args.purpose, ledger=led, log_path=args.log, source="cli")
    except (LedgerError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_decision(decision))
    return 0 if decision.allowed else 3


if __name__ == "__main__":
    sys.exit(main())
