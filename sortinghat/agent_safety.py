"""Agent-session safety helpers (Phase 0e).

* ``restricted_roots`` - paths treated as restricted (defaults + env
  ``SORTINGHAT_RESTRICTED_ROOT``, os.pathsep-separated).
* ``assert_not_restricted_in_agent`` - refuse to open restricted paths when the
  process looks like it runs inside a Claude Code session (``CLAUDECODE=1``).
* ``python -m sortinghat.agent_safety --write-local`` writes
  ``.claude/settings.local.json`` deny rules for the env-configured roots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ENV_ROOT = "SORTINGHAT_RESTRICTED_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = ["/restricted", "data/heedb"]


class RestrictedDataError(RuntimeError):
    pass


def restricted_roots(include_defaults: bool = True) -> list[Path]:
    raw = list(DEFAULT_ROOTS) if include_defaults else []
    raw += [p for p in os.environ.get(ENV_ROOT, "").split(os.pathsep) if p]
    out = []
    for r in raw:
        p = Path(r)
        out.append((p if p.is_absolute() else REPO_ROOT / p).resolve())
    return out


def in_agent_session() -> bool:
    return os.environ.get("CLAUDECODE") == "1" or os.environ.get("SORTINGHAT_AGENT_SESSION") == "1"


def is_restricted(path: str | Path) -> bool:
    p = Path(path).resolve()
    return any(p == r or r in p.parents for r in restricted_roots())


def assert_not_restricted_in_agent(path: str | Path) -> None:
    if in_agent_session() and is_restricted(path):
        raise RestrictedDataError(
            "Refusing to read a restricted data path from inside an agent session. "
            "Run this job from a plain terminal or scheduler (see CLAUDE.md).")


def deny_rules(roots: list[str]) -> list[str]:
    rules: list[str] = []
    for r in roots:
        r = r.rstrip("/")
        if r.startswith("/"):
            rules += [f"Read(/{r}/**)", f"Edit(/{r}/**)"]
        else:
            rules += [f"Read(./{r}/**)", f"Edit(./{r}/**)"]
        rules += [f"Bash(*{r}*)"]
    return rules


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-local", action="store_true",
                    help="write deny rules for SORTINGHAT_RESTRICTED_ROOT into .claude/settings.local.json")
    args = ap.parse_args(argv)
    extra = [p for p in os.environ.get(ENV_ROOT, "").split(os.pathsep) if p]
    rules = deny_rules(extra)
    if not args.write_local:
        print(json.dumps({"permissions": {"deny": rules}}, indent=2))
        return 0
    target = REPO_ROOT / ".claude" / "settings.local.json"
    cur = json.loads(target.read_text()) if target.exists() else {}
    deny = cur.setdefault("permissions", {}).setdefault("deny", [])
    for r in rules:
        if r not in deny:
            deny.append(r)
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(cur, indent=2) + "\n")
    print(f"wrote {len(rules)} deny rules to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
