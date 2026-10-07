import json
from pathlib import Path

from sortinghat import agent_safety as a

REPO = Path(__file__).resolve().parents[1]


def test_settings_json_denies_restricted_paths():
    s = json.loads((REPO / ".claude" / "settings.json").read_text())
    deny = s["permissions"]["deny"]
    assert "Read(//restricted/**)" in deny
    assert "Read(./data/heedb/**)" in deny
    assert any(d.startswith("Bash(") and "restricted" in d for d in deny)


def test_env_root_extends_restricted(monkeypatch, tmp_path):
    monkeypatch.setenv(a.ENV_ROOT, str(tmp_path / "x"))
    assert a.is_restricted(tmp_path / "x" / "tbl.parquet")
    assert a.is_restricted("/restricted/foo")
    assert not a.is_restricted(tmp_path / "other")


def test_deny_rules_shape():
    r = a.deny_rules(["/mnt/heedb"])
    assert "Read(//mnt/heedb/**)" in r and "Bash(*/mnt/heedb*)" in r
