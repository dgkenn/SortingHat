"""Tests for the data-rights ledger and production-build refusal rule (sortinghat/rights.py)."""

from __future__ import annotations

import re

import pytest
import yaml

from sortinghat import rights
from sortinghat.rights import (
    GATED_PURPOSES,
    LedgerError,
    RightsRefused,
    UnknownPurposeError,
    assert_buildable,
    check,
    load_ledger,
    main,
    requires_rights,
)

LISTED = {
    "heedb_v4_1", "bind_v1_0", "morgoth_1_0_0", "nesi_1_0_0", "gcs_from_ehr_1_0_0", "tme_cohort_1_0_0",
    "icare_2_0_bdsp", "icare_2_1_physionet", "sparcnet_1_1", "prophet_1_0",
    "tuh_tueg", "tuh_tuab", "tuh_tusz", "tuh_tuev", "tuh_tuar", "tdbrain_v3", "cbramod_checkpoint",
    "nmt", "vitaldb_physionet_1_0_0", "vitaldb_api", "certa_eeg", "korean_ncse_cohort_2024",
}
LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\t")


@pytest.fixture(autouse=True)
def log_file(tmp_path, monkeypatch):
    """Keep every check out of the real docs/rights_checks.log."""
    path = tmp_path / "rights_checks.log"
    monkeypatch.setattr(rights, "DEFAULT_LOG", path)
    return path


@pytest.fixture(scope="module")
def ledger():
    return load_ledger()


def _uses(**overrides):
    base = {u: "permitted" for u in rights.USES}
    base.update(overrides)
    return base


def _entry(**overrides):
    e = {
        "kind": "dataset", "name": "Synthetic test set", "version": "0.1", "version_hash": None,
        "lineage": "commercial_candidate", "licence": "test licence", "source_url": "https://example.org",
        "access": "open", "permitted_uses": _uses(), "status": "green", "open_questions": [],
        "evidence_file": "docs/research/citation_verification.md",
    }
    e.update(overrides)
    return e


def _write_ledger(tmp_path, artefacts):
    p = tmp_path / "ledger.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 1, "ledger_as_of": "2026-10-08", "artefacts": artefacts}))
    return p


# ---- real ledger --------------------------------------------------------------------------

def test_real_ledger_loads_and_lists_every_artefact(ledger):
    assert LISTED <= set(ledger)
    assert all(a.status in {"green", "yellow", "red"} for a in ledger.values())


def test_real_ledger_has_no_green_artefact_yet(ledger):
    # No counsel review is recorded (DECISION_LOG D-072), so nothing may pass a commercial gate.
    assert [a.id for a in ledger.values() if a.status == "green"] == []


def test_research_passes_with_research_only_data(ledger):
    d = assert_buildable(["heedb_v4_1", "icare_2_1_physionet"], "research", ledger=ledger)
    assert d.allowed and d.reasons["heedb_v4_1"] == ()


@pytest.mark.parametrize("purpose", sorted(GATED_PURPOSES))
@pytest.mark.parametrize("artefacts", [["heedb_v4_1"], ["morgoth_1_0_0"], ["cbramod_checkpoint"]])
def test_commercial_build_refused_for_research_only_and_yellow(ledger, purpose, artefacts):
    with pytest.raises(RightsRefused) as exc:
        assert_buildable(artefacts, purpose, ledger=ledger)
    assert exc.value.decision.allowed is False


def test_cbramod_refusal_names_yellow_status(ledger):
    d = check(["cbramod_checkpoint"], "production_build", ledger=ledger)
    reasons = d.reasons["cbramod_checkpoint"]
    assert "status=yellow" in reasons
    assert "commercial_training=unknown" in reasons
    assert "derivative_weights=unknown" in reasons


def test_heedb_refusal_names_not_permitted_uses(ledger):
    d = check(["heedb_v4_1"], "production_build", ledger=ledger)
    assert "status=red" in d.reasons["heedb_v4_1"]
    assert "commercial_training=not_permitted" in d.reasons["heedb_v4_1"]


def test_full_ledger_refused_for_every_gated_purpose(ledger):
    for purpose in GATED_PURPOSES:
        assert not check(sorted(ledger), purpose, ledger=ledger).allowed


# ---- fail closed ---------------------------------------------------------------------------

@pytest.mark.parametrize("purpose", ["research", "commercial_training", "production_build", "regulatory_submission"])
def test_unknown_artefact_fails_closed(ledger, purpose):
    d = check(["not_in_ledger_xyz"], purpose, ledger=ledger)
    assert d.allowed is False
    assert d.reasons["not_in_ledger_xyz"] == ("not in ledger (fail closed)",)


def test_unknown_artefact_is_refused_even_alongside_permitted_ones(ledger):
    d = check(["heedb_v4_1", "ghost_dataset"], "research", ledger=ledger)
    assert d.allowed is False
    assert d.reasons["heedb_v4_1"] == ()


def test_empty_manifest_refused(ledger):
    d = check([], "research", ledger=ledger)
    assert d.allowed is False


def test_unknown_purpose_raises_and_is_logged(ledger, log_file):
    with pytest.raises(UnknownPurposeError):
        check(["heedb_v4_1"], "marketing", ledger=ledger)
    line = log_file.read_text().splitlines()[-1]
    assert "verdict=ERROR" in line and "purpose=marketing" in line


# ---- gate semantics with a synthetic ledger ------------------------------------------------

def test_green_artefact_passes_gated_purposes(tmp_path):
    path = _write_ledger(tmp_path, {"clean_set": _entry()})
    led = load_ledger(path)
    for purpose in GATED_PURPOSES:
        assert check(["clean_set"], purpose, ledger=led).allowed


def test_one_unknown_use_blocks_green_for_every_purpose(tmp_path):
    # Partly cleared is treated as not green, so it blocks every commercial purpose.
    path = _write_ledger(tmp_path, {"partial": _entry(
        permitted_uses=_uses(sublicensing="unknown"), status="yellow")})
    led = load_ledger(path)
    for purpose in GATED_PURPOSES:
        d = check(["partial"], purpose, ledger=led)
        assert d.allowed is False
        assert "status=yellow" in d.reasons["partial"]


def test_gate_only_checks_the_uses_the_purpose_needs(tmp_path):
    # Red because sublicensing is not_permitted. commercial_training alone is permitted, so it is
    # still refused: the status must be green regardless of which use is asked.
    path = _write_ledger(tmp_path, {"narrow": _entry(
        lineage="commercial_candidate",
        permitted_uses=_uses(sublicensing="not_permitted"), status="red")})
    led = load_ledger(path)
    d = check(["narrow"], "commercial_training", ledger=led)
    assert d.allowed is False
    assert "sublicensing=not_permitted" not in d.reasons["narrow"]  # not needed for this purpose
    assert "status=red" in d.reasons["narrow"]


# ---- ledger validation ---------------------------------------------------------------------

def test_status_must_match_uses(tmp_path):
    path = _write_ledger(tmp_path, {"bad": _entry(permitted_uses=_uses(sublicensing="unknown"),
                                                  status="green")})
    with pytest.raises(LedgerError, match="status is 'green' but its uses"):
        load_ledger(path)


def test_research_only_cannot_permit_commercial_use(tmp_path):
    path = _write_ledger(tmp_path, {"leak": _entry(lineage="research_only", status="red")})
    with pytest.raises(LedgerError, match="research_only lineage"):
        load_ledger(path)


def test_yellow_lineage_cannot_be_green(tmp_path):
    path = _write_ledger(tmp_path, {"sa": _entry(lineage="yellow", status="green")})
    with pytest.raises(LedgerError, match="status is 'green'"):
        load_ledger(path)


def test_missing_evidence_file_rejected(tmp_path):
    path = _write_ledger(tmp_path, {"orphan": _entry(evidence_file="docs/does_not_exist.md")})
    with pytest.raises(LedgerError, match="evidence_file"):
        load_ledger(path)


def test_absolute_evidence_path_rejected(tmp_path):
    path = _write_ledger(tmp_path, {"abs": _entry(evidence_file=str(rights.REPO_ROOT / "DECISION_LOG.md"))})
    with pytest.raises(LedgerError, match="evidence_file"):
        load_ledger(path)


def test_bad_enum_and_missing_key_rejected(tmp_path):
    bad = _entry(lineage="green_ish")
    del bad["licence"]
    path = _write_ledger(tmp_path, {"broken": bad})
    with pytest.raises(LedgerError) as exc:
        load_ledger(path)
    assert "lineage" in str(exc.value) and "licence" in str(exc.value)


def test_unknown_use_value_rejected(tmp_path):
    path = _write_ledger(tmp_path, {"typo": _entry(permitted_uses=_uses(research="yes"))})
    with pytest.raises(LedgerError, match="permitted_uses values"):
        load_ledger(path)


# ---- decorator -----------------------------------------------------------------------------

def test_decorator_blocks_the_call_before_it_runs(ledger):
    ran = []

    @requires_rights("production_build", ["cbramod_checkpoint"], ledger=ledger)
    def build():
        ran.append(True)

    with pytest.raises(RightsRefused):
        build()
    assert ran == []


def test_decorator_reads_ids_from_arguments(ledger):
    @requires_rights("research", lambda manifest: manifest["ids"], ledger=ledger)
    def fit(manifest):
        return "fitted"

    assert fit({"ids": ["heedb_v4_1"]}) == "fitted"
    with pytest.raises(RightsRefused):
        fit({"ids": ["heedb_v4_1", "ghost"]})


def test_decorator_rejects_unknown_purpose_at_definition():
    with pytest.raises(UnknownPurposeError):
        requires_rights("marketing", ["heedb_v4_1"])


# ---- log -----------------------------------------------------------------------------------

def test_log_is_dated_and_append_only(ledger, log_file):
    check(["heedb_v4_1"], "research", ledger=ledger)
    first = log_file.read_text()
    check(["cbramod_checkpoint"], "production_build", ledger=ledger)
    text = log_file.read_text()
    assert text.startswith(first)
    lines = text.splitlines()
    assert len(lines) == 2 and all(LOG_LINE.match(x) for x in lines)
    assert "verdict=ALLOWED" in lines[0]
    assert "verdict=REFUSED" in lines[1] and "cbramod_checkpoint[" in lines[1]


def test_log_holds_ids_and_verdicts_only(ledger, log_file):
    check(["heedb_v4_1"], "production_build", ledger=ledger)
    line = log_file.read_text().splitlines()[0]
    assert set(line.split("\t")[1:2]) <= {"source=api"} or "source=api" in line
    assert "/" not in line  # no paths or data locations


def test_log_sanitises_hostile_ids(ledger, log_file):
    check(["bad\tid\nsecond_line"], "research", ledger=ledger)
    assert len(log_file.read_text().splitlines()) == 1


# ---- CLI -----------------------------------------------------------------------------------

def test_cli_exit_codes(log_file, capsys):
    assert main(["check", "--purpose", "production_build", "--ids", "heedb_v4_1,cbramod_checkpoint"]) == 3
    out = capsys.readouterr().out
    assert "decision: REFUSED" in out and "cbramod_checkpoint:" in out
    assert main(["check", "--purpose", "research", "--ids", "heedb_v4_1"]) == 0
    assert main(["check", "--purpose", "research", "--ids", "ghost"]) == 3


def test_cli_reads_manifest_file(tmp_path, capsys):
    manifest = tmp_path / "build.yaml"
    manifest.write_text(yaml.safe_dump({"artefacts": ["icare_2_1_physionet", "heedb_v4_1"]}))
    assert main(["check", "--purpose", "research", "--manifest", str(manifest)]) == 0
    assert main(["check", "--purpose", "commercial_training", "--manifest", str(manifest)]) == 3


def test_cli_usage_and_ledger_errors_return_2(tmp_path, capsys):
    assert main(["check", "--purpose", "research", "--ids", "heedb_v4_1",
                 "--ledger", str(tmp_path / "missing.yaml")]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_list_prints_every_artefact(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "cbramod_checkpoint" in out and "heedb_v4_1" in out
