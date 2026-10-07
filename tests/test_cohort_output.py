"""Cohort outputs: file locations and modes, nothing record-level printed or written outside local_only, CLI."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sortinghat.cohort import FrameSources, build_cohort, read_key_list, write_outputs
from sortinghat.cohort.build import KEY_LIST_COLUMNS
from sortinghat.cohort.output import known_ids

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_cohort.py"


def mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


@pytest.fixture(scope="module")
def cli_run(synth_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("cohort_out")
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "SORTINGHAT_AGENT_SESSION")}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--data", str(synth_dir), "--out", str(out)],
                          capture_output=True, text=True, check=True, env=env, cwd=REPO)
    return out, proc


def test_cli_writes_the_expected_files_with_private_modes(cli_run):
    out, _ = cli_run
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == ["cohort/flow.json", "cohort/flow.md", "local_only/cohort_study1.csv",
                     "local_only/recording_keys.csv"]
    assert mode(out / "local_only") == 0o700
    assert mode(out / "local_only" / "cohort_study1.csv") == 0o600
    assert mode(out / "local_only" / "recording_keys.csv") == 0o600


def test_cli_prints_aggregates_only(cli_run, synth):
    out, proc = cli_run
    tables = synth[0]
    ids = {str(i) for i in tables["omop_person"]["person_id"]} | set(tables["eeg_metadata"]["SessionID"].astype(str)) \
        | set(tables["eeg_metadata"]["BidsFolder"].astype(str))
    for text in (proc.stdout, proc.stderr):
        assert not any(i in text for i in ids)
    assert "strict cohort" in proc.stdout and "mode 0600" in proc.stdout


def test_no_identifier_or_timestamp_outside_local_only(cli_run, synth):
    out, _ = cli_run
    tables = synth[0]
    ids = [str(i) for i in tables["omop_person"]["person_id"]][:3000] + list(tables["eeg_metadata"]["SessionID"].astype(str))
    for p in list((out / "cohort").iterdir()):
        text = p.read_text()
        assert not any(i in text for i in ids), p.name
        assert "sub-" not in text and "ses-" not in text and "2014-" not in text


def test_local_only_files_hold_the_table_and_the_key_list(cli_run):
    out, _ = cli_run
    t = pd.read_csv(out / "local_only" / "cohort_study1.csv")
    k = read_key_list(out / "local_only" / "recording_keys.csv")
    assert t["person_id"].is_unique and len(t) == len(k) > 100
    assert list(k.columns) == KEY_LIST_COLUMNS
    assert {"t0", "onset", "hours_since_onset", "in_strict", "in_broad"} <= set(t.columns)


def test_flow_md_lists_the_pending_qc_step(cli_run):
    text = (cli_run[0] / "cohort" / "flow.md").read_text()
    assert "Fp1, Fp2, F7, F8, T3, T4, T5, T6, O1, O2" in text and "first qualifying EEG" in text


def test_write_outputs_rewrites_existing_files_privately(synth, tmp_path):
    res = build_cohort(FrameSources(synth[0]))
    write_outputs(res, tmp_path)
    os.chmod(tmp_path / "local_only" / "cohort_study1.csv", 0o644)           # a loosened file is tightened again
    write_outputs(res, tmp_path)
    assert mode(tmp_path / "local_only" / "cohort_study1.csv") == 0o600
    assert len(known_ids(res)) > 100


def test_gitignore_covers_the_output_locations():
    ignored = (REPO / ".gitignore").read_text().split()
    assert "out/" in ignored and "**/local_only/" in ignored


def test_cli_refuses_a_restricted_path_inside_an_agent_session(synth_dir, tmp_path):
    env = {**os.environ, "CLAUDECODE": "1", "SORTINGHAT_RESTRICTED_ROOT": str(synth_dir)}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--data", str(synth_dir), "--out", str(tmp_path)],
                          capture_output=True, text=True, env=env, cwd=REPO)
    assert proc.returncode != 0 and not (tmp_path / "local_only").exists()
