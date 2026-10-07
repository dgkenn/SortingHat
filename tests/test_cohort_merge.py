"""Patient merge history: applied before first-EEG selection when the table exists (D-106); a notice when absent."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sortinghat.cohort import CohortConfig, FrameSources, StoreSources, build_cohort
from sortinghat.cohort.build import MERGE_NOTICE
from sortinghat.cohort.sources import merge_pairs, resolve_merges
from sortinghat.data_io import LocalStore
from test_cohort_helpers import GCS, H, T0, World

REPO = Path(__file__).resolve().parents[1]


def world():
    w = World()
    w.patient(1)                                                      # later EEG of the surviving record
    w.eeg(2, t0=T0 - H(100)); w.visit(2, "ICU", T0 - H(103))           # earlier EEG under the retired id
    w.score(2, GCS, T0 - H(101), 8)
    w.patient(5)                                                      # unrelated
    return w


def tables(w, hist=None):
    t = w.tables()
    if hist is not None:
        t["patient_merge_history"] = hist
    return t


def test_without_a_merge_table_two_patients_and_a_notice():
    r = build_cohort(FrameSources(tables(world())))
    assert r.merge_status == "absent" and sorted(r.table["person_id"]) == [1, 2, 5]
    assert r.report["checks"]["merge_history_status"] == MERGE_NOTICE["absent"]


def test_merge_is_applied_before_first_eeg_selection():
    hist = pd.DataFrame({"OldBDSPPatientID": ["2"], "NewBDSPPatientID": ["1"]})
    r = build_cohort(FrameSources(tables(world(), hist)))
    t = r.table.set_index("person_id")
    assert r.merge_status == "applied" and sorted(t.index) == [1, 5]
    assert t.loc[1, "t0"] == T0 - H(100) and t.loc[1, "person_id_source"] == 2   # the retired id's EEG came first
    assert t.loc[1, "severity_strict"]                                           # its score was re-keyed too
    assert removed(r, "Not the patient's first") == 1                            # 1's own later EEG is dropped
    assert "re-keyed" in r.report["checks"]["merge_history_status"]
    assert r.fates.loc[1] == "included" and 2 not in r.fates.index


def removed(r, label):
    steps = r.flow_raw["steps"]
    return next(sum(a["remaining"].values()) - sum(b["remaining"].values())
                for a, b in zip(steps, steps[1:]) if b["label"].startswith(label))


def test_chains_are_resolved_and_cycles_do_not_hang():
    assert resolve_merges({3: 2, 2: 1}) == {3: 1, 2: 1}
    assert resolve_merges({1: 2, 2: 1})                                # terminates
    assert merge_pairs(pd.DataFrame({"merged_person_id": [3, "x"], "surviving_person_id": [2, 9]})) == {3: 2}
    assert merge_pairs(pd.DataFrame({"a": [1], "b": [2]})) is None     # columns not identifiable


def test_unrecognised_layout_is_reported_and_not_applied():
    r = build_cohort(FrameSources(tables(world(), pd.DataFrame({"a": [2], "b": [1]}))))
    assert r.merge_status == "unrecognised" and sorted(r.table["person_id"]) == [1, 2, 5]
    assert "NOT applied" in r.report["checks"]["merge_history_status"]


@pytest.fixture(scope="module")
def merged_store(synth_dir, tmp_path_factory):
    base = build_cohort(StoreSources(LocalStore(synth_dir)))
    both = base.table[base.table["SiteID"] == "S0001"].sort_values("person_id")
    a, b = int(both["person_id"].iloc[0]), int(both["person_id"].iloc[1])
    root = tmp_path_factory.mktemp("merged_store")
    shutil.copytree(synth_dir, root / "d")
    (root / "d" / "PatientMergeHistory").mkdir()
    pd.DataFrame({"RetiredPatientID": [b], "SurvivingPatientID": [a]}).to_csv(
        root / "d" / "PatientMergeHistory" / "merge_0.csv", index=False)
    return base, build_cohort(StoreSources(LocalStore(root / "d"))), a, b


def test_store_with_a_PatientMergeHistory_prefix(merged_store):
    base, merged, a, b = merged_store
    assert base.merge_status == "absent" and merged.merge_status == "applied"
    bt, mt = base.table.set_index("person_id"), merged.table.set_index("person_id")
    assert b in bt.index and b not in mt.index and len(mt) == len(bt) - 1
    assert mt.loc[a, "t0"] == min(bt.loc[a, "t0"], bt.loc[b, "t0"])


def test_cli_prints_the_one_line_notice(synth_dir, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "SORTINGHAT_AGENT_SESSION")}
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "build_cohort.py"), "--data", str(synth_dir),
                        "--out", str(tmp_path)], capture_output=True, text=True, check=True, env=env, cwd=REPO)
    assert sum("patient merge history" in ln for ln in p.stdout.splitlines()) == 1
    assert MERGE_NOTICE["absent"] in p.stdout
