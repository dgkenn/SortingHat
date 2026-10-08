"""Visit matching rules (any-overlap, null end, date-only, text ids, slack) and the --debug-flow report."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.cohort import CohortConfig, FrameSources, build_cohort
from sortinghat.cohort import rules
from sortinghat.cohort.flow import debug_report
from sortinghat.cohort.output import known_ids
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_cohort_helpers import GCS, H, T0, World

REPO = Path(__file__).resolve().parents[1]
ACUTE = ("ICU", "Inpatient", "ED")


def sess(t0=T0):
    return pd.DataFrame({"person_id": pd.array([1], dtype="Int64"), "t0": [pd.Timestamp(t0)]})


def vis(rows):
    base = {"person_id": 1, "visit_concept_id": 9201, "visit_source_value": None}
    return pd.DataFrame([{**base, **r} for r in rows])


def match(rows, **kw):
    return rules.match_visits(sess(), vis(rows), ACUTE, 6.0, **kw).iloc[0]


def test_a_short_nested_visit_no_longer_hides_the_admission():
    r = match([{"visit_start_datetime": T0 - H(100), "visit_end_datetime": T0 + H(100), "visit_concept_id": 9201},
               {"visit_start_datetime": T0 - H(2), "visit_end_datetime": T0 - H(1), "visit_concept_id": 9202}])
    assert r["visit_class"] == "Inpatient" and r["visit_match"] == "exact"      # latest-start rule found nothing


def test_acuity_priority_among_covering_visits_then_latest_start():
    r = match([{"visit_start_datetime": T0 - H(5), "visit_end_datetime": T0 + H(5), "visit_concept_id": 9202},
               {"visit_start_datetime": T0 - H(9), "visit_end_datetime": T0 + H(5), "visit_concept_id": 32037},
               {"visit_start_datetime": T0 - H(3), "visit_end_datetime": T0 + H(5), "visit_concept_id": 9201}])
    assert r["visit_class"] == "ICU" and r["visit_start"] == T0 - H(9)


def test_null_end_is_open_for_open_visit_days():
    row = {"visit_start_datetime": T0 - H(24 * 29), "visit_end_datetime": pd.NaT}
    assert match([row])["visit_class"] == "Inpatient"
    assert pd.isna(match([{**row, "visit_start_datetime": T0 - H(24 * 31)}])["visit_class"])
    assert match([{**row, "visit_start_datetime": T0 - H(24 * 31)}], open_days=None)["visit_class"] == "Inpatient"
    assert match([{**row, "visit_start_datetime": T0 - H(24 * 31)}], open_days=60)["visit_class"] == "Inpatient"


def test_date_only_visits():
    d = T0.normalize()
    # an admission ending on the EEG's own date (date-only end = 00:00) still covers the EEG at 12:00 that day
    r = match([{"visit_start_datetime": d - pd.Timedelta(days=3), "visit_end_datetime": d}])
    assert r["visit_class"] == "Inpatient"
    assert pd.isna(match([{"visit_start_datetime": d - pd.Timedelta(days=3), "visit_end_datetime": d}],
                         date_only_end_of_day=False)["visit_class"])
    # datetimes missing: the date columns are used (start of the start day, end of the end day)
    r = match([{"visit_start_datetime": pd.NaT, "visit_end_datetime": pd.NaT, "visit_start_date": str(d.date()),
                "visit_end_date": str(d.date())}])
    assert r["visit_class"] == "Inpatient" and r["visit_start"] == d


def test_end_before_start_is_a_zero_length_visit_and_text_person_ids_are_cast():
    v = vis([{"visit_start_datetime": T0 - H(2), "visit_end_datetime": T0 - H(30)}])
    v["person_id"] = "0000001"                                   # text with leading zeros
    assert pd.isna(rules.match_visits(sess(), v, ACUTE, 6.0).iloc[0]["visit_class"])
    v["visit_end_datetime"] = pd.NaT
    assert rules.match_visits(sess(), v, ACUTE, 6.0).iloc[0]["visit_class"] == "Inpatient"


def test_slack_widens_and_is_labelled():
    row = {"visit_start_datetime": T0 + H(3), "visit_end_datetime": T0 + H(30)}     # EEG 3 h before the visit starts
    assert pd.isna(match([row])["visit_class"])
    r = match([row], slack_h=6.0)
    assert r["visit_class"] == "Inpatient" and r["visit_match"] == "slack"
    assert match([{**row, "visit_start_datetime": T0 - H(1)}], slack_h=6.0)["visit_match"] == "exact"


def test_cohort_uses_the_robust_rules_end_to_end():
    w = World()
    w.eeg(1); w.score(1, GCS, T0 - H(1), 8)
    w.visit(1, "Inpatient", T0 - H(100), end=T0 + H(100)); w.visit(1, "Outpatient", T0 - H(2), end=T0 - H(1))
    w.eeg(2); w.score(2, GCS, T0 - H(1), 8); w.visit(2, "ICU", T0 + H(3), end=T0 + H(50))   # EEG precedes the visit
    assert build_cohort(FrameSources(w.tables())).table["person_id"].tolist() == [1]
    r = build_cohort(FrameSources(w.tables()), CohortConfig(visit_slack_h=6.0))
    t = r.table.set_index("person_id")
    assert sorted(t.index) == [1, 2] and t.loc[1, "visit_match"] == "exact" and t.loc[2, "visit_match"] == "slack"


# ------------------------------------------------------------------------------------------ debug flow
def test_debug_flow_lists_every_step_and_suppresses_only_individual_counts(synth):
    res = build_cohort(FrameSources(synth[0]))
    steps = [s["label"] for s in res.flow_raw["steps"]]
    rows = res.debug["sites"]["ALL"]["rows"]
    assert [r["step"] for r in rows] == steps and not any(" + " in r["step"] for r in rows)
    raw = [sum(s["remaining"].values()) for s in res.flow_raw["steps"]]
    for r, n, prev in zip(rows[1:], raw[1:], raw):
        assert r["n_excluded"] == (prev - n if prev - n >= 11 else SUPPRESSED)
        assert r["n_remaining"] == (n if n >= 11 else SUPPRESSED)
    assert any(r["n_excluded"] == SUPPRESSED for r in rows)          # small steps stay separate, shown as <11
    assert len(rows) > len(res.report["sites"]["ALL"]["rows"])        # the shareable report merges them
    assert res.debug["debug"].startswith("DIAGNOSTIC")
    assert_aggregate_only(res.debug, known_ids(res))


def test_debug_report_withholds_a_site_that_starts_below_11():
    raw = {"sites": ["A"], "steps": [{"label": "s", "unit": "u", "remaining": {"A": 5}, "start": True}],
           "partitions": [], "checks": {}}
    assert debug_report(raw)["sites"]["A"] == {"withheld": "starting count < 11"}


def test_cli_debug_flow(synth_dir, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "SORTINGHAT_AGENT_SESSION")}
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "build_cohort.py"), "--data", str(synth_dir),
                        "--out", str(tmp_path), "--debug-flow"], capture_output=True, text=True, check=True,
                       env=env, cwd=REPO)
    assert "DEBUG FLOW" in p.stdout and "No visit covering the EEG start | excluded" in p.stdout
    assert (tmp_path / "cohort" / "flow_debug.md").read_text().startswith("# Study 1 cohort flow")
    assert (tmp_path / "cohort" / "flow.md").exists()
