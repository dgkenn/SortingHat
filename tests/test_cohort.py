"""Study 1 cohort: every inclusion / exclusion step on hand-built patients, first-EEG-per-patient, onset proxy,
strict severity, broad phenotype, site variants, and agreement with the Phase 0a audit on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from sortinghat.audit import field_audit as fa
from sortinghat.cohort import CohortConfig, FrameSources, build_cohort
from sortinghat.cohort import rules
from sortinghat.cohort.build import INCLUDED, duration_unit_check
from test_cohort_helpers import FOUR, GCS, H, T0, World, remaining_after, removed_at

PRE_FIRST = ("Patient id", "EEG start", "Age", "No visit", "Visit care", "Non-acute", "OR or EMU")


def build(w, **cfg):
    return build_cohort(FrameSources(w.tables()), CohortConfig(**cfg))


@pytest.fixture(scope="module")
def scenario():
    w = World()
    P = w.patient
    P(1)                                                        # strict, ICU, GCS 8 at t0-1h
    P(2, start_blank=True)                                      # EEG start missing
    P(3, age_blank=True)                                        # age missing
    P(4, age=17.9)                                              # child
    P(5, age=18.0)                                              # boundary: adult
    P(6, visit_cls=None)                                        # no visit at all
    w.eeg(7); w.visit(7, "ICU", T0 - H(30), end=T0 - H(5)); w.score(7, GCS, T0 - H(1), 8)    # visit ended before t0
    P(8, visit_cls="Outpatient")                                # non-acute
    w.eeg(9); w.visit(9, "ICU", T0 - H(3), concept=False); w.score(9, GCS, T0 - H(1), 8)    # unclassifiable visit
    w.eeg(10); w.visit(10, None or "ICU", T0 - H(3), concept=False, text="Intensive Care Unit"); w.score(10, GCS, T0 - H(1), 8)
    P(11, service="OR")
    P(12, service="EMU")
    # first qualifying EEG: an earlier outpatient EEG does not qualify, the later ICU EEG is the index
    w.eeg(13, t0=T0 - H(240)); w.visit(13, "Outpatient", T0 - H(243))
    P(13)
    # two qualifying EEGs: the earlier one is the index even though its SessionID sorts last
    w.eeg(14, t0=T0 - H(100), sid="zzz"); w.visit(14, "ICU", T0 - H(103)); w.score(14, GCS, T0 - H(101), 8)
    P(14, t0=T0, sid="aaa")
    P(15, dur=None, end_blank=True)                             # duration unknown (no metadata, no clock)
    P(16, dur=659.0)                                            # 1 s short of minutes 1-11
    P(17, dur=660.0)                                            # exactly long enough
    P(18, dur=None, clock_dur=1800.0)                           # no metadata duration; clock duration 30 min
    P(19, visit_start_h=-30.0, gcs_at=2.0)                      # 30 h after arrival, strict via a score after t0
    P(20, visit_start_h=-60.0, gcs_at=2.0)                      # 60 h: outside the widest window
    w.patient(21, visit_start_h=-60.0, gcs=10.0, gcs_at=-5.0); w.score(21, GCS, T0 - H(50), 15)   # onset = first abnormal score
    w.patient(22, visit_start_h=-10.0, gcs=8.0, gcs_at=2.0); w.score(22, GCS, T0 - H(100), 10)     # earlier encounter ignored
    P(23, gcs=None)                                             # no score, no phenotype
    P(24, gcs=12.0)                                             # GCS 12 > 11
    P(25, gcs=11.0)                                             # GCS 11 boundary
    P(26, gcs=None); w.score(26, FOUR, T0 - H(1), 12)           # FOUR 12 boundary
    P(27, gcs=None); w.score(27, FOUR, T0 - H(1), 13)           # FOUR 13
    P(28, gcs=8.0, gcs_at=7.0)                                  # strict score but outside +-6 h
    P(29, gcs=8.0, gcs_at=5.5)                                  # inside +-6 h, AFTER t0
    P(30, gcs=14.0); w.cond_row(30, "R40.2", T0 - H(1))         # phenotype only (broad, not strict)
    P(31, gcs=14.0); w.cond_row(31, "G93.41", T0 - H(1))        # encephalopathy code is not a phenotype code
    P(32, gcs=14.0); w.cond_row(32, "R40.2", T0 + H(10))        # phenotype code outside the window
    P(33, gcs=None)                                             # components only
    for nm, v in (("GCS eye opening", 1), ("GCS motor response", 2), ("GCS verbal response", 2)):
        w.score(33, nm, T0 - H(1), v)
    P(34, gcs=None); w.score(34, "GCS eye opening", T0 - H(1), 1); w.score(34, "GCS motor response", T0 - H(1), 2)
    P(35, gcs=99.0)                                             # implausible value dropped
    w.eeg(36); w.visit(36, "ED", T0 - H(25), end=T0 - H(23.5)); w.visit(36, "ICU", T0 - H(23)); w.score(36, GCS, T0 + H(2), 8)
    P(37, id_blank=True)                                        # BDSPPatientID blank: from BidsFolder
    P(38, site="I0008", age=50.0)                               # I-site layout: start in metadata, DOB, no findings
    P(39, site="I0008", age=15.0)
    P(40, site="I0008", start_blank=True)
    w.eeg(41, start_blank=True); P(41)                          # an unstamped sibling session
    P(42, id_blank=True, bids_blank=True)                       # no resolvable id at all
    return w, build(w)


def fate(res, pid):
    return res.fates.loc[pid]


# ----------------------------------------------------------------------------------------------- exclusions
@pytest.mark.parametrize("pid,prefix", [
    (2, "EEG start time missing"), (3, "Age missing"), (4, "Age <"), (6, "No visit"), (7, "No visit"),
    (8, "Non-acute"), (9, "Visit care setting"), (11, "OR or EMU"), (12, "OR or EMU"),
    (15, "Recording duration unknown"), (16, "Recording shorter"), (20, "EEG more than 48 h"),
    (23, "Neither"), (24, "Neither"), (27, "Neither"), (28, "Neither"), (31, "Neither"), (32, "Neither"),
    (34, "Neither"), (35, "Neither"), (39, "Age <"), (40, "EEG start time missing")])
def test_each_exclusion_reason(scenario, pid, prefix):
    assert fate(scenario[1], pid).startswith(prefix)


@pytest.mark.parametrize("pid", [1, 5, 10, 13, 14, 17, 18, 19, 21, 22, 25, 26, 29, 30, 33, 36, 37, 38, 41])
def test_included(scenario, pid):
    assert fate(scenario[1], pid) == INCLUDED
    assert pid in set(scenario[1].table["person_id"])


def test_exact_step_counts(scenario):
    r = scenario[1]
    assert removed_at(r, "Patient id") == 1                       # pid 42
    assert removed_at(r, "EEG start") == 3                        # 2, 40 and 41's unstamped sibling
    assert removed_at(r, "Age missing") == 1
    assert removed_at(r, "Age <") == 2                            # 4, 39
    assert removed_at(r, "No visit") == 2                         # 6, 7
    assert removed_at(r, "Visit care") == 1                       # 9
    assert removed_at(r, "Non-acute") == 2                        # 8 and 13's earlier outpatient EEG
    assert removed_at(r, "OR or EMU") == 2
    assert removed_at(r, "Not the patient's first") == 1          # 14's later EEG
    assert removed_at(r, "Recording duration unknown") == 1
    assert removed_at(r, "Recording shorter") == 1
    assert removed_at(r, "No ACI onset") == 0
    assert removed_at(r, "EEG more than 48 h") == 1
    assert removed_at(r, "Neither") == 8
    assert remaining_after(r, "Neither") == len(r.table) == 19


def test_flow_is_monotone_and_per_site_sums_match(scenario):
    steps = scenario[1].flow_raw["steps"]
    tot = [sum(s["remaining"].values()) for s in steps]
    assert tot == sorted(tot, reverse=True)
    assert steps[0]["start"] and steps[0]["remaining"]["I0008"] == 3 + 0   # 38, 39, 40
    assert steps[-1]["remaining"]["I0008"] == 1


# --------------------------------------------------------------------------------- first qualifying EEG
def test_first_qualifying_eeg_is_by_time_not_id(scenario):
    t = scenario[1].table.set_index("person_id")
    assert t.loc[14, "t0"] == T0 - H(100) and t.loc[14, "SessionID"] == "zzz"
    assert t.loc[13, "t0"] == T0                                  # the non-qualifying outpatient EEG is skipped
    assert t.index.is_unique


def test_missing_start_session_is_flagged_not_used(scenario):
    t = scenario[1].table.set_index("person_id")
    assert t.loc[41, "n_unstamped_sessions"] == 1 and t.loc[1, "n_unstamped_sessions"] == 0


# ------------------------------------------------------------------------------------ onset and windows
def test_onset_and_window_flags(scenario):
    t = scenario[1].table.set_index("person_id")
    assert t.loc[1, "onset_basis"] == "abnormal_score" and t.loc[1, "hours_since_onset"] == pytest.approx(1.0)
    assert t.loc[19, "onset_basis"] == "visit_start" and t.loc[19, "hours_since_onset"] == pytest.approx(30.0)
    assert not t.loc[19, "onset_le_24h"] and t.loc[19, "onset_le_48h"] and not t.loc[19, "in_strict"]
    assert t.loc[19, "severity_strict"]
    assert t.loc[21, "onset_basis"] == "abnormal_score" and t.loc[21, "hours_since_onset"] == pytest.approx(5.0)
    assert t.loc[22, "onset_basis"] == "visit_start" and t.loc[22, "hours_since_onset"] == pytest.approx(10.0)
    assert t.loc[22, "onset_le_12h"] and not t.loc[22, "onset_le_6h"]
    assert (t["hours_since_onset"] >= 0).all() and (t["hours_since_onset"] <= 48).all()


def test_ed_to_icu_encounter_counts_from_ed_arrival(scenario):
    t = scenario[1].table.set_index("person_id")
    assert t.loc[36, "hours_since_onset"] == pytest.approx(25.0)
    assert not t.loc[36, "onset_le_24h"] and not t.loc[36, "in_broad"] and t.loc[36, "onset_le_48h"]
    res = build(_ed_icu_world(), visit_chain_gap_h=0.5)           # a 1.5 h gap is NOT bridged
    assert res.table.set_index("person_id").loc[36, "hours_since_onset"] == pytest.approx(23.0)


def _ed_icu_world():
    w = World()
    w.eeg(36); w.visit(36, "ED", T0 - H(25), end=T0 - H(23.5)); w.visit(36, "ICU", T0 - H(23)); w.score(36, GCS, T0 + H(2), 8)
    return w


def test_onset_rules(scenario):
    w = scenario[0]
    r = build(w, onset_rule="visit_start")
    t = r.table.set_index("person_id")
    assert fate(r, 21).startswith("EEG more than 48 h") and t.loc[1, "hours_since_onset"] == pytest.approx(3.0)
    assert (t["onset_basis"] == "visit_start").all()
    r = build(w, onset_rule="score_only")
    assert fate(r, 19).startswith("No ACI onset proxy") and fate(r, 1) == INCLUDED
    assert removed_at(r, "No ACI onset") > 0


# ------------------------------------------------------------------------------------- strict / broad
def test_strict_and_broad_flags(scenario):
    t = scenario[1].table.set_index("person_id")
    assert t.loc[25, "in_strict"] and t.loc[26, "in_strict"] and t.loc[29, "in_strict"]
    assert t.loc[33, "in_strict"] and t.loc[33, "gcs_min_window"] == 5       # eye + motor + verbal
    assert t.loc[30, "in_broad"] and not t.loc[30, "in_strict"] and t.loc[30, "phenotype"]
    assert (t["in_broad"] | ~t["in_strict"]).all()                           # strict is a subset of broad
    assert t.loc[1, "in_strict"] and t.loc[1, "in_broad"]


def test_score_rule_nearest():
    w = World()
    w.patient(1, gcs=8.0, gcs_at=-5.0); w.score(1, GCS, T0 - H(1), 14)
    assert build(w).table["in_strict"].tolist() == [True]                    # any score <= 11 in the window
    r = build(w, score_rule="nearest")
    assert fate(r, 1).startswith("Neither")                                  # nearest is 14


def test_scores_are_range_checked_not_clipped():
    meas = pd.DataFrame({"person_id": [1, 1], "measurement_datetime": [T0, T0], "measurement_date": [T0, T0],
                         "measurement_time": ["12:00:00"] * 2, "measurement_source_value": [GCS, GCS],
                         "value_as_number": [99.0, 8.0]})
    s = rules.extract_scores(rules.filter_score_rows(meas))
    assert s["value"].tolist() == [8.0]


# ---------------------------------------------------------------------------------- service fallback etc.
def test_service_fallback_only_when_enabled():
    w = World()
    w.eeg(1, service="LTM"); w.visit(1, "ICU", T0 - H(3), concept=False); w.score(1, GCS, T0 - H(1), 8)
    w.eeg(2, service="Routine"); w.visit(2, "ICU", T0 - H(3), concept=False); w.score(2, GCS, T0 - H(1), 8)
    assert len(build(w).table) == 0
    r = build(w, use_service_fallback=True)
    assert r.table["person_id"].tolist() == [1] and r.table["acute_basis"].tolist() == ["service"]
    assert fate(r, 2).startswith("Visit care setting")


def test_visit_class_matches_the_audit_rule():
    w = World()
    w.eeg(1); w.visit(1, "ED", T0 - H(2), concept=False, text="Emergency Room")
    sess = FrameSources(w.tables()).sessions()
    m = rules.match_visits(sess, FrameSources(w.tables()).visits([1]), ("ICU", "Inpatient", "ED"), 6.0)
    assert m["visit_class"].tolist() == ["ED"] == [fa.visit_class(0, "Emergency Room")]


def test_duration_scale_and_unit_check():
    w = World()
    for i in range(12):                                          # RecordingDuration in MINUTES at I0008 (11 min)
        w.patient(100 + i, site="I0008", dur=11.0, clock_dur=660.0)
    assert len(build(w).table) == 0
    assert len(build(w, duration_scale_by_site=(("I0008", 60.0),)).table) == 12
    S = FrameSources(w.tables()).sessions()
    assert "UNIT WARNING" in duration_unit_check(S, ["I0008"])["duration_over_clock_I0008"]


def test_site_variants_are_read(scenario):
    t = scenario[1].table.set_index("person_id")
    assert t.loc[38, "SiteID"] == "I0008" and t.loc[38, "age_years"] == pytest.approx(50.0, abs=0.05)
    assert pd.isna(t.loc[38, "ServiceName"])                     # no ServiceName column at I0008
    assert t.loc[38, "t0"] == T0 and t.loc[37, "t0"] == T0       # I0008 start from metadata; S-site start from findings


# ---------------------------------------------------------------------------- agreement with the audit
def test_matches_audit_candidates_on_synthetic(synth):
    tables = synth[0]
    res = build_cohort(FrameSources(tables))
    cand = fa.build_candidates(fa.from_raw_tables(tables)["eeg_metadata"])
    late = ("Recording", "No ACI", "EEG more than", "Neither", INCLUDED)
    reached = res.fates[res.fates.str.startswith(late)]
    assert set(reached.index) == set(cand["person_id"])
    t = res.table.set_index("person_id")["t0"]
    c = cand.set_index("person_id")["t0"].reindex(t.index)
    assert (t == c).all()


def test_strict_flag_matches_an_independent_recomputation(synth):
    tables = synth[0]
    res = build_cohort(FrameSources(tables))
    m = tables["omop_measurement"]
    m = m[m["measurement_source_value"].isin([GCS, FOUR])]
    out = {}
    for pid, t0 in zip(res.table["person_id"], res.table["t0"]):
        g = m[(m["person_id"] == pid) & ((m["measurement_datetime"] - t0).abs() <= pd.Timedelta(hours=6))]
        v = g.set_index("measurement_source_value")["value_as_number"]
        low = [(g["measurement_source_value"] == GCS) & (g["value_as_number"].between(3, 11)),
               (g["measurement_source_value"] == FOUR) & (g["value_as_number"].between(0, 12))]
        out[pid] = bool(low[0].any() or low[1].any())
    exp = pd.Series(out)
    got = res.table.set_index("person_id")["severity_strict"]
    assert (got == exp.reindex(got.index)).all()


def test_synthetic_invariants(synth):
    res = build_cohort(FrameSources(synth[0]))
    t = res.table
    assert t["person_id"].is_unique and len(t) > 100
    assert (t["in_strict"] <= t["in_broad"]).all()
    assert (t["duration_s"] >= 660).all()
    assert (t["age_years"] >= 18).all() and t["visit_class"].isin(["ICU", "Inpatient", "ED"]).all()
    assert (t["hours_since_onset"].between(0, 48)).all()
    assert set(t["SiteID"]) == {"S0001", "S0002", "I0002", "I0003", "I0008", "I0009"}   # both layouts present
    assert list(res.keys["person_id"]) == list(t["person_id"])
    assert (res.keys["window_start_s"] == 60.0).all() and (res.keys["window_duration_s"] == 600.0).all()


def test_key_list_edf_keys(scenario):
    k = scenario[1].keys.set_index("person_id")
    assert k.loc[1, "edf_key"].startswith("EEG/bids/S0001/sub-S0001") and k.loc[1, "edf_key"].endswith("_eeg.edf")
    assert not k.loc[1, "task_token_assumed"] and k.loc[38, "task_token_assumed"]      # I-sites: no EEGFolder
    assert k.loc[37, "BidsFolder"] == "sub-S0001" + "37"
