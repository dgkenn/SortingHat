"""Study 1 cohort: every inclusion / exclusion step on hand-built patients, first-EEG-per-patient, onset proxy,
strict severity, broad phenotype, site variants, and agreement with the Phase 0a audit on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from sortinghat.audit import field_audit as fa
from sortinghat.cohort import CohortConfig, FrameSources, build_cohort, flow_markdown
from sortinghat.cohort import rules
from sortinghat.cohort.build import INCLUDED, duration_unit_check
from test_cohort_helpers import FOUR, GCS, H, T0, World, remaining_after, removed_at

PRE_FIRST = ("Patient id", "EEG start", "Age", "No visit", "Visit care", "Non-acute", "OR or EMU")


# The hand-built scenario patients carry timestamped visits with real concept ids, so they are matched on the exact
# intervals (the D-112 date-only +-24 h cover is tested separately in test_cohort_hepatients.py / test_cohort_visits.py).
LEGACY = dict(study_sites=None, visit_dates_only=False, visit_slack_h=0.0)


def build(w, **cfg):
    return build_cohort(FrameSources(w.tables()), CohortConfig(**{**LEGACY, **cfg}))


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
    P(19, visit_start_h=-30.0, gcs_at=0.5)                      # 30 h after arrival, strict via a score 30 min after t0
    P(20, visit_start_h=-60.0, gcs_at=0.5)                      # 60 h: outside the widest window
    w.patient(21, visit_start_h=-60.0, gcs=10.0, gcs_at=-5.0); w.score(21, GCS, T0 - H(50), 15)   # onset = first abnormal score
    w.patient(22, visit_start_h=-10.0, gcs=8.0, gcs_at=0.5); w.score(22, GCS, T0 - H(100), 10)     # earlier encounter ignored
    P(23, gcs=None)                                             # no score, no phenotype
    P(24, gcs=12.0)                                             # GCS 12 > 11
    P(25, gcs=11.0)                                             # GCS 11 boundary
    P(26, gcs=None); w.score(26, FOUR, T0 - H(1), 12)           # FOUR 12 boundary
    P(27, gcs=None); w.score(27, FOUR, T0 - H(1), 13)           # FOUR 13
    P(28, gcs=8.0, gcs_at=7.0)                                  # strict score but outside every window
    P(29, gcs=8.0, gcs_at=5.5)                                  # inside +-6 h but after +1 h: strict_pm6 only
    P(30, gcs=14.0); w.cond_row(30, "R40.2", T0 - H(1))         # phenotype only (broad, not strict)
    P(31, gcs=14.0); w.cond_row(31, "G93.41", T0 - H(1))        # encephalopathy code is not a phenotype code
    P(32, gcs=14.0); w.cond_row(32, "R40.2", T0 + H(10))        # phenotype code outside the window
    P(33, gcs=None)                                             # components only
    for nm, v in (("GCS eye opening", 1), ("GCS motor response", 2), ("GCS verbal response", 2)):
        w.score(33, nm, T0 - H(1), v)
    P(34, gcs=None); w.score(34, "GCS eye opening", T0 - H(1), 1); w.score(34, "GCS motor response", T0 - H(1), 2)
    P(35, gcs=99.0)                                             # implausible value dropped
    w.eeg(36); w.visit(36, "ED", T0 - H(25), end=T0 - H(23.5)); w.visit(36, "ICU", T0 - H(23)); w.score(36, GCS, T0 + H(0.5), 8)
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
    (8, "Not acute care"), (9, "Not acute care"), (11, "OR or EMU"), (12, "OR or EMU"),
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
    assert removed_at(r, "Not acute care") == 3                   # 8, 9 (class unknown, short visit) and 13's earlier outpatient EEG
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
    res = build(_ed_icu_world(), visit_chain_gap_h=0.25)          # the 0.5 h gap is NOT bridged
    assert res.table.set_index("person_id").loc[36, "hours_since_onset"] == pytest.approx(23.0)


def _ed_icu_world():
    w = World()
    w.eeg(36); w.visit(36, "ED", T0 - H(25), end=T0 - H(23.5)); w.visit(36, "ICU", T0 - H(23)); w.score(36, GCS, T0 + H(0.5), 8)
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
    assert t.loc[25, "in_strict"] and t.loc[26, "in_strict"]
    assert not t.loc[29, "in_strict"] and t.loc[29, "in_strict_pm6"] and not t.loc[29, "in_broad"]   # sensitivity only
    assert t.loc[33, "in_strict"] and t.loc[33, "gcs_min_window"] == 5       # eye + motor + verbal
    assert t.loc[30, "in_broad"] and not t.loc[30, "in_strict"] and t.loc[30, "phenotype"]
    assert (t["in_broad"] | ~t["in_strict"]).all()                           # strict is a subset of broad
    assert t.loc[1, "in_strict"] and t.loc[1, "in_broad"]


def _strict(*scores, **cfg):
    w = World()
    w.patient(1, gcs=None)
    for nm, h, v in scores:
        w.score(1, nm, T0 + H(h), v)
    r = build(w, **cfg)
    t = r.table
    return (bool(t["in_strict"].iloc[0]), bool(t["in_strict_pm6"].iloc[0])) if len(t) else (False, False)


def test_primary_window_is_minus6_to_plus1_and_pm6_is_a_flagged_sensitivity():
    assert _strict((GCS, -6.0, 8)) == (True, True)                # -6 h inclusive
    assert _strict((GCS, -6.1, 8)) == (False, False)
    assert _strict((GCS, 1.0, 8)) == (True, True)                 # +1 h inclusive
    assert _strict((GCS, 1.1, 8)) == (False, True)                # after +1 h: strict_pm6 only
    assert _strict((GCS, 6.0, 8)) == (False, True) and _strict((GCS, 6.1, 8)) == (False, False)


def test_primary_uses_the_nearest_score_pre_t0_on_ties():
    assert _strict((GCS, -5.0, 8), (GCS, -1.0, 14)) == (False, True)     # nearest is 14; pm6 'any' still sees the 8
    assert _strict((GCS, -1.0, 8), (GCS, 1.0, 14)) == (True, True)       # tie -> pre-t0 (8)
    assert _strict((GCS, -1.0, 14), (GCS, 1.0, 8)) == (False, True)      # tie -> pre-t0 (14)
    assert _strict((GCS, -3.0, 14), (GCS, 0.5, 8)) == (True, True)       # nearest is the post-t0 8
    assert _strict((GCS, -1.0, 14), (FOUR, -2.0, 12)) == (True, True)    # nearest is taken per instrument
    assert _strict((GCS, -5.0, 8), (GCS, -1.0, 14), score_rule="any") == (True, True)   # any: all in the window


def test_both_strict_definitions_are_reported_in_the_flow(scenario):
    parts = scenario[1].flow_raw["partitions"]
    titles = [p["title"] for p in parts]
    assert "Strict definitions, primary onset window" in titles
    sd = next(p for p in parts if p["title"].startswith("Strict definitions"))["parts"]
    assert {k: sum(v.values()) for k, v in sd.items()} == {
        "both definitions": sum(scenario[1].table.eval("in_strict & in_strict_pm6")),
        "primary only (-6 h to +1 h)": sum(scenario[1].table.eval("in_strict & ~in_strict_pm6")),
        "strict_pm6 only (+-6 h)": sum(scenario[1].table.eval("~in_strict & in_strict_pm6"))}


def test_scores_are_range_checked_not_clipped():
    meas = pd.DataFrame({"person_id": [1, 1], "measurement_datetime": [T0, T0], "measurement_date": [T0, T0],
                         "measurement_time": ["12:00:00"] * 2, "measurement_source_value": [GCS, GCS],
                         "value_as_number": [99.0, 8.0]})
    s = rules.extract_scores(rules.filter_score_rows(meas))
    assert s["value"].tolist() == [8.0]


# ---------------------------------------------------------------------------------- service fallback etc.
def test_acute_care_proxy_is_visit_length_or_service_when_the_class_is_unknown():
    """D-111: all visit_concept_id are 0 in HEEDB, so the class is unknown and the proxy decides: the covering visit is
    longer than a day (end date after start date), or ServiceName names an acute setting (LTM, ICU, ED, inpatient)."""
    w = World()
    day = T0.normalize()

    def pt(pid, service, start, end):
        w.eeg(pid, service=service); w.visit(pid, "ICU", start, end=end, concept=False); w.score(pid, GCS, T0 - H(1), 8)
    pt(1, "Routine", day - pd.Timedelta(days=2), day + pd.Timedelta(days=3))     # multi-day visit
    pt(2, "Routine", day, day)                                                   # same-day visit, routine service
    pt(3, "LTM", day, day)                                                       # same-day visit, LTM service
    pt(4, "EMU", day - pd.Timedelta(days=2), day + pd.Timedelta(days=3))         # multi-day but EMU: excluded
    pt(5, "OR", day - pd.Timedelta(days=2), day + pd.Timedelta(days=3))
    pt(6, "Routine", day - pd.Timedelta(days=1), None)                           # open visit: length unknown
    r = build(w)
    t = r.table.set_index("person_id")
    assert sorted(t.index) == [1, 3]
    assert t.loc[1, "acute_basis"] == "visit_length" and t.loc[3, "acute_basis"] == "service"
    assert all(r.fates.loc[p].startswith(x) for p, x in ((2, "Not acute care"), (4, "OR or EMU"), (5, "OR or EMU"),
                                                          (6, "Not acute care")))
    assert build(w, use_service_proxy=False).table["person_id"].tolist() == [1]


def test_a_known_visit_class_still_decides_when_concept_ids_are_ever_non_zero():
    w = World()
    w.eeg(1); w.visit(1, "Outpatient", T0 - H(30), end=T0 + H(30)); w.score(1, GCS, T0 - H(1), 8)    # multi-day but Outpatient
    w.eeg(2); w.visit(2, "ED", T0 - H(2), end=T0 + H(1)); w.score(2, GCS, T0 - H(1), 8)               # same-day ED
    assert build(w).table["person_id"].tolist() == [2]


def test_visit_class_matches_the_audit_rule():
    w = World()
    w.eeg(1); w.visit(1, "ED", T0 - H(2), concept=False, text="Emergency Room")
    sess = FrameSources(w.tables()).sessions()
    m = rules.match_visits(sess, FrameSources(w.tables()).visits([1]), ("ICU", "Inpatient", "ED"), 6.0)
    assert m["visit_class"].tolist() == ["ED"] == [fa.visit_class(0, "Emergency Room")]


def test_duration_is_the_clock_not_the_metadata_value():
    """D-115: recording duration = EndTime - StartTime; the metadata duration (wrong unit or wrong value) is ignored."""
    w = World()
    for i in range(12):                                          # metadata says 11 (minutes?), the clock says 30 min
        w.patient(100 + i, site="I0008", dur=11.0, clock_dur=1800.0)
    for i in range(12):                                          # metadata says 2 h, the clock says 5 min
        w.patient(200 + i, site="S0001", dur=7200.0, clock_dur=300.0)
    r = build(w)
    assert sorted(r.table["person_id"]) == [100 + i for i in range(12)]
    assert (r.table["duration_s"] == 1800.0).all() and (r.table["duration_basis"] == "clock").all()
    S = FrameSources(w.tables()).sessions()
    assert "UNIT WARNING" in duration_unit_check(S, ["I0008"])["duration_over_clock_I0008"]   # informational only


def test_site_variants_are_read(scenario):
    t = scenario[1].table.set_index("person_id")
    assert t.loc[38, "SiteID"] == "I0008" and t.loc[38, "age_years"] == pytest.approx(50.0, abs=0.05)
    assert pd.isna(t.loc[38, "ServiceName"])                     # no ServiceName column at I0008
    assert t.loc[38, "t0"] == T0 and t.loc[37, "t0"] == T0       # I0008 start from metadata; S-site start from findings


# ---------------------------------------------------------------------------- agreement with the audit
def test_agrees_with_the_audit_candidates_on_synthetic(synth):
    """The audit matches the visit with the latest start; the cohort matches any covering visit (acuity first), so it
    can only ADD patients (an inpatient stay covering a later outpatient-labelled EEG) or move t0 earlier."""
    tables = synth[0]
    res = build_cohort(FrameSources(tables), CohortConfig(**LEGACY))
    cand = fa.build_candidates(fa.from_raw_tables(tables)["eeg_metadata"])
    late = ("Recording", "No ACI", "EEG more than", "Neither", INCLUDED)
    reached = res.fates[res.fates.str.startswith(late)]
    audit = set(cand["person_id"])
    assert audit <= set(reached.index) and len(set(reached.index) - audit) < 0.03 * len(audit)
    t = res.table.set_index("person_id")["t0"]
    c = cand.set_index("person_id")["t0"].reindex(t.index)
    assert (t[c.notna()] <= c[c.notna()]).all() and (t[c.notna()] == c[c.notna()]).mean() > 0.97


def test_strict_flags_match_an_independent_recomputation(synth):
    tables = synth[0]
    res = build_cohort(FrameSources(tables))
    m = tables["omop_measurement"]
    m = m[m["measurement_source_value"].isin([GCS, FOUR])]
    lim = {GCS: 11, FOUR: 12}
    prim, pm6 = {}, {}
    for pid, t0 in zip(res.table["person_id"], res.table["t0"]):
        g = m[m["person_id"] == pid]
        dt = (g["measurement_datetime"] - t0).dt.total_seconds() / 3600
        ok = False
        for name, cap in lim.items():
            h = g[(g["measurement_source_value"] == name) & dt.between(-6, 1)]
            if len(h):
                d = (dt[h.index]).abs()
                best = h.assign(_a=d, _p=(dt[h.index] > 0)).sort_values(["_a", "_p"]).iloc[0]
                ok |= best["value_as_number"] <= cap
        prim[pid] = bool(ok)
        w6 = g[dt.abs() <= 6]
        pm6[pid] = bool(((w6["measurement_source_value"] == GCS) & (w6["value_as_number"].between(3, 11))).any()
                        or ((w6["measurement_source_value"] == FOUR) & (w6["value_as_number"].between(0, 12))).any())
    t = res.table.set_index("person_id")
    assert (t["severity_strict"] == pd.Series(prim).reindex(t.index)).all()
    assert (t["severity_strict_pm6"] == pd.Series(pm6).reindex(t.index)).all()
    assert t["severity_strict"].sum() > 100 and (t["severity_strict_pm6"] & ~t["severity_strict"]).sum() > 0


def test_study_sites_default_excludes_i0008_i0009_as_a_separate_block(synth):
    res = build_cohort(FrameSources(synth[0]))                    # defaults: D-111 to D-115
    assert set(res.table["SiteID"]) <= {"I0002", "I0003", "S0001", "S0002"} and len(res.table) > 100
    rep = res.report
    assert rep["excluded_site_blocks"] == ["I0008", "I0009"] or rep["excluded_site_blocks"] == ["I0008+I0009"]
    blk = rep["sites"][rep["excluded_site_blocks"][0]]["rows"]
    assert blk[-1]["step"].startswith("Site is not a Study 1 site") and blk[-1]["n_remaining"] == "<11"
    assert "excluded block" in flow_markdown(rep)
    assert len(build_cohort(FrameSources(synth[0]), CohortConfig(study_sites=None)).table) > len(res.table)


def test_synthetic_invariants(synth):
    res = build_cohort(FrameSources(synth[0]), CohortConfig(study_sites=None))
    t = res.table
    assert t["person_id"].is_unique and len(t) > 100
    assert (t["in_strict"] <= t["in_broad"]).all() and (t["in_strict"] <= t["in_strict_pm6"]).all()
    assert (t["duration_s"] >= 660).all()
    assert (t["age_years"] >= 18).all() and (t["visit_inpatient_length"] | t["visit_class"].isin(["ICU", "Inpatient", "ED"])
                                           | t["acute_basis"].eq("service")).all()
    assert (t["hours_since_onset"].between(0, 48)).all()
    assert set(t["SiteID"]) == {"S0001", "S0002", "I0002", "I0003", "I0008", "I0009"}   # both layouts present
    assert list(res.keys["person_id"]) == list(t["person_id"])
    assert (res.keys["window_start_s"] == 60.0).all() and (res.keys["window_duration_s"] == 600.0).all()


def test_key_list_carries_the_clock_duration(scenario):
    k = scenario[1].keys.set_index("person_id")
    assert k.loc[17, "clock_duration_s"] == 660.0 and k.loc[1, "clock_duration_s"] == 1800.0


def test_key_list_edf_keys(scenario):
    k = scenario[1].keys.set_index("person_id")
    assert k.loc[1, "edf_key"].startswith("EEG/bids/S0001/sub-S0001") and k.loc[1, "edf_key"].endswith("_eeg.edf")
    assert not k.loc[1, "task_token_assumed"] and k.loc[38, "task_token_assumed"]      # I-sites: no EEGFolder
    assert k.loc[37, "BidsFolder"] == "sub-S0001" + "37"


def test_empty_and_tiny_inputs_do_not_break_the_build():
    r = build(World())
    assert len(r.table) == 0 and len(r.keys) == 0 and r.report["sites"]["ALL"] == {"withheld": "starting count < 11"}
    w = World()
    w.patient(1)
    r = build(w)                                                 # one patient: built, but every count is withheld
    assert len(r.table) == 1 and r.report["sites"]["ALL"] == {"withheld": "starting count < 11"}
