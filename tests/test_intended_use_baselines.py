"""D-145 on SYNTHETIC data: the current-encounter restriction, Baseline P (Presentation), the subgroup-definition columns, the
bounded presentation gate, and the streamed build (scripts/build_baselines.py) agreeing with the in-memory build."""
import contextlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from sortinghat import data_io
from sortinghat.baselines import (BaselineConfig, as_of, as_of_presentation, assert_presentation_masked, build_feature_set,
                                  label_dx_events)
from sortinghat.baselines.asof import EVENT_COLUMNS, empty_events
from sortinghat.baselines.encounter import resolve_encounter_start, restrict_to_current_encounter
from sortinghat.baselines.events import label_dx_items
from sortinghat.cohort import CohortConfig, StoreSources, build_cohort
from sortinghat.tableio import load_tables
from test_baselines_helpers import H, T0, add_rows, drug, meas, mini_tables
from test_silver_feasibility_inputs import rsf

bb = rsf.bb
CFG = BaselineConfig()
HIST = BaselineConfig(encounter_scope="with_history")
ENC = pd.Timestamp("2020-01-01 00:00:00")                 # the cohort's date-only encounter start for the current visit below
T4 = pd.Timestamp("2020-01-01 04:00:00")                  # an EEG 4 h into the encounter: the prior encounter's charts fall in every window
P_GROUPS = {"demographics", "pres_score", "pres_first_vital", "pres_first_glucose"}


def visit(vid, start, end, src="Inpatient"):
    return {"person_id": 1, "visit_occurrence_id": vid, "visit_start_datetime": pd.Timestamp(start),
            "visit_end_datetime": pd.Timestamp(end), "visit_concept_id": 0, "visit_source_value": src}


def two_visits(t0=T0):
    """One patient, EEG at ``t0`` (2020-01-01 12:00), in the second of two visits; the first ended the day before (a PRIOR encounter,
    not chained: the gap is longer than the cohort's chain gap)."""
    return add_rows(mini_tables(t0=t0), "omop_visit_occurrence", [
        visit(1, "2019-12-30 08:00", "2019-12-31 10:00"), visit(2, "2020-01-01 00:30", "2020-01-04 10:00")])


def cond(t, code):
    return {"person_id": 1, "condition_start_datetime": t, "condition_source_value": code, "condition_concept_id": 0}


def prior_events(t, t0=T4):
    """Events of the PRIOR encounter (before 2020-01-01 00:00), placed inside every feature window of an EEG at 04:00."""
    p = t0 - H(5)                                                        # 2019-12-31 23:00
    t = add_rows(t, "omop_measurement", [
        meas(1, "Glasgow Coma Scale Score", p, 3), meas(1, "FOUR Score", p, 2), meas(1, "RASS (Richmond Agitation Sedation Scale)", p, -4),
        meas(1, "Heart rate", p, 150), meas(1, "Systolic blood pressure", p, 60),
        meas(1, "POC glucose (fingerstick)", p - H(1), 40, "mg/dL"), meas(1, "LACTATE, WHOLE BLOOD", p - H(2), 9.0, "mmol/L")])
    t = add_rows(t, "omop_drug_exposure", [drug(1, "MIDAZOLAM 1 MG/ML INJ", p, None, 5.0)])
    return add_rows(t, "omop_condition_occurrence", [cond(p - H(5), "S06.5X0A"), cond(p - H(5), "I46.9")])


def same(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# ------------------------------------------------------------------------------------ current-encounter restriction
def test_the_encounter_start_follows_the_cohort_visit_rule():
    fs = build_feature_set(two_visits())
    assert fs.X_extra.loc[1, "meta__hours_since_encounter_start"] == pytest.approx(12.0)     # from 2020-01-01 00:00, not 2019-12-30
    from sortinghat.baselines import build_index
    idx = build_index(two_visits())
    assert idx.loc[0, "encounter_start"] == ENC and idx.loc[0, "encounter_basis"] == "visits"


def test_prior_encounter_events_never_enter_by_default():
    """The D-145 requirement: A, B, C and P are identical with and without any prior-encounter event."""
    base, withprior = two_visits(T4), prior_events(two_visits(T4))
    fs0, fs1 = build_feature_set(base), build_feature_set(withprior)
    same(fs1.X, fs0.X)                                                                      # Baselines A-D untouched
    keep = [c for c in fs0.X_extra.columns if c != "meta__label_dx_before_t0"]
    same(fs1.X_extra[keep], fs0.X_extra[keep])                                              # Baseline P untouched
    assert fs0.diagnostics["n_events_prior_encounter_dropped"] == 0 and fs1.diagnostics["n_events_prior_encounter_dropped"] >= 9
    # the subgroup definition DOES see a prior-encounter arrest code ("recorded before t0", any encounter)
    assert fs0.X_extra.loc[1, "meta__label_dx_before_t0"] == 0 and fs1.X_extra.loc[1, "meta__label_dx_before_t0"] == 1


def test_with_history_variant_admits_prior_encounter_events():
    fs = build_feature_set(prior_events(two_visits(T4)), HIST)
    r, e = fs.X.loc[1], fs.X_extra.loc[1]
    assert r["score__gcs__value"] == 3 and r["vital__hr__value"] == 150 and r["poc_glucose__poc_glucose__value"] == 40
    assert r["lab__lactate__value"] == 9 and r["sed__midazolam__qty_24h"] > 0 and r["hx__arrest_recent"] == 1
    assert e["pres_score__gcs__value"] == 3 and e["pres_score__rass__value"] == -4          # windowed P scores follow the scope
    # ... but P's FIRST vitals / glucose are "of the current encounter" by definition, also in the history variant
    assert e["pres_first__hr__miss"] == 1 and e["pres_first__poc_glucose__miss"] == 1
    assert fs.config.encounter_scope == "with_history" and fs.diagnostics["n_events_prior_encounter_dropped"] == 0


def test_current_encounter_events_enter_every_baseline():
    t = add_rows(two_visits(T4), "omop_measurement", [
        meas(1, "Glasgow Coma Scale Score", T4 - H(3.5), 9), meas(1, "Glasgow Coma Scale Score", T4 - H(0.5), 8),
        meas(1, "Heart rate", T4 - H(3), 100), meas(1, "Heart rate", T4 - H(2), 120),
        meas(1, "POC glucose (fingerstick)", T4 - H(3.25), 130, "mg/dL"), meas(1, "POC glucose (fingerstick)", T4 - H(1), 90, "mg/dL"),
        meas(1, "LACTATE, WHOLE BLOOD", T4 - H(2.5), 4.0, "mmol/L")])
    fs = build_feature_set(prior_events(t))
    r, e = fs.X.loc[1], fs.X_extra.loc[1]
    assert r["score__gcs__value"] == 8 and r["vital__hr__value"] == 120 and r["poc_glucose__poc_glucose__value"] == 90
    assert e["pres_first__hr__value"] == 100 and e["pres_first__poc_glucose__value"] == 130      # FIRST of the encounter, not the latest
    assert e["pres_score__gcs__value"] == 8
    assert r["lab__lactate__value"] == 4.0                                                   # the prior encounter's 9.0 is not seen


def test_a_first_value_before_the_encounter_start_or_after_t0_is_never_the_first():
    t = add_rows(two_visits(), "omop_measurement", [
        meas(1, "Heart rate", ENC - pd.Timedelta(seconds=1), 1), meas(1, "Heart rate", ENC, 55), meas(1, "Heart rate", T0 - H(1), 70),
        meas(1, "Heart rate", T0 + pd.Timedelta(microseconds=1), 2)])
    e = build_feature_set(t).X_extra.loc[1]
    assert e["pres_first__hr__value"] == 55                                                  # exactly at the start counts


def test_unknown_encounter_start_falls_back_and_is_labelled():
    from sortinghat.baselines import build_index
    idx = build_index(mini_tables())                                                          # no visit table rows at all
    assert idx.loc[0, "encounter_basis"] == "fallback" and idx.loc[0, "encounter_start"] == T0 - pd.Timedelta(days=3)
    start, basis = resolve_encounter_start(idx, pd.Series([T0 - H(5)]), None, 3.0)
    assert start.iloc[0] == T0 - H(5) and basis.iloc[0] == "cohort"
    start, _ = resolve_encounter_start(idx, pd.Series([T0 + H(5)]), None, 3.0)
    assert start.iloc[0] == T0                                                                # never after the EEG


def test_restriction_primitive():
    rows = [dict(person_id=1, domain="lab", key="a", t_event=ENC - H(1)), dict(person_id=1, domain="lab", key="b", t_event=ENC),
            dict(person_id=1, domain="lab", key="c", t_event=pd.NaT), dict(person_id=1, domain="dxlab", key="d", t_event=ENC - H(900)),
            dict(person_id=2, domain="lab", key="e", t_event=ENC)]
    ev = pd.DataFrame(rows).reindex(columns=EVENT_COLUMNS)
    ev["t_event"] = pd.to_datetime(ev["t_event"]).astype("datetime64[us]")
    out, dropped = restrict_to_current_encounter(ev, pd.Series({1: ENC}))
    assert out["key"].tolist() == ["b", "d"] and dropped == 3                                 # person 2 has no start: dropped
    assert restrict_to_current_encounter(empty_events(), pd.Series({1: ENC}))[1] == 0


def test_config_validation():
    with pytest.raises(ValueError):
        BaselineConfig(encounter_scope="everything")
    with pytest.raises(ValueError):
        BaselineConfig(presentation_score_after_h=-1.0)
    assert BaselineConfig().encounter_scope == "current" and BaselineConfig().presentation_score_after_h == 1.0


# ---------------------------------------------------------------------------------------------------- Baseline P
def test_p_score_window_is_nearest_in_minus6h_to_plus1h_and_nothing_else_sees_the_future():
    scores = [("Glasgow Coma Scale Score", -7, 15), ("Glasgow Coma Scale Score", -5, 10), ("Glasgow Coma Scale Score", -1, 7),
              ("Glasgow Coma Scale Score", 0.5, 5), ("Glasgow Coma Scale Score", 1.5, 3),
              ("FOUR Score", -0.5, 8), ("FOUR Score", 0.5, 12), ("RASS (Richmond Agitation Sedation Scale)", 2, -5)]
    t = two_visits()
    with_post = add_rows(t, "omop_measurement", [meas(1, n, T0 + H(h), v) for n, h, v in scores]
                         + [meas(1, "Heart rate", T0 + H(0.5), 777), meas(1, "POC glucose (fingerstick)", T0 + H(0.5), 9999),
                            meas(1, "SODIUM", T0 + H(0.5), 9999)])
    fs = build_feature_set(with_post)
    e, r = fs.X_extra.loc[1], fs.X.loc[1]
    assert e["pres_score__gcs__value"] == 5                      # +0.5 h is nearer than -1 h; +1.5 h and -7 h are outside the window
    assert e["pres_score__four__value"] == 8                     # a tie goes to the earlier chart (the cohort's rule)
    assert e["pres_score__rass__miss"] == 1                      # +2 h is outside
    # no other column sees any post-t0 event: Baselines A-D equal the build WITHOUT the post-t0 rows except for pre-t0 scores
    pre_only = add_rows(t, "omop_measurement", [meas(1, n, T0 + H(h), v) for n, h, v in scores if h < 0])
    ref = build_feature_set(pre_only)
    same(fs.X, ref.X)
    assert r["score__gcs__value"] == 7 and r["vital__hr__miss"] == 1 and r["poc_glucose__poc_glucose__miss"] == 1
    other = [c for c in fs.X_extra.columns if not c.startswith("pres_score__")]
    same(fs.X_extra[other], ref.X_extra[other])                   # only the three score blocks of P may differ


def test_p_strict_mode_is_t0_masked():
    t = add_rows(two_visits(), "omop_measurement", [meas(1, "Glasgow Coma Scale Score", T0 - H(1), 7),
                                                    meas(1, "Glasgow Coma Scale Score", T0 + H(0.1), 5)])
    assert build_feature_set(t, BaselineConfig(presentation_score_after_h=0.0)).X_extra.loc[1, "pres_score__gcs__value"] == 7
    assert build_feature_set(t).X_extra.loc[1, "pres_score__gcs__value"] == 5


def test_p_has_no_history_or_diagnosis_columns():
    fs = build_feature_set(prior_events(two_visits(T4)))
    prov = fs.all_provenance().set_index("feature")
    p = fs.columns("P")
    assert set(prov.loc[p, "group"]) <= P_GROUPS and len(p) == len(set(p))
    assert {"demo__age_years", "demo__sex_male"} <= set(p)
    assert not [c for c in p if c.startswith(("hx__", "lab__", "tox__", "sed__", "culture__", "ind__", "img__", "meta__", "pupil__"))]
    assert set(fs.columns("meta")) == {"meta__hours_since_encounter_start", "meta__label_dx_before_t0"}
    assert not set(fs.columns("meta")) & set(fs.columns("D")) and not set(fs.columns("meta")) & set(p)
    assert fs.X.shape[1] == len(fs.provenance) and list(fs.X_extra.columns) == fs.provenance_extra["feature"].tolist()
    with pytest.raises(ValueError):
        fs.columns("Q")


def test_presentation_gate_unit():
    def ev(rows):
        d = pd.DataFrame(rows).reindex(columns=EVENT_COLUMNS)
        for c in ("t_event", "t_avail", "t_end_raw"):
            d[c] = pd.to_datetime(d[c]).astype("datetime64[us]")
        d["approx"] = False
        return d
    e = ev([dict(person_id=1, domain="score", key="gcs", value=5.0, t_event=T0 + H(1), t_avail=T0 + H(1)),
            dict(person_id=1, domain="score", key="gcs", value=4.0, t_event=T0 + H(1.01), t_avail=T0 + H(1.01)),
            dict(person_id=1, domain="vital", key="hr", value=9.0, t_event=T0 + H(0.1), t_avail=T0 + H(0.1)),
            dict(person_id=1, domain="score", key="gcs", value=6.0, t_event=T0 - H(1), t_avail=T0 - H(1))])
    out = as_of_presentation(e, {1: T0}, 1.0)
    assert sorted(out["value"]) == [5.0, 6.0] and (out["domain"] == "score").all()           # the boundary is inclusive; other domains never pass
    assert out["t0"].eq(T0).all() and sorted(out["hours_since_event"]) == [-1.0, 1.0]
    assert_presentation_masked(out, 1.0)
    assert len(as_of(e, {1: T0})) == 1                                                        # the ordinary gate is untouched (only the -1 h score)
    with pytest.raises(ValueError):
        assert_presentation_masked(as_of(e, {1: T0}), 1.0)                                    # an ordinary masked frame is not accepted
    assert as_of_presentation(empty_events(), T0, 1.0).empty


# --------------------------------------------------------------------------------------- subgroup-definition columns
def test_label_family_dx_flag_and_isolation():
    assert {"dx_ich", "dx_sah", "dx_sdh_edh", "dx_infarct", "dx_tbi", "arrest_event", "asphyxia_event"} <= set(label_dx_items())
    assert not {"esrd", "proc_neurosurgical"} & set(label_dx_items())                         # helper / procedure items are not dx families
    base = two_visits()
    f0 = build_feature_set(base)
    assert f0.X_extra.loc[1, "meta__label_dx_before_t0"] == 0
    for code, when, want in (("I63.9", T0 - H(24 * 400), 1),            # old infarct, any encounter
                             ("I63.9", T0, 1),                          # at t0
                             ("I63.9", T0 + H(5), 0),                   # after t0
                             ("E11.9", T0 - H(5), 0),                   # not a label-family code
                             ("T58.01XA", T0 - H(5), 1),                # CO poisoning = asphyxia family
                             ("I63.9", None, 1)):                       # no time: counts (conservative for the subgroup)
        fs = build_feature_set(add_rows(base, "omop_condition_occurrence", [cond(when, code)]))
        assert fs.X_extra.loc[1, "meta__label_dx_before_t0"] == want, (code, when)
        same(fs.X, f0.X)                                               # the dx code is never a model input (A-D unchanged) ...
        keep = [c for c in f0.X_extra.columns if c != "meta__label_dx_before_t0"]
        same(fs.X_extra[keep], f0.X_extra[keep])                       # ... nor a P input
    ev = label_dx_events({"omop_condition_occurrence": add_rows(base, "omop_condition_occurrence", [cond(None, "I63.9")])
                          ["omop_condition_occurrence"]}, {1})
    assert ev["time_basis"].tolist() == ["unknown_time"] and ev["domain"].tolist() == ["dxlab"]


# ------------------------------------------------------------------------------------------------ cohort + streamed
@pytest.fixture(scope="module")
def tables(synth_dir):
    t = load_tables(synth_dir)
    t.pop("imaging")
    return t


@pytest.fixture(scope="module")
def cohort_res(synth_dir):
    return build_cohort(StoreSources(data_io.open_store(synth_dir)), CohortConfig(study_sites=None))


@pytest.fixture(scope="module")
def cohort_csv(tmp_path_factory, cohort_res):
    lo = tmp_path_factory.mktemp("iu") / "local_only"
    lo.mkdir()
    cohort_res.table.to_csv(lo / "cohort_study1.csv", index=False)
    return lo / "cohort_study1.csv"


def test_cohort_table_carries_the_encounter_start(cohort_res):
    t = cohort_res.table
    assert t["encounter_start"].notna().all() and (t["encounter_start"] <= t["t0"]).all()
    vs = t["onset_basis"] == "visit_start"
    assert vs.any() and (t.loc[vs, "onset"] == t.loc[vs, "encounter_start"]).all()          # the onset basis IS the encounter start


class FrameStream:
    def __init__(self, tables, store, rows=500):
        self.t, self.s3, self.rows = tables, store, rows

    def iter_rows(self, table, columns, person_ids, min_rows=0):
        d = self.t[table]
        d = d[d["person_id"].isin({int(p) for p in person_ids})]
        keep = [c for c in columns if c in d.columns]
        for i in range(0, len(d), self.rows):
            yield d.iloc[i:i + self.rows][keep].copy()


def test_streamed_build_equals_in_memory_build_with_first_vitals_and_the_presentation_window(synth_dir, cohort_csv, tables):
    store = data_io.open_store(synth_dir)
    coh = bb.load_cohort(cohort_csv, None, "table")
    index = bb.make_index(coh, bb.sex_lookup(store, coh))
    rows = []
    for pid, t0, enc in zip(index["person_id"], index["t0"], index["encounter_start"]):
        k = 0
        while enc + H(2 * k) <= t0 + H(3):                       # a vital every 2 h from the encounter start, past t0
            rows.append(meas(int(pid), "Heart rate", enc + H(2 * k), 100.0 + k))
            k += 1
        rows += [meas(int(pid), "Heart rate", enc - H(3), 1.0),                           # PRIOR encounter: never the first
                 meas(int(pid), "POC glucose (fingerstick)", enc + H(1), 111.0, "mg/dL"),
                 meas(int(pid), "POC glucose (fingerstick)", t0 - H(1), 99.0, "mg/dL"),
                 meas(int(pid), "Glasgow Coma Scale Score", t0 + H(0.5), 5), meas(int(pid), "Glasgow Coma Scale Score", t0 + H(2), 3)]
    aug = {**tables, "omop_measurement": add_rows(tables, "omop_measurement", rows)["omop_measurement"]}
    ref = build_feature_set(aug, CFG, index)
    fs, n_pref = bb.build_matrices(FrameStream(aug, store), index, CFG, None, True, chunk_rows=700)
    fs0, n_all = bb.build_matrices(FrameStream(aug, store), index, CFG, None, False, chunk_rows=700)
    same(fs.all_x(), ref.all_x())
    same(fs0.all_x(), ref.all_x())                                # with and without the memory prefilter
    assert n_pref < n_all
    e = ref.X_extra
    hrs = ((index["t0"] - index["encounter_start"]).dt.total_seconds() / 3600.0).to_numpy()
    assert (e["pres_first__hr__value"] == 100.0).all()
    g = e["pres_first__poc_glucose__value"].to_numpy()
    assert (g[hrs >= 2.0] == 111.0).all() and (g[(hrs >= 1.0) & (hrs < 2.0)] == 99.0).all()    # the earlier of the two charts
    assert np.isnan(g[hrs < 1.0]).all()                                                          # nothing charted yet by t0
    # the +0.5 h chart wins unless the synthetic cohort already has a chart nearer to t0 (then it is the cohort's own nearest GCS:
    # the P rule IS the cohort's severity rule); never the +2 h chart (3), also through the prefilter
    near = coh.set_index("person_id")["gcs_nearest_window"].reindex(e.index)
    got = e["pres_score__gcs__value"]
    assert ((got == 5.0) | (got == near)).all() and (got == 5.0).mean() > 0.5
    # the same streamed build as the history variant: the prior-encounter chart (value 1) is still never a FIRST vital
    fh, _ = bb.build_matrices(FrameStream(aug, store), index, HIST, None, True, chunk_rows=700)
    assert (fh.X_extra["pres_first__hr__value"] == 100.0).all()
    assert fh.diagnostics["n_events_prior_encounter_dropped"] == 0 and fs.diagnostics["n_events_prior_encounter_dropped"] > 0


def test_old_cohort_tables_get_their_encounter_start_from_the_visit_table(synth_dir, cohort_csv):
    store = data_io.open_store(synth_dir)
    coh = bb.load_cohort(cohort_csv, None, "table")
    index = bb.make_index(coh, bb.sex_lookup(store, coh))
    old = index.assign(encounter_start=pd.NaT)
    got = bb.ensure_encounters(StoreSources(store), old, CFG)
    assert (got["encounter_basis"] == "visits").all()
    pd.testing.assert_series_equal(got["encounter_start"], index["encounter_start"], check_names=False)
    # a start that cannot be resolved falls back to t0 minus 3 days and says so
    nov = bb.ensure_encounters(FrameStream({}, store), old, CFG)
    assert (nov["encounter_basis"] == "fallback").all()
    assert (nov["t0"] - nov["encounter_start"] == pd.Timedelta(days=3)).all()


def test_prune_events_keeps_the_first_of_the_encounter_and_the_presentation_grace():
    t0 = pd.Series({1: pd.Timestamp("2024-05-01 12:00")})
    enc = pd.Series({1: pd.Timestamp("2024-04-29 00:00")})
    ts = pd.to_datetime(["2024-04-28 10:00", "2024-04-29 02:00", "2024-04-29 03:00", "2024-04-30 12:00", "2024-05-01 11:00",
                         "2024-05-01 12:30", "2024-05-01 13:30", "2024-05-01 12:30"], format="%Y-%m-%d %H:%M")
    dom = ["vital", "vital", "vital", "vital", "vital", "score", "score", "vital"]
    ev = pd.DataFrame({"person_id": 1, "domain": dom, "key": ["hr", "hr", "hr", "hr", "hr", "gcs", "gcs", "hr"], "value": 1.0,
                       "quantity": np.nan, "t_event": ts, "t_avail": ts, "t_end_raw": pd.NaT, "time_basis": "x", "approx": False,
                       "unit": "u", "raw_name": "n"})
    out = bb.prune_events(ev, t0, CFG, True, enc)
    kept = sorted(out["t_event"].tolist())
    # prior-encounter chart (04-28) dropped; of the old charts only the EARLIEST (04-29 02:00) survives; 11:00 is in the window;
    # the score at +0.5 h passes (grace), at +1.5 h and the vital at +0.5 h do not
    assert kept == sorted(pd.to_datetime(["2024-04-29 02:00", "2024-05-01 11:00", "2024-05-01 12:30"], format="%Y-%m-%d %H:%M").tolist())
    assert out[out["t_event"] == pd.Timestamp("2024-05-01 12:30")]["domain"].tolist() == ["score"]
    plain = bb.prune_events(ev, t0, CFG, True)                          # without an encounter start: the old behaviour
    assert pd.Timestamp("2024-04-29 02:00") not in plain["t_event"].tolist()
    rows = pd.DataFrame({"person_id": [1, 1], "measurement_datetime": pd.to_datetime(["2024-05-01 12:30", "2024-05-01 13:30"],
                                                                                       format="%Y-%m-%d %H:%M"),
                         "measurement_date": pd.NaT})
    assert len(bb.prune_rows(rows, t0, "measurement_datetime", "measurement_date")) == 0
    assert len(bb.prune_rows(rows, t0, "measurement_datetime", "measurement_date", grace_h=1.0)) == 1


def run_cli(synth_dir, cohort_csv, out, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bb.main(["--data", str(synth_dir), "--cohort", str(cohort_csv), "--out", str(out), "--chunk-rows", "900", *extra])
    assert rc == 0
    return buf.getvalue()


def test_cli_default_is_current_encounter_and_with_history_is_a_labelled_variant(synth_dir, cohort_csv):
    lo = cohort_csv.parent
    text = run_cli(synth_dir, cohort_csv, lo / "bl_cur.parquet", "--sites", "all")
    text_h = run_cli(synth_dir, cohort_csv, lo / "bl_hist.parquet", "--sites", "all", "--with-history")
    side = json.loads((lo / "bl_cur.columns.json").read_text())
    side_h = json.loads((lo / "bl_hist.columns.json").read_text())
    assert side["encounter_scope"] == "current" and side["encounter_scope_label"] == "current encounter only"
    assert side_h["encounter_scope"] == "with_history" and side_h["encounter_scope_label"] == "with history"
    assert "current encounter only" in text and "WITH HISTORY" in text_h
    assert "encounter start from {'cohort':" in text                                      # the cohort table carries the column
    assert side["baselines"] == side_h["baselines"] and set(side["meta"]) == {"meta__hours_since_encounter_start", "meta__label_dx_before_t0"}
    cur, hist = pd.read_parquet(lo / "bl_cur.parquet"), pd.read_parquet(lo / "bl_hist.parquet")
    assert list(cur.columns) == list(hist.columns)
    assert rsf.baseline_scope(lo / "bl_cur.parquet") == "current encounter only" and rsf.baseline_scope(lo / "bl_hist.parquet") == "with history"


# ------------------------------------------------------------------------------------------------- leakage, Baseline P
@pytest.mark.parametrize("hours_after", [1e-6 / 3600, 0.01, 24.0])
def test_injected_post_t0_events_change_no_baseline_p_or_meta_column(synth, hours_after):
    """The helper adds its sentinels 1 h after (t0 + hours_after), i.e. beyond P's +1 h presentation window: nothing moves."""
    from test_baselines_helpers import augment_bedside, post_t0_events
    tables = augment_bedside(synth[0], seed=11)
    base = build_feature_set(tables)
    leaked = build_feature_set(post_t0_events(tables, hours_after=hours_after))
    same(leaked.X, base.X)
    same(leaked.X_extra, base.X_extra)
    assert (base.X_extra["pres_first__hr__miss"] < 1).any() and (base.X_extra["pres_score__gcs__miss"] < 1).any()
