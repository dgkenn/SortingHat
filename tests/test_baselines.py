"""Baselines A-D: exact-value feature tests on hand-built patients, provenance, imputation, synthetic scale."""

import numpy as np
import pandas as pd
import pytest

from sortinghat.baselines import (BASELINES, BaselineConfig, MedianImputer, as_of, build_feature_set,
                                  build_registry)
from sortinghat.baselines import lexicon as lx
from sortinghat.baselines.asof import empty_events
from sortinghat.baselines.vocab import discover_lab_vocabulary
from test_baselines_helpers import H, T0, add_rows, augment_bedside, drug, img, meas, mini_tables


def row(fs, pid=1):
    return fs.X.loc[pid]


def build(tables, **cfg):
    return build_feature_set(tables, BaselineConfig(**cfg))


# ------------------------------------------------------------------------------------------ Baseline A
def test_scores_nearest_within_6h_before_t0():
    t = mini_tables()
    t = add_rows(t, "omop_measurement", [
        meas(1, "Glasgow Coma Scale Score", T0 - H(4), 5), meas(1, "Glasgow Coma Scale Score", T0 - H(2), 8),
        meas(1, "RASS (Richmond Agitation Sedation Scale)", T0 - H(7), -4),        # outside the 6 h window
        meas(1, "FOUR Score", T0 + H(1), 10),                                        # after t0
        meas(1, "Eye Opening", T0, 3)])                                              # exactly at t0 is allowed
    r = row(build(t))
    assert r["score__gcs__value"] == 8 and r["score__gcs__age_h"] == pytest.approx(2.0) and r["score__gcs__miss"] == 0
    assert np.isnan(r["score__rass__value"]) and r["score__rass__miss"] == 1
    assert np.isnan(r["score__four__value"]) and r["score__four__miss"] == 1
    assert r["score__gcs_eye__value"] == 3 and r["score__gcs_eye__age_h"] == 0
    assert r["score__nesi__miss"] == 1                                               # not available -> indicator


def test_scores_out_of_range_are_missing_not_clipped():
    t = add_rows(mini_tables(), "omop_measurement", [meas(1, "Glasgow Coma Scale Score", T0 - H(1), 99)])
    assert row(build(t))["score__gcs__miss"] == 1


def test_demographics_and_sex_missing():
    fs = build(mini_tables(sex="Female", age=47.5))
    r = row(fs)
    assert r["demo__age_years"] == 47.5 and r["demo__sex_male"] == 0 and r["demo__sex_male__miss"] == 0
    r2 = row(build(mini_tables(sex="")))
    assert np.isnan(r2["demo__sex_male"]) and r2["demo__sex_male__miss"] == 1


def test_sedation_exact_values_admin_and_order_fallback():
    t = add_rows(mini_tables(), "omop_drug_exposure", [
        drug(1, "PROPOFOL 10 MG/ML IV EMULSION", T0 - H(3), T0 + H(2), 10.0),   # running at t0
        drug(1, "MIDAZOLAM 1 MG/ML INJ", T0 - H(10), T0 - H(8), 4.0),             # finished, in 24 h window only
        drug(1, "FENTANYL 50 MCG/ML INJ", T0 - H(1), None, 2.0),                  # order only, <= 2 h
        drug(1, "HYDROMORPHONE 1 MG/ML INJ", T0 - H(5), None, 3.0),               # order only, too old to be active
        drug(1, "MORPHINE SULFATE 4 MG/ML INJ", T0 - H(25), T0 - H(23), 8.0),     # half inside 24 h
        drug(1, "LORAZEPAM 2 MG/ML INJ", T0 - H(26), T0 - H(24.5), 5.0),          # entirely before the window
        drug(1, "ACETAMINOPHEN 325 MG TAB", T0 - H(1), T0, 1.0)])                 # not a sedative / opioid
    r = row(build(t))
    assert r["sed__propofol__on_t0"] == 1 and r["sed__propofol__qty_6h"] == 0 and r["sed__propofol__qty_24h"] == 0
    assert r["sed__midazolam__on_t0"] == 0 and r["sed__midazolam__qty_6h"] == 0 and r["sed__midazolam__qty_24h"] == 4
    assert r["sed__fentanyl__on_t0"] == 1 and r["sed__fentanyl__qty_6h"] == 2
    assert r["sed__hydromorphone__on_t0"] == 0 and r["sed__hydromorphone__qty_6h"] == 3
    assert r["sed__morphine__on_t0"] == 0 and r["sed__morphine__qty_24h"] == pytest.approx(4.0)
    assert r["sed__lorazepam__qty_24h"] == 0
    assert r["sed__sedative__on_t0"] == 1 and r["sed__opioid__on_t0"] == 1
    assert r["sed__approx_time"] == 1                                              # fentanyl / hydromorphone orders
    assert r["sed__n_agents_24h"] == 4                                             # propofol, midazolam, fentanyl, hydromorphone


def test_sedation_admin_only_is_not_approximate():
    t = add_rows(mini_tables(), "omop_drug_exposure", [drug(1, "DEXMEDETOMIDINE 4 MCG/ML IV", T0 - H(2), T0 - H(1), 3.0)])
    r = row(build(t))
    assert r["sed__approx_time"] == 0 and r["sed__dexmedetomidine__qty_6h"] == 3


def test_drug_time_basis_order_forces_approximate():
    t = add_rows(mini_tables(), "omop_drug_exposure", [drug(1, "PROPOFOL 10 MG/ML IV EMULSION", T0 - H(3), T0 - H(1), 6.0)])
    auto, order = row(build(t)), row(build(t, drug_time_basis="order"))
    assert auto["sed__approx_time"] == 0 and order["sed__approx_time"] == 1
    assert order["sed__propofol__on_t0"] == 0                                      # order 3 h ago, > 2 h


def test_ingredient_matching_brands_and_combinations():
    assert lx.match_ingredients("DIPRIVAN 1% EMULSION") == ["propofol"]
    assert lx.match_ingredients("Precedex") == ["dexmedetomidine"]
    assert set(lx.match_ingredients("HYDROCODONE/ACETAMINOPHEN 5-325")) == {"hydrocodone"}
    assert lx.match_ingredients("LEVETIRACETAM 500 MG") == [] and lx.match_ingredients(None) == []
    assert set(lx.NAMED_INGREDIENTS) == {"propofol", "midazolam", "lorazepam", "dexmedetomidine", "ketamine",
                                         "fentanyl", "hydromorphone", "morphine"}


def test_concept_name_resolves_ingredient_when_source_text_is_opaque():
    t = add_rows(mini_tables(), "omop_drug_exposure", [
        {"person_id": 1, "drug_exposure_start_datetime": T0 - H(2), "drug_exposure_end_datetime": T0 - H(1),
         "drug_source_value": "NDC 0409-4699", "quantity": 7.0, "drug_concept_id": 111}])
    t = add_rows(t, "omop_concept", [{"concept_id": 111, "concept_name": "ketamine 50 MG/ML Injectable Solution",
                                      "domain_id": "Drug", "vocabulary_id": "RxNorm", "standard_concept": "S"}])
    assert row(build(t))["sed__ketamine__qty_6h"] == 7


# ------------------------------------------------------------------------------------------ Baseline B
def test_bedside_features_latest_before_t0():
    t = add_rows(mini_tables(), "omop_measurement", [
        meas(1, "Heart rate", T0 - H(3), 100), meas(1, "Heart rate", T0 - H(1), 118), meas(1, "Heart rate", T0 + H(1), 40),
        meas(1, "Temperature", T0 - H(1), 104.0, "degF"), meas(1, "SpO2", T0 - H(8), 90),
        meas(1, "Pupil size left", T0 - H(1), 4), meas(1, "Pupil size right", T0 - H(1), 2),
        meas(1, "Pupil reactivity left", T0 - H(1), 0), meas(1, "Pupil reactivity right", T0 - H(1), 1),
        meas(1, "POC glucose (fingerstick)", T0 - H(30), 55, "mg/dL"), meas(1, "POC glucose (fingerstick)", T0 - H(2), 38, "mg/dL"),
        meas(1, "GLUCOSE", T0 - H(1), 500, "mg/dL")])                            # serum glucose is NOT POC
    r = row(build(t))
    assert r["vital__hr__value"] == 118
    assert r["vital__temp_c__value"] == pytest.approx(40.0)
    assert r["vital__spo2__miss"] == 1                                              # 8 h old
    assert r["pupil__pupil_size_l__value"] == 4 and r["pupil__size_asymmetry_mm"] == 2
    assert r["pupil__any_nonreactive"] == 1
    assert r["poc_glucose__poc_glucose__value"] == 38 and r["poc_glucose__poc_glucose__age_h"] == pytest.approx(2.0)


def test_history_flags_recent_vs_any_and_no_diagnosis_leak():
    t = mini_tables()
    t = add_rows(t, "omop_condition_occurrence", [
        {"person_id": 1, "condition_start_datetime": T0 - H(24 * 400), "condition_source_value": "I46.9"},   # old arrest
        {"person_id": 1, "condition_start_datetime": T0 - H(5), "condition_source_value": "S06.5X0A"},     # head trauma
        {"person_id": 1, "condition_start_datetime": T0 + H(5), "condition_source_value": "G40.909"}])
    t = add_rows(t, "omop_observation", [{"person_id": 1, "observation_datetime": T0 - H(1),
                                          "observation_source_value": "Witnessed convulsion", "value_as_string": "yes"}])
    r = row(build(t))
    assert r["hx__arrest_any"] == 1 and r["hx__arrest_recent"] == 0
    assert r["hx__head_trauma_recent"] == 1 and r["hx__trauma_recent"] == 1
    assert r["hx__convulsion_recent"] == 1


def test_convulsion_negative_observation_not_flagged():
    t = add_rows(mini_tables(), "omop_observation", [{"person_id": 1, "observation_datetime": T0 - H(1),
                                                      "observation_source_value": "Witnessed seizure", "value_as_string": "no"}])
    assert row(build(t))["hx__convulsion_recent"] == 0


# ------------------------------------------------------------------------------------------ Baseline C
def test_lab_uses_result_time_not_collection_time():
    res = "measurement_result_datetime"
    t = add_rows(mini_tables(), "omop_measurement", [
        {**meas(1, "SODIUM", T0 - H(3), 128.0), res: T0 - H(2)},                 # collected and resulted before t0
        {**meas(1, "LACTATE, WHOLE BLOOD", T0 - H(2), 9.0), res: T0 + H(1)},     # collected before, resulted AFTER t0
        {**meas(1, "AMMONIA, PLASMA", T0 - H(1), 200.0), res: T0 - H(0.5)}])
    r = row(build(t))
    assert r["lab__sodium__value"] == 128 and r["lab__ammonia__value"] == 200
    assert r["lab__lactate__miss"] == 1 and np.isnan(r["lab__lactate__value"])    # not yet resulted at t0
    assert r["lab__approx_time"] == 0                                              # all used real result times


def test_lab_collection_plus_lag_fallback_is_flagged_approximate():
    t = add_rows(mini_tables(), "omop_measurement", [
        meas(1, "SODIUM", T0 - H(3), 128.0),                                       # + 1 h lag -> before t0
        meas(1, "POTASSIUM", T0 - H(0.5), 6.5),                                    # + 1 h lag -> after t0: excluded
        meas(1, "AMMONIA, PLASMA", T0 - H(1.5), 90.0)])                            # + 2 h lag -> after t0: excluded
    r = row(build(t))
    assert r["lab__sodium__value"] == 128 and r["lab__potassium__miss"] == 1 and r["lab__ammonia__miss"] == 1
    assert r["lab__approx_time"] == 1 and r["lab__n_results_by_t0"] == 1


def test_lab_time_basis_result_requires_a_result_time():
    t = add_rows(mini_tables(), "omop_measurement", [meas(1, "SODIUM", T0 - H(30), 130.0)])
    assert row(build(t))["lab__sodium__value"] == 130
    assert row(build(t, lab_time_basis="result"))["lab__sodium__miss"] == 1


def test_lab_lookback_window_and_unit_conversion():
    t = add_rows(mini_tables(), "omop_measurement", [
        meas(1, "SODIUM", T0 - H(80), 111.0),                                      # beyond 72 h lookback
        meas(1, "GLUCOSE", T0 - H(10), 5.0, "mmol/L"),
        meas(1, "CREATININE", T0 - H(10), 176.8, "umol/L")])
    r = row(build(t))
    assert r["lab__sodium__miss"] == 1
    assert r["lab__glucose__value"] == pytest.approx(90.08, abs=0.01)
    assert r["lab__creatinine__value"] == pytest.approx(2.0, abs=1e-6)


def test_csf_and_serum_names_do_not_collide():
    assert lx.classify_measurement("CSF GLUCOSE").key == "csf_glucose"
    assert lx.classify_measurement("GLUCOSE").key == "glucose"
    assert lx.classify_measurement("POC glucose (fingerstick)").key == "poc_glucose"
    assert lx.classify_measurement("Glasgow Coma Scale - Eye Opening").key == "gcs_eye"
    assert lx.classify_measurement("Glasgow Coma Scale Score").key == "gcs"
    assert lx.classify_measurement("Urine drug screen benzodiazepine").key == "uds_benzodiazepine"
    assert lx.classify_measurement("something unrelated") is None


def test_tox_and_culture_features():
    t = add_rows(mini_tables(), "omop_measurement", [
        meas(1, "Ethanol level, serum", T0 - H(10), 210.0),
        meas(1, "Blood culture", T0 - H(40), 1.0),                                  # +24 h lag -> resulted before t0
        meas(1, "Urine culture", T0 - H(10), 1.0)])                                 # +24 h lag -> not resulted yet
    r = row(build(t))
    assert r["tox__ethanol__value"] == 210
    assert r["culture__culture_blood__resulted"] == 1 and r["culture__culture_blood__positive"] == 1
    assert r["culture__culture_urine__resulted"] == 0 and r["culture__culture_urine__miss"] == 1


def test_imaging_availability_final_time_vs_study_plus_lag():
    t = add_rows(mini_tables(), "imaging", [
        img(1, "CT head", T0 - H(2), T0 - H(1)),                                    # final before t0
        img(1, "MRI brain", T0 - H(1), T0 + H(3)),                                  # study done, read after t0
        img(1, "CTA head", T0 - H(3), None)])                                       # no final time: study + 1.5 h lag
    r = row(build(t))
    assert r["img__ct_head__final_by_t0"] == 1 and r["img__mri_brain__final_by_t0"] == 0
    assert r["img__cta_head__final_by_t0"] == 1 and r["img__approx_time"] == 1
    assert row(build(t, imaging_time_basis="result"))["img__cta_head__final_by_t0"] == 0


def test_extra_labs_vocabulary_is_added_to_baseline_c_only():
    cfg = BaselineConfig(extra_labs=("vitamin b12",))
    t = add_rows(mini_tables(), "omop_measurement", [meas(1, "Vitamin B12", T0 - H(20), 300.0)])
    fs = build_feature_set(t, cfg)
    assert fs.X.loc[1, "lab__x_vitamin_b12__value"] == 300
    p = fs.provenance.set_index("feature")
    assert p.loc["lab__x_vitamin_b12__value", "baseline"] == "C"
    assert "lab__x_vitamin_b12__value" not in build_feature_set(t).X.columns


def test_discover_lab_vocabulary_is_suppressed_and_min_11():
    rows = [meas(i, "Vitamin B12", T0, 1.0) for i in range(30)] + [meas(i, "Rare thing", T0, 1.0) for i in range(5)]
    m = pd.DataFrame(rows)
    v = discover_lab_vocabulary(m, min_patients=1)
    assert list(v["norm_name"]) == ["vitamin b12"]


# ------------------------------------------------------------------------------------------ Baseline D
def test_indication_only_in_baseline_d():
    fs = build(mini_tables(indication="Rule out NCSE"))
    assert fs.X.loc[1, "ind__rule_out_ncse"] == 1 and fs.X.loc[1, [c for c in fs.X if c.startswith("ind__")]].sum() == 1
    assert not any(c.startswith("ind__") for c in fs.columns("C"))
    assert [c for c in fs.columns("D") if c not in fs.columns("C")] == [f"ind__{l}" for l in lx.INDICATION_LEVELS]
    assert fs.X.loc[1, "ind__missing"] == 0
    assert build(mini_tables(indication=""), ).X.loc[1, "ind__missing"] == 1
    for text, lev in (("post-arrest", "post_arrest"), ("unexplained AMS", "unexplained_ams"), ("seizure", "seizure"),
                      ("spell", "spell"), ("pre-op screening", "other")):
        assert lx.indication_category(text) == lev


def test_index_uses_first_acute_adult_eeg_and_its_own_indication():
    t = mini_tables(indication="spell")
    t = add_rows(t, "eeg_metadata", [{
        "SiteID": "S0001", "BDSPPatientID": "1", "BidsFolder": "sub-S00011", "SessionID": "s1b", "PatientClass": "ICU",
        "ReferralIndication": "seizure", "SexDSC": "Male", "EEGFolder": "eeg", "DurationInSeconds": 100.0,
        "ServiceName": "Routine"}])
    t = add_rows(t, "reports_findings", [{"BDSPPatientID": "1", "SessionID": "s1b", "StartTime(EEG)": T0 + H(24 * 5),
                                          "AgeAtVisit": 61.0, "SexDSC": "Male"}])
    fs = build(t)
    assert len(fs.index) == 1 and fs.index.loc[0, "t0"] == T0 and fs.X.loc[1, "ind__spell"] == 1


# --------------------------------------------------------------------------- provenance / stability
@pytest.fixture(scope="module")
def synth_fs(synth):
    tables = augment_bedside(synth[0], seed=3)
    return build_feature_set(tables), tables


def test_provenance_covers_matrix_exactly_and_baselines_nest(synth_fs):
    fs, _ = synth_fs
    prov = fs.provenance
    assert list(fs.X.columns) == prov["feature"].tolist() and prov["feature"].is_unique
    assert set(prov["baseline"]) == set(BASELINES)
    for a, b in zip(BASELINES, BASELINES[1:]):
        assert set(fs.columns(a)) < set(fs.columns(b))
    assert fs.columns("D") == prov["feature"].tolist()
    assert prov.loc[prov["baseline"] == "A", "group"].isin(["demographics", "score", "sedation_opioid"]).all()
    assert prov.loc[prov["baseline"] == "B", "group"].isin(["vital", "pupil", "poc_glucose", "history"]).all()
    assert prov.loc[prov["baseline"] == "C", "group"].isin(["lab", "tox", "culture", "imaging"]).all()
    assert (prov.loc[prov["baseline"] == "D", "group"] == "indication").all()
    assert prov[["source_table", "time_basis", "window", "imputation", "description"]].notna().all().all()


def test_feature_names_are_stable_and_independent_of_data(synth_fs):
    fs, _ = synth_fs
    assert build_registry(BaselineConfig())["feature"].tolist() == build_feature_set(mini_tables()).X.columns.tolist()
    assert fs.X.columns.tolist() == build_registry(BaselineConfig())["feature"].tolist()


def test_synthetic_scale_sanity(synth_fs):
    fs, tables = synth_fs
    X = fs.X
    assert len(X) == len(fs.index) > 1000 and X.index.is_unique
    assert fs.index["t0"].notna().all() and (fs.index["person_id"].to_numpy() == X.index.to_numpy()).all()
    assert fs.diagnostics["n_events_dropped_post_t0"] > 0                            # the generator has post-t0 rows
    assert not np.isinf(X.to_numpy(float)).any()
    # every missing-indicator agrees with its value column
    for c in X.columns[X.columns.str.endswith("__value")]:
        m = c[:-len("value")] + "miss"
        if m in X:
            assert ((X[c].isna()).astype(float) == X[m]).all(), c
    # flags are 0/1, counts non-negative
    prov = fs.provenance.set_index("feature")
    for c in prov.index[prov["role"].isin(["flag", "approx_flag"])]:
        assert set(X[c].dropna().unique()) <= {0.0, 1.0}, c
    assert (X[prov.index[prov["role"].isin(["count", "dose"])]] >= 0).all().all()
    assert X["sed__sedative__on_t0"].sum() > 0 and X["vital__hr__value"].notna().sum() > 100
    ind = X[[c for c in X if c.startswith("ind__")]]
    assert (ind.sum(axis=1) == 1).all()


def test_to_long_is_tidy(synth_fs):
    fs, _ = synth_fs
    long = fs.to_long("B")
    assert list(long.columns[:3]) == ["person_id", "feature", "value"]
    assert len(long) == len(fs.X) * len(fs.columns("B"))
    assert set(long["baseline"]) == {"A", "B"}


# --------------------------------------------------------------------------------------- imputation
def test_median_imputer_fit_on_training_only_and_indicators_survive(synth_fs):
    fs, _ = synth_fs
    Xc = fs.matrix("D")
    train, test = Xc.iloc[: len(Xc) // 2], Xc.iloc[len(Xc) // 2:]
    imp = MedianImputer(fs.provenance).fit(train)
    out = imp.transform(test)
    assert not out.isna().any().any()
    assert imp.medians_["demo__age_years"] == pytest.approx(train["demo__age_years"].median())
    # missing indicators unchanged by imputation
    pd.testing.assert_frame_equal(out.filter(like="__miss"), test.filter(like="__miss"))
    # observed values are untouched
    obs = test["score__gcs__value"].notna()
    assert (out.loc[obs, "score__gcs__value"] == test.loc[obs, "score__gcs__value"]).all()
    # an all-missing training column falls back to 0, never NaN
    assert imp.medians_["score__nesi__value"] == 0.0


def test_imputer_requires_fit():
    with pytest.raises(RuntimeError):
        MedianImputer().transform(pd.DataFrame({"a": [1.0]}))


def test_empty_events_ok():
    assert as_of(empty_events(), T0).empty
    fs = build_feature_set(mini_tables())
    assert fs.X.loc[1, "sed__n_agents_24h"] == 0 and fs.X.loc[1, "score__gcs__miss"] == 1


def test_cli_writes_provenance_and_keeps_matrix_local_only(tmp_path, synth_dir, capsys):
    from sortinghat.baselines.__main__ import main
    assert main(["--data", str(synth_dir), "--out", str(tmp_path)]) == 0
    prov = pd.read_csv(tmp_path / "feature_provenance.csv")
    assert set(prov["baseline"]) == set(BASELINES)
    m = tmp_path / "local_only" / "features.csv"
    assert m.exists() and (m.stat().st_mode & 0o777) == 0o600
    out = capsys.readouterr().out
    assert "patients=" in out and "50000" not in out                     # no ids / record-level values printed
