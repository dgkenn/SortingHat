"""Structured silver-label extractor on synthetic fixtures (hand-built OMOP rows; the generator lacks these items)."""
import numpy as np
import pandas as pd
import pytest

from sortinghat import data_io, schema
from sortinghat.labels import extract as ex
from sortinghat.labels.anchors import EVENT_COLUMNS
from sortinghat.labels.circularity_audit import run_circularity_audit
from sortinghat.safe_output import AggregateOnlyError, SUPPRESSED, assert_aggregate_only

T0 = pd.Timestamp("2025-03-01 12:00:00")


def at(h):
    return T0 + pd.Timedelta(hours=h)


class Fx:
    """Tiny OMOP fixture builder (column names as in the real tables)."""

    def __init__(self):
        self.rows = {k: [] for k in ("meas", "drug", "cond", "proc", "obs", "visit", "concept", "rf")}
        self.pids: list = []

    def person(self, pid, site="S0001"):
        self.pids.append((pid, site))
        return pid

    def lab(self, pid, h, name, value=np.nan, unit=None, cid=0, scid=0, vtext=None, vcid=0, unit_cid=0):
        self.rows["meas"].append({"person_id": pid, "measurement_datetime": at(h), "measurement_source_value": name,
                                  "value_as_number": value, "unit_source_value": unit, "measurement_concept_id": cid,
                                  "measurement_source_concept_id": scid, "value_source_value": vtext,
                                  "value_as_concept_id": vcid, "unit_concept_id": unit_cid})

    def drug(self, pid, h, name, end_h=None, route=None, cid=0):
        self.rows["drug"].append({"person_id": pid, "drug_exposure_start_datetime": at(h),
                                  "drug_exposure_end_datetime": at(end_h) if end_h is not None else pd.NaT,
                                  "drug_source_value": name, "route_source_value": route, "drug_concept_id": cid})

    def cond(self, pid, h, code, visit_id=None, cid=0, scid=0):
        self.rows["cond"].append({"person_id": pid, "condition_start_datetime": at(h) if h is not None else pd.NaT,
                                  "condition_source_value": code, "condition_concept_id": cid,
                                  "condition_source_concept_id": scid, "visit_occurrence_id": visit_id})

    def proc(self, pid, h, code):
        self.rows["proc"].append({"person_id": pid, "procedure_datetime": at(h), "procedure_source_value": code,
                                  "procedure_concept_id": 0, "procedure_source_concept_id": 0})

    def obs(self, pid, h, text, value=None):
        self.rows["obs"].append({"person_id": pid, "observation_datetime": at(h), "observation_source_value": text,
                                 "value_as_string": value, "observation_concept_id": 0})

    def visit(self, pid, vid, start_h, end_h=None, prev=None):
        self.rows["visit"].append({"person_id": pid, "visit_occurrence_id": vid, "visit_start_datetime": at(start_h),
                                   "visit_end_datetime": at(end_h) if end_h is not None else pd.NaT,
                                   "preceding_visit_occurrence_id": prev})

    def concept(self, cid, name, domain, vocab, code):
        self.rows["concept"].append({"concept_id": cid, "concept_name": name, "domain_id": domain,
                                     "vocabulary_id": vocab, "concept_code": code})

    def tables(self, with_rf=False):
        cols = ex.COLUMNS
        t = {"omop_measurement": pd.DataFrame(self.rows["meas"], columns=cols["measurement"]),
             "omop_drug_exposure": pd.DataFrame(self.rows["drug"], columns=cols["drug_exposure"]),
             "omop_condition_occurrence": pd.DataFrame(self.rows["cond"], columns=cols["condition_occurrence"]),
             "omop_procedure_occurrence": pd.DataFrame(self.rows["proc"], columns=cols["procedure_occurrence"]),
             "omop_observation": pd.DataFrame(self.rows["obs"], columns=cols["observation"]),
             "omop_visit_occurrence": pd.DataFrame(self.rows["visit"], columns=cols["visit_occurrence"]),
             "omop_concept": pd.DataFrame(self.rows["concept"], columns=cols["concept"])}
        if with_rf:
            t["reports_findings"] = pd.DataFrame(self.rows["rf"])
        return t

    def cohort(self):
        return pd.DataFrame({"person_id": [p for p, _ in self.pids], "t0": [T0] * len(self.pids),
                             "SiteID": [s for _, s in self.pids]})

    def run(self, **kw):
        return ex.extract_silver(self.tables(), self.cohort(), **kw)


def one(fx_builder, label):
    """Silver label of a single-patient fixture. ``fx_builder(fx, pid)`` fills the rows."""
    fx = Fx()
    pid = fx.person(1)
    fx_builder(fx, pid)
    res = fx.run()
    return res, res.labels.loc["1", label], res.fired["1"][label]


# ------------------------------------------------------------------------------------------- E5 and units
def test_e5_glucose_unit_conversion_mmol_to_mgdl():
    assert one(lambda f, p: f.lab(p, -2, "GLUCOSE", 2.5, "mmol/L"), "E5")[1]          # 2.5 mmol/L = 45 mg/dL < 50
    assert not one(lambda f, p: f.lab(p, -2, "GLUCOSE", 3.5, "mmol/L"), "E5")[1]      # 63 mg/dL
    res, v, fired = one(lambda f, p: f.lab(p, -2, "GLUCOSE", 2.5, "mmol/L"), "E5")
    assert fired == ["E5_glucose_lo"]
    assert res.events.loc[res.events["item"] == "glucose", "value"].iloc[0] == pytest.approx(45.04, rel=1e-3)


def test_e5_ammonia_ug_dl_to_umol_l():
    assert one(lambda f, p: f.lab(p, -3, "AMMONIA, PLASMA", 200, "ug/dL"), "E5")[2] == ["E5_ammonia"]    # 117 umol/L
    assert not one(lambda f, p: f.lab(p, -3, "AMMONIA, PLASMA", 100, "ug/dL"), "E5")[1]                  # 58.7 umol/L
    assert one(lambda f, p: f.lab(p, -3, "AMMONIA, PLASMA", 120, "umol/L"), "E5")[1]


def test_unrecognised_or_missing_unit_never_fires():
    res, v, _ = one(lambda f, p: f.lab(p, -2, "GLUCOSE", 20, "furlongs"), "E5")
    assert not v and res.diagnostics.get("meas_dropped_unit_unrecognised") == 1
    res, v, _ = one(lambda f, p: f.lab(p, -2, "GLUCOSE", 2.5, None), "E5")
    assert not v and res.diagnostics.get("meas_dropped_unit_missing") == 1


def test_e5_window_and_paco2_with_ph():
    assert not one(lambda f, p: f.lab(p, -40, "AMMONIA, PLASMA", 200, "umol/L"), "E5")[1]       # before the window
    assert not one(lambda f, p: f.lab(p, 2, "GLUCOSE", 30, "mg/dL"), "E5")[1]                   # glucose<50 window ends +1 h

    def f(fx, p):
        fx.lab(p, -2, "PaCO2 arterial", 75, "mmHg")
        fx.lab(p, -2, "Arterial pH", 7.2)
    assert one(f, "E5")[2] == ["E5_paco2", "E5_paco2_hi", "E5_paco2_ph"]


def test_concept_route_with_loinc_default_unit_and_zero_filled_names():
    def f(fx, p):
        fx.concept(9001, "Glucose [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "2345-7")
        fx.lab(p, -2, "xyzzy", 40, None, cid=9001)                    # name is useless; unit comes from the LOINC entry
    res, v, fired = one(f, "E5")
    assert v and fired == ["E5_glucose_lo"]
    assert res.diagnostics["meas_match_basis_concept"] == 1


def test_unit_concept_id_resolves_through_ucum():
    def f(fx, p):
        fx.concept(5001, "milligram per deciliter", "Unit", "UCUM", "mg/dL")
        fx.lab(p, -2, "GLUCOSE", 30, None, unit_cid=5001)
    assert one(f, "E5")[1]


# ------------------------------------------------------------------------------------------------ E2
def test_arrest_from_icd_cpt_and_concept_id():
    assert one(lambda f, p: f.cond(p, -10, "I46.9"), "E2")[2] == ["E2_arrest"]
    assert one(lambda f, p: f.proc(p, -10, "92950"), "E2")[2] == ["E2_arrest"]
    assert not one(lambda f, p: f.cond(p, 5, "I46.9"), "E2")[1]                                  # arrest after t0

    def f(fx, p):
        fx.concept(321042, "Cardiac arrest", "Condition", "ICD10CM", "I46.9")
        fx.cond(p, -10, "garbled", cid=321042)
    assert one(f, "E2")[1]
    assert not one(lambda f, p: f.cond(p, -10, "Z86.74"), "E2")[1]                               # history, not an event


def test_asphyxia_codes():
    assert one(lambda f, p: f.cond(p, -20, "T71.162A"), "E2")[2] == ["E2_asphyxia"]
    assert one(lambda f, p: f.cond(p, -20, "T75.1XXA"), "E2")[2] == ["E2_asphyxia"]


def test_profound_shock_needs_30_minutes_below_map_50():
    def run(readings, **kw):
        def f(fx, p):
            for h, v in readings:
                fx.lab(p, h, "MAP", v, "mmHg")
        return one(f, "E2")
    m = lambda mins: -2 + mins / 60.0                                                      # noqa: E731
    assert run([(m(i), 45) for i in range(0, 35, 5)])[2] == ["E2_shock"]                   # 0..30 min low
    assert not run([(m(i), 45) for i in range(0, 25, 5)])[1]                               # only 20 min
    assert not run([(m(0), 45), (m(10), 45), (m(15), 70), (m(20), 45), (m(30), 45)])[1]    # a normal reading breaks the run
    assert not run([(m(0), 45), (m(40), 45)])[1]                                           # 40-min gap: not sustained
    assert not run([(m(i), 50) for i in range(0, 60, 5)])[1]                               # MAP 50 is not < 50
    assert not run([(m(i), 0) for i in range(0, 60, 5)])[1]                                # 0 = disconnected line, dropped
    assert not run([(m(i) - 80, 45) for i in range(0, 40, 5)])[1]                          # outside [-48, 0] h of t0


def test_shock_from_systolic_and_diastolic_when_no_map():
    def f(fx, p):
        for i in range(0, 40, 5):
            fx.lab(p, -2 + i / 60.0, "Systolic blood pressure", 70, "mmHg")
            fx.lab(p, -2 + i / 60.0, "Diastolic blood pressure", 35, "mmHg")          # MAP = 35 + 35/3 = 46.7
    assert one(f, "E2")[2] == ["E2_shock"]


# ------------------------------------------------------------------------------------------------ E6 / E7
def e6_fixture(fx, p, abx_days=4, route="IV", organ=True, agent="VANCOMYCIN 1 G IV"):
    fx.lab(p, -5, "Blood culture", np.nan, None, vtext="No growth")
    for d in range(abx_days):
        fx.drug(p, -4 + 24 * d, agent, end_h=-4 + 24 * d + 2, route=route)
    if organ:
        fx.lab(p, -3, "LACTATE, WHOLE BLOOD", 3.4, "mmol/L")


def test_e6_blood_culture_qad_and_organ_dysfunction():
    res, v, fired = one(e6_fixture, "E6")
    assert v and fired == ["E6_blood_culture", "E6_lactate", "E6_qad"]
    assert not one(lambda f, p: e6_fixture(f, p, abx_days=3), "E6")[1]                      # only 3 QADs
    assert not one(lambda f, p: e6_fixture(f, p, organ=False), "E6")[1]                     # no organ dysfunction
    assert not one(lambda f, p: e6_fixture(f, p, route="PO"), "E6")[1]                      # oral vancomycin does not qualify
    assert one(lambda f, p: e6_fixture(f, p, route="PO", agent="LEVOFLOXACIN 750 MG"), "E6")[1]   # any-route agent


def test_qad_requires_new_agent_and_bridges_one_missing_day():
    def f(fx, p):
        fx.lab(p, -5, "Blood culture")
        fx.lab(p, -3, "LACTATE, WHOLE BLOOD", 3.4, "mmol/L")
        for d in (0, 1, 3):                                      # day 2 missing, same agent: bridged -> 4 days
            fx.drug(p, -4 + 24 * d, "CEFEPIME 2 G IV", end_h=-3 + 24 * d, route="IV")
    assert one(f, "E6")[1]

    def g(fx, p):                                                # same drug already running the 2 prior days: not "new"
        fx.lab(p, -5, "Blood culture")
        fx.lab(p, -3, "LACTATE, WHOLE BLOOD", 3.4, "mmol/L")
        for d in range(-10, 4):
            fx.drug(p, -4 + 24 * d, "CEFEPIME 2 G IV", end_h=-3 + 24 * d, route="IV")
    res, v, _ = one(g, "E6")
    assert not v


def test_vasopressor_initiation_and_washout():
    def f(fx, p):
        e6_fixture(fx, p, organ=False)
        fx.drug(p, -3, "NOREPINEPHRINE 4 MG/250 ML IV", route="IV")
    res, v, fired = one(f, "E6")
    assert v and "E6_vasopressor" in fired

    def g(fx, p):                                                # rate-change rows within 24 h are not new starts
        for h in (-60, -50, -40, -30, -20):
            fx.drug(p, h, "NOREPINEPHRINE 4 MG/250 ML IV", route="IV")
    res, v, _ = one(g, "E6")
    assert (res.events["item"] == "vasopressor_initiation").sum() == 1
    res, v, _ = one(lambda f, p: f.drug(p, -3, "LIDOCAINE 1% WITH EPINEPHRINE", route="INFILTRATION"), "E6")
    assert (res.events["item"] == "vasopressor_initiation").sum() == 0


def test_e7_csf_with_rbc_correction_and_e6_exclusion():
    def f(fx, p, wbc=24, rbc=5000, prot=150):
        fx.lab(p, 1, "CSF WBC", wbc, "cells/uL")
        fx.lab(p, 1, "CSF RBC", rbc, "cells/uL")
        fx.lab(p, 1, "CSF PROTEIN", prot, "mg/dL")
    assert not one(lambda fx, p: f(fx, p), "E7")[1]                  # 24 - 5000/500 = 14 < 20
    assert one(lambda fx, p: f(fx, p, wbc=40), "E7")[1]              # 30
    assert not one(lambda fx, p: f(fx, p, wbc=40, prot=90), "E7")[1]  # no supporting protein / glucose / blood culture

    def both(fx, p):
        e6_fixture(fx, p)
        fx.lab(p, 4, "CSF culture", np.nan, None, vtext="Streptococcus pneumoniae")
    res, v, fired = one(both, "E7")
    assert v and fired == ["E7_csf_culture"]
    assert not res.labels.loc["1", "E6"]                           # E6 requires "without CNS infection"


def test_blood_culture_positive_supports_csf_pleocytosis_and_commensal_does_not():
    def f(fx, p, organism):
        fx.lab(p, 1, "CSF WBC", 60, "cells/uL")
        fx.lab(p, 0, "Blood culture", np.nan, None, vtext=organism)
    assert one(lambda fx, p: f(fx, p, "Staphylococcus aureus"), "E7")[1]
    assert not one(lambda fx, p: f(fx, p, "Coagulase-negative Staphylococcus"), "E7")[1]
    assert not one(lambda fx, p: f(fx, p, "No growth at 5 days"), "E7")[1]


def test_csf_pcr_and_autoimmune_antibody():
    assert one(lambda f, p: f.lab(p, 1, "HSV 1/2 PCR, CSF", np.nan, None, vtext="Detected"), "E7")[2] == ["E7_csf_pcr"]
    assert not one(lambda f, p: f.lab(p, 1, "HSV 1/2 PCR, CSF", np.nan, None, vtext="Not detected"), "E7")[1]
    assert not one(lambda f, p: f.lab(p, 1, "HSV PCR", np.nan, None, vtext="Detected"), "E7")[1]     # specimen not CSF
    assert one(lambda f, p: f.lab(p, 100, "NMDA receptor antibody, serum", np.nan, None, vtext="Positive"),
               "E7")[2] == ["E7_autoimmune_ab"]


# ---------------------------------------------------------------------------- baseline-relative flags
def organ_flag_case(fx, p, item_rows, esrd=False):
    fx.lab(p, -5, "Blood culture")
    for d in range(4):
        fx.drug(p, -4 + 24 * d, "VANCOMYCIN 1 G IV", end_h=-3 + 24 * d, route="IV")
    for h, name, v, u in item_rows:
        fx.lab(p, h, name, v, u)
    if esrd:
        fx.cond(p, -100, "N18.6")


def test_creatinine_doubling_vs_encounter_baseline_and_esrd_exclusion():
    rows = [(-60, "CREATININE", 1.0, "mg/dL"), (-20, "CREATININE", 2.1, "mg/dL")]
    res, v, fired = one(lambda f, p: organ_flag_case(f, p, rows), "E6")
    assert v and "E6_creatinine_doubling" in fired
    assert not one(lambda f, p: organ_flag_case(f, p, [(-60, "CREATININE", 1.0, "mg/dL"),
                                                      (-20, "CREATININE", 1.9, "mg/dL")]), "E6")[1]
    assert not one(lambda f, p: organ_flag_case(f, p, rows, esrd=True), "E6")[1]                  # ESRD excluded
    # SI units: 88 -> 177 umol/L is a doubling
    assert one(lambda f, p: organ_flag_case(f, p, [(-60, "CREATININE", 88, "umol/L"),
                                                  (-20, "CREATININE", 177, "umol/L")]), "E6")[1]


def test_baseline_is_the_encounter_not_an_earlier_admission():
    def f(fx, p):
        fx.visit(p, 10, -400, -300)                      # old admission with a low creatinine
        fx.visit(p, 11, -50, None)                       # current encounter
        organ_flag_case(fx, p, [(-300, "CREATININE", 0.5, "mg/dL"), (-45, "CREATININE", 1.5, "mg/dL"),
                                (-20, "CREATININE", 2.0, "mg/dL")])
    assert not one(f, "E6")[1]                           # 2.0 vs 1.5 within the encounter: no doubling


def test_bilirubin_and_platelet_rules():
    assert one(lambda f, p: organ_flag_case(f, p, [(-60, "BILIRUBIN, TOTAL", 0.9, "mg/dL"),
                                                  (-20, "BILIRUBIN, TOTAL", 2.2, "mg/dL")]), "E6")[1]
    assert not one(lambda f, p: organ_flag_case(f, p, [(-60, "BILIRUBIN, TOTAL", 0.5, "mg/dL"),
                                                      (-20, "BILIRUBIN, TOTAL", 1.4, "mg/dL")]), "E6")[1]   # doubled but < 2.0
    assert one(lambda f, p: organ_flag_case(f, p, [(-60, "PLATELETS", 220, "K/uL"),
                                                  (-20, "PLATELETS", 90, "K/uL")]), "E6")[1]
    assert not one(lambda f, p: organ_flag_case(f, p, [(-60, "PLATELETS", 220, "K/uL"),
                                                      (-20, "PLATELETS", 120, "K/uL")]), "E6")[1]           # >= 100
    assert not one(lambda f, p: organ_flag_case(f, p, [(-60, "PLATELETS", 90, "K/uL"),
                                                      (-20, "PLATELETS", 60, "K/uL")]), "E6")[1]            # baseline < 100


# ---------------------------------------------------------------------------------------- E4a versus E4b
def test_tox_screen_positive_without_inhospital_agent_is_e4a():
    res, v, fired = one(lambda f, p: f.lab(p, -3, "Urine drug screen - opiates", np.nan, None, vtext="POSITIVE"), "E4a")
    assert v and fired == ["E4a_tox"]
    assert not res.labels.loc["1", "e4b_tox_inhospital"]
    assert not one(lambda f, p: f.lab(p, -3, "Urine drug screen - opiates", np.nan, None, vtext="Negative"), "E4a")[1]


def test_tox_screen_for_an_agent_given_in_hospital_is_e4b_not_e4a():
    def f(fx, p):
        fx.drug(p, -6, "MORPHINE 2 MG/ML INJ", route="IV")                          # given BEFORE the specimen
        fx.lab(p, -3, "Urine drug screen - opiates", np.nan, None, vtext="POSITIVE")
    res, v, fired = one(f, "E4a")
    assert not v and fired == []
    assert res.labels.loc["1", "e4b_tox_inhospital"]
    assert res.diagnostics["tox_positive_inhospital_agent"] == 1

    def g(fx, p):                                                                    # drug AFTER the specimen does not explain it
        fx.drug(p, -1, "MORPHINE 2 MG/ML INJ", route="IV")
        fx.lab(p, -3, "Urine drug screen - opiates", np.nan, None, vtext="POSITIVE")
    assert one(g, "E4a")[1]

    def h(fx, p):                                                                    # a different class was given: still E4a
        fx.drug(p, -6, "MIDAZOLAM 1 MG/ML INJ", route="IV")
        fx.lab(p, -3, "Urine drug screen - opiates", np.nan, None, vtext="POSITIVE")
    assert one(h, "E4a")[1]

    def i(fx, p):
        fx.drug(p, -6, "MIDAZOLAM 1 MG/ML INJ", route="IV")
        fx.lab(p, -3, "Benzodiazepines, urine screen", np.nan, None, vtext="Positive")
        fx.lab(p, -3, "Cocaine metabolite, urine screen", np.nan, None, vtext="Positive")
    res, v, fired = one(i, "E4a")
    assert v and fired == ["E4a_tox"]                                                # cocaine stays E4a, benzo is E4b
    assert res.labels.loc["1", "e4b_tox_inhospital"]


def test_inhospital_agent_before_encounter_start_does_not_explain_by_default():
    def f(fx, p):
        fx.visit(p, 1, -20, None)
        fx.drug(p, -60, "MORPHINE 2 MG/ML INJ", route="PO")                         # home-med style record, previous days
        fx.lab(p, -3, "Urine drug screen - opiates", np.nan, None, vtext="POSITIVE")
    assert one(f, "E4a")[1]                                                         # documented limitation (see docs)
    fx = Fx(); p = fx.person(1); f(fx, p)
    res = fx.run(config=ex.ExtractConfig(exposure_lookback_days=5))
    assert not res.labels.loc["1", "E4a"]


def test_quantitative_toxic_levels():
    assert one(lambda f, p: f.lab(p, -3, "ETHANOL, SERUM", 350, "mg/dL"), "E4a")[2] == ["E4a_ethanol"]
    assert one(lambda f, p: f.lab(p, -3, "ETHANOL, SERUM", 0.35, "%"), "E4a")[2] == ["E4a_ethanol"]
    assert not one(lambda f, p: f.lab(p, -3, "ETHANOL, SERUM", 250, "mg/dL"), "E4a")[1]
    assert one(lambda f, p: f.lab(p, -3, "ACETAMINOPHEN LEVEL", 1100, "umol/L"), "E4a")[2] == ["E4a_apap"]     # 166 ug/mL
    assert one(lambda f, p: f.lab(p, -3, "SALICYLATE LEVEL", 350, "mg/L"), "E4a")[2] == ["E4a_salicylate"]    # 35 mg/dL


def test_antidote_given_is_recorded_but_response_is_not_codable():
    res, v, fired = one(lambda f, p: f.drug(p, -2, "NALOXONE 0.4 MG/ML INJ", route="IV"), "E4a")
    assert v and fired == ["E4a_antidote"]
    ev = res.events[res.events["item"] == "antidote_response"]
    assert len(ev) == 1 and ev["source"].iloc[0] == "drug:antidote_given"                 # "given", never "responded"
    for name in ("FLUMAZENIL 0.1 MG/ML", "PHYSOSTIGMINE", "FOMEPIZOLE", "ACETYLCYSTEINE 20% IV"):
        assert one(lambda f, p, n=name: f.drug(p, -2, n, route="IV"), "E4a")[1], name
    assert not one(lambda f, p: f.drug(p, 20, "NALOXONE 0.4 MG/ML INJ", route="IV"), "E4a")[1]   # outside [-6, +6]


def test_antidote_after_inhospital_agent_is_reversal_e4b_not_e4a():
    def f(fx, p):
        fx.drug(p, -5, "FENTANYL 50 MCG/ML INJ", route="IV")
        fx.drug(p, -2, "NALOXONE 0.4 MG/ML INJ", route="IV")
    res, v, _ = one(f, "E4a")
    assert not v and res.labels.loc["1", "e4b_antidote_reversal"]

    def g(fx, p):                                                    # flumazenil after in-hospital midazolam
        fx.drug(p, -5, "MIDAZOLAM 1 MG/ML INJ", route="IV")
        fx.drug(p, -2, "FLUMAZENIL 0.1 MG/ML", route="IV")
    assert not one(g, "E4a")[1]

    def h(fx, p):                                                    # naloxone first, opioid later: not a reversal
        fx.drug(p, -2, "NALOXONE 0.4 MG/ML INJ", route="IV")
        fx.drug(p, 3, "FENTANYL 50 MCG/ML INJ", route="IV")
    assert one(h, "E4a")[1]


def test_sedative_exposure_covariate():
    res, _, _ = one(lambda f, p: f.drug(p, -10, "PROPOFOL 10 MG/ML IV EMULSION", route="IV"), "E4a")
    assert res.labels.loc["1", "e4b_sedative_exposure"]


# ------------------------------------------------------------------------------------------------- E1
def test_e1_structured_alternative_dx_and_procedure_codes():
    assert one(lambda f, p: f.cond(p, -10, "I61.9"), "E1")[2] == ["E1_dx_ich"]
    assert one(lambda f, p: f.cond(p, 20, "I63.9"), "E1")[2] == ["E1_dx_infarct"]
    assert one(lambda f, p: f.cond(p, -10, "I60.7"), "E1")[2] == ["E1_dx_sah"]
    assert one(lambda f, p: f.cond(p, -10, "S06.5X0A"), "E1")[2] == ["E1_dx_sdh_edh"]
    assert one(lambda f, p: f.cond(p, -10, "S06.350A"), "E1")[2] == ["E1_dx_tbi"]
    assert one(lambda f, p: f.proc(p, -5, "00C40ZZ"), "E1")[2] == ["E1_proc_neurosurg"]
    assert one(lambda f, p: f.proc(p, -5, "03CG3ZZ"), "E1")[1]
    assert not one(lambda f, p: f.cond(p, -10, "I62.03"), "E1")[1]                     # chronic subdural
    assert not one(lambda f, p: f.cond(p, -10, "S06.5X0S"), "E1")[1]                   # sequela
    assert not one(lambda f, p: f.cond(p, -100, "I61.9"), "E1")[1]                     # before the [-72, +24] window
    assert not one(lambda f, p: f.cond(p, 30, "I61.9"), "E1")[1]
    assert not one(lambda f, p: f.proc(p, -5, "00963ZX"), "E1")[1]                     # diagnostic CSF tap


def test_e1_edema_or_compression_only_with_a_structural_dx():
    assert not one(lambda f, p: f.cond(p, -10, "G93.6"), "E1")[1]
    assert not one(lambda f, p: f.cond(p, -10, "G93.5"), "E1")[1]
    res, v, fired = one(lambda f, p: (f.cond(p, -10, "G93.6"), f.cond(p, -12, "I63.9")), "E1")
    assert v and fired == ["E1_dx_infarct", "E1_dx_mass"]
    assert not one(lambda f, p: (f.cond(p, -10, "G93.6"), f.cond(p, -100, "I63.9")), "E1")[1]   # supporting dx outside window


def test_dx_timing_falls_back_to_visit_start_and_is_flagged_approximate():
    def f(fx, p):
        fx.visit(p, 77, -30, 100)
        fx.cond(p, None, "I61.9", visit_id=77)                                         # no condition_start_datetime
    res, v, _ = one(f, "E1")
    assert v
    row = res.events[res.events["item"] == "dx_ich"].iloc[0]
    assert row["hours_from_t0"] == pytest.approx(-30.0) and bool(row["timing_approximate"])
    res2, v2, _ = one(lambda f, p: f.cond(p, -10, "I61.9"), "E1")
    assert not bool(res2.events[res2.events["item"] == "dx_ich"].iloc[0]["timing_approximate"])
    assert res.diagnostics["dx_time_from_visit_start"] == 1


def test_imaging_flags_are_gaps_and_never_emitted():
    res, _, _ = one(lambda f, p: f.cond(p, -10, "I61.9"), "E1")
    assert any(g.startswith("imaging_ich") for g in res.gaps) and len(res.gaps) == 6
    assert not res.events["item"].str.startswith("imaging_").any()
    assert res.label_status["E1"] == "partial"


def test_extra_events_can_supply_imaging_flags_later():
    fx = Fx(); p = fx.person(1)
    fx.cond(p, -10, "I61.9")
    extra = pd.DataFrame({"case_id": ["1"], "item": ["imaging_sah"], "value": [1.0], "hours_from_t0": [-3.0],
                          "acute": [True]})
    res = ex.extract_silver(fx.tables(), fx.cohort(), extra_events=extra)
    assert res.fired["1"]["E1"] == ["E1_dx_ich", "E1_sah"]                      # our dx row survives the acuity filter


# --------------------------------------------------------------------------------- banned evidence / EEG
def test_banned_encephalopathy_codes_never_anchor():
    res, v, _ = one(lambda f, p: (f.cond(p, -5, "G93.41"), f.cond(p, -5, "G92.8"), f.cond(p, -5, "G93.40")), "E5")
    assert not res.labels.loc["1", ["E1", "E2", "E4a", "E5", "E6", "E7"]].fillna(False).any()
    assert len(res.events) == 0


def test_eeg_or_encephalopathy_wording_is_removed_from_positive_evidence():
    res, v, _ = one(lambda f, p: f.obs(p, -5, "EEG: burst suppression after cardiac arrest"), "E2")
    assert not v and res.diagnostics.get("banned_dropped_eeg_derived") == 1
    res, v, _ = one(lambda f, p: f.obs(p, -5, "toxic metabolic encephalopathy, cardiac arrest"), "E2")
    assert not v and res.diagnostics.get("banned_dropped_nonspecific_dx") == 1
    assert one(lambda f, p: f.obs(p, -5, "Cardiac arrest, ROSC achieved"), "E2")[2] == ["E2_arrest"]
    assert not one(lambda f, p: f.obs(p, -5, "Cardiac arrest", value="No"), "E2")[1]                 # negated observation


# --------------------------------------------------------------------------------- reports_findings
def rf_frame(pid, flags):
    d = {"BDSPPatientID": str(pid), "SessionID": f"sess{pid}", "StartTime(EEG)": T0}
    d.update({f: None for f in schema.FINDING_FLAGS_ALL})
    d.update({f: "present" for f in flags})
    return d


def test_reports_findings_never_produce_a_positive():
    fx = Fx()
    p = fx.person(1)
    fx.lab(p, -2, "GLUCOSE", 90, "mg/dL")                                              # nothing anchors
    base = fx.run()
    fx.rows["rf"].append(rf_frame(1, ["bs", "gen slowing", "foc slowing", "gpd", "lpd", "low voltage", "seizure",
                                      "status", "abnormal", "grda", "lrda"]))
    res = ex.extract_silver(fx.tables(with_rf=True), fx.cohort())
    pd.testing.assert_frame_equal(base.labels, res.labels)
    assert not res.labels.loc["1", ["E1", "E2", "E4a", "E5", "E6", "E7"]].fillna(False).any()
    pd.testing.assert_frame_equal(base.events, res.events)                              # events identical with / without the EEG flags
    assert res.diagnostics["forbidden_tables_ignored"] == 1
    assert "reports_findings" in ex.FORBIDDEN_TABLES and "reports_findings" not in ex.STRUCTURED_TABLES
    assert "omop_note" in ex.FORBIDDEN_TABLES and "omop_note_nlp" in ex.FORBIDDEN_TABLES


def test_notes_and_imaging_tables_are_ignored_too():
    fx = Fx(); p = fx.person(1)
    t = fx.tables()
    t["omop_note"] = pd.DataFrame({"note_id": [1], "person_id": [1], "note_text": ["cardiac arrest, intracerebral hemorrhage"]})
    t["imaging"] = pd.DataFrame({"person_id": [1], "modality": ["CT head"], "study_datetime": [at(-3)]})
    res = ex.extract_silver(t, fx.cohort())
    assert not res.labels.loc["1", ["E1", "E2"]].fillna(False).any()
    assert res.diagnostics["forbidden_tables_ignored"] == 2


def test_eeg_comparator_is_separate_and_feeds_the_circularity_audit():
    fx = Fx()
    for i in range(1, 41):
        fx.person(i)
        fx.rf = None
        fx.rows["rf"].append(rf_frame(i, ["foc slowing"] if i % 2 else ["bs"]))
    fx.pids.append((99, "S0001"))                                                       # a case without any report
    cohort = fx.cohort()
    rf = pd.DataFrame(fx.rows["rf"])
    comp = ex.eeg_impression_comparator(rf, cohort)
    assert set(comp["label"]) == {"E1", "E2", "E5"} and set(comp.columns) == {"case_id", "label", "eeg_impr"}
    e1 = comp[comp["label"] == "E1"].set_index("case_id")["eeg_impr"]
    assert e1["1"] == 1.0 and e1["2"] == 0.0 and np.isnan(e1["99"])                     # no report -> NaN, not 0
    e2 = comp[comp["label"] == "E2"].set_index("case_id")["eeg_impr"]
    assert e2["2"] == 1.0 and e2["1"] == 0.0
    # plugs into the audit schema (pred / gold supplied by the caller)
    rng = np.random.default_rng(0)
    long = comp[comp["case_id"] != "99"].assign(pred=lambda d: rng.random(len(d)), gold=lambda d: (rng.random(len(d)) > .5).astype(int))
    out = run_circularity_audit(long, ["E1", "E2"], n_boot=20)
    assert set(out["per_label"]) == {"E1", "E2"}
    # session-less matching by start time works as well
    comp2 = ex.eeg_impression_comparator(rf.drop(columns="SessionID"), cohort)
    assert comp2[comp2["label"] == "E1"].set_index("case_id")["eeg_impr"]["1"] == 1.0


# ------------------------------------------------------------------------------------ cohort and cases
def test_cohort_validation_and_multiple_cases_per_person():
    with pytest.raises(ValueError):
        ex.prepare_cohort(pd.DataFrame({"person_id": [1]}))
    with pytest.raises(ValueError):
        ex.prepare_cohort(pd.DataFrame({"person_id": [1, 1], "t0": [T0, T0]}))
    fx = Fx(); fx.person(1)
    fx.lab(1, -2, "GLUCOSE", 30, "mg/dL")
    c = pd.DataFrame({"person_id": [1, 1], "case_id": ["a", "b"], "t0": [T0, T0 + pd.Timedelta(days=30)],
                      "SiteID": ["S0001", "S0001"]})
    res = ex.extract_silver(fx.tables(), c)
    assert bool(res.labels.loc["a", "E5"]) and not bool(res.labels.loc["b", "E5"])


def test_patients_outside_the_cohort_and_unmatched_rows_are_ignored():
    fx = Fx(); fx.person(1)
    fx.lab(1, -2, "GLUCOSE", 30, "mg/dL")
    fx.lab(2, -2, "GLUCOSE", 30, "mg/dL")                                              # not in the cohort
    fx.lab(1, -2, "SOME RANDOM TEST", 30, "mg/dL")
    res = fx.run()
    assert list(res.labels.index) == ["1"] and len(res.events) == 1


def test_empty_tables_give_all_false_labels():
    fx = Fx(); fx.person(1)
    res = fx.run()
    assert res.labels.loc["1", ["E2", "E4a", "E5", "E6", "E7"]].eq(False).all()
    assert len(res.events) == 0


# ---------------------------------------------------------------------------- streaming path and report
def write_omop(fx, root):
    fx_tables = fx.tables()
    for name, df in fx_tables.items():
        d = root / "OMOP" / "Merged" / name[len("omop_"):]
        d.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S").where(out[c].notna(), None)
        if name == "omop_concept" and out.empty:
            out = out.astype({"concept_id": "int64"})
        out.to_parquet(d / "part-00000.parquet", index=False)


def many_patients_fixture(n=30):
    fx = Fx()
    for i in range(1, n + 1):
        site = "S0001" if i <= 20 else "I0003"
        fx.person(i, site)
        fx.lab(i, -2, "GLUCOSE", 30 if i % 2 else 100, "mg/dL")
        if i % 3 == 0:
            fx.cond(i, -5, "I46.9")
        if i % 5 == 0:
            fx.cond(i, -5, "I61.9")
        fx.visit(i, 1000 + i, -30, None)
    fx.concept(9001, "Ammonia", "Measurement", "LOINC", "16362-6")
    fx.lab(1, -3, "mystery", 200, "ug/dL", cid=9001)
    return fx


def test_streaming_from_a_local_store_equals_in_memory(tmp_path):
    fx = many_patients_fixture()
    write_omop(fx, tmp_path)
    store = data_io.LocalStore(tmp_path)
    streamed = ex.extract_silver(store, fx.cohort())
    mem = fx.run()
    pd.testing.assert_frame_equal(streamed.labels, mem.labels)
    assert len(streamed.events) == len(mem.events)
    assert streamed.labels.loc["1", "E5"] and "E5_ammonia" in streamed.fired["1"]["E5"]       # concept route through omop_concept


def test_streaming_reads_only_structured_tables_with_pruned_columns(tmp_path):
    fx = many_patients_fixture()
    write_omop(fx, tmp_path)
    for forbidden in ("reports_findings", "note", "note_nlp"):                                # decoys that must not be touched
        d = tmp_path / "OMOP" / "Merged" / forbidden
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"person_id": [1], "note_text": ["secret"]}).to_parquet(d / "part-00000.parquet")

    class Spy(data_io.LocalStore):
        keys: list = []
        def get_object(self, Bucket=None, Key="", Range=None):
            Spy.keys.append(Key)
            return super().get_object(Bucket, Key, Range)
    Spy.keys = []
    ex.extract_silver(Spy(tmp_path), fx.cohort())
    touched = {k.split("/")[2] for k in Spy.keys if k.startswith("OMOP/")}
    assert touched <= {"concept", "visit_occurrence", "measurement", "drug_exposure", "condition_occurrence",
                       "procedure_occurrence", "observation"}
    assert not any("reports_findings" in k or "/note" in k for k in Spy.keys)
    assert not any(k.startswith("EEG/") for k in Spy.keys)


def test_requested_columns_exclude_free_text_note_fields():
    assert not any("note_text" in c for cols in ex.COLUMNS.values() for c in cols)


def test_silver_report_is_aggregate_only_with_small_cell_suppression():
    res = many_patients_fixture(30).run()
    rep = ex.silver_report(res)
    assert_aggregate_only(rep, known_ids=set(res.labels.index))
    # I0003 has 10 (< 11): complementary suppression also hides S0001 (20), so 10 cannot be recovered as 30 - 20
    assert rep["n_cases"] == 30 and rep["n_cases_per_site"] == {"I0003": SUPPRESSED, "S0001": SUPPRESSED}
    e5 = rep["per_label"]["E5"]
    assert e5["n_positive"] == 15 and e5["per_site"]["I0003"]["n_positive"] == SUPPRESSED
    e2 = rep["per_label"]["E2"]
    assert e2["n_positive"] == SUPPRESSED and e2["prevalence"] == SUPPRESSED                         # 10 arrests: below 11
    assert rep["anchor_firing"]["E5"]["E5_glucose_lo"]["n_cases"] == 15
    assert rep["anchor_firing"]["E2"]["E2_arrest"]["n_cases"] == SUPPRESSED                         # 10 < 11
    assert rep["label_status"]["E1"] == "partial" and rep["anchor_status"]["E1_ich"] == "unavailable"
    assert any("imaging_ich" in g for g in rep["gaps"])
    for lab in ("E1", "E2", "E4a", "E5", "E6", "E7"):
        assert lab in rep["per_label"]
    text = repr(rep)
    assert "50000" not in text and "T12:00" not in text


def test_complementary_suppression_hides_the_second_cell():
    assert ex._suppress_group({"a": 5, "b": 40, "c": 60}, 105) == {"a": SUPPRESSED, "b": SUPPRESSED, "c": 60}
    assert ex._suppress_group({"a": 5, "b": 3, "c": 60}, 68) == {"a": SUPPRESSED, "b": SUPPRESSED, "c": 60}
    assert ex._suppress_group({"a": 20, "b": 40}, 60) == {"a": 20, "b": 40}


def test_report_refuses_identifiers():
    with pytest.raises(AggregateOnlyError):
        assert_aggregate_only({"case": "sub-S000150000001"})


def test_report_marks_unavailable_labels_when_nothing_is_computable(monkeypatch):
    res = many_patients_fixture(30).run()
    res.label_status["E1"] = "unavailable"
    res.labels["E1"] = pd.array([pd.NA] * len(res.labels), dtype="boolean")
    assert ex.silver_report(res)["per_label"]["E1"] == "unavailable from structured data"


def test_cli_refuses_labels_outside_local_only(tmp_path):
    fx = many_patients_fixture(12)
    write_omop(fx, tmp_path)
    fx.cohort().to_csv(tmp_path / "cohort.csv", index=False)
    with pytest.raises(SystemExit):
        ex.main(["--data", str(tmp_path), "--cohort", str(tmp_path / "cohort.csv"), "--labels-out",
                 str(tmp_path / "labels.csv")])
    rc = ex.main(["--data", str(tmp_path), "--cohort", str(tmp_path / "cohort.csv"), "--out", str(tmp_path / "rep"),
                  "--labels-out", str(tmp_path / "local_only" / "labels.csv")])
    assert rc == 0 and (tmp_path / "rep" / "silver_report.json").exists()
    assert oct((tmp_path / "local_only" / "labels.csv").stat().st_mode & 0o777) == "0o600"


# ----------------------------------------------------------------------------------- generator smoke
def test_runs_on_the_synthetic_generator_output():
    from sortinghat.synthetic import generate
    tables, _ = generate(300, seed=3)
    first = tables["omop_measurement"].dropna(subset=["measurement_datetime"]).groupby("person_id")["measurement_datetime"].min()
    cohort = pd.DataFrame({"person_id": first.index.astype("int64"), "t0": first.values + np.timedelta64(6, "h")})
    cohort["SiteID"] = "S0001"
    res = ex.extract_silver(tables, cohort)                                  # reports_findings / imaging present, ignored
    assert len(res.labels) == len(cohort) and res.diagnostics["forbidden_tables_ignored"] >= 2
    assert len(res.events) > 0                                                # generator labs carry sodium/glucose/ammonia ...
    assert list(res.events.columns) == ex.EVENT_FRAME_COLUMNS
    assert set(EVENT_COLUMNS) <= set(res.events.columns)
    rep = ex.silver_report(res)
    assert_aggregate_only(rep)
