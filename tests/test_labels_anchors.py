import pandas as pd
import pytest

from sortinghat.labels.anchors import (EVENT_COLUMNS, evaluate_case, is_proposed, is_signed_off, load_anchor_config,
                                        loinc_to_item, silver_anchor_table)
from sortinghat.labels.ontology import SILVER_CIRCULARITY_LABELS


def ev(*rows):
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def test_config_covers_labels_and_is_signed_off():
    cfg = load_anchor_config()
    assert is_signed_off(cfg) and not is_proposed(cfg)
    assert cfg["status"].startswith("Evidence-reviewed 2026-10-08") and "external co-I review before publication" in cfg["status"]
    assert set(SILVER_CIRCULARITY_LABELS) - {"E7"} <= set(cfg["labels"]) and "E7" in cfg["labels"]
    assert set(cfg["labels"]) == set(SILVER_CIRCULARITY_LABELS)
    assert loinc_to_item(cfg)["2345-7"] == "glucose"


def one(rows, label):
    return evaluate_case(ev(*[("c",) + r for r in rows]), [label])[label]


@pytest.mark.parametrize("item,val,h,hit", [
    ("glucose", 49, -1, True), ("glucose", 50, -1, False), ("glucose", 601, -1, True), ("glucose", 600, -1, False),
    ("sodium", 119, 0, True), ("sodium", 120, 0, False), ("sodium", 161, 0, True), ("sodium", 160, 0, False),
    ("paco2", 71, 0, False),            # PaCO2 alone no longer anchors (needs acidemia)
    ("creatinine", 8.0, 0, False),      # E5_creatinine removed (stable ESRD)
    ("glucose", 30, 2, False), ("glucose", 30, 1, True),       # glucose_lo window ends +1 h
    ("ammonia", 150, -2, True), ("bun", 120, -2, True),
    ("glucose", 30, -100, False),      # outside timing window
    ("glucose", 30, 48, False),        # outside timing window (after)
])
def test_e5_thresholds_and_windows(item, val, h, hit):
    assert bool(one([(item, val, h)], "E5")) is hit


def test_e1_imaging_and_e2_arrest_windows():
    assert one([("imaging_ich", 1, 2)], "E1") == ["E1_ich"]
    assert not one([("imaging_ich", 1, 200)], "E1")
    assert one([("arrest_event", 1, -30)], "E2") == ["E2_arrest"]
    assert not one([("arrest_event", 1, 5)], "E2")        # arrest after t0 is not an anchor


def test_e4a_antidote_and_tox():
    assert one([("antidote_response", 1, -1)], "E4a")
    assert one([("ethanol_serum", 350, -3)], "E4a")
    assert not one([("ethanol_serum", 200, -3)], "E4a")


def test_e5_paco2_requires_acidemia_in_window():
    assert one([("paco2", 75, -2), ("ph_arterial", 7.25, -2)], "E5") == ["E5_paco2", "E5_paco2_hi", "E5_paco2_ph"]
    assert not one([("paco2", 75, -2), ("ph_arterial", 7.30, -2)], "E5")      # pH not < 7.30
    assert not one([("paco2", 70, -2), ("ph_arterial", 7.25, -2)], "E5")      # PaCO2 not > 70
    assert not one([("paco2", 75, -2), ("ph_arterial", 7.25, 8)], "E5")       # pH outside window
    assert not one([("paco2", 75, -20), ("ph_arterial", 7.25, -2)], "E5")     # PaCO2 outside window


E6_BASE = [("blood_culture_drawn", 1, -5), ("qad_ge4", 1, -4)]


@pytest.mark.parametrize("organ", [
    ("vasopressor_initiation", 1, -2), ("lactate", 2.0, -2), ("creatinine_doubling", 1, 0),
    ("bilirubin_doubling_ge2", 1, 0), ("platelets_drop", 1, 3)])
def test_e6_cdc_ase_each_organ_dysfunction(organ):
    assert one(E6_BASE + [organ], "E6")


def test_e6_requires_culture_qad_and_organ_dysfunction_in_windows():
    organ = ("lactate", 3.1, -2)
    assert not one(E6_BASE + [("lactate", 1.9, -2)], "E6")             # lactate below 2.0
    assert not one([E6_BASE[1], organ], "E6")                          # no culture
    assert not one([E6_BASE[0], organ], "E6")                          # no >=4 QAD
    assert not one(E6_BASE, "E6")                                      # no organ dysfunction
    assert not one(E6_BASE + [("lactate", 3.1, -60)], "E6")            # organ window is [-48, 24]
    assert not one(E6_BASE + [("lactate", 3.1, 30)], "E6")
    assert not one([("blood_culture_drawn", 1, -80), E6_BASE[1], organ], "E6")   # culture outside [-72, 24]
    assert one(E6_BASE + [("lactate", 3.1, -48)], "E6")
    fired = one(E6_BASE + [organ], "E6")
    assert fired == ["E6_blood_culture", "E6_lactate", "E6_qad"]


def test_e6_does_not_use_sirs_or_new_ventilation_or_old_items():
    cfg = load_anchor_config()
    assert "suspected_infection" not in cfg["items"]
    assert not one(E6_BASE + [("temp_c", 39, -3), ("heart_rate", 110, -3), ("resp_rate", 30, -3),
                              ("wbc", 15, -3)], "E6")
    assert not one(E6_BASE + [("mechanical_ventilation", 1, -3), ("blood_culture_pos", 1, -3)], "E6")


def test_e6_excluded_when_e7_fires():
    rows = E6_BASE + [("lactate", 3.1, -2), ("csf_culture_pos", 1, 4)]
    res = evaluate_case(ev(*[("c",) + r for r in rows]))
    assert res["E6"] == [] and res["E7"] == ["E7_csf_culture"]


def test_e2_shock_window_and_flag():
    assert one([("profound_shock", 1, -48)], "E2") == ["E2_shock"]
    assert not one([("profound_shock", 1, -49)], "E2")
    assert not one([("profound_shock", 1, 1)], "E2")


def test_config_flag_items_and_loincs():
    cfg = load_anchor_config()
    for k in ("blood_culture_drawn", "qad_ge4", "vasopressor_initiation", "creatinine_doubling",
              "bilirubin_doubling_ge2", "platelets_drop", "csf_rbc"):
        assert k in cfg["items"]
        assert cfg["items"][k]["loinc"] == [] or k == "csf_rbc"
    assert loinc_to_item(cfg)["1975-2"] == "bilirubin" and loinc_to_item(cfg)["777-3"] == "platelets"
    assert cfg["labels"]["E1"]["acuity_required"] is True


def acute_ev(rows):
    return pd.DataFrame([("c",) + r for r in rows], columns=list(EVENT_COLUMNS) + ["acute"])


def test_e1_acuity_filter_when_column_present():
    assert evaluate_case(acute_ev([("imaging_ich", 1, 2, True)]), ["E1"])["E1"] == ["E1_ich"]
    assert evaluate_case(acute_ev([("imaging_ich", 1, 2, False)]), ["E1"])["E1"] == []     # chronic
    assert evaluate_case(acute_ev([("imaging_ich", 1, 2, None)]), ["E1"])["E1"] == []      # unknown acuity
    mixed = acute_ev([("imaging_ich", 1, 2, False), ("imaging_sah", 1, 2, True)])
    assert evaluate_case(mixed, ["E1"])["E1"] == ["E1_sah"]
    # acuity is not applied to labels without acuity_required
    assert evaluate_case(acute_ev([("glucose", 30, 0, False)]), ["E5"])["E5"] == ["E5_glucose_lo"]
    # without the column the legacy behaviour holds
    assert one([("imaging_ich", 1, 2)], "E1") == ["E1_ich"]
    t = silver_anchor_table(acute_ev([("imaging_ich", 1, 2, False)]))
    assert not t.loc["c", "E1"]


def test_e7_pleocytosis_needs_support():
    assert not one([("csf_wbc", 50, 0)], "E7")
    assert one([("csf_wbc", 50, 0), ("csf_protein", 150, 0)], "E7")
    assert one([("csf_pcr_pos", 1, 10)], "E7")


SUPPORT = ("csf_protein", 150, 0)


def test_e7_rbc_correction_same_tap():
    # 520 RBC/500 = 1.04 -> corrected 19.96 < 20: no pleocytosis anchor
    assert not one([("csf_wbc", 21, 0), ("csf_rbc", 520, 0), SUPPORT], "E7")
    assert one([("csf_wbc", 21, 0), ("csf_rbc", 500, 0), SUPPORT], "E7")           # corrected exactly 20
    assert one([("csf_wbc", 200, 0), ("csf_rbc", 5000, 0), SUPPORT], "E7")         # corrected 190
    assert not one([("csf_wbc", 200, 0), ("csf_rbc", 99500, 0), SUPPORT], "E7")    # corrected ~1
    assert not one([("csf_wbc", 25, 0), ("csf_rbc", 5000, 0), SUPPORT], "E7")      # corrected 15


def test_e7_rbc_from_different_tap_or_absent_is_uncorrected():
    assert one([("csf_wbc", 21, 0), ("csf_rbc", 5000, 6), SUPPORT], "E7")          # other tap: uncorrected
    assert one([("csf_wbc", 21, 0), SUPPORT], "E7")                                # no RBC: uncorrected


def test_table_and_missing_case():
    e = ev(("a", "imaging_sah", 1, 0), ("b", "glucose", 30, 0))
    t = silver_anchor_table(e, case_ids=["a", "b", "z"])
    assert t.loc["a", "E1"] and t.loc["b", "E5"] and not t.loc["z"].any()
    assert t.attrs["fired"]["a"]["E1"] == ["E1_sah"]
    with pytest.raises(ValueError):
        silver_anchor_table(e.drop(columns="value"))
