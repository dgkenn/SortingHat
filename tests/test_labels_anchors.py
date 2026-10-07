import pandas as pd
import pytest

from sortinghat.labels.anchors import (EVENT_COLUMNS, evaluate_case, is_proposed, load_anchor_config,
                                        loinc_to_item, silver_anchor_table)
from sortinghat.labels.ontology import SILVER_CIRCULARITY_LABELS


def ev(*rows):
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def test_config_covers_labels_and_is_marked_proposed():
    cfg = load_anchor_config()
    assert is_proposed(cfg)
    assert set(SILVER_CIRCULARITY_LABELS) - {"E7"} <= set(cfg["labels"]) and "E7" in cfg["labels"]
    assert set(cfg["labels"]) == set(SILVER_CIRCULARITY_LABELS)
    assert loinc_to_item(cfg)["2345-7"] == "glucose"


def one(rows, label):
    return evaluate_case(ev(*[("c",) + r for r in rows]), [label])[label]


@pytest.mark.parametrize("item,val,h,hit", [
    ("glucose", 49, -1, True), ("glucose", 50, -1, False), ("glucose", 601, -1, True), ("glucose", 600, -1, False),
    ("sodium", 119, 0, True), ("sodium", 120, 0, False), ("sodium", 161, 0, True), ("sodium", 160, 0, False),
    ("paco2", 71, 0, True), ("paco2", 70, 0, False), ("ammonia", 150, -2, True), ("bun", 120, -2, True),
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


def test_e6_requires_infection_plus_dysfunction_and_excludes_cns():
    base = [("suspected_infection", 1, -5), ("lactate", 3.1, -2)]
    assert one(base, "E6")
    assert not one([("lactate", 3.1, -2)], "E6")                       # no suspected infection
    assert not one([("suspected_infection", 1, -5)], "E6")             # no organ dysfunction
    sirs = [("suspected_infection", 1, -5), ("temp_c", 39, -3), ("heart_rate", 110, -3)]
    assert "E6_sirs2" in one(sirs, "E6")
    assert not one([("suspected_infection", 1, -5), ("temp_c", 39, -3)], "E6")   # only 1 SIRS
    res = evaluate_case(ev(*[("c",) + r for r in base + [("csf_culture_pos", 1, 4)]]))
    assert res["E6"] == [] and res["E7"] == ["E7_csf_culture"]


def test_e7_pleocytosis_needs_support():
    assert not one([("csf_wbc", 50, 0)], "E7")
    assert one([("csf_wbc", 50, 0), ("csf_protein", 150, 0)], "E7")
    assert one([("csf_pcr_pos", 1, 10)], "E7")


def test_table_and_missing_case():
    e = ev(("a", "imaging_sah", 1, 0), ("b", "glucose", 30, 0))
    t = silver_anchor_table(e, case_ids=["a", "b", "z"])
    assert t.loc["a", "E1"] and t.loc["b", "E5"] and not t.loc["z"].any()
    assert t.attrs["fired"]["a"]["E1"] == ["E1_sah"]
    with pytest.raises(ValueError):
        silver_anchor_table(e.drop(columns="value"))
