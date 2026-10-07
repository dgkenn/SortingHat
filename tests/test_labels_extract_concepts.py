"""Concept map, unit conversion and result semantics of the structured silver-label extractor (synthetic, no data)."""
import numpy as np
import pandas as pd
import pytest

from sortinghat.labels.anchors import load_anchor_config, loinc_to_item
from sortinghat.labels.concepts import (ConceptIndex, ConceptMap, anchor_item_status, anchor_leaf_status, clean_unit,
                                         label_status, load_concept_map, normalize_code, normalize_source_code,
                                         validate_concept_map)


@pytest.fixture(scope="module")
def cm():
    return ConceptMap()


def test_concept_map_is_consistent_with_signed_off_anchor_config(cm):
    assert validate_concept_map(cm) == []
    cfg = load_anchor_config()
    raw = load_concept_map()
    covered = set(raw["measurement_items"]) | set(raw["qualitative_items"]) | set(raw["code_event_items"]) \
        | set(raw["derived_items"]) | set(raw["unavailable_items"])
    assert set(cfg["items"]) <= covered
    for code, item in loinc_to_item(cfg).items():            # every LOINC of silver_anchors.yaml is mapped to the same item
        assert any(c == code for (_, c, _) in cm.specs[item].codes)


def test_imaging_items_are_recorded_gaps_and_e1_is_anchored_on_codes(cm):
    st = anchor_item_status(cm)
    for it in ("imaging_ich", "imaging_sah", "imaging_sdh_edh", "imaging_infarct", "imaging_tbi_contusion",
               "imaging_mass_effect"):
        assert st[it] == "unavailable"
    assert st["antidote_response"] == "proxy" and st["glucose"] == "full"
    assert st["dx_ich"] == "proxy" and st["proc_neurosurgical"] == "proxy"
    ls = label_status(cm)
    assert ls["E1"] == "partial" and ls["E5"] == "full"           # E1 survives only through the dx/procedure alternative
    leaf = anchor_leaf_status(cm)
    assert leaf["E1_ich"] == "unavailable" and leaf["E1_dx_ich"] == "proxy" and leaf["E5_bun"] == "full"


def test_imaging_only_e1_would_be_unavailable(cm):
    cfg = load_anchor_config()
    import copy
    c2 = copy.deepcopy(cfg)
    c2["labels"]["E1"]["any_of"] = [n for n in c2["labels"]["E1"]["any_of"] if n["item"].startswith("imaging_")]
    assert label_status(cm, c2)["E1"] == "unavailable"


@pytest.mark.parametrize("item,value,unit,expect", [
    ("ammonia", 100, "umol/L", 100.0),
    ("ammonia", 100, "ug/dL", 58.718),            # NH3 17.031 g/mol
    ("ammonia", 100, "µg/dL", 58.718),       # micro sign
    ("ammonia", 100, "mcg/dL", 58.718),
    ("glucose", 5.5, "mmol/L", 99.088),
    ("glucose", 90, "mg/dL", 90.0),
    ("glucose", 1.0, "g/L", 100.0),
    ("temp_c", 98.6, "degF", 37.0),
    ("temp_c", 98.6, "°F", 37.0),
    ("temp_c", 37.2, "Cel", 37.2),
    ("bun", 10, "mmol/L", 28.014),
    ("creatinine", 177, "umol/L", 2.0),
    ("calcium", 3.5, "mmol/L", 14.028),
    ("lactate", 18.016, "mg/dL", 2.0),
    ("bilirubin", 34.2, "umol/L", 2.0),
    ("platelets", 150000, "/uL", 150.0),
    ("platelets", 150, "K/uL", 150.0),
    ("platelets", 150, "10*9/L", 150.0),
    ("paco2", 9.33, "kPa", 70.0),
    ("ethanol_serum", 0.3, "%", 300.0),
    ("ethanol_serum", 65, "mmol/L", 299.46),
    ("acetaminophen", 1000, "umol/L", 151.16),
    ("salicylate", 300, "mg/L", 30.0),
    ("sodium", 150, "mEq/L", 150.0),
    ("csf_protein", 1.5, "g/L", 150.0),
    ("csf_wbc", 0.05, "10*3/uL", 50.0),
])
def test_unit_conversions(cm, item, value, unit, expect):
    out, st = cm.convert_value(item, value, unit)
    assert st == "ok"
    assert out == pytest.approx(expect, rel=2e-3, abs=2e-3)


def test_unknown_missing_and_implausible_units_drop_the_row(cm):
    assert cm.convert_value("ammonia", 100, "furlongs")[1] == "unit_unrecognised"
    assert cm.convert_value("glucose", 5.5, None)[1] == "unit_missing"          # mmol/L vs mg/dL cannot be guessed
    assert cm.convert_value("glucose", 5.5, "")[1] == "unit_missing"
    assert cm.convert_value("glucose", 5.5, None, default_unit="mmol/L")[0] == pytest.approx(99.088, rel=1e-3)
    assert cm.convert_value("ph_arterial", 7.2, None) == (7.2, "ok")             # pH is unitless
    assert cm.convert_value("ph_arterial", 72, "pH")[1] == "implausible"
    assert cm.convert_value("glucose", 0.5, "mg/dL")[1] == "implausible"
    assert np.isnan(cm.convert_value("sodium", 999, "mmol/L")[0])


def test_unit_normalisation_and_codes():
    assert clean_unit("µmol/L") == "umol/l" and clean_unit(" 10^3/uL ") == "10*3/ul" and clean_unit(None) == ""
    assert normalize_code("ICD10CM", "I46.9") == "I469" and normalize_code("LOINC", " 2345-7 ") == "2345-7"
    assert normalize_source_code("ICD-10-CM: I46.9") == "I469" and normalize_source_code("ICD10:G93.41") == "G9341"


def test_name_routes_respect_specimen_excludes(cm):
    assert cm.classify_measurement_name("CSF GLUCOSE") == ("csf_glucose",)
    assert cm.classify_measurement_name("GLUCOSE, POC") == ("glucose",)
    assert cm.classify_measurement_name("Glucose, Urine") == ()
    assert cm.classify_measurement_name("Calcium, ionized") == ()
    assert cm.classify_measurement_name("Bilirubin, direct") == ()
    assert cm.classify_measurement_name("Creatinine, Urine") == ()
    assert cm.classify_measurement_name("pH, venous") == ()
    assert cm.classify_measurement_name("Arterial pH") == ("ph_arterial",)
    assert set(cm.classify_measurement_name("Blood Culture")) == {"blood_culture_drawn", "blood_culture_pos"}
    assert cm.classify_measurement_name("AMMONIA, PLASMA") == ("ammonia",)
    assert cm.classify_measurement_name("CSF WBC") == ("csf_wbc",)
    assert cm.classify_measurement_name("WBC") == ("wbc",)


def test_tox_agents_need_screen_context_and_skip_history(cm):
    assert cm.classify_tox_name("Urine Drug Screen - Opiates") == ("opiates",)
    assert cm.classify_tox_name("Benzodiazepine screen, urine") == ("benzodiazepines",)
    assert cm.classify_tox_name("History of opioid use") == ()
    assert cm.classify_tox_name("fentanyl") == ()                      # no screen / level context
    assert cm.classify_tox_name("Fentanyl level, serum") == ("fentanyl",)


@pytest.mark.parametrize("item,text,expect", [
    ("blood_culture_pos", "Staphylococcus aureus", "positive"),
    ("blood_culture_pos", "No growth at 5 days", "negative"),
    ("blood_culture_pos", "Gram negative rods", "positive"),            # 'negative' inside a positive result
    ("blood_culture_pos", "Coagulase-negative Staphylococcus", "commensal"),
    ("blood_culture_pos", "Preliminary: pending", "pending"),
    ("csf_pcr_pos", "Not detected", "negative"),
    ("csf_pcr_pos", "Detected", "positive"),
    ("csf_pcr_pos", "weird", "unknown"),
    ("tox:opiates", "POSITIVE", "positive"),
    ("tox:opiates", "Negative", "negative"),
])
def test_result_status(cm, item, text, expect):
    assert cm.result_status(item, text) == expect


def test_result_status_uses_concept_name_then_number(cm):
    assert cm.result_status("csf_pcr_pos", None, "Positive") == "positive"
    assert cm.result_status("csf_pcr_pos", "", None, 1500.0) == "positive"
    assert cm.result_status("csf_pcr_pos", "", None, 0.0) == "negative"
    assert cm.result_status("blood_culture_pos", "", None, 5.0) == "unknown"        # numbers do not make a culture positive
    assert cm.result_status("tox:cocaine", None, None, 120.0) == "positive"


def test_condition_and_procedure_code_hits(cm):
    h = cm.condition_source_hits
    assert h("I46.9") == ("arrest_event",) and h("I469") == ("arrest_event",) and h("4275") == ("arrest_event",)
    assert h("Z86.74") == () and h("G93.41") == () and h("E11.9") == ()
    assert h("T71.162A") == ("asphyxia_event",) and h("T58.01XA") == ("asphyxia_event",)
    assert h("I61.9") == ("dx_ich",) and h("I60.7") == ("dx_sah",) and h("S06.6X0A") == ("dx_sah",)
    assert h("I62.01") == ("dx_sdh_edh",) and h("I62.03") == ()                    # chronic subdural is not acute
    assert h("I62.9") == ()                                                        # unspecified intracranial hemorrhage not listed
    assert h("S06.5X0A") == ("dx_sdh_edh",) and h("S06.5X0S") == ()                # sequela excluded
    assert h("S06.4X1A") == ("dx_sdh_edh",) and h("S06.2X0A") == ("dx_tbi",) and h("S06.350A") == ("dx_tbi",)
    assert h("S06.0X0A") == () and h("S06.1X0A") == ()                             # concussion / traumatic edema alone: not listed
    assert h("I63.9") == ("dx_infarct",) and h("I63.411") == ("dx_infarct",) and h("I69.351") == ()
    assert h("G93.5") == ("dx_mass_effect",) and h("G93.6") == ("dx_mass_effect",) and h("G93.40") == ()
    assert h("N18.6") == ("esrd",) and h("Z99.2") == ("esrd",)
    p = cm.procedure_source_hits
    assert p("92950") == ("arrest_event",) and p("5A12012") == ("arrest_event",)
    assert p("00C40ZZ") == ("proc_neurosurgical",) and p("0NB00ZZ") == ("proc_neurosurgical",)
    assert p("009630Z") == ("proc_neurosurgical",) and p("00963ZX") == ()               # diagnostic CSF tap is not an EVD
    assert p("03CG3ZZ") == ("proc_neurosurgical",) and p("61312") == ("proc_neurosurgical",)
    assert p("99213") == ()


def test_arrest_wording_excludes_history(cm):
    assert cm.text_hits("observation", "Cardiac arrest, ROSC achieved") == ("arrest_event",)
    assert cm.text_hits("observation", "History of cardiac arrest") == ()
    assert cm.text_hits("observation", "family history cpr") == ()


def test_drug_name_hits_group_and_exclude(cm):
    d = cm.drug_text_hits
    assert d("NOREPINEPHRINE 4 MG/250 ML IV") == (("vasopressor", "norepinephrine"),)
    assert d("EPINEPHRINE 1 MG/ML INJ") == (("vasopressor", "epinephrine"),)
    assert d("LIDOCAINE 1% WITH EPINEPHRINE") == ()                              # local anaesthetic combination
    assert d("Naloxone 0.4 mg/mL inj") == (("antidote", "naloxone"),)
    assert ("antimicrobial", "piperacillin_tazobactam") in d("Zosyn 3.375 g IV PB")
    assert ("other", "morphine") in d("MORPHINE 2 MG/ML INJ")
    assert d("midazolam 1 mg/mL") == (("sedative", "midazolam"),)
    assert d("midodrine 5 mg tablet") == ()
    assert cm.abx_route_ok("vancomycin", "IV", "vancomycin") and not cm.abx_route_ok("vancomycin", "PO", "vancomycin")
    assert cm.abx_route_ok("metronidazole", "PO", "metronidazole")                  # any-route agent
    assert cm.abx_route_ok("cefepime", None, "CEFEPIME 2 G IV PIGGYBACK")
    assert not cm.abx_route_ok("cefepime", None, "CEFEPIME 2 G TABLET")


def test_concept_index_resolves_codes_units_and_drug_names(cm):
    ix = ConceptIndex(cm)
    ix.ingest(pd.DataFrame({
        "concept_id": [11, 12, 13, 14, 15, 16, 17, 18, 19],
        "concept_name": ["Ammonia", "Cardiac arrest", "Metabolic encephalopathy", "Norepinephrine 4 MG Injection",
                         "mg/dL", "Positive", "Cerebral infarction", "CPR", "Naloxone 0.4 MG/ML Injection"],
        "domain_id": ["Measurement", "Condition", "Condition", "Drug", "Unit", "Meas Value", "Condition", "Procedure",
                      "Drug"],
        "vocabulary_id": ["LOINC", "ICD10CM", "ICD10CM", "RxNorm", "UCUM", "LOINC", "ICD10CM", "CPT4", "RxNorm"],
        "concept_code": ["16362-6", "I46.9", "G93.41", "999", "mg/dL", "LA6576-8", "I63.9", "92950", "888"]}))
    assert ix.meas[11] == [("ammonia", "umol/l")]
    assert ix.cond[12] == ("arrest_event",) and ix.cond[17] == ("dx_infarct",) and 13 not in ix.cond   # G93.41 never mapped
    assert ix.drug[14] == (("vasopressor", "norepinephrine"),) and ix.drug[19] == (("antidote", "naloxone"),)
    assert ix.unit_of(15) == "mg/dL" and ix.value_name[16] == "Positive" and ix.proc[18] == ("arrest_event",)
    assert ix.n_resolved()["measurement"] == 1
