"""BIND step 0c vocabulary lister: guard and suppression on SYNTHETIC frames only (CLAUDE.md rule 1)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("bind_step0c_vocab", Path(__file__).parents[1] / "scripts" / "bind_step0c_vocab.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_guard_withholds_ids_dates_and_long_strings():
    assert V.withheld_reason("Acute") is None and V.withheld_reason("sub-acute") is None
    assert V.withheld_reason("1234567") == "digits"
    assert V.withheld_reason("2020-01-05") == "date" and V.withheld_reason("1/5/2020") == "date" and V.withheld_reason("Jan 5") == "date"
    assert V.withheld_reason("x" * 81) == "too_long" and V.withheld_reason("x" * 80) is None
    assert V.withheld_reason("sub-ABC123") == "id_pattern"


def test_distinct_values_lists_without_counts_and_withholds():
    s = pd.Series(["Acute", "Acute", "Chronic", None, "", "I0001123456", "2021-03-04"])
    d = V.distinct_values(s)
    assert d["values"] == ["Acute", "Chronic"] and d["withheld"] == 2 and d["has_blank"]
    assert "count" not in str(d).lower()
    big = V.distinct_values(pd.Series([f"v{i}" for i in range(50)]), max_distinct=10)
    assert big["status"] == "not_categorical" and big["values"] == []


def test_build_vocab_structure_is_suppressed_and_refuses_free_text():
    n = 30
    f = pd.DataFrame({"BDSPPatientID": [str(i) for i in range(n)], "Session_id": [f"s{i}" for i in range(n)],
                      "Acute/Chronic": ["Acute"] * n, "Nullfinding": ["Yes"] * 5 + ["No"] * (n - 5),
                      "Label_manual": [""] * (n - 3) + ["x"] * 3})
    d = pd.DataFrame({"bdsp_patient_id": [str(i) for i in range(n + 5)], "Session_id": [f"s{i}" for i in range(n + 5)]})
    r = pd.DataFrame({"bdsp_patient_id": [str(i) for i in range(n + 5)], "session_id": [f"s{i}" for i in range(n + 5)],
                      "Type": ["MRI HEAD"] * (n + 5)})
    out = V.build_vocab(f, r, d)
    assert out["findings"]["Acute/Chronic"]["values"] == ["Acute"] and out["reports"]["Type"]["values"] == ["MRI HEAD"]
    st = out["structure"]
    assert st["label_manual_rows_filled"] == "<11" and st["nullfinding_value_by_study"]["Yes"]["studies_any_row"] == "<11"
    assert st["share_demographics_studies_with_findings_row"] == "<11" or isinstance(st["share_demographics_studies_with_findings_row"], float)
    from sortinghat.safe_output import assert_aggregate_only
    assert_aggregate_only(out)
    with pytest.raises(ValueError):
        V.build_vocab(f.assign(Name="free text"), r, d)


def test_tokens_split_multilabel_cells_and_type_is_a_closed_capitals_list():
    s = pd.Series(["VASCULAR CONDITIONS, NEOPLASMS", "NEOPLASMS", "NORMAL,  X Y"])
    d = V.distinct_values(s, tokens=True)
    assert d["values"] == ["NEOPLASMS", "NORMAL", "VASCULAR CONDITIONS", "X Y"] and d["unit"] == "token"
    t = V.distinct_values(pd.Series(["MRI BRAIN", "Not available", "have reviewed the", "and"]), strict_type=True)
    assert t["values"] == ["MRI BRAIN", "Not available"] and t["withheld"] == 2


def test_lexicon_probe_lists_only_lesion_word_names():
    p = V.lexicon_probe(pd.Series(["ACUTE INFARCT, WHITE MATTER", "Subdural hematoma", "Meningioma", "ICH 1234567"]))
    assert p["values"] == ["ACUTE INFARCT", "Subdural hematoma"] and p["withheld"] == 1
