"""Case report form and REDCap data dictionary (no patient data involved)."""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sortinghat.adjudication import crf
from sortinghat.labels.ontology import ALL_LABELS, GoldState

REPO = Path(__file__).resolve().parents[1]
CFG = crf.load_crf()


def test_the_form_covers_every_ontology_label_plus_other():
    codes = crf.label_codes(CFG)
    assert set(ALL_LABELS) <= set(codes) and codes[-1] == "other"
    prim = [c["code"] for c in CFG["primary_etiology_choices"]]
    assert prim == [*ALL_LABELS, "other", "unknown"]                       # E1..E7 incl. E3 and E4b, other, unknown


def test_state_scale_matches_the_gold_state_vocabulary():
    names = crf.state_names(CFG)
    assert sorted(names.values()) == sorted(s.name.lower() for s in GoldState)
    assert CFG["positive_states"] == ["probable", "definite"]
    assert {GoldState.parse(n).is_positive for n in names.values() if n != "unassessable"} == {True, False}
    assert set(CFG["confidence_scale"]) == {1, 2, 3, 4, 5}


def test_expanded_fields_are_valid():
    fields = crf.expand_fields(CFG)
    assert crf.validate_fields(fields) == []
    names = [f["name"] for f in fields]
    assert names[0] == "case_id" and "rater_id" in names
    for lab in CFG["labels"]:
        f = lab["field"]
        assert f"{f}_state" in names and f"{f}_conf" in names                       # confidence 1-5 for each label
        by = {x["name"]: x for x in fields}
        assert list(by[f"{f}_conf"]["choices"]) == ["1", "2", "3", "4", "5"]
        assert by[f"{f}_state"]["required"] and by[f"{f}_conf"]["required"]
        assert "<> '9'" in by[f"{f}_conf"]["branching"]                             # no confidence for an unassessable label
    assert {"primary_etiology", "primary_conf", "evidence_used", "dx_certainty", "minutes_spent", "eeg_seen", "e1_sub"} <= set(names)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]{0,25}", n) for n in names)


def test_no_field_asks_for_eeg_information_or_an_identifier():
    for f in crf.expand_fields(CFG):
        if f["name"] in ("eeg_seen", "eeg_seen_text"):
            continue                                                                # the blinding-check fields name EEG on purpose
        assert "eeg" not in (f["label"] + str(f.get("note", ""))).lower().replace("eeg-blind", "").replace("eeg-based", "")
    dd = crf.data_dictionary(CFG)
    assert (dd["Identifier?"] == "").all()


def test_data_dictionary_layout():
    dd = crf.data_dictionary(CFG)
    assert dd.columns.tolist() == crf.REDCAP_COLUMNS and len(dd.columns) == 18
    assert dd.iloc[0]["Variable / Field Name"] == "case_id" and dd.iloc[0]["Field Type"] == "text"
    assert set(dd["Form Name"]) == {"adjudication"}
    assert set(dd["Field Type"]) <= crf.FIELD_TYPES
    ch = dd.set_index("Variable / Field Name")["Choices, Calculations, OR Slider Labels"]
    parsed = {code.strip(): lab.strip() for code, lab in (c.split(",", 1) for c in ch["e1_state"].split(" | "))}
    assert parsed == {"0": "absent", "1": "possible", "2": "probable", "3": "definite", "9": "unassessable"}
    assert ch["primary_etiology"].count(" | ") == 9 and "unknown, Unknown" in ch["primary_etiology"]
    # a section header opens each section, only on its first field
    heads = dd[dd["Section Header"] != ""]["Variable / Field Name"].tolist()
    assert heads[0] == "case_id" and len(heads) == len(CFG["sections"])
    # every branching reference names an existing field
    names = set(dd["Variable / Field Name"])
    for b in dd["Branching Logic (Show field only if...)"]:
        assert set(re.findall(r"\[([a-z0-9_]+)", b)) <= names


def test_choice_labels_contain_no_pipe_and_codes_are_unique():
    for f in crf.expand_fields(CFG):
        if f.get("choices"):
            assert len(set(f["choices"])) == len(f["choices"])
            assert not any("|" in v for v in f["choices"].values())


def test_validation_catches_a_broken_form():
    bad = [{"name": "a", "type": "text", "label": "x"}, {"name": "b", "type": "radio", "label": "y", "choices": None, "branching": "[zzz] = '1'"},
           {"name": "A!", "type": "bogus", "label": "z"}]
    errs = " ".join(crf.validate_fields(bad))
    assert "without choices" in errs and "unknown field zzz" in errs and "bad field name" in errs and "bad type" in errs


def test_committed_dictionary_is_up_to_date():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "adjudication_redcap.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_redcap_script_writes_the_same_dictionary(tmp_path):
    out = tmp_path / "dd.csv"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "adjudication_redcap.py"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and out.read_text() == (REPO / "docs" / "adjudication_redcap_dictionary.csv").read_text()


# ------------------------------------------------------------------------------------------------ export parsing
def export_frame():
    return pd.DataFrame({
        "case_id": ["C1", "C1", "C2", "C2", "C3", "", "C4"],
        "rater_id": ["R1", "R2", "R1", "R2", "R1", "R1", "R1"],
        "e1_state": ["3", "0", "2", "9", "1", "3", ""], "e1_conf": ["5", "4", "3", "", "2", "5", ""],
        "e7_state": ["definite", "ABSENT", "probable", "possible", "", "absent", "absent"], "e7_conf": ["4", "5", "3", "2", "", "1", "1"],
        "primary_etiology": ["E1", "E1", "E7", "unknown", "E1", "E1", "E1"], "minutes_spent": ["12", "9", "15", "20", "8", "5", "5"]})


def test_long_ratings_accepts_codes_and_names_and_drops_blanks():
    long = crf.long_ratings(export_frame(), CFG)
    e1 = long[long["label"] == "E1"].set_index(["case_id", "rater"])["state"]
    assert e1.to_dict() == {("C1", "R1"): "definite", ("C1", "R2"): "absent", ("C2", "R1"): "probable", ("C2", "R2"): "unassessable",
                            ("C3", "R1"): "possible"}                                  # blank case id and blank state dropped
    e7 = long[long["label"] == "E7"].set_index(["case_id", "rater"])["state"]
    assert e7[("C1", "R1")] == "definite" and e7[("C1", "R2")] == "absent" and ("C3", "R1") not in e7.index
    assert long.loc[(long["label"] == "E1") & (long["case_id"] == "C1") & (long["rater"] == "R1"), "conf"].iloc[0] == 5


def test_a_corrected_form_replaces_the_earlier_one():
    ex = pd.concat([export_frame().iloc[:1], export_frame().iloc[:1].assign(e1_state="1")], ignore_index=True)
    long = crf.long_ratings(ex, CFG, labels=["E1"])
    assert len(long) == 1 and long["state"].iloc[0] == "possible"


def test_case_fields():
    cf = crf.case_fields(export_frame(), CFG)
    assert set(cf.columns) >= {"case_id", "rater", "primary_etiology", "minutes_spent"} and len(cf) == 6
