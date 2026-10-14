"""BIND imaging-E1 loader (ASSUMED value vocabularies; CONFIRMED column names). SYNTHETIC fixtures only (CLAUDE.md rule 1)."""
import pandas as pd
import pytest

from sortinghat.labels import bind
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only

CFG = bind.load_config()
F_COLS = list(CFG.findings_columns.values())


def _f(person, session, date, name="", acute="", brain="Yes", null="No", level2="", general=""):
    row = dict.fromkeys(F_COLS, "")
    row.update({"BDSPPatientID": person, "Session_id": session, "ShiftedStudyDate": date, "Name": name, "General Name": general,
                "Acute/Chronic": acute, "Brain-related": brain, "Brain": brain, "Nullfinding": null, "Label_LLAMA_Level2": level2})
    return row


def _sessions(rows):
    d = pd.DataFrame(rows, columns=["bdsp_patient_id", "Session_id", "ShiftedStudyDate", "ShiftedDateOfBirth"])
    return bind.normalise_sessions(d, "I0001", CFG)


def _cases(rows):
    return pd.DataFrame(rows, columns=["case_id", "person_id", "t0"]).assign(t0=lambda d: pd.to_datetime(d["t0"], format="mixed"))


def _run(find_rows, sess_rows, case_rows, **kw):
    f = bind.classify_findings(bind.normalise_findings(pd.DataFrame(find_rows, columns=F_COLS), CFG), CFG)
    return bind.build_label_table(_cases(case_rows), _sessions(sess_rows), f, CFG, **kw).set_index("case_id")


def test_config_columns_are_the_confirmed_real_header_names():
    assert CFG.status == "VOCAB_OBSERVED_0C"
    assert list(CFG.findings_columns.values()) == [
        "BDSPPatientID", "Session_id", "ShiftedDateOfBirth", "ShiftedStudyDate", "Counts", "Abnormality", "Brain", "FindingID",
        "PresentPrior", "Change", "Name", "General Name", "Location", "Brain-related", "Magnitude", "Acute/Chronic", "Test",
        "Label_manual", "Nullfinding", "Label_LLAMA_Level2", "Label_LLAMA_Level1"]
    assert "Report_txt" in CFG.never_load and "Report_txt" not in CFG.findings_columns.values()
    assert CFG.window_days == (-1, 3)
    assert bind.bind_key("findings", "I0001", CFG) == "Imaging/I0001/Clinical/I0001_imaging_findings_De-id.csv"
    assert bind.bind_key("demographics", "I0004", CFG) == "Imaging/I0004/Clinical/I0004_demographics.csv"
    with pytest.raises(ValueError):
        bind.bind_key("findings", "S0001", CFG)


def test_report_text_is_refused():
    df = pd.DataFrame({"BDSPPatientID": ["1"], "Session_id": ["a"], "Report_txt": ["free text"]})
    with pytest.raises(ValueError):
        bind.normalise_findings(df, CFG)


def test_positive_negative_and_window_edges():
    sess = [("1", "s1", "2020-01-02", "1950-01-01"),   # +1 day: in window, acute infarct
            ("2", "s2", "2020-01-10", "1950-01-01"),   # +9 days: outside the window
            ("3", "s3", "2020-01-01", "1950-01-01"),   # same day, brain study, only a chronic finding
            ("4", "s4", "2020-01-04", "1950-01-01"),   # +3: last day of window, acute SAH
            ("5", "s5", "2019-12-31", "1950-01-01"),   # -1: first day of window, no findings rows but brain study (null)
            ("6", "s6", "2019-12-30", "1950-01-01")]   # -2: outside
    fin = [_f("1", "s1", "2020-01-02", "acute infarct in the left MCA territory", "acute"),
           _f("2", "s2", "2020-01-10", "acute infarct", "acute"),
           _f("3", "s3", "2020-01-01", "old infarct", "chronic"),
           _f("4", "s4", "2020-01-04", "subarachnoid hemorrhage", "acute"),
           _f("5", "s5", "2019-12-31", "", "", null="Yes")]
    cases = [("c1", 1, "2020-01-01 13:00"), ("c2", 2, "2020-01-01 13:00"), ("c3", 3, "2020-01-01 13:00"),
             ("c4", 4, "2020-01-01 13:00"), ("c5", 5, "2020-01-01 13:00"), ("c6", 6, "2020-01-01 13:00"), ("c7", 99, "2020-01-01")]
    lab = _run(fin, sess, cases)
    assert lab.loc["c1", "E1_img"] == True and lab.loc["c1", "cls_infarct"]            # noqa: E712
    assert lab.loc["c2", "reason"] == bind.REASON_NO_STUDY and pd.isna(lab.loc["c2", "E1_img"])
    assert lab.loc["c3", "E1_img"] == False                                           # noqa: E712  chronic never counts
    assert lab.loc["c4", "E1_img"] == True and lab.loc["c4", "cls_sah"]               # noqa: E712
    assert lab.loc["c5", "E1_img"] == False and lab.loc["c5", "n_brain_studies"] == 1  # noqa: E712  null-finding brain study
    assert pd.isna(lab.loc["c6", "E1_img"]) and pd.isna(lab.loc["c7", "E1_img"])
    wide = _run(fin, sess, cases, window_days=(-3, 10))
    assert wide.loc["c2", "E1_img"] == True                                           # noqa: E712
    assert pd.isna(wide.loc["c6", "E1_img"])      # a session with no findings row and no report Type is not known to be a brain study (plan step 0c)


def test_indeterminate_acuity_is_masked_not_negative_and_a_positive_wins():
    sess = [("1", "s1", "2020-01-01", "1950-01-01"), ("2", "s2", "2020-01-01", "1950-01-01")]
    fin = [_f("1", "s1", "2020-01-01", "intraparenchymal hemorrhage", ""),          # unknown acuity
           _f("2", "s2", "2020-01-01", "intraparenchymal hematoma", ""),
           _f("2", "s2", "2020-01-01", "subdural hematoma", "acute")]
    lab = _run(fin, sess, [("a", 1, "2020-01-01"), ("b", 2, "2020-01-01")])
    assert pd.isna(lab.loc["a", "E1_img"]) and lab.loc["a", "reason"] == bind.REASON_INDETERMINATE
    assert lab.loc["b", "E1_img"] == True                                             # noqa: E712
    cfg2 = bind.load_config(overrides={"indeterminate_masks_label": False})
    f = bind.classify_findings(bind.normalise_findings(pd.DataFrame(fin, columns=F_COLS), cfg2), cfg2)
    lab2 = bind.build_label_table(_cases([("a", 1, "2020-01-01")]), _sessions(sess), f, cfg2).set_index("case_id")
    assert lab2.loc["a", "E1_img"] == False                                           # noqa: E712  sensitivity: indeterminate -> negative


def test_exclusions_negation_nonbrain_and_secondary_class():
    sess = [(str(i), f"s{i}", "2020-01-01", "1950-01-01") for i in range(1, 7)]
    fin = [_f("1", "s1", "2020-01-01", "acute microhemorrhage", "acute"),            # excluded concept
           _f("2", "s2", "2020-01-01", "no acute infarct", "acute"),                # negated
           _f("3", "s3", "2020-01-01", "acute infarct of the lung", "acute", brain="No"),   # not brain-related
           _f("4", "s4", "2020-01-01", "acute hydrocephalus", "acute"),            # secondary class only
           _f("5", "s5", "2020-01-01", "acute infarct", "subacute"),               # subacute counts
           _f("6", "s6", "2020-01-01", "acute infarct", "chronic")]                # chronic qualifier wins
    cases = [(f"c{i}", i, "2020-01-01") for i in range(1, 7)]
    lab = _run(fin, sess, cases)
    got = lab["E1_img"].tolist()
    assert got[0] == False and got[1] == False and pd.isna(got[2]) and got[3] == False and got[4] == True and got[5] == False  # noqa: E712
    assert lab.loc["c6", "ctrl_chronic"] == True and lab.loc["c5", "ctrl_chronic"] == False   # noqa: E712  chronic infarct = control outcome
    assert lab.loc["c4", "cls_hydrocephalus"] and not lab.loc["c4", "E1_img"]
    broad = _run(fin, sess, cases, include_secondary_in_primary=True)
    assert broad.loc["c4", "E1_img"] == True                                           # noqa: E712


def test_ids_normalise_and_report_type_assessability():
    assert bind.norm_id("00123.0") == "123" and bind.norm_id(" 7 ") == "7" and bind.norm_id("") is None and bind.norm_id(float("nan")) is None
    cfg = bind.load_config(overrides={"session_assessable": {"from_findings_brain": False, "report_type_regex": "head|brain"}})
    dem = pd.DataFrame([["1", "s1", "2020-01-01", "1950-01-01"]], columns=["bdsp_patient_id", "Session_id", "ShiftedStudyDate", "ShiftedDateOfBirth"])
    rep = pd.DataFrame({"bdsp_patient_id": ["1"], "session_id": ["s1"], "Type": ["MRI BRAIN W/O CONTRAST"]})
    s = bind.normalise_sessions(dem, "I0001", cfg, reports=rep)
    assert bool(s["report_brain"].iloc[0])
    f = bind.classify_findings(bind.normalise_findings(pd.DataFrame([], columns=F_COLS), cfg), cfg)
    lab = bind.build_label_table(_cases([("c", 1, "2020-01-01")]), s, f, cfg).set_index("case_id")
    assert lab.loc["c", "E1_img"] == False                                            # noqa: E712  brain study by report Type, no findings


def test_aggregate_helpers_suppress_and_pass_the_aggregate_check():
    n = 40
    sess = [(str(i), f"s{i}", "2020-01-01", "1950-01-01") for i in range(n)]
    fin = [_f(str(i), f"s{i}", "2020-01-01", "acute infarct", "acute") for i in range(3)] + \
          [_f(str(i), f"s{i}", "2020-01-01", "", "", null="Yes") for i in range(3, n)]
    lab = _run(fin, sess, [(f"c{i}", i, "2020-01-01") for i in range(n)])
    s = bind.summarise(lab, CFG)
    assert s["positive"] == SUPPRESSED and s["assessable"] == n and s["classes"]["infarct"] == SUPPRESSED
    assert s["prevalence_among_assessable"] == SUPPRESSED
    assert_aggregate_only(s)
    q = bind.linkage_qc([str(i) for i in range(30)], [str(i) for i in range(20, 60)])
    assert q["with_any_imaging"] == SUPPRESSED and q["cohort_people"] == 40
    assert_aggregate_only(q)
    ids = [str(i) for i in range(100)]
    b = pd.Series(pd.to_datetime(["1950-01-01"] * 100), index=ids)
    r = b.copy(); r.iloc[:20] = r.iloc[:20] + pd.Timedelta(days=5)
    a = bind.shift_agreement(b, r)
    assert a["linked_with_dob"] == 100 and a["exact_agreement"] == 0.8 and a["within_1_day"] == 0.8
    assert_aggregate_only(a)


def test_concordance_suppresses_small_cells_and_computes_large_ones():
    idx = range(200)
    img = pd.Series([i < 100 for i in idx], dtype="boolean")
    silver = pd.Series([i < 80 or 100 <= i < 120 for i in idx], dtype="boolean")      # tp 80, fn 20, fp 20, tn 80
    c = bind.concordance(silver, img)
    assert c["silver_pos_img_pos"] == 80 and c["sensitivity"] == 0.8 and c["specificity"] == 0.8 and c["ppv"] == 0.8
    img2 = img.copy(); silver2 = silver.copy(); silver2[:] = False; silver2[:5] = True
    c2 = bind.concordance(silver2, img2)
    assert c2["silver_pos_img_pos"] == SUPPRESSED and c2["ppv"] == SUPPRESSED
    masked = img.copy(); masked[:50] = pd.NA
    assert bind.concordance(silver, masked)["n_both_assessable"] == 150
    assert_aggregate_only(c)


# ---- value vocabulary adapted in plan step 0c (D-160): the categories below are strings OBSERVED in the real Level2 / Acute-Chronic
# columns (copied as value strings only; no record, count or id). Acuity values observed: acute | subacute | chronic | unknown.
def _classify(name, acute="acute", level2="", brain="yes", null="No"):
    row = pd.DataFrame([_f("1", "s1", "2020-01-01", name, acute, brain=brain, null=null, level2=level2)], columns=F_COLS)
    f = bind.classify_findings(bind.normalise_findings(row, CFG), CFG)
    return {c: bool(f[f"pos_{c}"].iloc[0]) for c in CFG.class_names()}


@pytest.mark.parametrize("text,cls", [
    ("infarct", "infarct"), ("lacunar infarct", "infarct"), ("small vessel ischemic infarction", "infarct"),
    ("to early sub infarct", "infarct"), ("ischemic stroke", "infarct"), ("cva / ischemic stroke", "infarct"),
    ("hemorrhage", "ich"), ("parenchymal hemorrhage", "ich"), ("intracranial hemorrhage", "ich"),
    ("subdural hematoma", "sdh_edh"), ("epidural hematoma", "sdh_edh"), ("subarachnoid hemorrhage", "sah"),
    ("traumatic brain injury (tbi)", "tbi_contusion"), ("diffuse axonal injury", "tbi_contusion"), ("contusion", "tbi_contusion"),
    ("midline shift", "mass_effect"), ("herniation", "mass_effect"), ("tumor related edema", "mass_effect"), ("edema", "mass_effect"),
    ("hydrocephalus", "hydrocephalus"), ("ventriculomegaly", "hydrocephalus")])
def test_observed_lesion_names_map_to_their_class_when_acute(text, cls):
    got = _classify(text)
    assert got[cls], (text, got)


@pytest.mark.parametrize("text", [
    "no infarct", "infarction absent", "no evidence of ischemia", "no midline shift or brain herniation",       # negated findings exist as rows
    "small vessel ischemic change", "microvascular ischemic changes", "ischemic changes", "ischemic leukoencephalopathy",
    "hypoxic ischemic injury", "hypoxic-ischemic encephalopathy", "spinal cord infarct", "ischemic optic neuropathy",
    "microhemorrhage", "microbleeds", "old hematoma", "subgaleal hemorrhage", "intraventricular hemorrhage",
    "subarachnoid cyst", "subarachnoid space prominence", "subdural hygroma", "subdural effusion",
    "nontraumatic", "post-traumatic changes", "traumatic encephalomalacia", "papilledema", "spinal cord edema", "edema/gliosis",
    "normal pressure hydrocephalus", "sequela of remote infarction", "infarct sequelae"])
def test_observed_non_qualifying_names_are_not_primary_positives(text):
    got = _classify(text)
    assert not any(got[c] for c in CFG.primary_classes), (text, got)


def test_acuity_vocabulary_and_brain_related_unknown():
    assert _classify("infarct", "subacute")["infarct"] and _classify("infarct", "acute")["infarct"]
    assert not _classify("infarct", "chronic")["infarct"] and not _classify("infarct", "unknown")["infarct"]
    assert not _classify("infarct", "acute", brain="unknown")["infarct"] and not _classify("infarct", "acute", brain="no")["infarct"]
    row = pd.DataFrame([_f("1", "s1", "2020-01-01", "infarct", "unknown")], columns=F_COLS)
    f = bind.classify_findings(bind.normalise_findings(row, CFG), CFG)
    assert bool(f["ind_infarct"].iloc[0]) and not bool(f["pos_infarct"].iloc[0])
