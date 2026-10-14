"""scripts/tueg_label_from_filelist.py on SYNTHETIC reports and file lists only (no TUEG data, no real text).

Covers: TUEG path parsing, the generated NEDC query scripts (section restriction, negation / history / uncertainty cues, loose variant,
profile mode, standard-library-only), the label table (mutual exclusion, conflicts, setting) and the stop rule with suppression.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from sortinghat.safe_output import SUPPRESSED
from test_silver_feasibility_inputs import load_script

tl = load_script("tueg_label_from_filelist")
CFG = tl.load_config()


# ------------------------------------------------------------------------------------------------------------------ path parsing
@pytest.mark.parametrize("path, subj, sess, year, tok", [
    ("edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf", "aaaaaaaa", 1, 2015, 0),
    ("/data/tuh_eeg/v2.0.2/edf/001/aaaaamye/s012_2011_01_05/02_tcp_le/aaaaamye_s012_t007.edf", "aaaaamye", 12, 2011, 7),
    ("edf/eval/abnormal/01_tcp_ar/aaaaamye_s001_t000.edf", "aaaaamye", 1, None, 0),          # TUAB-style path, no session folder
    ("edf/000/aaaaaaaa/s002_2016_03_04/aaaaaaaa_s002.txt", "aaaaaaaa", 2, 2016, None),         # report-like file, no token
    ("edf/000/aaaaaaaa/s003_2014_07_08/", "aaaaaaaa", 3, 2014, None),                          # session folder only
])
def test_parse_paths(path, subj, sess, year, tok):
    p = tl.parse_tueg_path(path)
    assert p["subject"] == subj and p["session"] == sess and p["year"] == year and p["token"] == tok
    assert p["session_id"] == f"{subj}_s{sess:03d}"


@pytest.mark.parametrize("bad", ["", "# comment", "   ", "just some text", "edf/000/aaaaaaaa/s001_2015_12_30/bbbbbbbb_s001_t000.edf",
                                 "edf/000/aaaaaaaa/s001_2015_12_30/aaaaaaaa_s002_t000.edf"])
def test_parse_rejects_inconsistent(bad):
    assert tl.parse_tueg_path(bad) is None


def test_read_list_collapses_tokens(tmp_path):
    f = tmp_path / "E1.txt"
    f.write_text("\n".join([
        "edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf",
        "edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t001.edf",
        "edf/000/aaaaaaab/s004_2012_01_01/01_tcp_ar/aaaaaaab_s004_t000.edf", "garbage line", "", "# note"]))
    sess, n_ok, n_bad = tl.read_list(f)
    assert set(sess) == {"aaaaaaaa_s001", "aaaaaaab_s004"} and sess["aaaaaaaa_s001"]["n_files"] == 2
    assert (n_ok, n_bad) == (3, 1)


# --------------------------------------------------------------------------------------------------------------- query scripts
def report(history="", meds="", impression="", correlation="", intro="", extra_lines=()):
    body = [
        "INTRODUCTION: " + (intro or "Digital video EEG is performed at the bedside using standard 10-20 electrode placement."),
        "",
        "CLINICAL HISTORY: " + history,
        "",
        "MEDICATIONS: " + meds,
        "",
        "DESCRIPTION OF THE RECORD: The background is slow and disorganized.",
        "",
        "IMPRESSION: " + impression,
        "",
        "CLINICAL CORRELATION: " + correlation,
    ]
    return "\n".join(body + list(extra_lines)) + "\n"


def make_script(tmp_path, name, variant="strict", mode="list", names=None):
    names = names or [name]
    rt = tl.build_runtime_cfg(CFG, names, variant, mode)
    if mode == "profile":
        prof = CFG["query_plan"]["profile"]
        rt["profile"] = {"labels": prof["labels"], "controls": prof["controls"], "cross_settings": prof["cross_settings"],
                         "primary": CFG["stop_rule"]["primary_labels"]}
    path = tmp_path / f"q_{name}_{variant}.py"
    path.write_text(tl.render_script(rt, "test"))
    return path


def run_query(script, files):
    listing = script.parent / "list.txt"
    listing.write_text("\n".join(str(f) for f in files) + "\n")
    # -I: isolated mode; the script must run on the standard library alone
    r = subprocess.run([sys.executable, "-I", str(script), str(listing)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@pytest.fixture()
def reports(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()

    def w(name, **kw):
        p = d / f"{name}.txt"
        p.write_text(report(**kw))
        return p
    return w


def matches(tmp_path, set_name, variant, files):
    return {p.split("/")[-1][:-4] for p in run_query(make_script(tmp_path, set_name, variant), files)}


def test_section_restriction_only_history_and_meds(tmp_path, reports):
    hist = reports("hist", history="45 yo man with acute subdural hematoma, unresponsive.")
    meds = reports("meds", history="Fall.", meds="levetiracetam")
    imp = reports("imp", history="Syncope.", impression="Findings suggest a left hemispheric infarct.",
                  correlation="Correlate with subdural hematoma on CT.")
    desc = reports("intro", history="Syncope.", intro="Patient with stroke is monitored.")
    got = matches(tmp_path, "E1", "strict", [hist, meds, imp, desc])
    assert got == {"hist"}                                  # IMPRESSION / CLINICAL CORRELATION / INTRODUCTION never searched
    assert matches(tmp_path, "MED_ASM", "strict", [hist, meds, imp]) == {"meds"}


def test_negation_history_uncertainty_family(tmp_path, reports):
    cases = {
        "pos": "Patient with acute stroke and aphasia.",
        "pos_age": "A 55 year old man with stroke, found down.",               # "old" must not act as a remote cue
        "neg": "No evidence of stroke on CT.",
        "denies": "Denies head trauma.",
        "hx": "History of stroke.",
        "hx_break": "History of HTN, now with acute stroke.",                    # breaker "now": the history cue does not reach the term
        "hx_list": "History of HTN, DM, and stroke.",
        "ro": "R/O stroke.",
        "ro2": "Rule out subdural hematoma.",
        "fam": "Family history of stroke in mother.",
        "post": "Stroke was ruled out.",
        "remote": "Remote subdural hematoma years ago.",
        "heat": "Heat stroke in the field.",
        "mi": "Recent myocardial infarct.",
        "q": "? stroke",
        "abbr_cs": "Fell, SDH on CT.",
        "abbr_lc": "stroke vs migraine; sdh.",
    }
    files = {k: reports(k, history=v) for k, v in cases.items()}
    got = matches(tmp_path, "E1", "strict", list(files.values()))
    assert got == {"pos", "pos_age", "abbr_cs", "hx_break"}, got
    # loose: only negation / family / post-negation apply
    got_loose = matches(tmp_path, "E1", "loose", list(files.values()))
    assert {"hx", "hx_list", "ro", "ro2", "remote", "q", "abbr_lc"} <= got_loose        # abbr_lc: "stroke vs migraine"
    assert not (got_loose & {"neg", "denies", "fam", "post", "heat", "mi"})


def test_e2_recent_arrest_forms(tmp_path, reports):
    cases = {
        "sp": "62 yo s/p cardiac arrest with ROSC after 20 minutes.",
        "hx_recent": "History of cardiac arrest 3 days ago, now comatose.",
        "remote": "Cardiac arrest years ago.",
        "pea": "PEA arrest at home.",
        "pea_bare": "PEA of unknown cause.",
        "vf_noctx": "VF was noted on telemetry earlier.",
        "anoxic": "Anoxic brain injury.",
        "hrf": "Hypoxic respiratory failure due to pneumonia.",
        "neg": "No cardiac arrest.",
    }
    files = {k: reports(k, history=v) for k, v in cases.items()}
    got = matches(tmp_path, "E2", "strict", list(files.values()))
    assert got == {"sp", "hx_recent", "pea", "anoxic"}, got


def test_e5_and_e3_and_control_sets(tmp_path, reports):
    e5 = {"hep": "Hepatic encephalopathy with elevated ammonia.", "dka": "Admitted with DKA.", "esrd_hx": "History of ESRD on HD.",
          "hypo": "No hypoglycemia documented.", "na": "Sodium 118, hyponatremia."}
    files = {k: reports(k, history=v) for k, v in e5.items()}
    assert matches(tmp_path, "E5", "strict", list(files.values())) == {"hep", "dka", "na"}
    assert "esrd_hx" in matches(tmp_path, "E5", "loose", list(files.values()))
    e3 = {"se": "Status epilepticus.", "ncse": "r/o NCSE", "new": "New onset seizure at home.", "ep": "Epilepsy on keppra.", "se_ctx": "SE tubes in place."}
    f3 = {k: reports(k, history=v) for k, v in e3.items()}
    assert matches(tmp_path, "E3", "strict", list(f3.values())) == {"se", "new"}
    assert matches(tmp_path, "E3_broad", "strict", list(f3.values())) == {"se", "new", "ep"}
    ctl = {"ams": "AMS, unresponsive.", "coma": "Comatose after fall.", "none": "Headache.", "hist": "History of encephalopathy."}
    fc = {k: reports(k, history=v) for k, v in ctl.items()}
    assert matches(tmp_path, "CTRL", "strict", list(fc.values())) == {"ams", "coma"}


def test_settings_covariates_universe(tmp_path, reports):
    s = {"icu": "Transferred to the ICU, intubated.", "ed": "Brought to the ED by EMS.", "out": "Outpatient referral for spells.",
         "emu": "Admitted to the epilepsy monitoring unit.", "neg": "Not in the ICU.", "plain": "Headache."}
    files = {k: reports(k, history=v) for k, v in s.items()}
    assert matches(tmp_path, "SETTING_ICU", "strict", list(files.values())) == {"icu"}
    assert matches(tmp_path, "SETTING_ED", "strict", list(files.values())) == {"ed"}
    assert matches(tmp_path, "SETTING_CLINIC", "strict", list(files.values())) == {"out"}
    assert matches(tmp_path, "SETTING_EMU", "strict", list(files.values())) == {"emu"}
    a = {"a30": "30 yo female with spells.", "a70": "A 72 year old man with syncope.", "a85": "85-year-old woman.", "none": "Spells."}
    fa = {k: reports(k, history=v) for k, v in a.items()}
    assert matches(tmp_path, "AGE_18_39", "strict", list(fa.values())) == {"a30"}
    assert matches(tmp_path, "AGE_65_79", "strict", list(fa.values())) == {"a70"}
    assert matches(tmp_path, "AGE_80P", "strict", list(fa.values())) == {"a85"}
    assert matches(tmp_path, "SEX_F", "strict", list(fa.values())) == {"a30", "a85"}
    # universe: a report with no CLINICAL HISTORY section is not in the universe
    nohist = reports("nohist")
    nohist.write_text("INTRODUCTION: routine EEG.\nIMPRESSION: normal.\n")
    assert matches(tmp_path, "UNIVERSE_HISTORY", "strict", [nohist, files["plain"]]) == {"plain"}


def test_unreadable_file_is_skipped(tmp_path, reports):
    ok = reports("ok", history="Subdural hematoma.")
    assert run_query(make_script(tmp_path, "E1"), [tmp_path / "missing.txt", ok]) == [str(ok)]


def test_profile_mode_prints_counts_only(tmp_path, reports):
    files = []
    spec = {
        "aaaaaaaa_s001": "ICU. Acute stroke.",
        "aaaaaaab_s001": "ED. Cardiac arrest, ROSC.",
        "aaaaaaac_s001": "Admitted. Subdural hematoma and cardiac arrest.",
        "aaaaaaad_s001": "ICU. AMS, unresponsive.",
        "aaaaaaae_s001": "Outpatient. Status epilepticus.",
    }
    for key, h in spec.items():
        d = tmp_path / "reports" / key[:8] / (key[9:] + "_2015_01_01")
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{key}.txt"
        p.write_text(report(history=h))
        files.append(p)
    prof_cfg = CFG["query_plan"]["profile"]
    names = list(dict.fromkeys(prof_cfg["labels"] + prof_cfg["controls"]
                               + [n for n in tl._all_sets(CFG) if tl._all_sets(CFG)[n]["group"] == "settings"] + ["UNIVERSE_HISTORY"]))
    script = make_script(tmp_path, "PROFILE", mode="profile", names=names)
    out = run_query(script, files)
    assert out[0].startswith("PROFILE") and not any("aaaaaaa" in line for line in out)       # no file names or ids in the output
    prof = tl.parse_profile("\n".join(out))
    assert prof["N"][("sessions",)] == 5
    assert prof["SET"][("E1",)] == 2 and prof["SET"][("E2",)] == 2 and prof["SET"][("E3",)] == 1
    assert prof["PURE"][("E1", "all")] == 1 and prof["PURE"][("E1", "ACUTE")] == 1
    assert prof["PURE"][("E2", "ACUTE")] == 1                     # ED + arrest; the E1+E2 session is not pure
    assert prof["OVERLAP"][("E1", "E2")] == 1
    assert prof["CONTROL"][("clean", "all")] == 1 and prof["CONTROL"][("clean", "ACUTE")] == 1


def test_emit_queries_manifest_and_stdlib_only(tmp_path):
    man = tl.emit_queries(CFG, tmp_path / "q")
    files = {s["file"] for s in man["scripts"]}
    assert "tueg_query_PROFILE_strict.py" in files and "tueg_query_E1_strict.py" in files and "tueg_query_E1_loose.py" in files
    assert not any(f.startswith("tueg_query_E4a") for f in files)                      # query: false
    import ast
    for s in man["scripts"]:
        tree = ast.parse((tmp_path / "q" / s["file"]).read_text())
        mods = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert mods <= {"json", "re", "sys"}, mods
    assert json.loads((tmp_path / "q" / "MANIFEST.json").read_text())["config_sha256_16"] == man["config_sha256_16"]


def test_config_integrity():
    sets = tl._all_sets(CFG)
    plan = CFG["query_plan"]
    for rnd in ("round1", "round2_strict", "round2_loose"):
        assert all(n in sets for n in plan[rnd]), rnd
    # no search of EEG-derived sections
    assert set(CFG["sections"]["use"]) == {"CLINICAL HISTORY", "MEDICATIONS"}
    assert "IMPRESSION" in CFG["sections"]["never_search"] and "CLINICAL CORRELATION" in CFG["sections"]["never_search"]
    for lab in ("E1", "E2", "E5", "E3"):
        assert sets[lab]["terms"] and "policy" in sets[lab]


# ------------------------------------------------------------------------------------------------------------- table and stop rule
def mk(subj_sess_year, n_files=1):
    return {s: {"subject": s[:8], "session": int(s[10:13]), "year": y, "n_files": n_files} for s, y in subj_sess_year}


def synthetic_lists(n_e1=60, n_e2=55, n_e5=8, n_ctrl=400):
    """Unique subjects aaaa0001... with controlled overlaps."""
    L = {k: {} for k in ("UNIVERSE_HISTORY", "E1", "E2", "E5", "E3", "E3_broad", "E5_nonspecific", "CTRL",
                         "SETTING_ICU", "SETTING_ED", "SETTING_ADMIT", "SETTING_CLINIC", "SETTING_EMU")}

    def add(name, sessions):
        for s in sessions:
            L[name][s] = {"subject": s[:8], "session": int(s[10:13]), "year": 2015, "n_files": 1}
    e1 = [f"e1{i:06d}_s001" for i in range(n_e1)]
    e2 = [f"e2{i:06d}_s001" for i in range(n_e2)]
    e5 = [f"e5{i:06d}_s001" for i in range(n_e5)]
    ct = [f"c0{i:06d}_s001" for i in range(n_ctrl)]
    for lab, ss in (("E1", e1), ("E2", e2), ("E5", e5), ("CTRL", ct)):
        add(lab, ss)
    allsess = e1 + e2 + e5 + ct
    add("UNIVERSE_HISTORY", allsess)
    add("SETTING_ICU", e1[:50] + e2[:30] + e5 + ct[:300])
    add("SETTING_ED", e2[25:] + e1[40:])                 # overlap with ICU: acute either way
    add("SETTING_CLINIC", e1[55:60])                 # e1[55:60] also ED (40:) -> mixed
    return L, e1, e2, e5, ct


def test_build_table_roles_and_conflicts():
    L, e1, e2, e5, ct = synthetic_lists()
    # E1 + E2 overlap (multi_primary), E3 on a primary session, control with a label hit, subject conflict
    L["E2"][e1[0]] = L["E1"][e1[0]]                                           # multi
    L["E3"][e1[1]] = L["E1"][e1[1]]                                           # e3 overlap with a primary
    L["E3"]["e3000001_s001"] = {"subject": "e3000001", "session": 1, "year": 2015, "n_files": 1}
    L["E3_broad"]["e3000001_s001"] = L["E3"]["e3000001_s001"]
    L["UNIVERSE_HISTORY"]["e3000001_s001"] = L["E3"]["e3000001_s001"]
    L["E5_nonspecific"][ct[0]] = L["CTRL"][ct[0]]                             # control with nonspecific metabolic wording -> excluded
    L["CTRL"][f"{e1[2][:8]}_s002"] = {"subject": e1[2][:8], "session": 2, "year": 2016, "n_files": 1}   # control from a positive subject
    L["UNIVERSE_HISTORY"][f"{e1[2][:8]}_s002"] = L["CTRL"][f"{e1[2][:8]}_s002"]
    df = tl.build_table(L, CFG)
    r = df.set_index("session_id")
    assert r.loc[e1[0], "role"] == "positive_multi" and "multi_primary" in r.loc[e1[0], "conflicts"]
    assert r.loc[e1[1], "role"] == "positive_pure" and "e3_overlap_primary" in r.loc[e1[1], "conflicts"]
    assert r.loc["e3000001_s001", "role"] == "e3_positive"
    assert r.loc[ct[0], "role"] == "excluded_control_label_hit" and "control_has_label_hit" in r.loc[ct[0], "conflicts"]
    sub = f"{e1[2][:8]}_s002"
    assert r.loc[sub, "role"] == "excluded_subject_conflict" and "subject_has_positive_session" in r.loc[sub, "conflicts"]
    df_keep = tl.build_table(L, CFG, subject_conflict="keep")
    assert df_keep.set_index("session_id").loc[sub, "role"] == "control"
    # mutual exclusion: a session has exactly one role; a pure positive has exactly one primary label
    assert df["role"].notna().all()
    pure = df[df["role"] == "positive_pure"]
    assert (pure[["E1", "E2", "E5"]].sum(axis=1) == 1).all() and (pure["primary_label"] != "").all()
    # setting derivation
    assert r.loc[e1[0], "setting"] == "acute"
    assert r.loc[e1[57], "setting"] == "mixed" and "setting_conflict" in r.loc[e1[57], "conflicts"]
    assert r.loc[ct[350], "setting"] == "unknown"


def test_subject_multi_label_flag():
    L, e1, e2, e5, ct = synthetic_lists(n_e1=3, n_e2=3, n_e5=3, n_ctrl=3)
    L["E2"]["e1000000_s002"] = {"subject": "e1000000", "session": 2, "year": 2016, "n_files": 1}   # same subject as e1[0], other label
    df = tl.build_table(L, CFG).set_index("session_id")
    assert "subject_multi_label" in df.loc["e1000000_s001", "conflicts"] and "subject_multi_label" in df.loc["e1000000_s002", "conflicts"]


def test_outside_universe_flag():
    L, e1, *_ = synthetic_lists(n_e1=3, n_e2=1, n_e5=1, n_ctrl=2)
    del L["UNIVERSE_HISTORY"][e1[0]]
    df = tl.build_table(L, CFG).set_index("session_id")
    assert "outside_universe" in df.loc[e1[0], "conflicts"]


def test_stop_rule_and_suppression():
    L, e1, e2, e5, ct = synthetic_lists(n_e1=60, n_e2=55, n_e5=8, n_ctrl=400)
    df = tl.build_table(L, CFG)
    summ = tl.summarize(df, L, {k: 0 for k in L}, CFG)
    pl = summ["per_label"]
    assert pl["E1"]["label_testable"] and pl["E2"]["label_testable"] and not pl["E5"]["label_testable"]
    sr = summ["stop_rule"]
    assert sr["labels_passing"] == ["E1", "E2"] and sr["route_continues"] is True and sr["stop_if_any_primary_below"] is True
    # E5 has only 8 sessions: suppressed everywhere it is shown
    assert pl["E5"]["pure_by_setting"]["total"] == SUPPRESSED and pl["E5"]["n_any_hit"] == SUPPRESSED
    # all three below threshold -> route stops
    L2, *_ = synthetic_lists(n_e1=20, n_e2=15, n_e5=8, n_ctrl=400)
    s2 = tl.summarize(tl.build_table(L2, CFG), L2, {}, CFG)
    assert s2["stop_rule"]["route_continues"] is False and s2["stop_rule"]["labels_passing"] == []
    # not enough controls: 3 x positives not met
    L3, *_ = synthetic_lists(n_e1=60, n_e2=55, n_e5=60, n_ctrl=40)
    s3 = tl.summarize(tl.build_table(L3, CFG), L3, {}, CFG)
    assert not s3["per_label"]["E1"]["acute_controls_ge_ratio_x_positives"]


def test_partition_hides_total_when_a_part_is_suppressed():
    p = tl._partition(30, {"acute": 25, "clinic": 5})
    assert p["acute"] == 25 and p["clinic"] == SUPPRESSED and p["total"] == SUPPRESSED
    q = tl._partition(30, {"acute": 20, "clinic": 15})
    assert q["total"] == 30


def test_cli_build_requires_local_only_and_is_aggregate(tmp_path, capsys):
    L, e1, e2, e5, ct = synthetic_lists(n_e1=60, n_e2=55, n_e5=60, n_ctrl=500)
    ld = tmp_path / "lists"
    ld.mkdir()
    for name, sess in L.items():
        lines = [f"edf/000/{s[:8]}/s{int(s[10:13]):03d}_2015_06_01/01_tcp_ar/{s}_t000.edf" for s in sess]
        (ld / f"{name}.txt").write_text("\n".join(lines) + "\n")
    summ = tmp_path / "out" / "tueg_summary.json"
    with pytest.raises(SystemExit):
        tl.main(["build", "--lists-dir", str(ld), "--table-out", str(tmp_path / "table.csv"), "--summary-out", str(summ)])
    tab = tmp_path / "local_only" / "tueg" / "labels.csv"
    assert tl.main(["build", "--lists-dir", str(ld), "--table-out", str(tab), "--summary-out", str(summ)]) == 0
    assert oct(tab.stat().st_mode & 0o777) == "0o600"
    out = capsys.readouterr().out
    assert "e1000000" not in out and "c0000000" not in out                            # no ids on stdout
    js = json.loads(summ.read_text())
    assert js["stop_rule"]["route_continues"] is True and "e1000000" not in summ.read_text()
    # a missing required list is refused unless --allow-missing
    (ld / "E5.txt").unlink()
    with pytest.raises(SystemExit):
        tl.main(["build", "--lists-dir", str(ld), "--summary-out", str(summ)])
    assert tl.main(["build", "--lists-dir", str(ld), "--summary-out", str(summ), "--allow-missing"]) == 0


def test_profile_check_cli(tmp_path):
    txt = textwrap.dedent("""\
        PROFILE\tmode=counts_only\tvariant=strict
        N\tsessions\t5000
        PURE\tE1\tall\t300
        PURE\tE1\tACUTE\t80
        PURE\tE2\tall\t90
        PURE\tE2\tACUTE\t55
        PURE\tE5\tall\t40
        PURE\tE5\tACUTE\t9
        CONTROL\tclean\tall\t900
        CONTROL\tclean\tACUTE\t400
        """)
    f = tmp_path / "profile.txt"
    f.write_text(txt)
    out = tmp_path / "res.json"
    assert tl.main(["profile-check", "--file", str(f), "--summary-out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["labels_passing"] == ["E1", "E2"] and res["route_continues"] is True
    assert res["pure_acute"]["E5"] == SUPPRESSED
