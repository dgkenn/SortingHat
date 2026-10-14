"""scripts/run_bind_e1.py end to end on the SYNTHETIC silver-feasibility inputs and a SYNTHETIC mirror of the BIND tables (no real data)."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from sortinghat import checkpoint as ck
from sortinghat.labels import bind
from sortinghat.safe_output import assert_aggregate_only
from test_silver_feasibility_inputs import load_script, make_inputs, rsf

rb = load_script("run_bind_e1")
CFG = bind.load_config()


def write_bind_mirror(root, cohort: pd.DataFrame, silver_e1: np.ndarray, seed: int = 0, p_study=0.8, p_pos_if_silver=0.65, p_pos_else=0.03):
    """Synthetic BIND I0001 tables. Imaging positives are correlated with silver E1 (so the planted EEG signal can show), a share of patients
    has no study in the window, some have an indeterminate-acuity mention, some a chronic-only lesion. No Report_txt anywhere."""
    rng = np.random.default_rng(seed)
    d = root / "Imaging" / "I0001" / "Clinical"
    d.mkdir(parents=True, exist_ok=True)
    F = list(CFG.findings_columns.values())
    demo, fin = [], []
    pos_names = ["acute infarct", "subdural hematoma", "intraparenchymal hemorrhage", "midline shift", "contusion", "subarachnoid hemorrhage"]
    for i, (pid, t0) in enumerate(zip(cohort["person_id"], pd.to_datetime(cohort["t0"]))):
        if rng.random() > p_study:
            continue
        off = int(rng.integers(-1, 4)) if rng.random() < 0.9 else 9
        day = (t0.normalize() + pd.Timedelta(days=off)).strftime("%Y-%m-%d")
        ses = f"ses{i}"
        demo.append([str(pid), ses, "1950-01-01", day])
        row = lambda **kw: {**dict.fromkeys(F, ""), "BDSPPatientID": str(pid), "Session_id": ses, "ShiftedStudyDate": day, "Brain": "yes",   # noqa: E731
                            "Brain-related": "yes", "Nullfinding": "No", **kw}
        u = rng.random()
        if u < (p_pos_if_silver if silver_e1[i] > 0.5 else p_pos_else):
            fin.append(row(Name=pos_names[int(rng.integers(0, len(pos_names)))], **{"Acute/Chronic": "acute"}))
        elif u > 0.93:
            fin.append(row(Name="intraparenchymal hemorrhage", **{"Acute/Chronic": "unknown"}))
        elif u > 0.88:
            fin.append(row(Name="old infarct", **{"Acute/Chronic": "chronic"}))
        else:
            fin.append(row(Name="no acute abnormality", Nullfinding="Yes", **{"Acute/Chronic": "unknown"}))
        fin[-1]["Label_manual"] = fin[-1]["Name"]
        fin[-1]["Label_LLAMA_Level2"] = fin[-1]["Name"]
    pd.DataFrame(demo, columns=["bdsp_patient_id", "Session_id", "ShiftedDateOfBirth", "ShiftedStudyDate"]).to_csv(d / "I0001_demographics.csv", index=False)
    pd.DataFrame(fin, columns=F).to_csv(d / "I0001_imaging_findings_De-id.csv", index=False)
    return root


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("bind_e1")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=3, n_per_site=450)
    silver = ms.y_silver[:, 0]
    mirror = write_bind_mirror(d / "bind", cohort, silver)
    return paths, ms, cohort, mirror, d


def argv(world, out, *extra, ck_root=None):
    paths, _ms, _c, mirror, d = world
    base = ["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver", str(paths["silver"]),
            "--baselines", str(paths["baselines"]), "--omop-source", "none", "--bind-data", str(mirror), "--out", str(out),
            "--n-boot", "300", "--n-boot-primary", "300", "--n-boot-secondary", "150", "--null-reps", "1", "--splits", "loso", "temporal",
            "--checkpoint-roots", str(ck_root or (d / "no_roots")), "--refit-scratch", str(d / "local_only" / "refit"),
            "--feasibility-report", str(d / "none.json")]
    return base + list(extra)


@pytest.fixture(scope="module")
def run_refit(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("bind_out_refit")
    assert rb.main(argv(world, out, "--allow-refit")) == 0
    return out


def test_step1_counts_are_written_first_and_are_suppressed(world, run_refit):
    j = json.loads((run_refit / "step1_counts.json").read_text())
    assert j["step"].startswith("1 (feasibility count") and j["s3"]["stop"] is False
    assert set(j["step1"]["strata"]) == {"IU_large", "whole_broad"}
    assert j["provenance"]["config_as_registered"] is True
    assert_aggregate_only(j)


def test_report_has_the_primary_pair_for_both_schemes_and_a_planted_gain(world, run_refit):
    J = json.loads((run_refit / "report.json").read_text())
    for sch in ("loso", "temporal"):
        assert set(J["primary"][sch]) == {"IU_large", "whole_broad"}
    whole = J["primary"]["loso"]["whole_broad"]
    assert whole["pair"]["estimable"] and whole["pair"]["delta_auroc"]["point"] is not None
    assert whole["pair"]["delta_auroc"]["point"] > 0                      # EEG signal planted in the silver E1 label that the imaging label tracks
    assert whole["verdict"] in {"supportive", "not_supportive", "harm"}
    assert "inconclusive, label unvalidated" in json.dumps(J["label_validity"]) or J["label_validity"]["validated"]
    assert J["predictions"]["route"].startswith("regenerated") and J["predictions"]["fits_regenerated"] > 0
    # every block of the plan is present
    for blk in ("secondary_arms", "eeg_only", "per_class", "negative_control", "sedation_intensity", "concordance", "drift", "selection", "sensitivities",
                "limitations", "settings", "step1"):
        assert blk in J, blk
    assert {"window [0, +1]", "indeterminate mapped to negative", "indeterminate mapped to positive", "hydrocephalus added to the label",
            "first imaging study only", "no mass-effect class", "imaging on or before the EEG day [-1, 0]"} <= set(J["sensitivities"])
    assert set(J["concordance"]) == {"scored_rows", "all_analysed"}
    assert J["concordance"]["scored_rows"]["overall"]["n_both_assessable"] != "<11"
    assert "MGB-only sites" in J["sensitivities"] and "not_run" in J["sensitivities"]["MGB-only sites"]
    assert J["label_validity"]["manual_vs_llama"]["available"] is True
    md = (run_refit / "report.md").read_text()
    assert "Primary pair" in md and "Report_txt" not in md


def test_no_record_level_content_in_the_outputs(world, run_refit):
    _p, _ms, cohort, _m, _d = world
    text = (run_refit / "report.json").read_text() + (run_refit / "report.md").read_text() + (run_refit / "step1_counts.json").read_text()
    for pid in cohort["person_id"].astype(str).head(200):
        assert pid not in text
    assert "ses0" not in text and "1950-01-01" not in text


def test_stored_predictions_route_reproduces_the_numbers_exactly_and_never_refits(world, run_refit, tmp_path):
    out2 = tmp_path / "again"
    scratch = world[4] / "local_only" / "refit"
    assert any(scratch.rglob("*.pkl"))
    assert rb.main(argv(world, out2, ck_root=scratch)) == 0                   # no --allow-refit: every fit must come from the stored units
    J1 = json.loads((run_refit / "report.json").read_text())
    J2 = json.loads((out2 / "report.json").read_text())
    assert J2["predictions"]["route"].startswith("loaded") and J2["predictions"]["fits_regenerated"] == 0 and J2["predictions"]["fits_from_cache"] > 0
    assert J1["primary"] == J2["primary"]


def test_missing_stored_fit_aborts_instead_of_refitting(world, tmp_path):
    with pytest.raises(rb.CacheMiss):
        rb.main(argv(world, tmp_path / "x", ck_root=tmp_path / "empty"))
    assert not (tmp_path / "x" / "report.json").exists()


def test_check_cache_mode_reads_no_bind_data(world, tmp_path):
    scratch = world[4] / "local_only" / "refit"
    assert rb.main(argv(world, tmp_path / "c", "--check-cache", ck_root=scratch)) == 0
    assert not (tmp_path / "c" / "step1_counts.json").exists()


def test_counts_only_and_stop_rule_s3(world, tmp_path, capsys):
    o1 = tmp_path / "counts"
    assert rb.main(argv(world, o1, "--counts-only")) == 0
    assert (o1 / "step1_counts.json").exists() and not (o1 / "report.json").exists()
    o2 = tmp_path / "stop"
    assert rb.main(argv(world, o2, "--min-positives", "100000")) == 0
    assert "Stopped by rule S3" in (o2 / "report.md").read_text() and not (o2 / "report.json").exists()
    assert json.loads((o2 / "step1_counts.json").read_text())["s3"]["stop"] is True


def test_config_hash_guard(world, tmp_path):
    cfgp = tmp_path / "bind_e1.yaml"
    cfgp.write_text(bind.CONFIG_PATH.read_text() + "\n# edited\n")
    with pytest.raises(SystemExit):
        rb.main(argv(world, tmp_path / "g", "--bind-config", str(cfgp), "--counts-only"))
    assert rb.main(argv(world, tmp_path / "g2", "--bind-config", str(cfgp), "--counts-only", "--allow-config-change")) == 0
    assert json.loads((tmp_path / "g2" / "step1_counts.json").read_text())["provenance"]["config_as_registered"] is False


def test_out_under_local_only_is_refused(world, tmp_path):
    with pytest.raises(SystemExit):
        rb.main(argv(world, tmp_path / "local_only" / "o", "--counts-only"))


def test_lookup_checkpoint_never_writes_the_originals(tmp_path):
    root = tmp_path / "run1"
    c = ck.Checkpoint("silver_feasibility", "k", root, quiet=True)
    c.put("fit:abc", ({"x": np.arange(3)}, {"i": 1}))
    before = sorted(p.name for p in root.rglob("*"))
    lk = rb.LookupCheckpoint([root])
    got = lk.get("fit:abc")
    assert got[1] == {"i": 1} and lk.n_hits == 1
    with pytest.raises(rb.CacheMiss):
        lk.get("fit:missing")
    with pytest.raises(rb.CacheMiss):
        lk.put("fit:new", 1)
    scratch = tmp_path / "scratch"
    lk2 = rb.LookupCheckpoint([root], scratch, allow_put=True)
    assert lk2.get("fit:new") is ck.MISS
    lk2.put("fit:new", ({"y": 1}, {}))
    assert lk2.get("fit:new")[0] == {"y": 1} and any(scratch.rglob("*.pkl"))
    assert sorted(p.name for p in root.rglob("*") if not p.name.startswith("meta")) == [n for n in before if not n.startswith("meta")]


def test_first_study_filter_keeps_only_the_earliest_brain_study():
    F = list(CFG.findings_columns.values())
    def row(person, ses, day, name, acuity):
        r = dict.fromkeys(F, "")
        r.update({"BDSPPatientID": person, "Session_id": ses, "ShiftedStudyDate": day, "Name": name, "Acute/Chronic": acuity, "Brain": "yes",
                  "Brain-related": "yes", "Nullfinding": "No"})
        return r
    fin = pd.DataFrame([row("1", "a", "2020-01-02", "no acute abnormality", "unknown"), row("1", "b", "2020-01-03", "acute infarct", "acute")], columns=F)
    demo = pd.DataFrame([["1", "a", "1950-01-01", "2020-01-02"], ["1", "b", "1950-01-01", "2020-01-03"]],
                        columns=["bdsp_patient_id", "Session_id", "ShiftedDateOfBirth", "ShiftedStudyDate"])
    sess = bind.normalise_sessions(demo, "I0001", CFG)
    f = bind.classify_findings(bind.normalise_findings(fin, CFG), CFG)
    cases = pd.DataFrame({"case_id": [0], "person_id": [1], "t0": pd.to_datetime(["2020-01-01 10:00"])})
    full = rb.label_table(cases, sess, f, CFG)
    s2, f2 = rb.first_study_filter(cases, sess, f, CFG, CFG.window_days)
    first = rb.label_table(cases, s2, f2, CFG)
    assert full.loc[0, "E1_img"] == True and first.loc[0, "E1_img"] == False and first.loc[0, "n_brain_studies"] == 1   # noqa: E712


def test_manual_agreement_is_aggregate_and_flags_disagreement():
    F = list(CFG.findings_columns.values())
    rows = []
    for i in range(100):
        r = dict.fromkeys(F, "")
        manual = "subdural hematoma" if i < 42 else "mild atrophy"
        level2 = "subdural hematoma" if (i < 30 or 42 <= i < 54) else "mild atrophy"          # 12 rows each way where the two names disagree
        r.update({"BDSPPatientID": str(i), "Session_id": "s", "Brain-related": "yes", "Brain": "yes", "Nullfinding": "No",
                  "Label_manual": manual, "Label_LLAMA_Level2": level2})
        rows.append(r)
    f = bind.classify_findings(bind.normalise_findings(pd.DataFrame(rows, columns=F), CFG), CFG)
    m = rb.manual_agreement(f, CFG)
    c = m["classes"]["sdh_edh"]
    assert m["available"] and c["manual_pos_llama_pos"] == 30 and c["manual_pos_llama_neg"] == 12 and c["manual_neg_llama_pos"] == 12
    assert 0.3 < c["kappa"] < 0.8 and m["any_primary_class_kappa_below_0.60"] is True
    assert_aggregate_only(m)
