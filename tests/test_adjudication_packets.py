"""EEG-blind rater packets: assembly, blinding, pending notes, dry run, store mode, file modes (SYNTHETIC data only)."""
import json
import re
import stat
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.adjudication import blinding, packets as pk, sources as src
from test_adjudication_fixtures import PID0, make_tables, write_cache, write_store
from test_silver_feasibility_inputs import load_script

ap = load_script("adjudication_packets")
PC = pk.PacketConfig()


def raw_for(tables, sample, pid, cfg=PC):
    uniq = sample.drop_duplicates("person_id")
    ids, to_surv = src.id_lookup(uniq)
    t0 = uniq.set_index("person_id")["t0"].to_dict()
    source = src.OmopSource("dict", tables=tables)
    raw, missing = ap.collect_structured(source, ids, to_surv, t0, cfg)
    return {t: raw[t].get(pid) for t in raw}, missing, t0


def packet_for(tables, sample, i=0, notes=None, configured=False):
    pid = int(sample["person_id"].iloc[i])
    raw, _m, _t0 = raw_for(tables, sample, pid)
    return pk.build_packet("C1234X", ap.person_row(sample.iloc[i]), raw, notes, PC, notes_source_configured=configured)


@pytest.fixture(scope="module")
def data():
    return make_tables(n=12, seed=1)


# ------------------------------------------------------------------------------------------------ structured sections
def test_window_edges_and_prior_history(data):
    t, s = data
    p = packet_for(t, s)
    sec = p.sections
    # the Sodium values at -100 h and +900 h are outside [-72 h, +30 d]; -2 h, +6 h and +400 h are inside
    sod = sec["labs"][sec["labs"]["name"] == "Sodium"]
    assert sorted(sod["rel_h"].round(0).tolist()) == [-2.0, 6.0, 400.0]
    # diagnoses: window row, a 30-day-old prior-history row, nothing older than a year
    dx = sec["diagnoses"]
    assert set(dx["code"]) == {"I63.9", "J18.9"}
    assert dx.set_index("code").loc["J18.9", "period"] == "prior history" and dx.set_index("code").loc["I63.9", "period"] == "window"
    # drugs: a drug running at the window start is kept, one that ended long before is not
    names = set(sec["medications"]["drug"])
    assert "insulin glargine" in names and "ceftriaxone" not in names and any("propofol" in n for n in names)


def test_blocked_rows_never_reach_a_packet(data):
    t, s = data
    p = packet_for(t, s)
    everything = " ".join(" ".join(df.astype(str).to_numpy().ravel()) for df in p.sections.values())
    for bad in ("BIS", "Suppression", "EEG", "G93.41", "R94.01", "95816"):
        assert bad not in everything
    assert p.dropped == {"nonspecific_encephalopathy_dx": 1, "abnormal_eeg_dx_code": 1, "eeg_procedure_code": 1, "eeg_text": 4}
    assert blinding.audit_text(pk.packet_texts(p)) == 0


def test_scores_vitals_and_labs_are_separated_and_vitals_binned(data):
    t, s = data
    sec = packet_for(t, s).sections
    assert set(sec["scores"]["name"]) == {"Glasgow Coma Scale Total"}
    assert set(sec["vitals"]["name"]) == {"Heart Rate"} and (sec["vitals"]["n"] >= 1).all()
    assert sec["vitals"]["bin_start_h"].min() >= PC.lo
    assert set(sec["labs"]["name"]) == {"Sodium"}
    # the 6-hour bins hold every in-window heart rate exactly once (41 half-hourly values from -10 h to +10 h)
    assert sec["vitals"]["n"].sum() == 41


def test_times_are_relative_and_no_absolute_dates_appear(data):
    t, s = data
    p = packet_for(t, s)
    p, page = pk.finalize(p, PC, [str(PID0)])
    assert p.withheld is None
    assert not re.search(r"20\d\d-\d\d-\d\d", page)
    assert "2024" not in page


def test_patient_without_rows_gets_an_empty_but_valid_packet(data):
    t, s = data
    p = packet_for(t, s, i=11)          # the last patient has no rows (and is a calibration case)
    assert all(len(df) == 0 for df in p.sections.values()) and p.counts["notes"] == 0
    p, page = pk.finalize(p, PC, [])
    assert p.withheld is None and "None in the window" in page


# ------------------------------------------------------------------------------------------------ notes
def prepared_notes(t, s, i=0, cfg=PC):
    raw_n = t["omop_note"]
    uniq = s.drop_duplicates("person_id")
    ids, to_surv = src.id_lookup(uniq)
    notes: dict = {}
    ap.ingest_notes(raw_n, to_surv, uniq.set_index("person_id")["t0"].to_dict(), cfg, notes, Counter())
    return notes.get(int(s["person_id"].iloc[i]), [])


def test_eeg_notes_are_excluded_and_eeg_sentences_redacted(data):
    t, s = data
    notes = prepared_notes(t, s)
    titles = [n["title"] for n in notes]
    assert "EEG report" not in titles and "Old Note" not in titles                 # the EEG note and the note outside the window are gone
    text = " ".join(n["text"] for n in notes)
    for bad in ("EEG", "slowing", "triphasic", "Burst", "BIS"):
        assert bad not in text
    assert "Found down at home" in text and "Sodium 118" in text and "continue levetiracetam" in text      # the clean sentences survive
    assert [n["title"] for n in notes] == ["Admission H&P", "Neurology consult", "Progress Note"]          # time ordered


def test_pending_status_when_no_note_source_and_the_html_says_so(data):
    t, s = data
    p = packet_for(t, s, notes=None, configured=False)
    assert p.notes_status == "pending_no_source" and p.counts["notes"] == 0
    _, page = pk.finalize(p, PC, [])
    assert "NOTES PENDING" in page
    p2 = packet_for(t, s, notes=[], configured=True)
    assert p2.notes_status == "none_in_window"


def test_note_cap_truncates_in_time_order():
    notes = [{"rel_h": float(i), "date_only": False, "title": "n", "text": "x" * 100} for i in range(10)]
    kept, trunc = pk.cap_notes(notes, pk.PacketConfig(max_notes=4))
    assert len(kept) == 4 and trunc and [n["rel_h"] for n in kept] == [0.0, 1.0, 2.0, 3.0]
    kept, trunc = pk.cap_notes(notes, pk.PacketConfig(max_note_chars=250))
    assert len(kept) == 2 and trunc


def test_packet_with_the_patients_own_id_is_withheld(data):
    t, s = data
    notes = [{"rel_h": 1.0, "date_only": False, "title": "Note", "text": f"Registered as {PID0} on arrival."}]
    p, _ = pk.finalize(packet_for(t, s, notes=notes, configured=True), PC, [str(PID0)])
    assert p.withheld == "contains_patient_id"


def test_packet_that_still_mentions_eeg_is_withheld(data):
    t, s = data
    notes = [{"rel_h": 1.0, "date_only": False, "title": "Note", "text": "The tracing was reviewed."},
             {"rel_h": 2.0, "date_only": False, "title": "Note", "text": "BIS 40 recorded."}]            # bypasses the filter on purpose
    p, _ = pk.finalize(packet_for(t, s, notes=notes, configured=True), PC, [])
    assert p.withheld == "eeg_audit"


# ------------------------------------------------------------------------------------------------ pseudonyms, assignments
def test_case_ids_are_unique_reproducible_and_checkable():
    ids = pk.make_case_ids(300, 5)
    assert len(set(ids)) == 300 and ids == pk.make_case_ids(300, 5) and ids != pk.make_case_ids(300, 6)
    assert all(pk.valid_case_id(i) for i in ids)
    bad = ids[0][:-1] + ("A" if ids[0][-1] != "A" else "B")
    assert not pk.valid_case_id(bad) and not pk.valid_case_id("C12345")


def make_id_map(n=200, n_dup=20, n_cal=5, seed=3):
    ids = pk.make_case_ids(n + n_dup + n_cal, seed)
    rows = [{"case_id": ids[i], "role": "sample", "duplicate_of_case": ""} for i in range(n)]
    rows += [{"case_id": ids[n + j], "role": "duplicate", "duplicate_of_case": ids[j]} for j in range(n_dup)]
    rows += [{"case_id": ids[n + n_dup + j], "role": "calibration", "duplicate_of_case": ""} for j in range(n_cal)]
    return pd.DataFrame(rows)


def test_assignments_show_duplicates_late_and_orders_differ_by_rater():
    im = make_id_map()
    asg = pk.make_assignments(im, ("R1", "R2"), seed=9, batch_size=20)
    for r in ("R1", "R2"):
        a = asg[asg["rater"] == r].sort_values("order")
        assert sorted(a["case_id"]) == sorted(im["case_id"]) and a["order"].tolist() == list(range(1, 226))
        pos = dict(zip(a["case_id"], a["order"]))
        cal = im[im["role"] == "calibration"]["case_id"]
        assert max(pos[c] for c in cal) == 5 and a[a["case_id"].isin(cal)]["batch"].eq(0).all()          # calibration first, batch 0
        for _, d in im[im["role"] == "duplicate"].iterrows():
            assert pos[d["case_id"]] - pos[d["duplicate_of_case"]] > 20                                     # at least a batch apart
    o1 = asg[asg["rater"] == "R1"].sort_values("order")["case_id"].tolist()
    o2 = asg[asg["rater"] == "R2"].sort_values("order")["case_id"].tolist()
    assert o1 != o2
    pd.testing.assert_frame_equal(asg, pk.make_assignments(im, ("R1", "R2"), seed=9, batch_size=20))      # deterministic


# ------------------------------------------------------------------------------------------------ the script
def write_inputs(root: Path, n=40, empty=()):
    t, s = make_tables(n=n, seed=2, empty_idx=empty or (n - 1,))
    lo = root / "local_only"
    lo.mkdir(parents=True, exist_ok=True)
    s2 = s.copy()
    s2["t0"] = s2["t0"].dt.strftime("%Y-%m-%d %H:%M:%S")
    s2.to_csv(lo / "sample.csv", index=False)
    write_cache(t, root / "cache")
    nd = root / "notes"
    nd.mkdir(exist_ok=True)
    t["omop_note"].to_parquet(nd / "notes.parquet")
    return t, s


def argv(root: Path, *extra):
    lo = root / "local_only"
    return ["--sample", str(lo / "sample.csv"), "--omop-cache", str(root / "cache"), "--summary-out", str(root / "agg" / "summary.json"),
            "--out-dir", str(lo / "packets"), "--id-map-out", str(lo / "id_map.csv"), "--assignments-out", str(lo / "assign.csv"),
            "--redcap-ids-out", str(lo / "rc.csv"), *extra]


def test_dry_run_reports_aggregate_availability_and_writes_no_packet(tmp_path, capsys):
    empty = tuple(range(1, 41))                                  # 40 of the 78 sampled patients have no data and no notes
    t, s = write_inputs(tmp_path, n=80, empty=empty)
    assert ap.main(argv(tmp_path, "--dry-run")) == 0
    out = capsys.readouterr().out
    rep = json.loads((tmp_path / "agg" / "summary.json").read_text())
    assert not (tmp_path / "local_only" / "packets").exists() and not (tmp_path / "local_only" / "id_map.csv").exists()
    assert rep["mode"] == "dry_run" and rep["n_cases"]["pooled"] == 78
    av = rep["availability"]
    assert av["any_structured"]["all_sites"]["n"] == 38 and av["any_structured"]["all_sites"]["den"] == 78
    for k in ("diagnoses", "medications", "labs", "scores", "vitals"):
        assert av[k]["all_sites"]["n"] == 38
    assert av["notes_timestamp_in_window"]["all_sites"]["n"] == 38
    assert "timestamps only" in rep["sources"]["notes_text_status"] and rep["sources"]["notes"] == "cache"
    assert rep["blinding"]["rows_or_sentences_removed_by_reason"]["eeg_text"] == 152           # 4 per patient x 38
    # stdout: counts only, no id of any kind
    assert re.search(r"any_structured: 38 of 78", out)
    for pid in t["omop_measurement"]["person_id"].unique()[:5]:
        assert str(pid) not in out
    assert not re.search(r"\bC\d{4}[A-Z]\b", out) and "s0001" not in out


def test_dry_run_refuses_to_stream_notes(tmp_path):
    write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        ap.main(argv(tmp_path, "--dry-run", "--notes-source", "store"))


def test_full_build_without_notes_says_pending_and_still_builds_the_structured_part(tmp_path, capsys):
    t, s = write_inputs(tmp_path, n=14)
    assert ap.main(argv(tmp_path)) == 0
    capsys.readouterr()
    packets = sorted((tmp_path / "local_only" / "packets").iterdir())
    assert len(packets) == 14
    first = next(p for p in packets if (p / "structured" / "labs.csv").stat().st_size > 20)
    assert not (first / "manifest.json").exists()                        # the manifest names EEG-related drop reasons: not in the rater's folder
    man = json.loads((tmp_path / "local_only" / "manifests" / f"{first.name}.json").read_text())
    assert man["notes_status"] == "pending_no_source" and man["counts"]["labs"] >= 1 and not (first / "notes").exists()
    assert "NOTES PENDING" in (first / "packet.html").read_text()


def test_full_build_with_note_directory_writes_filtered_blind_packets(tmp_path, capsys):
    t, s = write_inputs(tmp_path, n=14)
    assert ap.main(argv(tmp_path, "--notes-source", "dir", "--notes-dir", str(tmp_path / "notes"))) == 0
    out = capsys.readouterr().out
    root = tmp_path / "local_only"
    idm = pd.read_csv(root / "id_map.csv")
    assert idm["packet_status"].eq("written").sum() == 14 and idm["case_id"].is_unique
    pids = {str(p) for p in s["person_id"]}
    n_files = 0
    for f in sorted((root / "packets").rglob("*")):
        if f.is_file():
            n_files += 1
            text = f.read_text()
            body = text.split("<h2>Clinical notes</h2>")[1] if f.name == "packet.html" else text      # the fixed banner above says "EEG-blind"
            assert blinding.audit_text([body]) == 0
            assert stat.S_IMODE(f.stat().st_mode) == 0o600
            for pid in pids:
                assert pid not in text
        elif f.is_dir():
            assert stat.S_IMODE(f.stat().st_mode) == 0o700
    assert n_files > 14 * 8
    # note text is filtered, not summarised, and a note appears for each case with notes
    note_files = list((root / "packets").rglob("note_*.txt"))
    assert note_files and all("EEG" not in f.read_text() for f in note_files)
    for p in (root / "id_map.csv", root / "assign.csv", root / "rc.csv"):
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    # stdout: counts only
    for cid in idm["case_id"]:
        assert cid not in out
    for needle in ("Found down", "Sodium 118", "levetiracetam"):
        assert needle not in out
    # the duplicate gets its own pseudonym and an identical body
    dup = idm[idm["role"] == "duplicate"].iloc[0]
    orig = idm[idm["case_id"] == dup["duplicate_of_case"]].iloc[0]
    a = (root / "packets" / dup["case_id"] / "structured" / "labs.csv").read_text()
    b = (root / "packets" / orig["case_id"] / "structured" / "labs.csv").read_text()
    assert dup["case_id"] != orig["case_id"] and a == b
    asg = pd.read_csv(root / "assign.csv")
    assert set(asg["rater"]) == {"R1", "R2"} and set(asg["case_id"]) == set(idm["case_id"])
    assert pd.read_csv(root / "rc.csv")["case_id"].is_unique


def test_script_requires_local_only_paths(tmp_path):
    t, s = make_tables(n=14)
    (tmp_path / "x").mkdir()
    s.to_csv(tmp_path / "x" / "sample.csv", index=False)
    with pytest.raises(SystemExit):
        ap.main(["--sample", str(tmp_path / "x" / "sample.csv"), "--dry-run"])


def test_store_mode_streams_structured_data_and_notes_from_a_local_mirror(tmp_path, capsys):
    t, s = write_inputs(tmp_path, n=14)
    store = write_store(t, tmp_path / "mirror")
    # an empty cache: everything must come from the store
    assert ap.main(argv(tmp_path, "--data", str(store), "--structured-source", "store", "--notes-source", "store")) == 0
    capsys.readouterr()
    idm = pd.read_csv(tmp_path / "local_only" / "id_map.csv")
    assert idm["packet_status"].eq("written").all()
    notes = list((tmp_path / "local_only" / "packets").rglob("note_*.txt"))
    assert notes and all("EEG" not in f.read_text() for f in notes)
    man = json.loads(next((tmp_path / "local_only" / "manifests").glob("*.json")).read_text())
    assert man["notes_status"] in ("included", "none_in_window")


def test_missing_values_in_text_columns_do_not_break_or_leak_into_a_packet():
    """Real tables carry NaN in text columns (pandas keeps them missing under astype(str)): the packet must still build and audit."""
    t, s = make_tables(n=12, seed=1)
    t = {k: v.copy() for k, v in t.items()}
    t["omop_drug_exposure"].loc[t["omop_drug_exposure"].index[::3], "route_source_value"] = np.nan
    t["omop_drug_exposure"].loc[t["omop_drug_exposure"].index[1], "drug_source_value"] = np.nan
    t["omop_condition_occurrence"].loc[t["omop_condition_occurrence"].index[0], "condition_source_value"] = np.nan
    m = t["omop_measurement"]
    m.loc[m.index[5], ["measurement_source_value"]] = np.nan
    m.loc[m["measurement_source_value"] == "Sodium", "value_as_number"] = np.nan                 # a lab with no value at all
    t["omop_observation"].loc[t["omop_observation"].index[0], "value_as_string"] = np.nan
    p = packet_for(t, s)
    p, page = pk.finalize(p, PC, [str(PID0)])
    assert p.withheld is None
    body = page.split("<h2>Clinical notes</h2>")[1]
    assert ">nan<" not in body and "None<" not in body
    assert (p.sections["medications"]["drug"] != "").all()


def test_html_collapses_repeated_diagnosis_codes_and_indexes_the_notes(data):
    t, s = data
    t = {k: v.copy() for k, v in t.items()}
    extra = t["omop_condition_occurrence"].query("condition_source_value == 'I63.9'").copy()
    reps = []
    for d in range(1, 6):                                                     # the same code listed on five later days
        e = extra.copy()
        e["condition_start_datetime"] = e["condition_start_datetime"] + pd.Timedelta(days=d)
        reps.append(e)
    t["omop_condition_occurrence"] = pd.concat([t["omop_condition_occurrence"], *reps], ignore_index=True)
    notes = prepared_notes(t, s)
    p = packet_for(t, s, notes=notes, configured=True)
    _, page = pk.finalize(p, PC, [])
    assert len(p.sections["diagnoses"]) > 3                                    # every listing stays in the CSV table
    assert page.count("I63.9") == 1 and "Distinct days" in page and "Prior history (before the window)" in page
    assert "Index:" in page and "href='#n3'" in page and "id='n3'" in page


def test_notes_window_can_be_shortened_independently():
    c = pk.PacketConfig(after_days=30, notes_after_days=7)
    assert c.notes_hi == 7 * 24.0 and c.hi == 30 * 24.0
    assert pk.PacketConfig().notes_hi == 30 * 24.0
