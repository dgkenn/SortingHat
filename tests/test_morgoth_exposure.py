"""Exposure accounting on SYNTHETIC id lists (never real MORGOTH lists)."""
import importlib.util
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.morgoth import exposure as ex
from sortinghat.safe_output import assert_aggregate_only

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_morgoth.py"
spec = importlib.util.spec_from_file_location("extract_morgoth", SCRIPT)
xm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xm)


def cohort(n=40):
    return pd.DataFrame({"person_id": np.arange(1000001, 1000001 + n), "SiteID": ["S0001"] * (n // 2) + ["I0002"] * (n - n // 2),
                         "person_id_source": np.arange(1000001, 1000001 + n)})


def test_ids_from_cells_handles_bids_and_plain_forms():
    got = ex.ids_from_cells(["sub-S0001001234567_ses-1_x.mat", "I0002 1234", "S0001001234568", None, float("nan"),
                             "no id here", "S00010123"])
    assert got == [1234567, 1234568]                  # 'S0001' + >= 6 digits; short tails and free text are not ids


def test_scan_csv_and_xlsx_lists(tmp_path):
    pd.DataFrame({"BidsFolder": ["sub-S0001001000005", "sub-S0001001000006"], "label": ["a", "b"]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"BDSPPatientID": [1000010, 1000011.0], "x": [1, 2]}).to_excel(tmp_path / "b.xlsx", index=False)
    idx = ex.scan_lists([tmp_path / "a.csv", tmp_path / "b.xlsx"])
    assert idx.ids == {1000005, 1000006, 1000010, 1000011}
    assert idx.n_files == 2 and idx.n_files_with_ids == 2 and not idx.has_split_info


def test_split_column_gives_train_ids(tmp_path):
    pd.DataFrame({"file": ["S0001001000001_a", "S0001001000002_b", "S0001001000003_c"],
                  "split": ["train", "test", "Train"]}).to_csv(tmp_path / "s.csv", index=False)
    idx = ex.scan_lists([tmp_path / "s.csv"])
    assert idx.has_split_info and idx.train_ids == {1000001, 1000003} and idx.ids == {1000001, 1000002, 1000003}


def test_merge_map_and_pre_merge_id():
    idx = ex.TrainingIndex()
    ex.scan_frame(pd.DataFrame({"patient": ["S0001001000007"]}), idx, merge_map={1000007: 1000099})
    c = cohort(4)
    c.loc[1, "person_id"] = 1000099                    # post-merge id
    c.loc[2, "person_id_source"] = 1000007             # id before the merge map
    f = ex.flag_cohort(c, idx)
    assert bool(f.loc[1, "in_morgoth_lists"]) and bool(f.loc[2, "in_morgoth_lists"])


def test_flags_and_summary_are_correct_and_aggregate_only(tmp_path):
    idx = ex.TrainingIndex()
    ex.add_names(idx, [f"morgoth1/data/pretrain/S0001{i:06d}_seg.mat" for i in range(1000001, 1000015)])
    f = ex.flag_cohort(cohort(40), idx)
    assert f["in_morgoth_lists"].sum() == 14 and f["in_morgoth_lists"].iloc[:14].all()
    s = ex.summarize(f, idx)
    assert_aggregate_only(s)
    assert s["n_in_lists"] == 14 and s["n_cohort"] == 40 and "warning" not in s
    assert set(s["by_site"]) == {"site_1", "site_2"}


def test_small_cells_suppressed():
    idx = ex.TrainingIndex()
    ex.add_names(idx, ["S0001001000001"])
    s = ex.summarize(ex.flag_cohort(cohort(40), idx), idx)
    assert s["n_in_lists"] == "<11" and s["in_lists_proportion"] == "<11"


def test_zero_parsed_ids_is_a_warning_not_zero_exposure(tmp_path):
    pd.DataFrame({"hash": ["abc", "def"]}).to_csv(tmp_path / "h.csv", index=False)
    idx = ex.scan_lists([tmp_path / "h.csv"])
    assert len(idx.ids) == 0
    assert "warning" in ex.summarize(ex.flag_cohort(cohort(20), idx), idx)


def test_write_flags_only_under_local_only(tmp_path):
    f = ex.flag_cohort(cohort(12), ex.TrainingIndex())
    with pytest.raises(ValueError):
        ex.write_flags(f, tmp_path / "flags.parquet")
    p = ex.write_flags(f, tmp_path / "local_only" / "flags.parquet")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and len(pd.read_parquet(p)) == 12


def test_inspect_prints_names_only(tmp_path, capsys):
    pd.DataFrame({"BidsFolder": ["sub-S0001001000005"], "split": ["train"], "note": ["SECRETVALUE"]}).to_csv(tmp_path / "a.csv", index=False)
    d = ex.inspect_columns(tmp_path / "a.csv")["sheet0"]
    assert d["columns"] == ["BidsFolder", "split", "note"] and d["split_columns"] == ["split"] and "BidsFolder" in d["id_columns"]
    rc = xm.main(["exposure", "--cohort", "x", "--lists", str(tmp_path / "a.csv"), "--out", "x", "--inspect"])
    out = capsys.readouterr().out
    assert rc == 0 and "BidsFolder" in out and "SECRETVALUE" not in out and "1000005" not in out


def test_script_exposure_end_to_end_stdout_has_no_ids(tmp_path, capsys):
    lo = tmp_path / "local_only"
    lo.mkdir()
    c = cohort(40)
    c.to_csv(lo / "cohort.csv", index=False)
    pd.DataFrame({"BidsFolder": [f"sub-S0001{i:06d}" for i in range(1000001, 1000021)]}).to_csv(lo / "list.csv", index=False)
    rc = xm.main(["exposure", "--cohort", str(lo / "cohort.csv"), "--lists", str(lo / "list.csv"),
                  "--out", str(lo / "mg" / "exposure.parquet"), "--summary", str(tmp_path / "s.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "n_in_lists: 20" in out and "1000001" not in out and "S0001" not in out
    flags = pd.read_parquet(lo / "mg" / "exposure.parquet")
    assert flags["in_morgoth_lists"].sum() == 20
    with pytest.raises(SystemExit):
        xm.main(["exposure", "--cohort", str(lo / "cohort.csv"), "--lists", str(lo / "list.csv"),
                 "--out", str(tmp_path / "plain.parquet")])


def test_master_table_morgoth_column_is_membership_and_split(tmp_path):
    """Master-table layout (SYNTHETIC): a blank Morgoth cell means not a MORGOTH patient; its value is the split."""
    master = pd.DataFrame({"BDSPPatientID": [1000001, 1000002, 1000003, 1000004, 1000005],
                           "SiteID": ["S0001"] * 5,
                           "Morgoth": ["pretrain", "train", "test", np.nan, None],
                           "SpikeNet": ["train", np.nan, np.nan, "train", "test"]})
    task = pd.DataFrame({"bdsp_mrn": [1000006, 1000007], "file_name": ["sub-S0001001000006_ses-1", "S0001001000007_1_20240101"],
                         "label": [1, 0]})
    with pd.ExcelWriter(tmp_path / "datasets_deidentified_list.xlsx") as w:
        master.to_excel(w, sheet_name="a", index=False)
    task.to_csv(tmp_path / "IIIC__list.csv", index=False)
    idx = ex.scan_lists([tmp_path / "datasets_deidentified_list.xlsx", tmp_path / "IIIC__list.csv"])
    assert idx.ids == {1000001, 1000002, 1000003, 1000006, 1000007}          # blank-Morgoth rows are not members
    assert idx.train_ids == {1000001, 1000002}                                # train + pretrain
    assert idx.unsplit_ids == {1000006, 1000007} and not idx.split_complete   # per-task lists carry no split
    c = pd.DataFrame({"person_id": np.arange(1000001, 1000009), "person_id_source": np.arange(1000001, 1000009),
                      "SiteID": ["S0001"] * 8})
    f = ex.flag_cohort(c, idx).set_index("person_id")
    assert f["in_morgoth_lists"].tolist() == [True, True, True, False, False, True, True, False]
    assert f["in_morgoth_train_or_pretrain"].tolist() == [True, True, False, False, False, False, False, False]
    assert f["in_morgoth_pretrain"].tolist() == [True, False, False, False, False, False, False, False]
    assert f["in_morgoth_test_only"].tolist() == [False, False, True, False, False, False, False, False]
    assert f["in_morgoth_train_split"].isna().all()                           # incomplete split info: no narrow flag
    src = ex.source_counts(c, idx)
    assert "datasets_deidentified_list.xlsx:pretrain" in src and "IIIC__list.csv" in src


def test_complete_split_emits_train_flag_and_summary_has_sources():
    idx = ex.TrainingIndex()
    ex.scan_frame(pd.DataFrame({"BDSPPatientID": list(range(1000001, 1000021)), "Morgoth": ["train"] * 12 + ["test"] * 8}), idx,
                  label="m.xlsx")
    c = cohort(40)
    f = ex.flag_cohort(c, idx)
    assert idx.split_complete and f["in_morgoth_train_split"].sum() == 12 and f["in_morgoth_test_only"].sum() == 8
    s = ex.summarize(f, idx)
    assert_aggregate_only(s)
    assert s["split_complete"] and s["n_in_lists"] == 20
    assert_aggregate_only(ex.source_counts(c, idx))
