"""Row-alignment regression: SessionID is a per-patient counter (many patients have session "1"), so a join on
SessionID alone gave thousands of patients the same BidsFolder / SessionID / edf_key (real run: 12,826 patients,
33 distinct BidsFolders). Reproduced here at scale with realistic id formats, and the integrity check must abort."""

import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sortinghat import data_io, schema
from sortinghat.audit import field_audit as fa
from sortinghat.cohort import integrity
from sortinghat.cohort import CohortConfig, FrameSources, StoreSources, build_cohort, sources
from sortinghat.cohort.integrity import CohortIntegrityError, verify_rows
from sortinghat.data_io import LocalStore
from sortinghat import tableio
from test_cohort_helpers import GCS, H, T0, World

N = 6000
ID0 = 1_111_111_111              # realistic 10-digit BDSPPatientID: BidsFolder 'sub-S0001' + '1111111111...'
CFG = CohortConfig(study_sites=None)


def pid(i):
    return ID0 + 7 * i


def t0_of(i):
    return T0 + pd.Timedelta(minutes=3 * i)


@pytest.fixture(scope="module")
def world():
    w = World()
    for i in range(N):
        site = "S0001" if i % 3 else "S0002"
        w.eeg(pid(i), site=site, t0=t0_of(i), sid="1", id_blank=(i % 2 == 0))       # every patient's first session is "1"
        w.eeg(pid(i), site=site, t0=t0_of(i) + pd.Timedelta(days=40), sid="2", id_blank=(i % 2 == 0))
        if i % 5 == 0:
            w.eeg(pid(i), site=site, t0=t0_of(i) + pd.Timedelta(days=90), sid="3", id_blank=(i % 2 == 0))
        w.visit(pid(i), "ICU", t0_of(i) - H(3), end=t0_of(i) + H(50))
        w.score(pid(i), GCS, t0_of(i) - H(1), 8)
    return w


def check_rows(table):
    assert len(table) > 5000
    site = np.where(table["SiteID"] == "S0001", "S0001", "S0002")
    ids = table["person_id"].to_numpy()
    assert (table["BidsFolder"] == [f"sub-{s}{p}" for s, p in zip(site, ids)]).all()
    i = (ids - ID0) // 7
    assert (table["t0"] == [t0_of(int(k)) for k in i]).all()
    assert (table["SessionID"] == "1").all()
    assert table["BidsFolder"].nunique() == len(table) == table[["BidsFolder", "SessionID"]].drop_duplicates().shape[0]


def test_frame_source_rows_stay_aligned_with_patients(world):
    r = build_cohort(FrameSources(world.tables()), CFG)
    check_rows(r.table)
    assert r.keys["edf_key"].is_unique and r.keys["edf_key"].str.contains("sub-S000").all()


def test_store_source_with_several_row_groups_stays_aligned(world, tmp_path):
    d = write_store(world, tmp_path / "s")
    r = build_cohort(StoreSources(LocalStore(d)), CFG)
    check_rows(r.table)
    mem = build_cohort(FrameSources(world.tables()), CFG)
    assert r.table[["person_id", "BidsFolder", "SessionID", "t0"]].equals(mem.table[["person_id", "BidsFolder", "SessionID", "t0"]])


def write_store(world, d):
    """The HEEDB layout on disk. Written here rather than with tableio, whose writer maps findings rows to sites
    through a SessionID -> site lookup (it assumes SessionID is unique, which real data is not)."""
    t = world.tables()
    for site, g in t["eeg_metadata"].groupby("SiteID"):
        out = d / "EEG/eeg-metadata"
        out.mkdir(parents=True, exist_ok=True)
        data_io.denormalise_site_table("eeg_metadata", site, tableio._as_text("eeg_metadata", g)).to_csv(
            out / f"{site}_eeg_metadata_2026_04_30.csv", index=False)
        out = d / "EEG/HEEDB_Metadata"
        out.mkdir(parents=True, exist_ok=True)
        r = t["reports_findings"]
        r = r[r["BDSPPatientID"].isin(g["BidsFolder"].str.replace(f"sub-{site}", "", regex=False))]
        data_io.denormalise_site_table("reports_findings", site, tableio._as_text("reports_findings", r)).to_csv(
            out / f"{site}_EEG__reports_findings.csv", index=False)
    for name in ("omop_visit_occurrence", "omop_measurement", "omop_condition_occurrence"):
        df = tableio._as_text(name, t[name])
        out = d / f"OMOP/Merged/{name[5:]}"
        out.mkdir(parents=True, exist_ok=True)
        for k in range(2):                                                         # 2 parts x several row groups
            pq.write_table(pa.Table.from_pandas(df.iloc[k::2], preserve_index=False), out / f"part-{k:05d}.parquet",
                           row_group_size=500)
        assert sum(pq.ParquetFile(p).num_row_groups for p in out.glob("*.parquet")) >= (3 if name.endswith("visit_occurrence") else 1)
    return d


def _buggy_site_sessions(meta, rf, site):
    """The pre-fix implementation: metadata columns joined on SessionID ALONE."""
    m = fa.merge_eeg(meta, rf, site).drop(columns=["ServiceName"], errors="ignore")
    extra = pd.DataFrame({"SessionID": meta["SessionID"].astype("string"), "BidsFolder": meta["BidsFolder"].astype("string"),
                          "EEGFolder": meta["EEGFolder"].astype("string"), "ServiceName": meta["ServiceName"].astype("string"),
                          "duration_raw_s": pd.to_numeric(meta["DurationInSeconds"], errors="coerce")}
                         ).drop_duplicates("SessionID")
    out = m.merge(extra, on="SessionID", how="left")
    out["SiteID"] = site
    out = out.rename(columns={"StartTime": "t0", "EndTime": "t_end", "AgeAtVisit": "age_years"})
    out["t0"] = schema.parse_datetimes(out["t0"])
    out["t_end"] = schema.parse_datetimes(out["t_end"])
    return out[sources.SESSION_COLUMNS]


def test_the_old_join_corrupts_and_the_integrity_check_aborts(world, monkeypatch):
    monkeypatch.setattr(sources, "site_sessions", _buggy_site_sessions)
    s = FrameSources(world.tables()).sessions()
    assert s["BidsFolder"].nunique() < 10                                         # the symptom: a handful of folders
    with pytest.raises(CohortIntegrityError) as e:
        build_cohort(FrameSources(world.tables()), CFG)
    msg = str(e.value)
    assert "ROW INTEGRITY VIOLATED" in msg and "sub-" not in msg and "1111111" not in msg   # aggregate message only
    # with the check switched off the old join really does produce a misaligned cohort (the regression itself)
    monkeypatch.setattr(integrity, "verify_rows", lambda *a, **k: None)
    monkeypatch.setattr(integrity, "verify_output", lambda *a, **k: None)
    with pytest.raises(AssertionError):
        check_rows(build_cohort(FrameSources(world.tables()), CFG).table)


def test_verify_rows_catches_each_kind_of_misalignment():
    src = pd.DataFrame({"SiteID": ["S0001"] * 2, "BidsFolder": ["sub-S000111", "sub-S000122"], "SessionID": ["1", "1"],
                        "pid_folder": pd.array([11, 22], dtype="Int64"), "start_src": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    good = pd.DataFrame({"SiteID": ["S0001"] * 2, "BidsFolder": src["BidsFolder"], "SessionID": ["1", "1"],
                         "person_id_source": pd.array([11, 22], dtype="Int64"), "t0": src["start_src"]})
    verify_rows(good, src, "t")
    for col, val in (("BidsFolder", "sub-S000122"), ("person_id_source", 99), ("t0", pd.Timestamp("2021-01-01")),
                     ("SessionID", "7")):
        bad = good.copy()
        bad.loc[0, col] = val
        with pytest.raises(CohortIntegrityError):
            verify_rows(bad, src, "t")


def test_edf_key_is_the_extractors_first_candidate(world):
    from sortinghat import data_io
    r = build_cohort(FrameSources(world.tables()), CFG)
    k = r.keys.iloc[0]
    assert k["edf_key"] == data_io.bids_edf_candidates(k["SiteID"], k["BidsFolder"], k["SessionID"],
                                                       None if pd.isna(k["EEGFolder"]) else k["EEGFolder"])[0][1]


def test_store_and_frame_sources_agree_on_the_synthetic_cohort(synth, synth_dir):
    """The streaming (compact, pruned, row-group) readers must give the same cohort as the in-memory tables, under
    the default (D-111..D-115) configuration: date-only visit cover included."""
    a = build_cohort(FrameSources(synth[0])).table.reset_index(drop=True)
    b = build_cohort(StoreSources(LocalStore(synth_dir))).table.reset_index(drop=True)
    cols = ["SiteID", "person_id", "SessionID", "BidsFolder", "t0", "acute_basis", "visit_inpatient_length", "onset",
            "onset_basis", "hours_since_onset", "severity_strict", "severity_strict_pm6", "phenotype", "in_strict",
            "in_strict_pm6", "in_broad", "duration_s"]
    assert len(a) == len(b) > 100
    pd.testing.assert_frame_equal(a[cols].astype(str), b[cols].astype(str))


# ---------------------------------------------------------------------------------------------------- D-116 and precision
def _small_world(n=4):
    w = World()
    for i in range(n):
        w.patient(pid(i), site="S0001", t0=t0_of(i), sid="1")
    return w


def _flow_removed(r, label):
    from test_cohort_helpers import removed_at
    return removed_at(r, label)


def test_patient_id_disagreeing_with_bidsfolder_is_excluded_not_an_abort():
    w = _small_world()
    other = pid(99)
    w.patient(other, site="S0001", t0=t0_of(99), sid="1")
    t = w.tables()
    for name in ("eeg_metadata", "reports_findings"):                       # the I0002 situation: BidsFolder keeps the old id
        d = t[name]
        d.loc[d["BDSPPatientID"].astype(str) == str(other), "BDSPPatientID"] = str(other + 1)
    r = build_cohort(FrameSources(t), CFG)
    assert _flow_removed(r, "Patient id in BDSPPatientID disagrees with BidsFolder") == 1
    assert other not in set(r.table["person_id"]) and other + 1 not in set(r.table["person_id"])
    assert len(r.table) == 4


def test_report_start_differing_from_metadata_start_passes():
    w = _small_world()
    t = w.tables()
    t["eeg_metadata"].loc[0, "StartTime"] = t0_of(0) + pd.Timedelta(days=3)           # metadata start differs from the report's
    rf = t["reports_findings"]
    dup = rf.iloc[[1]].copy()                                                          # a second report row, later start, listed first
    dup[schema.START_EEG] = dup[schema.START_EEG] + pd.Timedelta(days=5)
    t["reports_findings"] = pd.concat([dup, rf], ignore_index=True)
    r = build_cohort(FrameSources(t), CFG)
    assert len(r.table) == 4


def test_misaligned_bidsfolder_between_patients_still_aborts(monkeypatch):
    w = _small_world()
    real = sources.site_sessions

    def swapped(meta, rf, site):
        s = real(meta, rf, site)
        b = s["BidsFolder"].to_numpy(object).copy()
        b[[0, 1]] = b[[1, 0]]                                                          # both patients have SessionID "1"
        s["BidsFolder"] = b
        return s

    monkeypatch.setattr(sources, "site_sessions", swapped)
    with pytest.raises(CohortIntegrityError) as e:
        build_cohort(FrameSources(w.tables()), CFG)
    msg = str(e.value)
    assert "patient-id mismatch (key found, id differs from the BidsFolder id) <11" in msg
    assert "key missing" in msg and "start-time mismatch" in msg and "sub-" not in msg.replace("BidsFolder", "")


def test_failure_message_breaks_down_causes_with_suppressed_counts():
    src = pd.DataFrame({"SiteID": ["S0001"] * 3, "BidsFolder": ["sub-S000111", "sub-S000122", "sub-S000133"],
                        "SessionID": ["1"] * 3, "pid_folder": pd.array([11, 22, 33], dtype="Int64"),
                        "start_src": pd.to_datetime(["2020-01-01"] * 3)})
    rows = pd.DataFrame({"SiteID": ["S0001"] * 3, "BidsFolder": ["sub-S000111", "sub-S000122", "sub-S000199"],
                         "SessionID": ["1"] * 3, "person_id_source": pd.array([11, 99, 55], dtype="Int64"),
                         "t0": pd.to_datetime(["2021-01-01", "2020-01-01", "2020-01-01"])})
    with pytest.raises(CohortIntegrityError) as e:
        verify_rows(rows, src, "t")
    msg = str(e.value)
    assert "key missing (no source row) <11" in msg and "patient-id mismatch" in msg and "start-time mismatch" in msg


def test_source_index_keeps_metadata_and_every_report_start():
    meta = pd.DataFrame({"BidsFolder": ["sub-S000111"], "SessionID": ["1"], "StartTime": ["2020-01-01 00:00:00"]})
    rf = pd.DataFrame({"BDSPPatientID": ["11", "11"], "SessionID": ["1", "1"],
                       schema.START_EEG: ["2020-02-01 00:00:00", "2020-03-01 00:00:00"]})
    idx = integrity.source_index(meta, rf, "S0001")
    assert sorted(idx["start_src"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-02-01", "2020-03-01"]
