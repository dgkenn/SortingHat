import pandas as pd

from sortinghat import data_io, schema
from sortinghat.synthetic import generate
from sortinghat.tableio import load_tables


def test_shape_and_schema(synth):
    tables, _ = synth
    assert set(tables) == set(schema.TABLE_NAMES)
    for name, df in tables.items():
        assert list(df.columns) == schema.columns(name)
    eeg = tables["eeg_metadata"]
    assert eeg["SessionID"].nunique() == len(eeg)
    assert eeg["BidsFolder"].nunique() == 3000
    assert eeg["SiteID"].nunique() == 4
    assert tables["omop_person"]["person_id"].nunique() == 3000


def test_real_names_and_blank_columns(synth):
    tables, _ = synth
    assert "DurationInSeconds" in tables["eeg_metadata"] and "DurationInSecond" not in tables["eeg_metadata"]
    assert {"StartTime(EEG)", "EndTime(EEG)", "ServiceName(EEG)"} <= set(tables["reports_findings"].columns)
    eeg = tables["eeg_metadata"]
    assert eeg["StartTime"].isna().all() and eeg["EndTime"].isna().all()          # blank in the real table
    blank = eeg["BDSPPatientID"].isna()
    assert set(eeg.loc[blank, "SiteID"]) == {"S0001", "S0002"}
    assert eeg.loc[blank, "BidsFolder"].str.startswith("sub-S000").all()


def test_deterministic():
    a, _ = generate(200, seed=7)
    b, _ = generate(200, seed=7)
    c, _ = generate(200, seed=8)
    pd.testing.assert_frame_equal(a["omop_measurement"], b["omop_measurement"])
    assert not a["omop_measurement"].equals(c["omop_measurement"])


def test_injected_defects_present(synth):
    tables, truth = synth
    assert tables["reports_findings"]["StartTime(EEG)"].isna().mean() > 0.01
    assert tables["omop_drug_exposure"]["drug_exposure_end_datetime"].isna().mean() > 0.1
    assert tables["imaging"]["report_final_datetime"].isna().mean() > 0.03
    assert len(truth["table_shift_ids"]) > 30


def test_on_disk_layout_matches_data_io(synth_dir):
    s3 = data_io.LocalStore(synth_dir)
    assert data_io.discover_sites(s3) == ["I0002", "I0003", "S0001", "S0002"]
    assert data_io.resolve_key(s3, "eeg_metadata", "S0001") == "EEG/eeg-metadata/S0001_eeg_metadata_2026_04_30.csv"
    assert data_io.resolve_key(s3, "reports_findings", "S0002") == "EEG/HEEDB_Metadata/S0002_EEG__reports_findings.csv"
    assert len(data_io.omop_parts(s3, "drug_exposure")) == 2                       # multi-part like the real tables
    hdr = data_io.table_columns(s3, "reports_findings", "S0001")
    assert hdr == schema.columns("reports_findings") and "StartTime(EEG)" in hdr
    assert data_io.table_columns(s3, "omop_measurement") == schema.columns("omop_measurement")
    assert data_io.table_columns(s3, "eeg_metadata", "S9999") is None


def test_roundtrip_through_data_io(synth, synth_dir):
    loaded = load_tables(synth_dir)
    assert set(loaded) == set(schema.TABLE_NAMES)
    pd.testing.assert_series_equal(loaded["omop_note"]["note_datetime"].reset_index(drop=True),
                                   synth[0]["omop_note"]["note_datetime"].reset_index(drop=True),
                                   check_dtype=False, check_exact=False)
    assert loaded["omop_drug_exposure"]["person_id"].dtype == "Int64"
    assert len(loaded["reports_findings"]) == len(synth[0]["reports_findings"])
    # datetimes are stored as text in the real format
    raw = pd.read_csv(synth_dir / "EEG/HEEDB_Metadata/S0001_EEG__reports_findings.csv", dtype=str)
    assert raw["StartTime(EEG)"].dropna().str.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$").all()
