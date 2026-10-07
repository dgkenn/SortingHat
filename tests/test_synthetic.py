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
    assert eeg["SiteID"].nunique() == 6
    assert tables["omop_person"]["person_id"].nunique() == 3000


def test_real_names_and_blank_columns(synth):
    tables, _ = synth
    assert "DurationInSeconds" in tables["eeg_metadata"] and "DurationInSecond" not in tables["eeg_metadata"]
    assert {"StartTime(EEG)", "EndTime(EEG)", "ServiceName(EEG)"} <= set(tables["reports_findings"].columns)
    eeg = tables["eeg_metadata"]
    md = eeg["SiteID"].isin(["I0008", "I0009"])
    assert eeg.loc[~md, "StartTime"].isna().all() and eeg.loc[~md, "EndTime"].isna().all()   # blank in the real table
    assert eeg.loc[md, "StartTime"].notna().mean() > 0.9 and eeg.loc[md, "EndTime"].notna().all()   # I0008/I0009 real
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
    assert data_io.discover_sites(s3) == ["I0002", "I0003", "I0008", "I0009", "S0001", "S0002"]
    assert data_io.resolve_key(s3, "eeg_metadata", "S0001") == "EEG/eeg-metadata/S0001_eeg_metadata_2026_04_30.csv"
    assert data_io.resolve_key(s3, "reports_findings", "S0002") == "EEG/HEEDB_Metadata/S0002_EEG__reports_findings.csv"
    assert len(data_io.omop_parts(s3, "drug_exposure")) == 2                       # multi-part like the real tables
    hdr = data_io.table_columns(s3, "reports_findings", "S0001")
    assert hdr == [a for a, *_ in schema.expected_columns("reports_findings", "S0001")] and "StartTime(EEG)" in hdr
    assert data_io.table_columns(s3, "reports_findings", "I0008") is None           # no reports_findings file there
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


# ------------------------------------------------------------------ per-site variants (real headers)
def _hdr(synth_dir, site):
    return pd.read_csv(next((synth_dir / "EEG/eeg-metadata").glob(f"{site}_*.csv")), nrows=0).columns.tolist()


def test_site_variant_headers_on_disk(synth_dir):
    s_hdr, i2, i3, i8 = (_hdr(synth_dir, x) for x in ("S0001", "I0002", "I0003", "I0008"))
    assert _hdr(synth_dir, "I0009") == i8 and _hdr(synth_dir, "S0002") == s_hdr
    assert "EEGFolder" in s_hdr and "DateOfDeath" in s_hdr and "ServiceName" in s_hdr and "BDSPLastModifiedDTS" not in s_hdr
    assert "EEGFolder" not in i2 and "ServiceName" not in i2 and "DateOfDeath" not in i2 and "HasPersystAnnotations" not in i2
    assert "AgeInDaysAtVisit" in i3 and "ServiceName" in i3
    assert {"InstituteID", "StartDateTime", "EndDateTime", "DateOfBirth", "RecordingDuration"} <= set(i8)
    assert not {"SiteID", "AgeAtVisit", "StartTime", "EndTime", "DurationInSeconds", "ServiceName"} & set(i8)
    for h in (s_hdr, i2, i3, i8):
        assert "PatientClass" not in h and "ReferralIndication" not in h and "EEGFolder" not in h or h is s_hdr
    assert not (synth_dir / "EEG/HEEDB_Metadata/I0008_EEG__reports_findings.csv").exists()
    assert not (synth_dir / "EEG/HEEDB_Metadata/I0009_EEG__reports_findings.csv").exists()
    rf3 = pd.read_csv(synth_dir / "EEG/HEEDB_Metadata/I0003_EEG__reports_findings.csv", nrows=0).columns
    assert "N1" in rf3 and "EEGDateTime(Reports)" in rf3 and "ServiceName(EEG)" not in rf3
    rf1 = pd.read_csv(synth_dir / "EEG/HEEDB_Metadata/S0001_EEG__reports_findings.csv", nrows=0).columns
    assert {"BeginDTS(Reports)", "ProcedureDSC(Reports)", "ServiceName(EEG)", "SiteID"} <= set(rf1)


def test_normaliser_maps_every_variant_onto_canonical_names(synth, synth_dir):
    s3 = data_io.LocalStore(synth_dir)
    canon = set(schema.columns("eeg_metadata"))
    for site in ("S0001", "I0002", "I0003", "I0008", "I0009"):
        raw = data_io.read_csv_table("eeg_metadata", site, s3=s3)
        norm = data_io.read_site_table("eeg_metadata", site, s3=s3)
        assert set(norm.columns) <= canon and "SiteID" in norm and (norm["SiteID"] == site).all()
        assert len(norm) == len(raw)
    i8 = data_io.read_site_table("eeg_metadata", "I0008", s3=s3)
    assert {"StartTime", "EndTime", "DurationInSeconds", "DateOfBirth", "InstituteID"} <= set(i8.columns)
    assert not {"StartDateTime", "EndDateTime", "RecordingDuration"} & set(i8.columns)
    # DeidentifiedName(Reports) is never read into memory
    rf = data_io.read_site_table("reports_findings", "S0001", s3=s3)
    assert "DeidentifiedName(Reports)" not in rf.columns and "ReportBeginDTS" in rf.columns
    assert set(rf.columns) <= set(schema.columns("reports_findings"))
    # round trip: a site's file equals the in-memory canonical rows for the columns it has
    mem = synth[0]["eeg_metadata"]
    for site in ("I0008", "S0001"):
        g = mem[mem["SiteID"] == site].reset_index(drop=True)
        got = schema.coerce_types("eeg_metadata", data_io.read_site_table("eeg_metadata", site, s3=s3))
        assert (got["SessionID"].astype(str) == g["SessionID"].astype(str)).all()
        for c in got.columns:
            if c in ("PatientClass", "ReferralIndication", "SiteID"):
                continue
            a, b = got[c].reset_index(drop=True), g[c]
            assert a.isna().equals(b.isna()), c


def test_normaliser_unknown_site_is_passthrough_and_denormalise_inverts():
    df = pd.DataFrame({"BidsFolder": ["sub-X1"], "Whatever": [1]})
    assert data_io.normalise_site_table("eeg_metadata", "S9999", df) is df
    canon = pd.DataFrame({"StartTime": [pd.Timestamp("2020-01-01")], "EndTime": [pd.Timestamp("2020-01-02")],
                          "DurationInSeconds": [5.0], "BidsFolder": ["b"], "SessionID": ["s"], "SiteID": ["I0009"]})
    out = data_io.denormalise_site_table("eeg_metadata", "I0009", canon)
    assert {"StartDateTime", "EndDateTime", "RecordingDuration"} <= set(out.columns) and "SiteID" not in out.columns
    back = data_io.normalise_site_table("eeg_metadata", "I0009", out)
    assert back["StartTime"].iloc[0] == pd.Timestamp("2020-01-01") and back["SiteID"].iloc[0] == "I0009"
    assert "PatientClass" not in out.columns


def test_age_comes_from_whichever_column_the_site_has():
    from sortinghat.audit.field_audit import meta_age
    t = pd.Timestamp("2020-06-01")
    i8 = pd.DataFrame({"StartTime": [t], "DateOfBirth": [t - pd.Timedelta(days=int(40 * 365.25))]})
    assert abs(meta_age(i8).iloc[0] - 40.0) < 0.01
    i3 = pd.DataFrame({"AgeAtVisit": [float("nan")], "AgeInDaysAtVisit": [365.25 * 7]})
    assert abs(meta_age(i3).iloc[0] - 7.0) < 1e-9
    assert meta_age(pd.DataFrame({"AgeAtVisit": ["61"]})).iloc[0] == 61.0
