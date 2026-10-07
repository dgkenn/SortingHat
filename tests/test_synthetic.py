import pandas as pd

from sortinghat import schema
from sortinghat.synthetic import generate
from sortinghat.tableio import load_tables


def test_shape_and_schema(synth):
    tables, _ = synth
    assert set(tables) == set(schema.TABLE_NAMES)
    for name, df in tables.items():
        assert list(df.columns) == schema.columns(name)
    eeg = tables["eeg_metadata"]
    assert eeg["BDSPPatientID"].nunique() == 3000
    assert eeg["SiteID"].nunique() == 4
    assert eeg["BDSPPatientID"].str.startswith("SYN").all()


def test_deterministic():
    a, _ = generate(200, seed=7)
    b, _ = generate(200, seed=7)
    c, _ = generate(200, seed=8)
    pd.testing.assert_frame_equal(a["labs"], b["labs"])
    assert not a["labs"].equals(c["labs"])


def test_injected_defects_present(synth):
    tables, truth = synth
    eeg = tables["eeg_metadata"]
    assert eeg["StartTime"].isna().mean() > 0.01
    assert tables["medications"]["admin_time"].isna().mean() > 0.1
    assert tables["imaging"]["report_final_time"].isna().mean() > 0.03
    assert len(truth["table_shift_ids"]) > 30


def test_roundtrip_csv(synth, synth_dir):
    loaded = load_tables(synth_dir)
    pd.testing.assert_series_equal(loaded["notes"]["note_time"].reset_index(drop=True),
                                   synth[0]["notes"]["note_time"].reset_index(drop=True),
                                   check_dtype=False, check_exact=False)
