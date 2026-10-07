import numpy as np
import pandas as pd
import pytest

from sortinghat.safe_output import (SUPPRESSED, AggregateOnlyError, assert_aggregate_only,
                                    group_counts, safe_print, safe_quantiles, safe_write_json,
                                    suppress_count, suppress_proportion)


def test_count_suppression_boundary():
    assert suppress_count(10) == SUPPRESSED == "<11"
    assert suppress_count(0) == "<11"
    assert suppress_count(11) == 11


def test_proportion_suppresses_small_cells_and_complement():
    assert suppress_proportion(5, 1000) == "<11"
    assert suppress_proportion(995, 1000) == "<11"   # complement of 5
    assert suppress_proportion(10, 10) == "<11"
    assert suppress_proportion(50, 100) == 0.5


def test_quantiles_suppressed_and_extremes_refused():
    assert set(safe_quantiles(range(5)).values()) == {"<11"}
    q = safe_quantiles(range(100))
    assert q["q50"] == pytest.approx(49.5)
    with pytest.raises(AggregateOnlyError):
        safe_quantiles(range(100), qs=(0.0, 0.5))
    with pytest.raises(AggregateOnlyError):
        safe_quantiles(range(100), qs=(1.0,))


def test_group_counts_suppresses_small_groups():
    df = pd.DataFrame({"site": ["A"] * 20 + ["B"] * 3})
    g = group_counts(df, "site").set_index("site")["n"]
    assert g["A"] == 20 and g["B"] == "<11"


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"BDSPPatientID": ["x"], "n": [1]}),
    pd.DataFrame({"session_id": ["x"], "n": [1]}),
    pd.DataFrame({"when": pd.to_datetime(["2020-01-01"]), "n": [1]}),
    pd.DataFrame({"label": ["SYNS0001000123"], "n": [1]}),
    pd.DataFrame({"label": ["sub-S0001000123"], "n": [1]}),
    {"BDSPPatientID": [1, 2]},
    {"ok": "see SYNS0001000123"},
])
def test_guard_rejects_row_level(bad):
    with pytest.raises(AggregateOnlyError):
        assert_aggregate_only(bad)


def test_guard_rejects_known_ids_and_index():
    with pytest.raises(AggregateOnlyError):
        assert_aggregate_only("value Z123456789 here", known_ids={"Z123456789"})
    s = pd.Series([1, 2], index=["abc12345", "def67890"], name="n")
    with pytest.raises(AggregateOnlyError):
        assert_aggregate_only(s, known_ids={"abc12345"})


def test_guard_allows_aggregates():
    assert_aggregate_only({"site": {"S0001": {"n": "<11", "p": 0.5}}, "n_patients": 3})
    assert_aggregate_only(pd.DataFrame({"site": ["S0001"], "n": [50]}))
    assert_aggregate_only(np.float64(0.2))


def test_guarded_emitters(tmp_path, capsys):
    with pytest.raises(AggregateOnlyError):
        safe_write_json(tmp_path / "x.json", {"id": "sub-ABC123"})
    assert not (tmp_path / "x.json").exists()
    with pytest.raises(AggregateOnlyError):
        safe_print("leak SYNS0001000001")
    assert capsys.readouterr().out == ""
