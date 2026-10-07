"""Cohort flow: small-cell suppression, merging, pooling, partitions, and aggregate-only output."""

import json

import pytest

from sortinghat.cohort import FrameSources, build_cohort, flow_markdown
from sortinghat.cohort.flow import (ALL, WITHHELD, FlowRecorder, Step, flow_report, merged_rows, pool_sites,
                                    suppress_parts)
from sortinghat.cohort.output import PENDING_STEPS, known_ids
from sortinghat.safe_output import SUPPRESSED, AggregateOnlyError, assert_aggregate_only


def view(*steps):
    return [(lab, unit, n, i == 0) for i, (lab, unit, n) in enumerate(steps)]


def test_steps_with_fewer_than_11_exclusions_are_merged_forward():
    rows = merged_rows(view(("start", "s", 1000), ("a", "s", 995), ("b", "s", 900), ("c", "s", 899)))
    assert [r["n_excluded"] for r in rows[1:]] == [101]          # a (5) + b (95) + c (1, folded back)
    assert rows[1]["step"] == "a + b + c" and rows[1]["n_remaining"] == 899


def test_trailing_small_step_folds_back_and_all_small_is_withheld():
    rows = merged_rows(view(("start", "s", 1000), ("a", "s", 500), ("b", "s", 495)))
    assert rows[1]["n_excluded"] == 505 and rows[1]["n_remaining"] == 495
    rows = merged_rows(view(("start", "s", 1000), ("a", "s", 995), ("b", "s", 990)))
    assert rows[1]["n_excluded"] == 10 and rows[1]["remaining_withheld"]


def test_no_shown_exclusion_is_small_and_no_small_count_is_derivable(synth):
    res = build_cohort(FrameSources(synth[0]))
    for blk in res.report["sites"].values():
        if WITHHELD in blk:
            continue
        for r in blk["rows"]:
            for key in ("n_excluded", "n_remaining"):
                v = r[key]
                assert v is None or v == WITHHELD or v == SUPPRESSED or v >= 11
            # excluded + remaining of consecutive shown rows reconcile exactly, so only >= 11 exclusions are shown
        shown = [r for r in blk["rows"] if isinstance(r["n_remaining"], int)]
        for prev, cur in zip(shown, shown[1:]):
            assert isinstance(cur["n_excluded"], int) and prev["n_remaining"] - cur["n_remaining"] == cur["n_excluded"]


def test_every_partition_cell_is_suppressed_or_at_least_11(synth):
    res = build_cohort(FrameSources(synth[0]))
    for tbl in res.report["partitions"].values():
        for cells in tbl.values():
            for v in cells.values():
                assert isinstance(v, str) or v >= 11


def test_suppress_parts_complementary():
    assert suppress_parts({"a": 500, "b": 300, "c": 4}) == {"a": 500, "b": SUPPRESSED, "c": SUPPRESSED}
    assert suppress_parts({"a": 500, "b": 300, "c": 200}) == {"a": 500, "b": 300, "c": 200}
    # hidden parts that sum to < 11 would reveal that sum through the total: a shown part is hidden too
    assert set(suppress_parts({"a": 888, "b": 1, "c": 1}).values()) == {SUPPRESSED}
    assert suppress_parts({"a": 500, "b": 300, "c": 200}, forced={"a"})["a"] == SUPPRESSED


def test_small_sites_are_pooled_so_all_minus_others_reveals_nothing():
    def step(label, rem, start=False):
        return Step(label, "patients", rem, start)
    steps = [step("start", {"A": 500, "B": 400, "C": 20, "D": 9}, True), step("end", {"A": 300, "B": 200, "C": 8, "D": 6})]
    groups = pool_sites(["A", "B", "C", "D"], steps)
    assert sorted(map(sorted, groups)) == [["A"], ["B"], ["C", "D"]]
    rec = FlowRecorder(["A", "B", "C", "D"])
    rec.steps = steps
    rep = flow_report(rec.raw())
    assert set(rep["sites"]) == {ALL, "A", "B", "C+D"} and rep["pooled_site_groups"] == ["C+D"]
    assert rep["sites"]["C+D"]["rows"][-1]["n_remaining"] == 14


def test_all_cell_hidden_when_exactly_one_site_group_cell_is_small():
    rec = FlowRecorder(["A", "B"])
    rec.steps = [Step("start", "patients", {"A": 300, "B": 300}, True), Step("end", "patients", {"A": 250, "B": 200})]
    import pandas as pd
    s = lambda a, b: pd.Series(["A"] * a + ["B"] * b)       # noqa: E731
    rec.partition("Split", {"x": s(240, 5), "y": s(10, 195)})
    rep = flow_report(rec.raw())["partitions"]["Split"]
    assert rep["B"]["x"] == SUPPRESSED and rep["ALL"]["x"] == SUPPRESSED      # ALL(x) - A(x) would give B(x)
    assert rep["B"]["y"] == 195 or rep["B"]["y"] == SUPPRESSED


def test_report_is_aggregate_only_and_contains_no_identifier(synth):
    res = build_cohort(FrameSources(synth[0]))
    ids = known_ids(res)
    assert len(ids) > 100
    assert_aggregate_only(res.report, ids)
    text = flow_markdown(res.report, PENDING_STEPS)
    assert_aggregate_only(text, ids)
    blob = json.dumps(res.report)
    assert not any(i in blob or i in text for i in list(ids)[:500])
    assert "EEG QC in the streaming extractor" in text
    # the record-level frames are recognised as record-level by the guard
    with pytest.raises(AggregateOnlyError):
        assert_aggregate_only(res.table)
    with pytest.raises(AggregateOnlyError):
        assert_aggregate_only(res.keys)


def test_building_prints_nothing(synth, capsys):
    build_cohort(FrameSources(synth[0]))
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
