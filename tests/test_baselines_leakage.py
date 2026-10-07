"""Leakage tests: nothing after t0 may reach a baseline feature.

Strategy: inject post-t0 events (sentinel values) into every event domain of the synthetic data and assert the
feature matrix is bit-for-bit unchanged and never contains a sentinel; plus unit tests of the single gate
``as_of`` and structural checks that no other module compares events with t0.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sortinghat.baselines as bl
from sortinghat.baselines import BaselineConfig, as_of, assert_masked, build_events, build_feature_set, build_index
from sortinghat.baselines import features as features_mod
from sortinghat.baselines.asof import EVENT_COLUMNS, empty_events
from test_baselines_helpers import (H, SENTINELS, T0, add_rows, augment_bedside, drug, img, meas, mini_tables,
                                    post_t0_events)


@pytest.fixture(scope="module")
def base_tables(synth):
    return augment_bedside(synth[0], seed=11)


@pytest.fixture(scope="module")
def base_fs(base_tables):
    return build_feature_set(base_tables)


# ----------------------------------------------------------------------------------- injection tests
def test_injected_post_t0_events_change_nothing(base_tables, base_fs):
    leaked = post_t0_events(base_tables)
    assert len(leaked["omop_measurement"]) > len(base_tables["omop_measurement"])
    fs = build_feature_set(leaked)
    pd.testing.assert_frame_equal(fs.X, base_fs.X, check_exact=True)
    assert fs.diagnostics["n_events_dropped_post_t0"] > base_fs.diagnostics["n_events_dropped_post_t0"]


def test_no_sentinel_value_in_any_feature(base_tables):
    X = build_feature_set(post_t0_events(base_tables)).X.to_numpy(float)
    for s in SENTINELS:
        assert not np.any(np.isclose(X, s)), s


@pytest.mark.parametrize("hours_after", [1e-6 / 3600, 0.01, 24.0, 24.0 * 400])
def test_injection_at_any_offset_after_t0(base_tables, base_fs, hours_after):
    """Even microseconds after t0 (and year-scale offsets, which also clears every assay lag)."""
    fs = build_feature_set(post_t0_events(base_tables, hours_after=hours_after))
    pd.testing.assert_frame_equal(fs.X, base_fs.X, check_exact=True)


def test_post_t0_edits_to_pre_t0_rows_change_nothing():
    """Interval end / quantity of a running infusion are future information: editing them is invisible."""
    def tables(end, qty):
        return add_rows(mini_tables(), "omop_drug_exposure",
                        [drug(1, "PROPOFOL 10 MG/ML IV EMULSION", T0 - H(3), end, qty)])
    a = build_feature_set(tables(T0 + H(1), 5.0)).X
    b = build_feature_set(tables(T0 + H(900), 424242.0)).X
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.loc[1, "sed__propofol__on_t0"] == 1 and a.loc[1, "sed__propofol__qty_24h"] == 0


def test_result_and_final_times_after_t0_hide_pre_t0_collections():
    res = "measurement_result_datetime"
    t = add_rows(mini_tables(), "omop_measurement", [
        {**meas(1, "SODIUM", T0 - H(48), 9999.0), res: T0 + H(0.001)},
        {**meas(1, "LACTATE, WHOLE BLOOD", T0 - H(48), 9999.0), res: T0 + H(500)}])
    t = add_rows(t, "imaging", [img(1, "CT head", T0 - H(48), T0 + H(0.001))])
    r = build_feature_set(t).X.loc[1]
    assert r["lab__sodium__miss"] == 1 and r["lab__lactate__miss"] == 1
    assert r["img__ct_head__final_by_t0"] == 0 and r["lab__n_results_by_t0"] == 0


def test_event_exactly_at_t0_is_available_and_one_microsecond_later_is_not():
    t = add_rows(mini_tables(), "omop_measurement", [
        meas(1, "Heart rate", T0, 91.0), meas(1, "Systolic blood pressure", T0 + pd.Timedelta(microseconds=1), 120.0)])
    r = build_feature_set(t).X.loc[1]
    assert r["vital__hr__value"] == 91 and r["vital__sbp__miss"] == 1


def test_date_only_rows_cannot_leak_same_day_values():
    """A value with only a date (no time) is placed at the END of its day: same-day-as-t0 values are unusable."""
    row = {"person_id": 1, "measurement_datetime": None, "measurement_date": T0.normalize(),
           "measurement_source_value": "Heart rate", "value_as_number": 150.0}
    t = add_rows(mini_tables(), "omop_measurement", [row])
    assert build_feature_set(t).X.loc[1, "vital__hr__miss"] == 1


def test_permutation_of_rows_does_not_change_features(base_tables, base_fs):
    rng = np.random.default_rng(0)
    shuffled = {k: (v.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)
                    if k.startswith("omop_") or k == "imaging" else v) for k, v in base_tables.items()}
    pd.testing.assert_frame_equal(build_feature_set(shuffled).X, base_fs.X, check_exact=True)


# --------------------------------------------------------------------------------- the as_of gate
def _ev(rows):
    d = pd.DataFrame(rows)
    for c in EVENT_COLUMNS:
        if c not in d:
            d[c] = np.nan if c in ("value", "quantity") else (pd.NaT if c.startswith("t_") else None)
    d["approx"] = d["approx"].fillna(False).astype(bool)
    for c in ("t_event", "t_avail", "t_end_raw"):
        d[c] = pd.to_datetime(d[c]).astype("datetime64[us]")
    return d[EVENT_COLUMNS]


def test_as_of_filters_on_availability_not_event_time():
    ev = _ev([
        dict(person_id=1, domain="lab", key="a", value=1.0, t_event=T0 - H(2), t_avail=T0 - H(1)),    # keep
        dict(person_id=1, domain="lab", key="b", value=2.0, t_event=T0 - H(2), t_avail=T0 + H(1)),    # collected before, avail after
        dict(person_id=1, domain="lab", key="c", value=3.0, t_event=T0 - H(2), t_avail=pd.NaT),       # unknown availability
        dict(person_id=1, domain="lab", key="d", value=4.0, t_event=T0, t_avail=T0),                  # boundary: keep
        dict(person_id=2, domain="lab", key="e", value=5.0, t_event=T0, t_avail=T0)])                 # no t0 for person 2
    out = as_of(ev, {1: T0})
    assert sorted(out["key"]) == ["a", "d"]
    assert (out["t_avail"] <= out["t0"]).all()
    assert out.attrs["as_of"] is True
    assert_masked(out)


def test_as_of_censors_future_interval_ends_and_quantities():
    ev = _ev([
        dict(person_id=1, domain="drug", key="propofol", quantity=10.0, t_event=T0 - H(3), t_avail=T0 - H(3),
             t_end_raw=T0 + H(5), time_basis="admin"),
        dict(person_id=1, domain="drug", key="midazolam", quantity=4.0, t_event=T0 - H(9), t_avail=T0 - H(9),
             t_end_raw=T0 - H(8), time_basis="admin"),
        dict(person_id=1, domain="drug", key="fentanyl", quantity=2.0, t_event=T0 - H(1), t_avail=T0 - H(1),
             time_basis="order")])
    out = as_of(ev, T0).set_index("key")
    assert "t_end_raw" not in out.columns
    assert bool(out.loc["propofol", "ongoing"]) and pd.isna(out.loc["propofol", "t_end"]) and pd.isna(out.loc["propofol", "quantity"])
    assert not out.loc["midazolam", "ongoing"] and out.loc["midazolam", "t_end"] == T0 - H(8) and out.loc["midazolam", "quantity"] == 4
    assert not out.loc["fentanyl", "ongoing"] and out.loc["fentanyl", "quantity"] == 2
    assert_masked(out.reset_index().assign())


def test_as_of_accepts_scalar_t0_and_series():
    ev = _ev([dict(person_id=1, domain="lab", key="a", value=1.0, t_event=T0 - H(1), t_avail=T0 - H(1)),
              dict(person_id=2, domain="lab", key="a", value=1.0, t_event=T0 - H(1), t_avail=T0 - H(1))])
    assert len(as_of(ev, T0)) == 2
    assert len(as_of(ev, pd.Series({1: T0 - H(2), 2: T0}))) == 1
    assert as_of(empty_events(), T0).empty


def test_assert_masked_rejects_raw_events_and_future_rows():
    ev = _ev([dict(person_id=1, domain="lab", key="a", value=1.0, t_event=T0 + H(1), t_avail=T0 + H(1))])
    with pytest.raises(ValueError):
        assert_masked(ev)                                                        # raw events never reach extractors
    out = as_of(ev, T0 + H(2))
    out["t0"] = T0
    with pytest.raises(AssertionError):
        assert_masked(out)


def test_raw_event_frame_contains_post_t0_rows_but_gate_removes_them(base_tables):
    idx = build_index(base_tables)
    ev, _ = build_events(base_tables, idx)
    t0 = idx.set_index("person_id")["t0"]
    assert (ev["t_avail"] > ev["person_id"].map(t0)).any()                      # the synthetic data really has post-t0 rows
    masked = as_of(ev, t0)
    assert (masked["t_avail"] <= masked["t0"]).all() and len(masked) < len(ev)
    assert (masked["hours_since_event"] >= -1e-9).all()
    assert not (masked["t_end"] > masked["t0"]).any()


def test_build_feature_set_routes_every_event_through_one_as_of_call(base_tables, monkeypatch):
    calls = []
    real = features_mod.as_of

    def spy(events, t0):
        calls.append((len(events), t0))
        return real(events, t0)
    monkeypatch.setattr(features_mod, "as_of", spy)
    fs = build_feature_set(base_tables)
    assert len(calls) == 1 and calls[0][0] == fs.diagnostics["n_events_total"]


def test_extractors_reject_unmasked_input(base_tables):
    idx = build_index(base_tables)
    ev, _ = build_events(base_tables, idx)
    with pytest.raises(ValueError):
        assert_masked(ev)


# -------------------------------------------------------------------------------- structural checks
PKG = Path(bl.__file__).parent
CMP = re.compile(r"(t_avail|t_event|t_end\w*|_start|_datetime)\W{0,3}\s*[<>]=?\s*[^=].*\bt0\b|\bt0\b.*[<>]=?\s*.*(t_avail|t_event|t_end)")


def test_only_asof_module_compares_events_with_t0():
    for f in sorted(PKG.glob("*.py")):
        if f.name == "asof.py":
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            assert not CMP.search(code), f"{f.name}:{n} compares an event time with t0 outside as_of: {line.strip()}"


def test_extractors_only_take_masked_frames():
    src = (PKG / "features.py").read_text()
    assert "assert_masked(as_of(" in src and src.count("as_of(") == 1
    assert "events" not in re.findall(r"def _(?:score_vital_pupil|history|sedation|labs|imaging)\((.*?)\)", src)[0]


def test_event_builder_does_not_know_t0():
    """events.py may mention t0 only inside build_index (which defines it); event builders never see it."""
    import ast
    tree = ast.parse((PKG / "events.py").read_text())
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "build_index"]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id == "t0":
                raise AssertionError(f"{fn.name} uses a t0 variable")
            if isinstance(node, ast.Constant) and node.value == "t0":
                raise AssertionError(f"{fn.name} reads a t0 column")
