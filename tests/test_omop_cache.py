"""The shared cross-step OMOP row-group cache (sortinghat/omop_cache.py, D-149) on SYNTHETIC data: cached reads equal uncached
reads, a second step reads nothing from the store, the warm-up resumes after a kill, a changed source object (ETag/size)
invalidates, the prefetch is bounded and ordered, and a request the cache cannot serve falls back to the store."""
import shutil
import stat
import threading
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from sortinghat import checkpoint as ck
from sortinghat import data_io, omop_cache
from sortinghat.checkpoint import SimulatedKill
from sortinghat.cohort.sources import iter_filtered_batches
from test_checkpoint import RG, Counter, mrg_store  # noqa: F401  (fixture + counter shared with the checkpoint tests)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(ck.ENV_ROOT, str(tmp_path / "ck"))
    monkeypatch.setenv(omop_cache.ENV_ROOT, str(tmp_path / "omop"))
    monkeypatch.setenv(ck.ENV_VERSION, "omop-v1")
    monkeypatch.delenv(ck.ENV_FAIL_AFTER, raising=False)
    monkeypatch.delenv(omop_cache.ENV_SWITCH, raising=False)
    monkeypatch.delenv(omop_cache.ENV_WORKERS, raising=False)
    return tmp_path


@pytest.fixture
def counter(monkeypatch):
    return Counter(monkeypatch)


def step(store, name="stepA", **kw):
    return ck.open_step(name, {"n": name}, store=store, quiet=True, **kw)


def read(store, table, cols, ids=None):
    parts = [b.to_pandas() for b in data_io.iter_omop_batches(table, s3=store, columns=cols, person_ids=ids)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def read_filtered(store, table, people, cols, **kw):
    parts = [t.to_pandas() for t in iter_filtered_batches(store, table, people, cols, **kw)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@pytest.fixture
def cands(mrg_store, env):
    return omop_cache.candidate_ids(data_io.LocalStore(mrg_store))


def some_ids(store_dir, table, cands, n):
    have = pd.read_parquet(store_dir / f"OMOP/Merged/{table}/part-00000.parquet", columns=["person_id"])["person_id"].unique()
    return sorted(int(p) for p in np.intersect1d(have, cands))[:n]


# ============================================================================================ equality
def test_candidate_set_is_adults_at_study_sites_in_its_own_small_file(mrg_store, env, cands):
    assert cands.dtype == np.int64 and (np.diff(cands) > 0).all() and len(cands) > 100
    files = list((env / "omop").glob("candidates-*.npy"))
    assert len(files) == 1 and stat.S_IMODE(files[0].stat().st_mode) == 0o600
    everyone = pd.concat([pd.read_parquet(f, columns=["person_id"]) for f in (mrg_store / "OMOP/Merged/person").glob("*.parquet")])
    assert 0 < len(cands) < everyone["person_id"].nunique()                  # a strict subset: minors / non-Study-1 sites left out


def test_cached_reads_equal_uncached_reads_and_a_warm_cache_touches_nothing(mrg_store, env, counter, cands):
    store = data_io.LocalStore(mrg_store)
    from sortinghat.cohort import sources as cs
    cases = [("measurement", cs._MEAS_COLS, some_ids(mrg_store, "measurement", cands, 400)),
             ("visit_occurrence", cs._VISIT_COLS, [int(p) for p in cands]),
             ("condition_occurrence", ["person_id", "condition_source_value"], some_ids(mrg_store, "condition_occurrence", cands, 50)),
             ("concept", ["concept_id", "concept_name"], None)]
    refs = {t: read(store, t, c, i) for t, c, i in cases}                      # no checkpoint active: the plain reader
    assert all(len(r) for r in refs.values())
    counter.reset()
    with ck.use(step(store)):
        cold = {t: read(store, t, c, i) for t, c, i in cases}
    assert counter.rg > 0
    for t, _c, _i in cases:
        pd.testing.assert_frame_equal(cold[t], refs[t])
    counter.reset()
    with ck.use(step(store, "stepB")):
        warm = {t: read(store, t, c, i) for t, c, i in cases}
    assert counter.rg == 0 and counter.gets == 0                               # nothing from the store, not even footers
    for t, _c, _i in cases:
        pd.testing.assert_frame_equal(warm[t], refs[t])


def test_filtered_reader_cached_equals_uncached(mrg_store, env, counter, cands):
    store = data_io.LocalStore(mrg_store)
    ids = [int(p) for p in cands]
    drug = dict(text_col="drug_source_value", pattern="propofol|midazolam|fentanyl|dexmedetomidine|lorazepam")
    meas = dict(text_col="measurement_source_value", pattern="gcs|glasgow", id_col="measurement_concept_id", ids=[0, 1, 2, 3])
    cols_d = ["person_id", "drug_exposure_start_datetime", "drug_source_value"]
    cols_m = ["person_id", "measurement_datetime", "measurement_source_value"]
    plain = lambda: (read_filtered(store, "drug_exposure", ids, cols_d, **drug),         # noqa: E731
                     read_filtered(store, "measurement", ids, cols_m, **meas),
                     read_filtered(store, "note", ids, ["person_id", "note_datetime"]))
    ref = plain()
    assert all(len(r) for r in ref)
    with ck.use(step(store)):
        cold = plain()
        counter.reset()
        warm = plain()
    assert counter.rg == 0
    for r, c, w in zip(ref, cold, warm):
        pd.testing.assert_frame_equal(c, r)
        pd.testing.assert_frame_equal(w, r)


def test_a_second_step_with_other_columns_and_fewer_people_reads_nothing(mrg_store, env, counter, cands):
    store = data_io.LocalStore(mrg_store)
    big = some_ids(mrg_store, "measurement", cands, 600)
    with ck.use(step(store, "cohort_like")):
        read(store, "measurement", ["person_id", "measurement_datetime", "measurement_source_value"], big)
    sub, cols = big[:37], ["value_as_number", "person_id", "measurement_datetime"]       # fewer people, other columns, other order
    ref = read(store, "measurement", cols, sub)
    counter.reset()
    with ck.use(step(store, "baselines_like")):
        got = read(store, "measurement", cols, sub)
    assert counter.rg == 0
    pd.testing.assert_frame_equal(got, ref)
    assert list(got.columns) == cols


def test_requests_the_cache_cannot_serve_read_the_store(mrg_store, env, counter, cands):
    store = data_io.LocalStore(mrg_store)
    ids = some_ids(mrg_store, "measurement", cands, 50)
    outsider = sorted(set(pd.read_parquet(mrg_store / "OMOP/Merged/measurement/part-00000.parquet", columns=["person_id"])
                          ["person_id"].unique()) - {int(c) for c in cands})[:5]
    assert outsider                                                          # people at non-candidate (non-Study-1 / minor) sites
    cols = ["person_id", "measurement_datetime"]
    cases = ({"ids": ids + [int(x) for x in outsider], "cols": cols},             # a person outside the candidate set
             {"ids": ids, "cols": cols + ["not_in_the_superset"]})                # a column outside the superset
    refs = [read(store, "measurement", c["cols"], c["ids"]) for c in cases]       # plain reader
    with ck.use(step(store)):
        read(store, "measurement", cols, ids)                                # fills the cache
        assert omop_cache.reader_for(ck.active(), store, "b", "measurement", columns=cols,
                                     ids=ids + [int(outsider[0])], max_get_bytes=1 << 20) is None
        assert omop_cache.reader_for(ck.active(), store, "b", "measurement", columns=cols + ["not_in_the_superset"],
                                     ids=ids, max_get_bytes=1 << 20) is None
        for c, ref in zip(cases, refs):
            counter.reset()
            pd.testing.assert_frame_equal(read(store, "measurement", c["cols"], c["ids"]), ref)
            assert counter.rg > 0                                            # served by the store, as before


def test_switch_off_by_environment(mrg_store, env, monkeypatch, cands):
    store = data_io.LocalStore(mrg_store)
    monkeypatch.setenv(omop_cache.ENV_SWITCH, "off")
    assert omop_cache.reader_for(ck.active(), store, "b", "measurement", columns=["person_id"], ids=[1],
                                 max_get_bytes=1 << 20) is None
    with ck.use(step(store)):
        assert omop_cache.reader_for(ck.active(), store, "b", "measurement", columns=["person_id"], ids=[int(cands[0])],
                                     max_get_bytes=1 << 20) is None
    assert not (env / "omop" / "measurement").exists()


# ===================================================================================== invalidation, privacy
def test_changed_etag_or_size_invalidates(mrg_store, env, counter, tmp_path):
    d = tmp_path / "s"
    shutil.copytree(mrg_store, d)
    store = data_io.LocalStore(d)
    cands = omop_cache.candidate_ids(store)
    ids = some_ids(d, "measurement", cands, 300)
    cols = ["person_id", "measurement_datetime", "value_as_number"]
    with ck.use(step(store)):
        before = read(store, "measurement", cols, ids)
        counter.reset()
        read(store, "measurement", cols, ids)
        assert counter.rg == 0
    f = d / "OMOP/Merged/measurement/part-00001.parquet"                     # the source object is replaced
    t = pq.read_table(f)
    pq.write_table(t.slice(0, t.num_rows - 5), f, row_group_size=RG)
    ref = read(store, "measurement", cols, ids)                              # plain reader on the new bytes
    counter.reset()
    with ck.use(step(store, "after_change")):
        after = read(store, "measurement", cols, ids)
    assert counter.rg > 0                                                    # only the changed part is read again ...
    pd.testing.assert_frame_equal(after, ref)
    assert len(before) >= len(after)
    counter.reset()
    with ck.use(step(store, "after_change2")):
        pd.testing.assert_frame_equal(read(store, "measurement", cols, ids), ref)
    assert counter.rg == 0                                                   # ... and then served locally


def test_cache_is_private_and_progress_lines_are_aggregate_only(mrg_store, env, capsys, cands):
    store = data_io.LocalStore(mrg_store)
    with ck.use(step(store)):
        read(store, "measurement", ["person_id", "measurement_datetime"], some_ids(mrg_store, "measurement", cands, 100))
    root = env / "omop"
    for p in root.rglob("*"):
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == (0o700 if p.is_dir() else 0o600), p
    assert not [p for p in root.rglob("*") if ".tmp." in p.name]
    out = capsys.readouterr().out
    assert "omop_measurement:" in out and "row groups (cached" in out
    assert str(root) not in out and "part-0000" not in out and "OMOP/Merged" not in out


# ============================================================================================= warm-up
def test_warm_up_resumes_after_a_kill_and_then_serves_every_step(mrg_store, env, counter, monkeypatch):
    tables = ["measurement", "visit_occurrence", "concept"]
    argv = ["warm", "--data", str(mrg_store), "--tables", *tables]
    monkeypatch.setenv(omop_cache.ENV_ROOT, str(env / "omop_ref"))
    assert omop_cache.main(argv) == 0
    full = counter.rg
    assert full > 20
    monkeypatch.setenv(omop_cache.ENV_ROOT, str(env / "omop"))
    counter.reset()
    monkeypatch.setenv(ck.ENV_FAIL_AFTER, "15")
    with pytest.raises(SimulatedKill):
        omop_cache.main(argv)
    killed = counter.rg
    monkeypatch.delenv(ck.ENV_FAIL_AFTER)
    counter.reset()
    assert omop_cache.main(argv) == 0
    assert 0 < killed and 0 < counter.rg < full
    assert killed + counter.rg == full                                       # every row group fetched exactly once overall
    counter.reset()
    assert omop_cache.main(argv) == 0                                        # a finished warm-up is a no-op
    assert counter.rg == 0 and counter.gets == 0
    # steps now run entirely from disk, with identical rows
    store = data_io.LocalStore(mrg_store)
    cands = omop_cache.candidate_ids(store)
    ids = some_ids(mrg_store, "measurement", cands, 500)
    ref = read(store, "measurement", ["person_id", "measurement_datetime", "value_as_number"], ids)
    counter.reset()
    with ck.use(step(store)):
        got = read(store, "measurement", ["person_id", "measurement_datetime", "value_as_number"], ids)
        read(store, "concept", ["concept_id", "concept_name"])
    assert counter.rg == 0
    pd.testing.assert_frame_equal(got, ref)


def test_warm_up_rejects_unknown_tables(mrg_store, env):
    with pytest.raises(SystemExit):
        omop_cache.main(["warm", "--data", str(mrg_store), "--tables", "not_a_table"])


def test_no_resume_refreshes_each_unit_once_per_run(mrg_store, env, counter, cands):
    store = data_io.LocalStore(mrg_store)
    cols = ["person_id", "measurement_datetime"]
    ids = some_ids(mrg_store, "measurement", cands, 200)
    with ck.use(step(store)):
        read(store, "measurement", cols, ids)
    full = counter.rg
    counter.reset()
    with ck.use(step(store, no_resume=True)):
        read(store, "measurement", cols, ids)
        first = counter.rg
        read(store, "measurement", cols, ids)                                # same run, same table: already refreshed
    assert first > 0 and counter.rg == first <= full * 2


# ============================================================================================= prefetch
def test_prefetch_is_ordered_and_bounded_by_the_worker_count(mrg_store, env, monkeypatch, cands):
    store = data_io.LocalStore(mrg_store)
    cols = omop_cache.superset_columns()["measurement"]
    live = {"now": 0, "max": 0, "produced": 0, "consumed": 0, "max_unconsumed": 0}
    lock = threading.Lock()
    real = data_io.read_rowgroup

    def slow(pf, rg, use, ids, pid_arr, outer):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.01)                                                     # network latency
        try:
            return real(pf, rg, use, ids, pid_arr, outer)
        finally:
            with lock:
                live["now"] -= 1
                live["produced"] += 1
                live["max_unconsumed"] = max(live["max_unconsumed"], live["produced"] - live["consumed"])
    monkeypatch.setattr(data_io, "read_rowgroup", slow)

    def run(workers, name):
        monkeypatch.setenv(omop_cache.ENV_WORKERS, str(workers))
        monkeypatch.setenv(omop_cache.ENV_ROOT, str(env / f"omop_{name}"))   # a cold cache each time
        for k in live:
            live[k] = 0
        order = []
        with ck.use(step(store, name)):
            r = omop_cache.reader_for(ck.active(), store, "b", "measurement", columns=cols,
                                      ids=omop_cache.candidate_ids(store), max_get_bytes=1 << 20)
            assert r.workers == workers
            for key, rg, tbl in r.iter_rowgroups(None):
                time.sleep(0.005)                                            # a slow consumer must not let the pool run ahead
                order.append((key, rg, None if tbl is None else tbl.num_rows))
                with lock:
                    live["consumed"] += 1
        return order, r.max_in_flight, dict(live)
    seq, _inflight1, live1 = run(1, "w1")
    par, inflight4, live4 = run(4, "w4")
    assert par == seq and len(seq) > 20                                      # same row groups, same order
    assert live1["max"] == 1 and live4["max"] == 4                           # concurrency really is N ...
    assert inflight4 <= 4 and live4["max_unconsumed"] <= 4                   # ... and nothing queues beyond N row groups


def test_a_failing_part_goes_to_on_error_and_the_others_still_arrive(mrg_store, env, monkeypatch, cands):
    store = data_io.LocalStore(mrg_store)
    monkeypatch.setenv(omop_cache.ENV_WORKERS, "1")                          # deterministic order of the reads
    real = data_io.read_rowgroup
    calls = {"n": 0}

    def flaky(pf, rg, use, ids, pid_arr, outer):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("boom")
        return real(pf, rg, use, ids, pid_arr, outer)
    monkeypatch.setattr(data_io, "read_rowgroup", flaky)
    errors = []
    with ck.use(step(store)):
        r = omop_cache.reader_for(ck.active(), store, "b", "measurement", columns=omop_cache.superset_columns()["measurement"],
                                  ids=cands, max_get_bytes=1 << 20)
        got = [(k, rg) for k, rg, _t in r.iter_rowgroups(lambda key, exc: errors.append((key, type(exc).__name__)))]
    assert len(errors) == 1 and errors[0][1] == "OSError"
    failed_part = errors[0][0]
    assert [rg for k, rg in got if k == failed_part] == [0, 1]               # the rest of the failed part is skipped
    assert any(k != failed_part for k, _ in got)                             # the other part still arrives


# ============================================================================================ coverage
def test_superset_covers_every_column_any_step_reads():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_baselines as bb
    from sortinghat.audit import alignment as al
    from sortinghat.audit import field_audit as fa
    from sortinghat.cohort import sources as cs
    from sortinghat.labels import extract as lx
    sup = {t: set(c) for t, c in omop_cache.superset_columns().items()}
    need = {"visit_occurrence": [cs._VISIT_COLS, lx.COLUMNS["visit_occurrence"], al.VISIT_COLS],
            "measurement": [cs._MEAS_COLS, lx.COLUMNS["measurement"], fa.MEAS_COLS, bb.MEAS_COLS],
            "condition_occurrence": [cs._COND_COLS, lx.COLUMNS["condition_occurrence"], bb.COND_COLS],
            "drug_exposure": [fa.DRUG_COLS, lx.COLUMNS["drug_exposure"], bb.DRUG_COLS],
            "procedure_occurrence": [lx.COLUMNS["procedure_occurrence"], bb.PROC_COLS],
            "observation": [fa.OBS_COLS, lx.COLUMNS["observation"], bb.OBS_COLS],
            "note": [fa.NOTE_COLS], "person": [al.BIRTH_COLS], "death": [al.DEATH_COLS],
            "concept": [lx.COLUMNS["concept"], ["concept_id", "concept_name", "domain_id", "vocabulary_id"]]}
    for table, lists in need.items():
        for cols in lists:
            assert set(cols) <= sup[table], (table, set(cols) - sup[table])
    for tbl, dt, dd in al.EVENT_SOURCES.values():
        assert {"person_id", dt, dd} <= sup[tbl]
    assert sup["note"] == {"person_id", "note_datetime", "note_date"}         # never note text
