"""Checkpoint / resume (sortinghat/checkpoint.py and its use in data_io, the cohort, audit, silver-label and baseline
steps, and the silver-feasibility model fits). SYNTHETIC data only.

Every step test follows the same script: run uninterrupted (reference); run again with the test hook that raises
``SimulatedKill`` after N stored units (a restart); rerun with the same arguments; assert the outputs are identical to the
reference and that work already stored was not redone (counted at the pyarrow row-group read, or at the model fit).
"""
import shutil
import stat

import pandas as pd
import pyarrow.parquet as pq
import pytest

from sortinghat import checkpoint as ck
from sortinghat import data_io
from sortinghat.checkpoint import SimulatedKill
from sortinghat.cohort.sources import iter_filtered_batches

RG = 250                                                    # rows per row group in the rewritten synthetic store


@pytest.fixture(scope="module")
def mrg_store(synth_dir, tmp_path_factory):
    """The synthetic store with every OMOP parquet part rewritten into many small row groups."""
    d = tmp_path_factory.mktemp("multi_rg") / "store"
    shutil.copytree(synth_dir, d)
    for f in (d / "OMOP" / "Merged").rglob("*.parquet"):
        t = pq.read_table(f)
        pq.write_table(t, f, row_group_size=RG)
    return d


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "ckroot"
    monkeypatch.setenv(ck.ENV_ROOT, str(r))
    monkeypatch.delenv(ck.ENV_FAIL_AFTER, raising=False)
    monkeypatch.setenv(ck.ENV_VERSION, "test-v1")
    return r


class Counter:
    """Counts pyarrow row-group reads and parquet GETs of the store."""

    def __init__(self, monkeypatch):
        self.rg = 0
        self.gets = 0
        real_rg = pq.ParquetFile.read_row_group
        real_get = data_io.LocalStore.get_object
        me = self

        def read_row_group(pf, i, *a, **kw):
            me.rg += 1
            return real_rg(pf, i, *a, **kw)

        def get_object(store, Bucket=None, Key="", Range=None):
            if Key.endswith(".parquet"):
                me.gets += 1
            return real_get(store, Bucket=Bucket, Key=Key, Range=Range)
        monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_row_group)
        monkeypatch.setattr(data_io.LocalStore, "get_object", get_object)

    def reset(self):
        self.rg = self.gets = 0


@pytest.fixture
def counter(monkeypatch):
    return Counter(monkeypatch)


# ===================================================================================================== the helper
def test_digest_is_stable_and_sensitive():
    a = {"x": [1, 2.5, "s"], "df": pd.DataFrame({"a": [1, 2], "b": ["u", "v"]})}
    b = {"df": pd.DataFrame({"a": [1, 2], "b": ["u", "v"]}), "x": [1, 2.5, "s"]}
    assert ck.digest(a) == ck.digest(b)
    assert ck.digest(a) != ck.digest({**a, "x": [1, 2.5, "t"]})
    assert ck.digest(a) != ck.digest({**a, "df": pd.DataFrame({"a": [1, 3], "b": ["u", "v"]})})
    assert ck.digest(1) != ck.digest(1.0) != ck.digest("1")


def test_dirs_and_files_are_private_and_writes_are_atomic(root):
    cp = ck.open_step("unit", {"a": 1}, quiet=True)
    cp.put("n1", {"v": [1, 2, 3]})
    assert cp.get("n1") == {"v": [1, 2, 3]}
    assert cp.get("absent") is ck.MISS
    for d in (cp.dir, cp.dir.parent, cp.dir.parent.parent, cp.path("n1").parent):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert stat.S_IMODE(cp.path("n1").stat().st_mode) == 0o600
    assert not [p for p in cp.dir.rglob("*") if ".tmp." in p.name]              # no temporary file left behind
    cp.path("n1").write_bytes(b"truncated garbage")                              # a corrupt unit is a miss, then replaced
    assert cp.get("n1") is ck.MISS
    cp.put("n1", 7)
    assert cp.get("n1") == 7


def test_key_changes_with_args_inputs_and_code_version(root, tmp_path, monkeypatch):
    f = tmp_path / "in.csv"
    f.write_text("a,b\n1,2\n")
    base = ck.open_step("k", {"a": 1}, inputs=[f], quiet=True).key
    assert ck.open_step("k", {"a": 1}, inputs=[f], quiet=True).key == base
    assert ck.open_step("k", {"a": 2}, inputs=[f], quiet=True).key != base
    assert ck.open_step("k", {"a": 1, "workers": 9, "no_resume": True}, inputs=[f], quiet=True).key == base   # execution knobs
    f.write_text("a,b\n1,3\n")                                                   # same size, other content
    assert ck.open_step("k", {"a": 1}, inputs=[f], quiet=True).key != base
    f.write_text("a,b\n1,2\n")
    assert ck.open_step("k", {"a": 1}, inputs=[f], quiet=True).key == base       # rewritten with identical bytes: same key
    monkeypatch.setenv(ck.ENV_VERSION, "test-v2")
    assert ck.open_step("k", {"a": 1}, inputs=[f], quiet=True).key != base


def test_stale_directories_are_pruned_and_no_resume_wipes(root):
    old = ck.open_step("p", {"a": 1}, quiet=True)
    old.put("u", 1)
    new = ck.open_step("p", {"a": 2}, quiet=True, prune_idle_h=0)               # other key, idle >= 0 h: removed
    assert not old.dir.exists() and new.dir.exists()
    keep = ck.open_step("p", {"a": 3}, quiet=True)                              # default 24 h: the idle sibling survives
    assert new.dir.exists() and keep.dir.exists()
    new.put("u", 5)
    fresh = ck.open_step("p", {"a": 2}, quiet=True, no_resume=True)
    assert fresh.dir == new.dir and fresh.get("u") is ck.MISS


def test_stage_runs_once_and_kill_hook_is_a_baseexception(root):
    cp = ck.open_step("s", {}, quiet=True, fail_after=2)
    calls = []
    assert cp.stage("one", 1, lambda: calls.append(1) or "a") == "a"
    with pytest.raises(SimulatedKill):
        cp.stage("two", 2, lambda: calls.append(2) or "b")
    assert not issubclass(SimulatedKill, Exception)
    cp2 = ck.open_step("s", {}, quiet=True)
    assert cp2.stage("one", 1, lambda: calls.append(3) or "zzz") == "a"
    assert cp2.stage("two", 2, lambda: calls.append(4) or "zzz") == "b"        # the unit written before the kill is kept
    assert calls == [1, 2]
    assert cp2.stage("one", "other key", lambda: "fresh") == "fresh"            # same name, other key object: not reused


# ====================================================================================== OMOP readers (data_io)
def _read(store, **kw):
    cols = ["person_id", "measurement_datetime", "measurement_source_value", "value_as_number"]
    parts = [b.to_pandas() for b in data_io.iter_omop_batches("measurement", s3=store, columns=cols, **kw)]
    return pd.concat(parts, ignore_index=True)


def _open(store, name="reader", **kw):
    return ck.open_step(name, {"t": "x"}, store=store, **kw)


def test_unfiltered_reader_resumes_without_rereading_row_groups(mrg_store, root, counter, capsys):
    store = data_io.LocalStore(mrg_store)
    ref = _read(store)
    n_rg = counter.rg
    assert n_rg >= 8
    counter.reset()

    cp = _open(store, fail_after=5)
    with ck.use(cp), pytest.raises(SimulatedKill):
        _read(store)
    assert counter.rg == 5 and cp.n_puts == 5

    counter.reset()
    with ck.use(_open(store)):
        got = _read(store)
    pd.testing.assert_frame_equal(got, ref)
    assert counter.rg == n_rg - 5                                    # the 5 stored row groups were not read again

    counter.reset()
    with ck.use(_open(store)):
        again = _read(store)
    pd.testing.assert_frame_equal(again, ref)
    assert counter.rg == 0 and counter.gets == 0                     # fully stored: the store is not touched at all
    out = capsys.readouterr().out
    assert f"omop_measurement: {n_rg}/{n_rg} row groups (cached {n_rg})" in out
    assert "resuming with" in out


def test_filtered_readers_resume_and_replay_the_same_rows(mrg_store, root, counter):
    store = data_io.LocalStore(mrg_store)
    ids = sorted(pd.read_parquet(mrg_store / "OMOP/Merged/measurement/part-00000.parquet", columns=["person_id"])
                 ["person_id"].unique())[:300]
    ref = _read(store, person_ids=ids)
    assert len(ref) > 0
    counter.reset()
    cp = _open(store, "filtered", fail_after=6)
    with ck.use(cp), pytest.raises(SimulatedKill):
        _read(store, person_ids=ids)
    done_first = counter.rg
    counter.reset()
    with ck.use(_open(store, "filtered")):
        got = _read(store, person_ids=ids)
    pd.testing.assert_frame_equal(got, ref)
    counter.reset()
    with ck.use(_open(store, "filtered")):
        pd.testing.assert_frame_equal(_read(store, person_ids=ids), ref)
    assert counter.rg == 0 and counter.gets == 0 and done_first > 0
    # another id set is another call: nothing from the first is reused for it
    counter.reset()
    with ck.use(_open(store, "filtered")):
        other = _read(store, person_ids=ids[:5])
    assert counter.rg > 0 and len(other) < len(ref)


def test_two_stage_predicate_reader_resumes(mrg_store, root, counter):
    store = data_io.LocalStore(mrg_store)
    ids = sorted(pd.read_parquet(mrg_store / "OMOP/Merged/measurement/part-00001.parquet", columns=["person_id"])
                 ["person_id"].unique())
    cols = ["person_id", "measurement_datetime", "measurement_source_value"]

    def run():
        return [t.to_pandas() for t in iter_filtered_batches(store, "measurement", ids, cols, text_col="measurement_source_value",
                                                             pattern="gcs|glasgow|lactate|map|heart", prefix=None)]
    ref = pd.concat(run(), ignore_index=True)
    assert len(ref) > 0
    counter.reset()
    with ck.use(_open(store, "two_stage", fail_after=4)), pytest.raises(SimulatedKill):
        run()
    counter.reset()
    with ck.use(_open(store, "two_stage")):
        got = pd.concat(run(), ignore_index=True)
    pd.testing.assert_frame_equal(got, ref)
    counter.reset()
    with ck.use(_open(store, "two_stage")):
        pd.testing.assert_frame_equal(pd.concat(run(), ignore_index=True), ref)
    assert counter.rg == 0 and counter.gets == 0


def test_stale_checkpoints_are_never_reused(mrg_store, root, counter, monkeypatch, tmp_path):
    store_dir = tmp_path / "s"
    shutil.copytree(mrg_store, store_dir)
    store = data_io.LocalStore(store_dir)
    with ck.use(_open(store)):
        ref = _read(store)
    full = counter.rg
    counter.reset()
    with ck.use(_open(store)):
        _read(store)
    assert counter.rg == 0                                           # sanity: reused when nothing changed

    monkeypatch.setenv(ck.ENV_VERSION, "test-v2")                    # other code version
    counter.reset()
    with ck.use(_open(store)):
        pd.testing.assert_frame_equal(_read(store), ref)
    assert counter.rg == full
    monkeypatch.setenv(ck.ENV_VERSION, "test-v1")

    f = store_dir / "OMOP/Merged/measurement/part-00001.parquet"      # an input part is replaced (different bytes)
    t = pq.read_table(f)
    pq.write_table(t.slice(0, t.num_rows - 3), f, row_group_size=RG)
    counter.reset()
    with ck.use(_open(store)):
        changed = _read(store)
    assert counter.rg > 0 and len(changed) == len(ref) - 3

    with ck.use(_open(store, "reader2")):                             # other step name, other arguments
        counter.reset()
        _read(store)
    assert counter.rg > 0


def test_no_resume_ignores_a_complete_checkpoint(mrg_store, root, counter):
    store = data_io.LocalStore(mrg_store)
    with ck.use(_open(store)):
        ref = _read(store)
    full = counter.rg
    counter.reset()
    with ck.use(_open(store, no_resume=True)):
        pd.testing.assert_frame_equal(_read(store), ref)
    assert counter.rg == full
    counter.reset()
    with ck.use(_open(store)):                                        # the --no-resume run was itself checkpointed
        _read(store)
    assert counter.rg == 0


def test_no_active_checkpoint_changes_nothing(mrg_store, root, counter):
    store = data_io.LocalStore(mrg_store)
    a = _read(store)
    b = _read(store)
    pd.testing.assert_frame_equal(a, b)
    assert not root.exists()


def test_progress_lines_are_aggregate_only(mrg_store, root, capsys):
    store = data_io.LocalStore(mrg_store)
    with ck.use(_open(store)):
        _read(store)
    out = capsys.readouterr().out
    assert "row groups (cached 0)" in out
    assert str(mrg_store) not in out and "part-0000" not in out and "OMOP/Merged" not in out
