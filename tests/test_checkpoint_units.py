"""Per-row-group restart units of the cohort, baselines and audit steps (SYNTHETIC data): a relaunch skips every stored row
group without reading or recomputing it, progress is logged as aggregate lines, and the accumulated result is unchanged."""
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sortinghat import checkpoint as ck
from sortinghat import data_io, omop_cache
from sortinghat.audit import alignment as al
from sortinghat.audit import field_audit as fa
from sortinghat.checkpoint import SimulatedKill
from sortinghat.cohort import StoreSources
from sortinghat.omop_cache import SKIPPED

RG = 120


@pytest.fixture(scope="module")
def store_dir(synth_dir, tmp_path_factory):
    d = tmp_path_factory.mktemp("cp_units") / "store"
    shutil.copytree(synth_dir, d)
    for f in (d / "OMOP" / "Merged").rglob("*.parquet"):
        pq.write_table(pq.read_table(f), f, row_group_size=RG)
    return d


@pytest.fixture(params=["shared_cache", "step_cache"])
def env(request, tmp_path, monkeypatch):
    monkeypatch.delenv(ck.ENV_FAIL_AFTER, raising=False)
    monkeypatch.setenv(ck.ENV_VERSION, "units-v1")
    monkeypatch.setenv(ck.ENV_ROOT, str(tmp_path / "ck"))
    monkeypatch.setenv(omop_cache.ENV_ROOT, str(tmp_path / "omop"))
    monkeypatch.setenv(omop_cache.ENV_SWITCH, "on" if request.param == "shared_cache" else "off")
    return monkeypatch


def person_ids(store):
    return [int(p) for p in sorted(set(pd.concat(
        [b.to_pandas()[["person_id"]] for b in data_io.iter_omop_batches("measurement", s3=store, columns=["person_id"])]
    )["person_id"].dropna().astype("int64")))]


def test_cohort_iter_units_resume_at_the_last_finished_row_group(store_dir, env, capsys):
    store = data_io.open_store(store_dir)
    src = StoreSources(store)
    pids = person_ids(store)
    calls = []

    def fn(raw):
        calls.append(len(raw))
        return raw[["person_id"]].assign(n=len(raw))

    def run(tag, fail_after=None):
        cp = ck.open_step("cohort", {"t": tag}, store=store, fail_after=fail_after, quiet=True)
        with ck.use(cp):
            return [r for res in src.iter_units("scores", pids, "omop_measurement", ["person_id", "value_as_number"], pids, fn)
                    for r in res]
    ref = run("ref")
    n_full = len(calls)
    assert n_full > 4
    calls.clear()
    with pytest.raises(SimulatedKill):
        run("kill", fail_after=12)
    killed = len(calls)
    calls.clear()
    got = run("kill")                                                   # same arguments: resumes
    assert 0 < killed < n_full and 0 < len(calls) < n_full
    assert killed + len(calls) <= n_full + 1                            # nothing stored was recomputed
    assert len(got) == len(ref) and all(a.equals(b) for a, b in zip(got, ref))
    calls.clear()
    assert len(run("kill")) == len(ref) and not calls                   # finished: nothing recomputed
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("cohort: scores")]
    assert lines and all("row groups (resumed" in l for l in lines)


def test_collect_resumes_without_recomputing(store_dir, env):
    store = data_io.open_store(store_dir)
    pids = person_ids(store)
    cols = ["person_id", "measurement_datetime"]
    calls = []

    def compact(tbl):
        calls.append(tbl.num_rows)
        return pd.DataFrame({"person_id": tbl.column("person_id").to_numpy(), "n": tbl.num_rows})

    def collect(tag, fail_after=None):
        cp = ck.open_step("field_audit", {"t": tag}, store=store, fail_after=fail_after, quiet=True)
        with ck.use(cp):
            return fa._collect(lambda skip: fa.iter_filtered_units(store, "measurement", pids, cols, skip=skip), compact,
                               ["person_id", "n"], "measurement", ("k",))
    ref = collect("ref")
    n_full = len(calls)
    calls.clear()
    with pytest.raises(SimulatedKill):
        collect("kill", fail_after=12)
    killed = len(calls)
    calls.clear()
    got = collect("kill")
    assert 0 < killed < n_full and 0 < len(calls) < n_full and killed + len(calls) <= n_full + 1
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True))
    calls.clear()
    collect("kill")
    assert not calls


def test_nearest_gaps_snapshot_roundtrip():
    rng = np.random.default_rng(1)
    n = 40
    cands = pd.DataFrame({"person_id": np.arange(n), "t0": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 99, n), "h")})

    def chunk():
        pid = rng.integers(0, n, 300)
        t = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(-500, 500, 300), "h")
        return pid, t.to_numpy("datetime64[us]"), (t + pd.Timedelta(days=2)).to_numpy("datetime64[us]")
    chunks = [chunk() for _ in range(6)]

    def feed(acc, cs):
        for pid, s, e in cs:
            acc.add_events("measurement", pid, s)
            acc.add_visits(pid, s, e)
    full = al.NearestGaps(cands)
    feed(full, chunks)
    part = al.NearestGaps(cands)
    feed(part, chunks[:3])
    resumed = al.NearestGaps(cands)
    for src in ("measurement", "visits"):
        resumed.set_state(src, part.get_state(src))
    feed(resumed, chunks[3:])
    pd.testing.assert_frame_equal(resumed.frame(), full.frame())
    pd.testing.assert_frame_equal(resumed.order_frame(), full.order_frame())
