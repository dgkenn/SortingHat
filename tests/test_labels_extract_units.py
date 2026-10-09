"""Silver-label step: vectorised concept ingest equals the row-by-row reference, and the streaming classification is
checkpointed per (table, row group) so a restart resumes at the last finished unit. SYNTHETIC data only."""
import re
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sortinghat import checkpoint as ck
from sortinghat import data_io, omop_cache
from sortinghat.checkpoint import SimulatedKill
from sortinghat.cohort import CohortConfig, StoreSources, build_cohort
from sortinghat.labels import extract as ex
from sortinghat.labels.concepts import ConceptIndex, ConceptMap, RXNORM_VOCABS, normalize_code


# ====================================================================================== ingest vs reference
def reference_ingest(ix: ConceptIndex, df: pd.DataFrame) -> None:
    """The original row-by-row ``ConceptIndex.ingest`` (kept here as the specification of the vectorised one)."""
    d = df[[c for c in ix.CONCEPT_COLUMNS if c in df.columns]].copy()
    for c in ("concept_name", "domain_id", "vocabulary_id", "concept_code"):
        d[c] = d[c].astype(object).where(d[c].notna(), "").astype(str) if c in d else ""
    d["concept_id"] = pd.to_numeric(d["concept_id"], errors="coerce")
    d = d[d["concept_id"].notna()]
    ix.n_ingested += len(d)
    ucum = d[d["vocabulary_id"] == "UCUM"]
    for cid, code in zip(ucum["concept_id"].astype("int64"), ucum["concept_code"]):
        ix.unit_code[int(cid)] = code
    mv = d[d["domain_id"] == "Meas Value"]
    for cid, nm in zip(mv["concept_id"].astype("int64"), mv["concept_name"]):
        ix.value_name[int(cid)] = nm
    d = d[d["vocabulary_id"].isin(ix._vocabs)]
    if d.empty:
        return
    norm = pd.Series([normalize_code(v, c) for v, c in zip(d["vocabulary_id"], d["concept_code"])], index=d.index)
    cm = ix.cm
    exact = {k for k in list(cm.meas_code) + list(cm.cond_exact) + list(cm.proc_exact)}
    mask = pd.Series([k in exact for k in zip(d["vocabulary_id"], norm)], index=d.index)
    for s, p, _, _ in cm.cond_prefix + cm.proc_prefix:
        mask |= (d["vocabulary_id"] == s) & norm.str.startswith(p)
    mask |= (d["vocabulary_id"] == "RxNorm") & d["concept_code"].isin(list(cm.rxnorm_code))
    for cid, vocab, code in zip(d.loc[mask, "concept_id"].astype("int64"), d.loc[mask, "vocabulary_id"], norm[mask]):
        cid = int(cid)
        if (vocab, code) in cm.meas_code:
            ix.meas.setdefault(cid, []).extend(cm.meas_code[(vocab, code)])
        ch = cm._code_hits([(vocab, code)], cm.cond_exact, cm.cond_prefix)
        if ch:
            ix.cond[cid] = tuple(dict.fromkeys(ix.cond.get(cid, ()) + ch))
        ph = cm._code_hits([(vocab, code)], cm.proc_exact, cm.proc_prefix)
        if ph:
            ix.proc[cid] = tuple(dict.fromkeys(ix.proc.get(cid, ()) + ph))
    rx = d[d["vocabulary_id"].isin(RXNORM_VOCABS)]
    for cid, code, vocab in zip(rx["concept_id"].astype("int64"), rx["concept_code"], rx["vocabulary_id"]):
        if vocab == "RxNorm" and code in cm.rxnorm_code:
            ix.drug[int(cid)] = tuple(dict.fromkeys(ix.drug.get(int(cid), ()) + tuple(cm.rxnorm_code[code])))
    if len(rx):
        low = rx["concept_name"].str.lower()
        for g, ing, rxp, exc, _ in cm.drugs:
            for cid, nm in zip(low.index, low):
                if rxp.search(nm) and not (exc is not None and exc.search(nm)):
                    c = int(rx.at[cid, "concept_id"])
                    cur = ix.drug.get(c, ())
                    if (g, ing) not in cur:
                        ix.drug[c] = cur + ((g, ing),)


def concept_frame(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Concept rows covering every branch: matching and non-matching codes in the mapped vocabularies, messy codes (case,
    padding, dots, prefixes), drug names (plain, combination, excluded wording, non-ASCII), units, Meas Value concepts."""
    rng = np.random.default_rng(seed)
    cm = ConceptMap()
    rows = []
    loinc = [c for (s, c) in cm.meas_code if s == "LOINC"]
    snomed = [c for (s, c) in list(cm.cond_exact) + list(cm.proc_exact) if s == "SNOMED"]
    cpt = [c for (s, c) in cm.proc_exact if s == "CPT4"]
    icd10 = [p for s, p, *_ in cm.cond_prefix if s == "ICD10CM"]
    pcs = [p for s, p, *_ in cm.proc_prefix if s == "ICD10PCS"]
    rx = list(cm.rxnorm_code)
    drugs = ["norepinephrine 4 MG", "Epinephrine 1 MG/ML Injectable", "naloxone", "vancomycin 1 g", "ceftriaxone", "Cefazolin Oral",
             "morphine sulfate", "propofol", "lorazepam", "aspirin", "jetinephrine", "Épinéphrine", "levophed", "fentanyl patch"]
    cid = 1000
    for i in range(n):
        k = i % 12
        cid += 1
        if k == 0:
            rows.append((cid, "x", "Measurement", "LOINC", rng.choice(loinc + ["999-9"])))
        elif k == 1:
            rows.append((cid, "x", "Condition", "ICD10CM", rng.choice(icd10 + ["Z00"]) + rng.choice(["", ".1", ".0XA"])))
        elif k == 2:
            rows.append((cid, "x", "Procedure", "ICD10PCS", rng.choice(pcs + ["ZZZ"]) + "3ZZ"))
        elif k == 3:
            rows.append((cid, "x", "Procedure", "CPT4", rng.choice(cpt + ["12345"]) .lower() + rng.choice(["", " "])))
        elif k == 4:
            rows.append((cid, "x", "Condition", "SNOMED", rng.choice(snomed + ["1"])))
        elif k == 5:
            rows.append((cid, rng.choice(drugs) + " " + rng.choice(["Oral Tablet", "Injection", ""]), "Drug",
                         rng.choice(list(RXNORM_VOCABS)), rng.choice(rx + ["123456"])))
        elif k == 6:
            rows.append((cid, str(rng.choice(drugs)), "Drug", "RxNorm Extension", f"OMOP{i}"))
        elif k == 7:
            rows.append((cid, "mg/dL", "Unit", "UCUM", "mg/dL"))
        elif k == 8:
            rows.append((cid, rng.choice(["Positive", "Negative", "Detected"]), "Meas Value", "LOINC", f"LA{i}"))
        elif k == 9:
            rows.append((cid, None, None, "NDC", None))
        elif k == 10:
            rows.append((cid, "x", "Condition", "ICD9CM", rng.choice(["427.5", "348.1", "999"])))
        else:
            rows.append((cid, "y", "Procedure", "HCPCS", rng.choice(["99291", "A0000"])))
    df = pd.DataFrame(rows, columns=["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_code"])
    df.loc[df.index[::97], "concept_code"] = None
    return df


@pytest.mark.parametrize("route", ["pandas", "arrow"])
def test_vectorised_ingest_equals_the_row_by_row_reference(route):
    df = concept_frame()
    ref = ConceptIndex(ConceptMap())
    reference_ingest(ref, df)
    new = ConceptIndex(ConceptMap())
    cuts = [0, len(df) // 3, 2 * len(df) // 3, len(df)]
    for part in (df.iloc[a:b] for a, b in zip(cuts, cuts[1:])):      # several batches, like the streamed read
        new.ingest(part if route == "pandas" else pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False))
    assert new.state() == ref.state()
    assert len(ref.drug) > 5 and len(ref.cond) > 5 and len(ref.proc) > 5 and ref.unit_code and ref.value_name


def test_concept_index_state_roundtrip():
    ix = ConceptIndex(ConceptMap())
    ix.ingest(concept_frame(600))
    other = ConceptIndex(ConceptMap())
    other.load_state(ix.state())
    assert other.state() == ix.state()


# ============================================================================== per-unit checkpoints (restart)
RG = 120


@pytest.fixture(scope="module")
def store_dir(synth_dir, tmp_path_factory):
    d = tmp_path_factory.mktemp("units") / "store"
    shutil.copytree(synth_dir, d)
    for f in (d / "OMOP" / "Merged").rglob("*.parquet"):
        pq.write_table(pq.read_table(f), f, row_group_size=RG)
    return d


@pytest.fixture(scope="module")
def cohort(store_dir):
    return build_cohort(StoreSources(data_io.open_store(store_dir)), CohortConfig(study_sites=None)).table


@pytest.fixture(params=["shared_cache", "step_cache"])
def env(request, tmp_path, monkeypatch):
    monkeypatch.delenv(ck.ENV_FAIL_AFTER, raising=False)
    monkeypatch.setenv(ck.ENV_VERSION, "units-v1")
    monkeypatch.setenv(ck.ENV_ROOT, str(tmp_path / "ck"))
    monkeypatch.setenv(omop_cache.ENV_ROOT, str(tmp_path / "omop"))
    monkeypatch.setenv(omop_cache.ENV_SWITCH, "on" if request.param == "shared_cache" else "off")
    return monkeypatch


class CallCounter:
    """Counts classifier calls (one per Arrow batch) per table."""

    def __init__(self, mp):
        self.n = {}
        for name in ("classify_visits", "classify_measurements", "classify_conditions", "classify_procedures",
                     "classify_observations", "classify_drugs"):
            real = getattr(ex, name)

            def wrap(d, ctx, _real=real, _name=name):
                self.n[_name] = self.n.get(_name, 0) + 1
                return _real(d, ctx)
            wrap.__name__ = name
            mp.setattr(ex, name, wrap)

    @property
    def total(self):
        return sum(self.n.values())


def run(store_dir, cohort, fail_after=None, no_resume=False, tag="a"):
    store = data_io.open_store(store_dir)
    cp = ck.open_step("silver_labels", {"t": tag}, store=store, no_resume=no_resume, fail_after=fail_after, quiet=True)
    with ck.use(cp):
        return ex.extract_silver(store, cohort)


def test_restart_resumes_at_the_last_finished_unit(store_dir, cohort, env, capsys):
    ref = run(store_dir, cohort, tag="ref")
    full = CallCounter(env)
    run(store_dir, cohort, tag="full")
    n_full = full.total
    assert n_full > 8                                                   # several row groups per table

    ctr = CallCounter(env)
    with pytest.raises(SimulatedKill):
        run(store_dir, cohort, fail_after=14, tag="kill")
    killed = ctr.total
    assert 0 < killed < n_full
    ctr.n.clear()
    res = run(store_dir, cohort, tag="kill")                            # same arguments: resumes
    assert 0 < ctr.total < n_full                                       # earlier units were not redone ...
    assert killed + ctr.total <= n_full + 1                            # ... (at most the unit being written when killed)
    pd.testing.assert_frame_equal(res.labels, ref.labels)
    pd.testing.assert_frame_equal(res.events.reset_index(drop=True), ref.events.reset_index(drop=True))
    assert res.diagnostics == ref.diagnostics

    ctr.n.clear()
    again = run(store_dir, cohort, tag="kill")                          # finished run: nothing is classified again
    assert ctr.total == 0
    pd.testing.assert_frame_equal(again.labels, ref.labels)
    ctr.n.clear()
    fresh = run(store_dir, cohort, tag="kill", no_resume=True)
    assert ctr.total == n_full
    pd.testing.assert_frame_equal(fresh.labels, ref.labels)


def test_progress_lines_are_aggregate_and_per_table(store_dir, cohort, env, capsys):
    run(store_dir, cohort, tag="prog")
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("silver_labels:")]
    for table in ("concept", "visit_occurrence", "measurement", "condition_occurrence", "procedure_occurrence",
                  "observation", "drug_exposure"):
        assert any(re.fullmatch(rf"silver_labels: {table} \d+(/\d+)? row groups \(resumed \d+\)( done)?", l) for l in lines), table
    assert any("concept map done" in l for l in lines)
