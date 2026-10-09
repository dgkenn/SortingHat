"""OMOP concept mapping for the STRUCTURED silver-label extractor (``configs/anchor_concepts.yaml``).

Maps every anchor item of ``configs/silver_anchors.yaml`` to OMOP rows:

* codes (LOINC / ICD-10-CM / ICD-9-CM / ICD-10-PCS / CPT4 / SNOMED / RxNorm) resolve to ``concept_id`` through
  ``omop_concept`` (``concept_code`` + ``vocabulary_id``): ``ConceptIndex.ingest`` is fed concept batches;
* rows with zero-filled concept ids fall back to the ``*_source_value`` text (code strings, then name regexes);
* quantitative values are converted to the canonical unit of the anchor config (ammonia ug/dL -> umol/L, glucose
  mmol/L -> mg/dL, temperature F -> C, ...). An unrecognised unit DROPS the row; nothing is guessed;
* qualitative results (cultures, PCR, antibodies, tox screens) are read as positive / negative / pending.

Everything here is a *mapping rule*, not data, and is PROVISIONAL until a human verifies the codes against the local
concept table (``docs/silver_extraction.md``). No record-level data is stored or printed by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

CONCEPT_PATH = Path(__file__).resolve().parents[2] / "configs" / "anchor_concepts.yaml"
ICD_SYSTEMS = ("ICD10CM", "ICD9CM", "ICD10PCS", "ICD9Proc")
RXNORM_VOCABS = ("RxNorm", "RxNorm Extension")
STATUS_FULL, STATUS_PROXY, STATUS_UNAVAILABLE = "full", "proxy", "unavailable"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_concept_map(path: str | Path | None = None) -> dict:
    return _load(str(path or CONCEPT_PATH))


# ----------------------------------------------------------------------------------------------------- units
def clean_unit(u) -> str:
    """Case/whitespace/micro-sign normalisation of a unit string (alias lookup is done by ``ConceptMap``)."""
    if u is None or (isinstance(u, float) and np.isnan(u)) or u is pd.NA:
        return ""
    s = str(u).strip().lower().replace("µ", "u").replace("μ", "u").replace("°", "")
    s = re.sub(r"\s+", "", s).replace("mcmol", "umol").replace("mcg", "ug").replace("^", "*")
    return s


def normalize_code(vocab: str, code) -> str:
    """Canonical form of a ``concept_code`` for comparison (ICD/PCS: dots removed, upper case)."""
    s = "" if code is None or (isinstance(code, float) and np.isnan(code)) else str(code).strip()
    if vocab in ICD_SYSTEMS:
        return re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return s.upper() if vocab in ("CPT4", "HCPCS") else s


_SRC_PREFIX = re.compile(r"^(icd-?10(-?cm|-?pcs)?|icd-?9(-?cm)?|cpt4?|hcpcs|snomed|loinc)\s*[:_ -]\s*", re.I)


def normalize_source_code(value) -> str:
    """Normalise a ``*_source_value`` that may hold a code (``ICD10:I46.9`` -> ``I469``). Upper case, alnum only."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = _SRC_PREFIX.sub("", str(value).strip())
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def _conv_tuple(v) -> tuple[float, float, float]:
    if isinstance(v, dict):
        return float(v.get("pre_add", 0.0)), float(v.get("mul", 1.0)), float(v.get("add", 0.0))
    return 0.0, float(v), 0.0


@dataclass
class Spec:
    """One measurement-like item (quantitative lab/vital, support series, or qualitative result)."""
    key: str
    kind: str                                    # quant | support | qual
    name_re: re.Pattern | None
    excl_re: re.Pattern | None
    unit: str = ""                               # canonical unit token ("" for flags)
    unit_label: str = ""
    conv: dict = field(default_factory=dict)     # normalised unit token -> (pre_add, mul, add)
    plausible: tuple[float, float] | None = None
    unit_optional: bool = False
    unused: bool = False
    codes: list = field(default_factory=list)    # [(system, normalised code, unit token or "")]
    raw: dict = field(default_factory=dict)


class ConceptMap:
    """Compiled view of ``anchor_concepts.yaml``: regexes, unit tables and code tables."""

    def __init__(self, raw: dict | None = None):
        self.raw = raw or load_concept_map()
        r = self.raw
        self.unit_alias = {clean_unit(k): clean_unit(v) for k, v in r.get("unit_aliases", {}).items()}
        self.specs: dict[str, Spec] = {}
        for kind, sec in (("quant", "measurement_items"), ("support", "support_items"), ("qual", "qualitative_items")):
            for key, d in r.get(sec, {}).items():
                self.specs[key] = self._spec(key, kind, d)
        self.result_re = {k: re.compile(v, re.I) for k, v in r.get("result_text", {}).items()}
        self.tox = {a: (re.compile(d["name_regex"], re.I), tuple(d["in_hospital_drugs"]))
                    for a, d in r.get("tox_agents", {}).items()}
        self.tox_context_re = re.compile(r["tox_context_regex"], re.I)
        self.tox_exclude_re = re.compile(r["tox_exclude_regex"], re.I)
        # code-event items
        self.event_items: dict[str, dict] = r.get("code_event_items", {})
        self.cond_exact: dict[tuple[str, str], list[str]] = {}
        self.cond_prefix: list[tuple[str, str, str, re.Pattern | None]] = []
        self.proc_exact: dict[tuple[str, str], list[str]] = {}
        self.proc_prefix: list[tuple[str, str, str, re.Pattern | None]] = []
        self.event_text: dict[str, tuple[re.Pattern, re.Pattern | None, tuple[str, ...]]] = {}
        for item, d in self.event_items.items():
            for tag, exact, pref in (("condition_codes", self.cond_exact, self.cond_prefix),
                                     ("procedure_codes", self.proc_exact, self.proc_prefix)):
                for c in d.get(tag) or []:
                    sysn, code = c["system"], normalize_code(c["system"], c["code"])
                    ex = re.compile(c["exclude_regex"]) if c.get("exclude_regex") else None
                    if c.get("prefix"):
                        pref.append((sysn, code, item, ex))
                    else:
                        exact.setdefault((sysn, code), []).append(item)
            if d.get("text_regex"):
                self.event_text[item] = (re.compile(d["text_regex"], re.I),
                                         re.compile(d["text_exclude"], re.I) if d.get("text_exclude") else None,
                                         tuple(d.get("text_tables") or ()))
        # measurement code tables: (system, code) -> [(item, unit token)]
        self.meas_code: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for key, sp in self.specs.items():
            for system, code, unit in sp.codes:
                self.meas_code.setdefault((system, code), []).append((key, unit))
        # drug registry: (group, ingredient, regex, exclude regex or None, extra)
        self.drugs: list[tuple[str, str, re.Pattern, re.Pattern | None, dict]] = []
        self.rxnorm_code: dict[str, list[tuple[str, str]]] = {}
        dg = r.get("drug_groups", {})
        ex_main = re.compile(r["drug_exclude_regex"], re.I)
        ex_abx = re.compile(r["antimicrobial_exclude_regex"], re.I)
        for group in ("vasopressor", "antidote", "sedative"):
            g = dg.get(group, {})
            extra = re.compile(g["extra_exclude"], re.I) if g.get("extra_exclude") else None
            for ing, d in g.get("ingredients", {}).items():
                ex = _either(ex_main, extra) if group in ("vasopressor", "antidote") else None
                self.drugs.append((group, ing, re.compile(d["regex"], re.I), ex, dict(d)))
                if d.get("rxnorm"):
                    self.rxnorm_code.setdefault(str(d["rxnorm"]), []).append((group, ing))
        for ing, rx in dg.get("other_ingredients", {}).items():
            self.drugs.append(("other", ing, re.compile(rx, re.I), None, {}))
        for ing, d in r.get("antimicrobials", {}).items():
            self.drugs.append(("antimicrobial", ing, re.compile(d["regex"], re.I), ex_abx, dict(d)))
        self.exposure_sets = {k: tuple(v) for k, v in dg.get("exposure_sets", {}).items()}
        self.abx_route = {ing: d.get("route", "iv") for ing, d in r.get("antimicrobials", {}).items()}
        self.abx_iv_re = re.compile(r["antimicrobial_iv_route_regex"], re.I)
        self.abx_oral_re = re.compile(r["antimicrobial_oral_route_regex"], re.I)
        self._name_cache: dict[str, tuple] = {}
        self._drug_cache: dict[str, tuple] = {}

    # ------------------------------------------------------------------------------------------ specs
    def _spec(self, key: str, kind: str, d: dict) -> Spec:
        conv = {}
        for u, c in (d.get("conversions") or {}).items():
            tok = self.unit_alias.get(clean_unit(u), clean_unit(u))
            conv[tok] = _conv_tuple(c)
        unit = d.get("unit", "")
        unit_tok = self.unit_alias.get(clean_unit(unit), clean_unit(unit)) if unit else ""
        codes = []
        for c in d.get("codes") or []:
            u = c.get("unit", "")
            codes.append((c["system"], normalize_code(c["system"], c["code"]),
                          self.unit_alias.get(clean_unit(u), clean_unit(u)) if u else ""))
        return Spec(key=key, kind=kind,
                    name_re=re.compile(d["name_regex"], re.I) if d.get("name_regex") else None,
                    excl_re=re.compile(d["name_exclude"], re.I) if d.get("name_exclude") else None,
                    unit=unit_tok, unit_label=unit, conv=conv,
                    plausible=tuple(d["plausible"]) if d.get("plausible") else None,
                    unit_optional=bool(d.get("unit_optional")), unused=bool(d.get("unused")), codes=codes, raw=d)

    def unit_token(self, u) -> str:
        c = clean_unit(u)
        return self.unit_alias.get(c, c)

    # ------------------------------------------------------------------------------- unit conversion
    def _unit_tokens(self, units) -> np.ndarray:
        """``unit_token`` of every element, computed once per unique unit string (object array)."""
        arr = np.asarray(units, dtype=object)
        codes, uniq = pd.factorize(arr)
        tab = np.array([self.unit_token(u) for u in uniq] + [""], dtype=object)       # last slot: missing (code -1)
        return tab[codes]

    def convert_value(self, item: str, value: float, unit, default_unit: str = "") -> tuple[float, str]:
        """Convert one value to the item's canonical unit. Returns ``(value, status)``; status is ``ok``,
        ``unit_missing``, ``unit_unrecognised`` or ``implausible`` (value is NaN unless ``ok``)."""
        out, st = self.convert_series(item, pd.Series([value], dtype=float), pd.Series([unit], dtype=object),
                                      pd.Series([default_unit], dtype=object))
        return float(out.iloc[0]), str(st.iloc[0])

    def convert_series(self, item: str, values: pd.Series, units: pd.Series,
                       default_units: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
        """Vectorised ``convert_value``. ``default_units`` (unit stored with the matched LOINC code) fills a blank
        unit; a blank unit with no default is dropped unless the item is ``unit_optional`` (pH)."""
        sp = self.specs[item]
        v = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
        tok = self._unit_tokens(units)
        if default_units is not None:
            dft = self._unit_tokens(default_units)
            tok = np.where(tok == "", dft, tok)
        out = np.full(len(v), np.nan)
        status = np.empty(len(v), dtype=object)
        status[:] = "ok"
        for t in pd.unique(tok):
            m = tok == t
            if t == "":
                if sp.unit_optional:
                    out[m] = v[m]
                else:
                    status[m] = "unit_missing"
                continue
            c = sp.conv.get(t)
            if c is None:
                status[m] = "unit_unrecognised"
                continue
            pre, mul, add = c
            out[m] = (v[m] + pre) * mul + add
        if sp.plausible is not None:
            lo, hi = sp.plausible
            bad = (status == "ok") & ~np.isnan(out) & ((out < lo) | (out > hi))
            status[bad] = "implausible"
        status[(status == "ok") & np.isnan(out)] = "no_value"
        out[status != "ok"] = np.nan
        return pd.Series(out, index=values.index), pd.Series(status, index=values.index)

    # ----------------------------------------------------------------------------- name classification
    def classify_measurement_name(self, name) -> tuple[str, ...]:
        """Items whose name regex matches ``name`` (and whose exclude regex does not). Tox agents are separate."""
        if name is None or (isinstance(name, float) and np.isnan(name)):
            return ()
        key = str(name)
        hit = self._name_cache.get(key)
        if hit is None:
            low = key.lower()
            hit = tuple(k for k, sp in self.specs.items()
                        if sp.name_re is not None and sp.name_re.search(low)
                        and not (sp.excl_re is not None and sp.excl_re.search(low)))
            self._name_cache[key] = hit
        return hit

    def classify_tox_name(self, name) -> tuple[str, ...]:
        """Tox-screen agents named in a measurement name (needs screen/drug context; history/plan text excluded)."""
        if name is None or (isinstance(name, float) and np.isnan(name)):
            return ()
        low = str(name).lower()
        if self.tox_exclude_re.search(low) or not self.tox_context_re.search(low):
            return ()
        return tuple(a for a, (rx, _) in self.tox.items() if rx.search(low))

    # ------------------------------------------------------------------------------ result semantics
    def result_status(self, item: str, text, concept_name=None, number=np.nan) -> str:
        """``positive`` | ``negative`` | ``pending`` | ``commensal`` | ``unknown`` for a qualitative result row.

        Text sources, in order: ``value_source_value``, the name of ``value_as_concept_id``. The number is used only for
        items that allow it (``numeric_positive``, tox levels): >0 is positive, 0 is negative."""
        sp = self.specs.get(item)
        raw = sp.raw if sp is not None else {}
        t = ""
        for cand in (text, concept_name):
            if cand is not None and not (isinstance(cand, float) and np.isnan(cand)) and str(cand).strip():
                t = str(cand).strip().lower()
                break
        num_ok = raw.get("numeric_positive", item.startswith("tox:"))
        has_num = number is not None and not (isinstance(number, float) and np.isnan(number))
        if not t:
            if has_num and num_ok:
                return "positive" if float(number) > 0 else "negative"
            return "unknown"
        if raw.get("commensal_regex") and re.search(raw["commensal_regex"], t, re.I):
            return "commensal"
        if self.result_re["negative"].search(t):
            return "negative"
        if self.result_re["pending"].search(t):
            return "pending"
        if self.result_re["positive"].search(t):
            return "positive"
        if raw.get("nonneg_text_is_positive"):
            return "positive"
        if has_num and num_ok and float(number) > 0:
            return "positive"
        return "unknown"

    # ---------------------------------------------------------------------------- code-event helpers
    def _prefix_index(self, prefix: list):
        """Prefix rules indexed by (system, prefix text) with the distinct prefix lengths per system, so a code is tested with
        one dict lookup per length instead of one ``startswith`` per rule. Cached per rule list."""
        cache = self.__dict__.setdefault("_pidx", {})
        got = cache.get(id(prefix))
        if got is not None and got[0] is prefix and got[1] == len(prefix):
            return got[2], got[3]
        by: dict = {}
        lens: dict = {}
        for pos, (s, p, it, ex) in enumerate(prefix):
            by.setdefault((s, p), []).append((pos, it, ex))
            lens.setdefault(s, set()).add(len(p))
        lens = {s: sorted(v) for s, v in lens.items()}
        cache[id(prefix)] = (prefix, len(prefix), by, lens)
        return by, lens

    def _code_hits(self, system_codes: Iterable[tuple[str, str]], exact, prefix) -> tuple[str, ...]:
        hits: list[str] = []
        by, lens = self._prefix_index(prefix)
        for system, code in system_codes:
            for it in exact.get((system, code), ()):
                if it not in hits:
                    hits.append(it)
            found = []
            for n in lens.get(system, ()):
                if n <= len(code):
                    found.extend(by.get((system, code[:n]), ()))
            for _pos, it, ex in sorted(found, key=lambda x: x[0]):
                if not (ex is not None and ex.search(code)) and it not in hits:
                    hits.append(it)
        return tuple(hits)

    def condition_source_hits(self, value) -> tuple[str, ...]:
        """Event items whose ICD code matches a ``condition_source_value`` (ICD-9 and ICD-10 are mixed in HEEDB)."""
        c = normalize_source_code(value)
        return self._code_hits([("ICD10CM", c), ("ICD9CM", c)], self.cond_exact, self.cond_prefix) if c else ()

    def procedure_source_hits(self, value) -> tuple[str, ...]:
        """Event items whose CPT / ICD-10-PCS / ICD-9 procedure code matches a ``procedure_source_value``."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ()
        s = str(value).strip()
        cands = {normalize_source_code(s)}
        first = re.split(r"[\s,;|-]+", _SRC_PREFIX.sub("", s))[0]
        cands.add(re.sub(r"[^A-Za-z0-9]", "", first).upper())
        cands.discard("")
        pairs = [(sy, c) for c in cands for sy in ("CPT4", "HCPCS", "ICD10PCS", "ICD9Proc")]
        return self._code_hits(pairs, self.proc_exact, self.proc_prefix)

    def text_hits(self, table: str, text) -> tuple[str, ...]:
        """Event items matched by free text in ``observation`` / ``procedure`` source values (arrest wording)."""
        if text is None or (isinstance(text, float) and np.isnan(text)) or not str(text).strip():
            return ()
        low = str(text).lower()
        out = []
        for it, (rx, ex, tables) in self.event_text.items():
            if (not tables or table in tables) and rx.search(low) and not (ex is not None and ex.search(low)):
                out.append(it)
        return tuple(out)

    # ---------------------------------------------------------------------------------- drug matching
    def drug_text_hits(self, text) -> tuple[tuple[str, str], ...]:
        """``(group, ingredient)`` pairs named in a drug text (combination products return several)."""
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ()
        key = str(text)
        hit = self._drug_cache.get(key)
        if hit is None:
            low = key.lower()
            hit = tuple((g, ing) for g, ing, rx, ex, _ in self.drugs
                        if rx.search(low) and not (ex is not None and ex.search(low)))
            self._drug_cache[key] = hit
        return hit

    def abx_route_ok(self, ingredient: str, route, text) -> bool:
        """IV/IM-only agents need an IV/IM route (or an IV/injection marker in the text when the route is blank);
        ``any``-route agents qualify by any route."""
        if self.abx_route.get(ingredient, "iv") == "any":
            return True
        r = "" if route is None or (isinstance(route, float) and np.isnan(route)) else str(route).lower()
        if r.strip():
            return bool(self.abx_iv_re.search(r)) and not self.abx_oral_re.search(r)
        t = str(text or "").lower()
        return bool(self.abx_iv_re.search(t)) and not self.abx_oral_re.search(t)


def _either(a: re.Pattern, b: re.Pattern | None) -> re.Pattern:
    return a if b is None else re.compile(f"(?:{a.pattern})|(?:{b.pattern})", re.I)


# ======================================================================================== concept index
class ConceptIndex:
    """Resolved ``concept_id`` -> item tables, built from ``omop_concept`` batches (``ingest``).

    Only concepts that match the map are kept (memory stays small even on the multi-million-row concept table).
    Also keeps UCUM unit codes and the names of ``Meas Value`` concepts (``Positive``, ``Negative``, ...)."""

    CONCEPT_COLUMNS = ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_code"]

    def __init__(self, cm: ConceptMap | None = None):
        self.cm = cm or ConceptMap()
        self.meas: dict[int, list[tuple[str, str]]] = {}      # id -> [(item, unit token)]
        self.cond: dict[int, tuple[str, ...]] = {}
        self.proc: dict[int, tuple[str, ...]] = {}
        self.drug: dict[int, tuple[tuple[str, str], ...]] = {}
        self.unit_code: dict[int, str] = {}
        self.value_name: dict[int, str] = {}
        self.n_ingested = 0
        self._vocabs = set(self.cm.raw.get("vocabularies", []))

    # -- state (checkpointed once per run: the classified concept map)
    def state(self) -> dict:
        return {"meas": self.meas, "cond": self.cond, "proc": self.proc, "drug": self.drug, "unit_code": self.unit_code,
                "value_name": self.value_name, "n_ingested": self.n_ingested}

    def load_state(self, st: dict) -> None:
        for k, v in st.items():
            setattr(self, k, v)

    # -- vectorised ingest
    def _prepare(self) -> None:
        """Lookup structures for the vectorised mask (built once): exact codes per vocabulary, prefix rules, RxNorm codes."""
        if getattr(self, "_prep", None) is not None:
            return
        cm = self.cm
        exact: dict[str, list[str]] = {}
        for v, c in list(cm.meas_code) + list(cm.cond_exact) + list(cm.proc_exact):
            exact.setdefault(v, []).append(c)
        pats = [rxp.pattern for _, _, rxp, _, _ in cm.drugs]
        self._prep = {"exact": exact, "prefix": list(cm.cond_prefix) + list(cm.proc_prefix),
                      "rxcodes": list(cm.rxnorm_code),
                      "any_rx_re2": "|".join(f"(?:{p})" for p in pats) if pats else None}

    def _as_arrow(self, df):
        """Concept rows as an Arrow table: ``concept_id`` int64 without nulls, the four text columns as non-null strings."""
        import pyarrow as pa
        import pyarrow.compute as pc
        if isinstance(df, pa.RecordBatch):
            df = pa.Table.from_batches([df])
        if isinstance(df, pa.Table) and "concept_id" in df.column_names and pa.types.is_integer(df.schema.field("concept_id").type):
            cols = [c for c in self.CONCEPT_COLUMNS if c in df.column_names]
            t = df.select(cols)
            t = t.filter(pc.is_valid(t.column("concept_id")))
            data = {"concept_id": pc.cast(t.column("concept_id"), pa.int64())}
            for c in ("concept_name", "domain_id", "vocabulary_id", "concept_code"):
                if c in t.column_names:
                    data[c] = pc.fill_null(pc.cast(t.column(c), pa.string()), "")
                else:
                    data[c] = pa.array([""] * t.num_rows, type=pa.string())
            return pa.table(data)
        if isinstance(df, pa.Table):
            df = df.to_pandas()
        d = df[[c for c in self.CONCEPT_COLUMNS if c in df.columns]].copy()
        for c in ("concept_name", "domain_id", "vocabulary_id", "concept_code"):
            d[c] = d[c].astype(object).where(d[c].notna(), "").astype(str) if c in d else ""
        d["concept_id"] = pd.to_numeric(d["concept_id"], errors="coerce")
        d = d[d["concept_id"].notna()]
        d["concept_id"] = d["concept_id"].astype("int64")
        return pa.table({"concept_id": pa.array(d["concept_id"].to_numpy(), type=pa.int64()),
                         **{c: pa.array(d[c].astype(object).tolist(), type=pa.string())
                            for c in ("concept_name", "domain_id", "vocabulary_id", "concept_code")}})

    def ingest(self, df) -> None:
        """Add one batch of ``omop_concept`` rows (a DataFrame, Arrow table or record batch with columns ``CONCEPT_COLUMNS``;
        extra columns ignored). Vectorised in Arrow (vocabulary / code / regex masks, RE2 name prefilter), so only the few rows
        that can match reach Python. (Restricting to the concept ids that occur in the cohort's fact tables was measured and
        rejected: scanning the id columns of the cached fact tables costs more than classifying the whole vocabulary.)"""
        import pyarrow as pa
        import pyarrow.compute as pc
        if df is None or len(df) == 0:
            return
        t = self._as_arrow(df)
        self.n_ingested += t.num_rows
        if t.num_rows == 0:
            return
        vocab, domain = t.column("vocabulary_id"), t.column("domain_id")
        ucum = t.filter(pc.equal(vocab, "UCUM"))
        for cid, code in zip(ucum.column("concept_id").to_pylist(), ucum.column("concept_code").to_pylist()):
            self.unit_code[int(cid)] = code
        mv = t.filter(pc.equal(domain, "Meas Value"))
        for cid, nm in zip(mv.column("concept_id").to_pylist(), mv.column("concept_name").to_pylist()):
            self.value_name[int(cid)] = nm
        d = t.filter(pc.is_in(vocab, value_set=pa.array(sorted(self._vocabs), type=pa.string())))
        if d.num_rows == 0:
            return
        self._prepare()
        prep, cm = self._prep, self.cm
        vocab, code = d.column("vocabulary_id"), d.column("concept_code")
        stripped = pc.utf8_trim_whitespace(code)
        is_icd = pc.is_in(vocab, value_set=pa.array(ICD_SYSTEMS, type=pa.string()))
        is_up = pc.is_in(vocab, value_set=pa.array(["CPT4", "HCPCS"], type=pa.string()))
        norm = pc.if_else(is_icd, pc.utf8_upper(pc.replace_substring_regex(stripped, r"[^A-Za-z0-9]", "")),
                          pc.if_else(is_up, pc.utf8_upper(stripped), stripped))
        mask = pa.array([False] * d.num_rows)
        for v, codes in prep["exact"].items():
            mask = pc.or_(mask, pc.and_(pc.equal(vocab, v), pc.is_in(norm, value_set=pa.array(codes, type=pa.string()))))
        for s, p, _, _ in prep["prefix"]:
            mask = pc.or_(mask, pc.and_(pc.equal(vocab, s), pc.starts_with(norm, pattern=p)))
        if prep["rxcodes"]:
            mask = pc.or_(mask, pc.and_(pc.equal(vocab, "RxNorm"),
                                        pc.is_in(code, value_set=pa.array(prep["rxcodes"], type=pa.string()))))
        mask = pc.fill_null(mask, False)
        sel_id = d.column("concept_id").filter(mask).to_pylist()
        sel_v = vocab.filter(mask).to_pylist()
        sel_n = norm.filter(mask).to_pylist()
        for cid, v, c in zip(sel_id, sel_v, sel_n):
            cid = int(cid)
            if (v, c) in cm.meas_code:
                self.meas.setdefault(cid, []).extend(cm.meas_code[(v, c)])
            ch = cm._code_hits([(v, c)], cm.cond_exact, cm.cond_prefix)
            if ch:
                self.cond[cid] = tuple(dict.fromkeys(self.cond.get(cid, ()) + ch))
            ph = cm._code_hits([(v, c)], cm.proc_exact, cm.proc_prefix)
            if ph:
                self.proc[cid] = tuple(dict.fromkeys(self.proc.get(cid, ()) + ph))
        # drugs: RxNorm ingredient codes + concept-name regex over RxNorm / RxNorm Extension drug concepts
        rx = d.filter(pc.is_in(vocab, value_set=pa.array(RXNORM_VOCABS, type=pa.string())))
        if rx.num_rows == 0:
            return
        rx_cid, rx_code, rx_voc = (rx.column(c).to_pylist() for c in ("concept_id", "concept_code", "vocabulary_id"))
        rxm = pc.fill_null(pc.and_(pc.equal(rx.column("vocabulary_id"), "RxNorm"),
                                   pc.is_in(rx.column("concept_code"),
                                            value_set=pa.array(prep["rxcodes"], type=pa.string()))), False)
        for i in np.flatnonzero(rxm.to_numpy(zero_copy_only=False)):
            c = int(rx_cid[i])
            self.drug[c] = tuple(dict.fromkeys(self.drug.get(c, ()) + tuple(cm.rxnorm_code[rx_code[i]])))
        if prep["any_rx_re2"] is None:
            return
        names = rx.column("concept_name")
        low = pc.utf8_lower(names)
        try:           # RE2 prefilter (a superset of what Python's re accepts for these patterns); exact check follows
            pre = pc.fill_null(pc.match_substring_regex(low, prep["any_rx_re2"], ignore_case=True), True)
            cand = np.flatnonzero(pre.to_numpy(zero_copy_only=False))
        except pa.ArrowInvalid:
            cand = np.arange(rx.num_rows)
        low_c = low.take(pa.array(cand))
        low_l = low_c.to_pylist()
        surv = [(int(c), nm) for c, nm in zip(cand, low_l)]
        for g, ing, rxp, ex, _ in cm.drugs:           # per ingredient: RE2 prefilter, then the exact Python check on its hits
            try:
                hit = np.flatnonzero(pc.fill_null(pc.match_substring_regex(low_c, rxp.pattern, ignore_case=True),
                                                  True).to_numpy(zero_copy_only=False))
            except pa.ArrowInvalid:
                hit = range(len(surv))
            for j in hit:
                i, nm = surv[j]
                if rxp.search(nm) and not (ex is not None and ex.search(nm)):
                    c = int(rx_cid[i])
                    cur = self.drug.get(c, ())
                    if (g, ing) not in cur:
                        self.drug[c] = cur + ((g, ing),)

    # ------------------------------------------------------------------------------------------ lookups
    def n_resolved(self) -> dict[str, int]:
        """Counts only (aggregate): resolved concept ids per domain, for the diagnostics."""
        return {"measurement": len(self.meas), "condition": len(self.cond), "procedure": len(self.proc),
                "drug": len(self.drug), "unit": len(self.unit_code), "meas_value": len(self.value_name)}

    def unit_of(self, unit_concept_id) -> str:
        try:
            return self.unit_code.get(int(unit_concept_id), "")
        except (TypeError, ValueError):
            return ""


# ===================================================================================== status / validation
def validate_concept_map(cm: ConceptMap | None = None, anchor_cfg: dict | None = None) -> list[str]:
    """Problems linking ``anchor_concepts.yaml`` and ``silver_anchors.yaml`` (empty list = consistent).

    Every anchor item must be mapped, derived, or recorded as unavailable; every LOINC listed in the anchor config must
    appear in the concept map; canonical units must agree."""
    from .anchors import load_anchor_config
    cm = cm or ConceptMap()
    cfg = anchor_cfg or load_anchor_config()
    r = cm.raw
    covered = (set(r["measurement_items"]) | set(r["qualitative_items"]) | set(r["code_event_items"])
               | set(r["derived_items"]) | set(r["unavailable_items"]))
    problems = []
    for item, spec in cfg["items"].items():
        if item not in covered:
            problems.append(f"anchor item {item!r} has no entry in anchor_concepts.yaml")
            continue
        sp = cm.specs.get(item)
        if sp is not None:
            have = {c for (_, c, _) in sp.codes}
            for loinc in spec.get("loinc", []):
                if loinc not in have:
                    problems.append(f"{item}: LOINC {loinc} of silver_anchors.yaml missing from the concept map")
            if sp.kind == "quant" and spec.get("unit") and cm.unit_token(spec["unit"]) != sp.unit:
                problems.append(f"{item}: unit {sp.unit_label!r} differs from silver_anchors.yaml {spec['unit']!r}")
    for item in r["unavailable_items"]:
        if item not in cfg["items"]:
            problems.append(f"unavailable item {item!r} is not an anchor item")
    return problems


def anchor_item_status(cm: ConceptMap | None = None, anchor_cfg: dict | None = None) -> dict[str, str]:
    """item -> ``full`` | ``proxy`` | ``unavailable`` (structured-data computability)."""
    from .anchors import load_anchor_config
    cm = cm or ConceptMap()
    cfg = anchor_cfg or load_anchor_config()
    proxy = set(cm.raw.get("item_status", {}).get("proxy", []))
    out = {}
    for item in cfg["items"]:
        if item in cm.raw["unavailable_items"]:
            out[item] = STATUS_UNAVAILABLE
        elif item in proxy:
            out[item] = STATUS_PROXY
        else:
            out[item] = STATUS_FULL
    return out


def anchor_leaf_status(cm: ConceptMap | None = None, anchor_cfg: dict | None = None) -> dict[str, str]:
    """anchor leaf id (e.g. ``E1_dx_ich``) -> status of its item."""
    from .anchors import load_anchor_config
    cfg = anchor_cfg or load_anchor_config()
    st = anchor_item_status(cm, cfg)
    out: dict[str, str] = {}

    def walk(node):
        for key in ("any_of", "all_of"):
            for ch in node.get(key, []):
                walk(ch)
        if "item" in node:
            out[node["id"]] = st[node["item"]]
    for lab in cfg["labels"].values():
        walk(lab)
    return out


def label_status(cm: ConceptMap | None = None, anchor_cfg: dict | None = None) -> dict[str, str]:
    """label -> ``full`` | ``partial`` | ``unavailable``.

    ``unavailable`` when the rule tree cannot be satisfied from structured data (any_of: all children unavailable;
    all_of: some child unavailable); ``partial`` when it can, but some anchor is a proxy or unavailable."""
    from .anchors import load_anchor_config
    cfg = anchor_cfg or load_anchor_config()
    st = anchor_item_status(cm, cfg)

    def sat(node) -> bool:                   # satisfiable from structured data?
        if "any_of" in node:
            return any(sat(c) for c in node["any_of"])
        if "all_of" in node:
            return all(sat(c) for c in node["all_of"])
        return st[node["item"]] != STATUS_UNAVAILABLE

    def leaves(node):
        for key in ("any_of", "all_of"):
            for c in node.get(key, []):
                yield from leaves(c)
        if "item" in node:
            yield node["item"]
    out = {}
    for lab, node in cfg["labels"].items():
        if not sat(node):
            out[lab] = "unavailable"
        elif all(st[i] == STATUS_FULL for i in leaves(node)):
            out[lab] = "full"
        else:
            out[lab] = "partial"
    return out
