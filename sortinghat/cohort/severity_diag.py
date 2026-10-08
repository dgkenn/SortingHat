"""Per-site severity / score-vocabulary diagnostics for the post-first-EEG candidate set (aggregate only).

Run through ``scripts/diag_cohort.py --severity``. It builds the cohort (D-111 to D-115 rules), takes the set after the
"first qualifying EEG per patient" step, and streams ``measurement``, ``observation`` and ``condition_occurrence``
for those patients chunk by chunk (per-candidate boolean flags and capped string counters only; no frame is kept),
so the peak stays within the cohort build's bound. Per site it reports:

* ``n_candidates`` and the unmerged flow (every step, only counts < 11 suppressed);
* the share of candidates with ANY GCS / FOUR / RASS row, separately in ``measurement`` and ``observation`` and in
  either, at any time, within +-6 h of t0 and within [-6 h, +1 h] (the primary strict window);
* the top-20 source values matched by the score lexicon (vocabulary strings, with the lexicon key) and the top-20
  UNMATCHED source values containing gcs / glasgow / coma / four / rass / richmond (case-insensitive);
* the share with any phenotype code in the window, at any time, and with any condition row at all;
* quantiles of hours from the onset proxy to the EEG and the onset-proxy basis counts.

Strings are listed only when their count is >= 11 (digits masked); keys are never free text.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..baselines import lexicon as lx
from ..safe_output import SUPPRESS_BELOW, SUPPRESSED, assert_aggregate_only, safe_quantiles, suppress_count
from . import rules
from .build import build_cohort
from .config import CohortConfig
from .diag import QS, _count_share, _Counts, _safe_text
from .sources import StoreSources, remap_ids

SCORE_KEYS_ANY = {"gcs", "gcs_eye", "gcs_motor", "gcs_verbal", "four", "rass"}          # the Phase 0a row: GCS, FOUR, RASS
KEYWORDS = re.compile(r"gcs|glasgow|coma|four|rass|richmond", re.I)
_MEAS = ["person_id", "measurement_datetime", "measurement_date", "measurement_time", "measurement_source_value"]
_OBS = ["person_id", "observation_datetime", "observation_date", "observation_source_value"]
_COND = ["person_id", "condition_start_datetime", "condition_source_value"]


class _SiteStrings:
    """Capped per-site counters of matched (value -> lexicon key) and unmatched keyword strings."""

    def __init__(self, nsite: int, cap: int = 5000):
        self.matched = [dict() for _ in range(nsite)]
        self.unmatched = [dict() for _ in range(nsite)]
        self.cap = cap

    def add(self, site_ix: np.ndarray, names: pd.Series, keys: pd.Series) -> None:
        d = pd.DataFrame({"s": site_ix, "n": names.astype(str).to_numpy(), "k": keys.to_numpy(object)})
        is_score = d["k"].notna()
        for (s, n, k), c in d[is_score].groupby(["s", "n", "k"]).size().items():
            self._bump(self.matched[s], f"{k}\t{n}", int(c))
        un = d[~is_score & d["n"].str.contains(KEYWORDS)]
        for (s, n), c in un.groupby(["s", "n"]).size().items():
            self._bump(self.unmatched[s], n, int(c))

    def _bump(self, dct: dict, key: str, c: int) -> None:
        if key in dct or len(dct) < self.cap:
            dct[key] = dct.get(key, 0) + c

    @staticmethod
    def top(dct: dict, top: int = 20, keyed: bool = False) -> list[dict]:
        vc = pd.Series(dct, dtype="int64").sort_values(ascending=False)
        vc = vc[vc >= SUPPRESS_BELOW].iloc[:top]
        out = []
        for k, n in vc.items():
            if keyed:
                key, val = k.split("\t", 1)
                out.append({"lexicon_key": key, "value": _safe_text(val), "n": int(n)})
            else:
                out.append({"value": _safe_text(k), "n": int(n)})
        return out


def _times(raw: pd.DataFrame, dt: str, date: str, tod: str | None) -> pd.Series:
    t = rules._dt(raw, dt)
    if date in raw:
        d = pd.to_datetime(raw[date], errors="coerce").dt.normalize()
        if tod and tod in raw:
            d = d + pd.to_timedelta(raw[tod].astype("string"), errors="coerce")
            t = t.where(t.notna(), d.astype("datetime64[us]"))
    return t


def run_severity_diag(store, sites: list[str] | None = None, cfg: CohortConfig | None = None,
                      min_rows: int = 500_000) -> tuple[dict, set[str]]:
    cfg = cfg or CohortConfig()
    src = StoreSources(store, sites)
    res = build_cohort(src, cfg)
    cand = res.stages["first_eeg"].reset_index(drop=True)
    cand["person_id"] = cand["person_id"].astype("int64")
    onset = res.stages["onset"]
    site_list = sorted(cand["SiteID"].astype(str).unique())
    n = len(cand)
    order = np.argsort(cand["person_id"].to_numpy())
    pids = cand["person_id"].to_numpy()[order]
    site_ix_sorted = pd.Categorical(cand["SiteID"].astype(str), categories=site_list).codes.astype("int64")[order]
    t0 = cand["t0"].astype("datetime64[us]").to_numpy()[order]
    enc = cand["encounter_start"].astype("datetime64[us]").to_numpy()[order]
    mm, _ = src.merge_map()
    rev: dict[int, list[int]] = {}
    for old, new in mm.items():
        rev.setdefault(new, []).append(old)
    fetch_ids = sorted(set(pids.tolist()) | {o for p in pids.tolist() for o in rev.get(int(p), [])})
    known = {str(p) for p in fetch_ids}

    flags = {k: np.zeros(n, bool) for k in ("m_any", "m_pm6", "m_prim", "o_any", "o_pm6", "o_prim",
                                           "c_any", "c_pheno_any", "c_pheno_win")}
    strings = _SiteStrings(len(site_list))
    null_src = {"measurement": np.zeros(len(site_list), "int64"), "observation": np.zeros(len(site_list), "int64")}
    rows_seen = {"measurement": np.zeros(len(site_list), "int64"), "observation": np.zeros(len(site_list), "int64")}
    key_cache: dict[str, str | None] = {}

    def score_key(name) -> str | None:
        if name not in key_cache:
            r = lx.classify_measurement(name)
            key_cache[name] = r.key if r is not None and r.domain == "score" else None
        return key_cache[name]

    def locate(raw):
        rp = pd.to_numeric(raw["person_id"], errors="coerce")
        ok = rp.notna().to_numpy(bool)
        v = rp.fillna(-1).astype("int64").to_numpy()
        ix = np.minimum(np.searchsorted(pids, v), n - 1)
        hit = ok & (pids[ix] == v)
        return ix, hit

    for table, cols, text, dt, date, tod, tag in (
            ("omop_measurement", _MEAS, "measurement_source_value", "measurement_datetime", "measurement_date",
             "measurement_time", "m"),
            ("omop_observation", _OBS, "observation_source_value", "observation_datetime", "observation_date",
             None, "o")):
        kind = "measurement" if tag == "m" else "observation"
        for raw in src.iter_rows(table, cols, fetch_ids, min_rows=min_rows):
            raw = remap_ids(raw, mm)
            ix, hit = locate(raw)
            if not hit.any():
                continue
            raw, ix = raw[hit], ix[hit]
            names = raw[text].astype("string") if text in raw else pd.Series(pd.NA, index=raw.index, dtype="string")
            rows_seen[kind] += np.bincount(site_ix_sorted[ix], minlength=len(site_list))
            null_src[kind] += np.bincount(site_ix_sorted[ix][names.isna().to_numpy(bool)], minlength=len(site_list))
            keys = names.map(lambda x: score_key(x) if isinstance(x, str) else None)
            strings.add(site_ix_sorted[ix][names.notna().to_numpy(bool)], names.dropna(), keys[names.notna()])
            is_score = keys.isin(list(SCORE_KEYS_ANY)).to_numpy(bool)
            if not is_score.any():
                continue
            t = _times(raw, dt, date, tod)
            dh = ((t - pd.Series(t0[ix], index=raw.index)).dt.total_seconds() / 3600.0).to_numpy()
            si = ix[is_score]
            flags[f"{tag}_any"][si] = True
            has_t = ~np.isnan(dh[is_score])
            d = dh[is_score]
            flags[f"{tag}_pm6"][si[has_t & (np.abs(d) <= 6)]] = True
            flags[f"{tag}_prim"][si[has_t & (d >= -6) & (d <= 1)]] = True

    phen_t = lambda raw: rules.filter_phenotype_rows(raw)        # noqa: E731
    for raw in src.iter_rows("omop_condition_occurrence", _COND, fetch_ids, min_rows=min_rows):
        raw = remap_ids(raw, mm)
        ix, hit = locate(raw)
        if not hit.any():
            continue
        raw, ix = raw[hit], ix[hit]
        flags["c_any"][ix] = True
        pm = phen_t(raw)
        if len(pm):
            ix2, _ = locate(pm)
            t = rules._dt(pm, "condition_start_datetime").to_numpy()
            flags["c_pheno_any"][ix2] = True
            win = (t >= enc[ix2]) & (t <= t0[ix2] + np.timedelta64(int(cfg.phenotype_after_h * 3600), "s"))
            flags["c_pheno_win"][ix2[win & ~np.isnat(t)]] = True

    report: dict = {"note": "per-site aggregates for the post-first-EEG candidate set (D-111..D-115 rules); counts < 11 "
                            "shown as \"<11\"; strings listed only with n >= 11",
                    "sites": {}}
    dbg = res.debug["sites"]
    ons_site = onset["SiteID"].astype(str)
    for k, site in enumerate(site_list):
        m = site_ix_sorted == k
        den = int(m.sum())
        pick = lambda f: _count_share(int(flags[f][m].sum()), den)          # noqa: E731
        both = lambda a, b: _count_share(int((flags[a] | flags[b])[m].sum()), den)  # noqa: E731
        o = onset[ons_site == site]
        report["sites"][site] = {
            "n_candidates": suppress_count(den),
            "measurement": {"any_gcs_four_rass_row": pick("m_any"), "within_pm6h": pick("m_pm6"),
                            "within_minus6h_plus1h": pick("m_prim"),
                            "n_rows_for_candidates": suppress_count(int(rows_seen["measurement"][k])),
                            "n_rows_with_null_source_value": suppress_count(int(null_src["measurement"][k]))},
            "observation": {"any_gcs_four_rass_row": pick("o_any"), "within_pm6h": pick("o_pm6"),
                            "within_minus6h_plus1h": pick("o_prim"),
                            "n_rows_for_candidates": suppress_count(int(rows_seen["observation"][k])),
                            "n_rows_with_null_source_value": suppress_count(int(null_src["observation"][k]))},
            "either_table": {"any_gcs_four_rass_row": both("m_any", "o_any"), "within_pm6h": both("m_pm6", "o_pm6"),
                             "within_minus6h_plus1h": both("m_prim", "o_prim")},
            "score_vocabulary_matched_top20": _SiteStrings.top(strings.matched[k], keyed=True),
            "unmatched_keyword_source_values_top20": _SiteStrings.top(strings.unmatched[k]),
            "condition": {"any_condition_row": pick("c_any"), "phenotype_code_in_window": pick("c_pheno_win"),
                          "phenotype_code_any_time": pick("c_pheno_any")},
            "hours_onset_proxy_to_eeg": safe_quantiles(o["hours_since_onset"].dropna().to_numpy(float), qs=QS),
            "onset_proxy_basis_counts": {str(b): suppress_count(int(c)) for b, c in
                                         o["onset_basis"].fillna("none").value_counts().items()},
            "n_with_onset_proxy": suppress_count(int(o["hours_since_onset"].notna().sum())),
            "flow_steps": dbg.get(site, {}).get("rows", dbg.get(site))}
    assert_aggregate_only(report, known)
    return report, known
