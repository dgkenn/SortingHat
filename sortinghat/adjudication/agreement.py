"""Agreement and silver-vs-gold statistics for the gold adjudication pilot (D-157).  Aggregate-only outputs.

Definitions
  binary call   positive = probable or definite; absent and possible are negative; unassessable is excluded from that label's statistic
  inter-rater   Cohen's kappa (binary) per label with a stratified case bootstrap CI, prevalence- and bias-adjusted kappa (PABAK =
                2 po - 1), the prevalence and bias indices, linear-weighted kappa on the 4 ordinal states, Fleiss' kappa for >= 2 raters,
                categorical kappa for the primary etiology; also design-weighted (Horvitz-Thompson sampling weights) versions, because the
                sample is enriched for rare silver labels and an unweighted kappa describes the enriched sample, not IU-large
  intra-rater   the same statistics between a rater's two ratings of the same patient (original vs duplicate packet)
  consensus     R1/R2 agree -> that call; disagree -> the third rater; fewer than two valid calls -> not assessable
  silver vs gold  sensitivity, specificity, PPV, NPV of the silver label against the consensus gold call, re-weighted to IU-large
Disclosure: every count < k (default 11) is shown "<11"; a statistic whose underlying cells are below k is suppressed; no case id, rater
free text or date is emitted.  The CIs are percentile bootstrap intervals resampling cases within sampling strata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..labels import pilot
from ..labels.ontology import primary_endpoint_labels
from ..safe_output import SUPPRESSED, SUPPRESS_BELOW, suppress_count

ORD = {"absent": 0, "possible": 1, "probable": 2, "definite": 3}
POSITIVE = {"probable", "definite"}
MAIN_LABELS = ("E1", "E2", "E3", "E4a", "E4b", "E5", "E6", "E7", "other")
NaN = float("nan")


# ====================================================================================================== panel
class Panel:
    """Ratings as arrays aligned to ``cases``: ``bin[label][rater]`` (1 / 0 / nan) and ``ordn[label][rater]`` (0..3 / nan)."""

    def __init__(self, long: pd.DataFrame, labels, raters):
        self.labels, self.raters = list(labels), list(raters)
        sub = long[long["rater"].isin(self.raters) & long["label"].isin(self.labels)]
        self.cases = sorted(sub["case_id"].astype(str).unique())
        pos = {c: i for i, c in enumerate(self.cases)}
        self.bin = {l: {r: np.full(len(self.cases), NaN) for r in self.raters} for l in self.labels}
        self.ordn = {l: {r: np.full(len(self.cases), NaN) for r in self.raters} for l in self.labels}
        for row in sub.itertuples(index=False):
            i = pos[str(row.case_id)]
            s = str(row.state).lower()
            if s in ORD:
                self.bin[row.label][row.rater][i] = float(s in POSITIVE)
                self.ordn[row.label][row.rater][i] = float(ORD[s])

    def index(self) -> dict:
        return {c: i for i, c in enumerate(self.cases)}


# ====================================================================================================== kappa kernels
def _kappa_cells(n00, n01, n10, n11):
    """Vectorised binary kappa and friends from the 2x2 cells (first index = rater A, second = rater B); works on scalars or arrays."""
    n00, n01, n10, n11 = (np.asarray(x, float) for x in (n00, n01, n10, n11))
    n = n00 + n01 + n10 + n11
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (n00 + n11) / n
        pa, pb = (n10 + n11) / n, (n01 + n11) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = np.where(pe < 1.0, (po - pe) / (1 - pe), np.nan)
        pabak = 2 * po - 1
        return {"n": n, "po": po, "kappa": kappa, "pabak": pabak, "prevalence_index": (n00 - n11) / n,
                "bias_index": (n01 - n10) / n, "pos_a": pa, "pos_b": pb}


def _cells(a, b, w):
    ok = ~(np.isnan(a) | np.isnan(b))
    a1, b1 = a == 1, b == 1
    return [float(np.sum(w[ok & ~a1 & ~b1])), float(np.sum(w[ok & ~a1 & b1])), float(np.sum(w[ok & a1 & ~b1])), float(np.sum(w[ok & a1 & b1]))]


def _cells_boot(a, b, w, idx):
    A, Bm, W = a[idx], b[idx], w[idx]
    ok = ~(np.isnan(A) | np.isnan(Bm))
    a1, b1 = A == 1, Bm == 1
    return tuple((W * (ok & m)).sum(axis=1) for m in (~a1 & ~b1, ~a1 & b1, a1 & ~b1, a1 & b1))


def _wkappa_table(T: np.ndarray, kind: str = "linear") -> float:
    tot = T.sum()
    if tot <= 0:
        return NaN
    obs = T / tot
    exp = np.outer(obs.sum(1), obs.sum(0))
    k = T.shape[0]
    i, j = np.indices((k, k))
    w = np.abs(i - j) / max(k - 1, 1) if kind == "linear" else ((i - j) / max(k - 1, 1)) ** 2
    den = (w * exp).sum()
    return NaN if den == 0 else float(1 - (w * obs).sum() / den)


def _ord_table(a, b, w, k=4):
    ok = ~(np.isnan(a) | np.isnan(b))
    return np.bincount((a[ok].astype(int) * k + b[ok].astype(int)), weights=w[ok], minlength=k * k).reshape(k, k)


def fleiss_kappa(M: np.ndarray, categories) -> float:
    """Fleiss' kappa for an n x m matrix of category codes (rows with a missing value are dropped)."""
    M = np.asarray(M, float)
    M = M[~np.isnan(M).any(axis=1)]
    n, m = M.shape
    if n == 0 or m < 2:
        return NaN
    cnt = np.stack([(M == c).sum(axis=1) for c in categories], axis=1).astype(float)
    Pi = ((cnt ** 2).sum(axis=1) - m) / (m * (m - 1))
    pj = cnt.sum(axis=0) / (n * m)
    pe = float((pj ** 2).sum())
    return NaN if pe >= 1 else float((Pi.mean() - pe) / (1 - pe))


def categorical_kappa(a, b, k: int) -> float:
    ok = (a >= 0) & (b >= 0)
    if not ok.any():
        return NaN
    T = np.bincount(a[ok] * k + b[ok], minlength=k * k).reshape(k, k).astype(float)
    tot = T.sum()
    obs = T / tot
    pe = float((obs.sum(1) * obs.sum(0)).sum())
    return NaN if pe >= 1 else float((np.trace(obs) - pe) / (1 - pe))


# ====================================================================================================== bootstrap
def boot_indices(n: int, strata: np.ndarray | None, B: int, seed: int) -> np.ndarray:
    """(B, n) resampled case indices, resampling WITHIN strata (the sampling design's strata) so each replicate keeps the stratum sizes."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.integers(0, n, size=(B, n))
    s = np.asarray(strata).astype(str)
    blocks = []
    for u in np.unique(s):
        g = np.flatnonzero(s == u)
        blocks.append(g[rng.integers(0, len(g), size=(B, len(g)))])
    return np.hstack(blocks) if blocks else np.zeros((B, 0), int)


def _ci(vals, level: float = 0.95) -> list | None:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 0.9 * max(len(vals), 1) or len(v) < 20:
        return None
    lo, hi = np.percentile(v, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [round(float(lo), 3), round(float(hi), 3)]


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ====================================================================================================== inter / intra rater
def binary_agreement(a, b, w=None, *, idx=None, strata=None, B=2000, seed=0, k=SUPPRESS_BELOW, boot=True) -> dict:
    """Agreement of two binary call vectors (nan = excluded).  Unweighted unless ``w`` (sampling weights) is given."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    n = int(ok.sum())
    n_pos = int((ok & ((a == 1) | (b == 1))).sum())
    out = {"n_pairs": suppress_count(n, k), "n_positive_either": suppress_count(n_pos, k)}
    if n < k or n_pos < k or (n - n_pos) < k:
        return {**out, "kappa": SUPPRESSED, "kappa_ci95": SUPPRESSED, "pabak": SUPPRESSED, "percent_agreement": SUPPRESSED,
                "prevalence_index": SUPPRESSED, "bias_index": SUPPRESSED, "prevalence_rater_a": SUPPRESSED, "prevalence_rater_b": SUPPRESSED}
    ww = np.ones(len(a)) if w is None else np.asarray(w, float)
    c = _kappa_cells(*_cells(a, b, ww))
    out.update({"kappa": _r(c["kappa"]), "pabak": _r(c["pabak"]), "percent_agreement": _r(c["po"]),
                "prevalence_index": _r(c["prevalence_index"]), "bias_index": _r(c["bias_index"]),
                "prevalence_rater_a": _r(c["pos_a"]), "prevalence_rater_b": _r(c["pos_b"])})
    if boot:
        ix = idx if idx is not None else boot_indices(len(a), strata, B, seed)
        cb = _kappa_cells(*_cells_boot(a, b, ww, ix))
        out["kappa_ci95"] = _ci(cb["kappa"]) or SUPPRESSED
        out["pabak_ci95"] = _ci(cb["pabak"]) or SUPPRESSED
    return out


def ordinal_agreement(a, b, w=None, *, idx=None, strata=None, B=2000, seed=0, k=SUPPRESS_BELOW, kind="linear") -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    n = int(ok.sum())
    if n < k:
        return {"n_pairs": SUPPRESSED, f"kappa_{kind}_weighted": SUPPRESSED, "ci95": SUPPRESSED}
    ww = np.ones(len(a)) if w is None else np.asarray(w, float)
    pt = _wkappa_table(_ord_table(a, b, ww), kind)
    ix = idx if idx is not None else boot_indices(len(a), strata, B, seed)
    reps = [_wkappa_table(_ord_table(a[i], b[i], ww[i]), kind) for i in ix]
    return {"n_pairs": suppress_count(n, k), f"kappa_{kind}_weighted": _r(pt), "ci95": _ci(reps) or SUPPRESSED}


def interrater(panel: Panel, ra: str, rb: str, *, weights=None, strata=None, B=2000, seed=0, k=SUPPRESS_BELOW) -> dict:
    ix = boot_indices(len(panel.cases), strata, B, seed)
    out = {}
    for lab in panel.labels:
        out[lab] = {"binary": binary_agreement(panel.bin[lab][ra], panel.bin[lab][rb], weights, idx=ix, k=k),
                    "ordinal": ordinal_agreement(panel.ordn[lab][ra], panel.ordn[lab][rb], weights, idx=ix, k=k)}
    return out


def fleiss_report(panel: Panel, raters, *, strata=None, B=1000, seed=0, k=SUPPRESS_BELOW) -> dict:
    ix = boot_indices(len(panel.cases), strata, B, seed)
    out = {}
    for lab in panel.labels:
        M = np.column_stack([panel.bin[lab][r] for r in raters])
        n = int((~np.isnan(M).any(axis=1)).sum())
        npos = int(((~np.isnan(M).any(axis=1)) & (np.nansum(M, axis=1) > 0)).sum())
        if n < k or npos < k or (n - npos) < k:
            out[lab] = {"n_cases": suppress_count(n, k), "fleiss_kappa": SUPPRESSED, "ci95": SUPPRESSED}
            continue
        reps = [fleiss_kappa(M[i], (0, 1)) for i in ix]
        out[lab] = {"n_cases": suppress_count(n, k), "fleiss_kappa": _r(fleiss_kappa(M, (0, 1))), "ci95": _ci(reps) or SUPPRESSED}
    return out


def duplicate_pairs(long: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """(original case, duplicate case) pairs from the id map (columns case_id, role, duplicate_of_case)."""
    d = id_map[id_map["role"] == "duplicate"][["case_id", "duplicate_of_case"]].rename(columns={"case_id": "dup", "duplicate_of_case": "orig"})
    return d.reset_index(drop=True)


def intrarater(long: pd.DataFrame, pairs: pd.DataFrame, labels, raters, *, seed=0, B=2000, k=SUPPRESS_BELOW) -> dict:
    """Kappa between each rater's rating of the original and of its duplicate packet: pooled over raters and per rater."""
    out = {"pooled": {}, "by_rater": {}}
    sub = long[long["label"].isin(labels)]
    key = {(str(r.case_id), r.rater, r.label): str(r.state).lower() for r in sub.itertuples(index=False)}

    def vecs(rs, lab):
        a, b = [], []
        for r in rs:
            for o, d in zip(pairs["orig"], pairs["dup"]):
                so, sd = key.get((o, r, lab)), key.get((d, r, lab))
                a.append(NaN if so not in ORD else float(so in POSITIVE))
                b.append(NaN if sd not in ORD else float(sd in POSITIVE))
        return np.array(a), np.array(b)
    for lab in labels:
        a, b = vecs(raters, lab)
        out["pooled"][lab] = binary_agreement(a, b, None, B=B, seed=seed, k=k)
    for i, r in enumerate(raters):
        out["by_rater"][f"rater_{i + 1}"] = {}
        for lab in labels:
            a, b = vecs([r], lab)
            out["by_rater"][f"rater_{i + 1}"][lab] = binary_agreement(a, b, None, B=B, seed=seed, k=k)
    return out


def primary_agreement(cf: pd.DataFrame, ra: str, rb: str, categories, *, strata_by_case: dict | None = None, B=2000, seed=0,
                      k=SUPPRESS_BELOW) -> dict:
    """Categorical kappa for the single primary etiology (all categories), with a stratified case bootstrap."""
    cats = {c: i for i, c in enumerate(categories)}
    w = cf.pivot_table(index="case_id", columns="rater", values="primary_etiology", aggfunc="first")
    if ra not in w or rb not in w:
        return {"n_pairs": SUPPRESSED, "kappa": SUPPRESSED}
    w = w.dropna(subset=[ra, rb])
    a = w[ra].map(lambda v: cats.get(str(v), -1)).to_numpy(int)
    b = w[rb].map(lambda v: cats.get(str(v), -1)).to_numpy(int)
    keep = (a >= 0) & (b >= 0)
    a, b, ids = a[keep], b[keep], w.index.to_numpy()[keep]
    n = len(a)
    if n < k:
        return {"n_pairs": SUPPRESSED, "kappa": SUPPRESSED, "percent_agreement": SUPPRESSED}
    strata = np.array([strata_by_case.get(str(c), "") for c in ids]) if strata_by_case else None
    ix = boot_indices(n, strata, B, seed)
    reps = [categorical_kappa(a[i], b[i], len(cats)) for i in ix]
    return {"n_pairs": suppress_count(n, k), "kappa": _r(categorical_kappa(a, b, len(cats))), "ci95": _ci(reps) or SUPPRESSED,
            "percent_agreement": _r(float((a == b).mean())), "categories": list(categories)}


# ====================================================================================================== consensus gold
def consensus(panel_all: Panel, main=("R1", "R2"), third="R3") -> dict:
    """{label: (gold array 1/0/nan, source array of 'agree' | 'third' | 'majority' | 'pending' | 'missing')} over ``panel_all.cases``."""
    out = {}
    ra, rb = main
    for lab in panel_all.labels:
        a, b = panel_all.bin[lab][ra], panel_all.bin[lab][rb]
        c = panel_all.bin[lab].get(third, np.full(len(a), NaN))
        gold = np.full(len(a), NaN)
        src = np.full(len(a), "missing", dtype=object)
        va, vb, vc = ~np.isnan(a), ~np.isnan(b), ~np.isnan(c)
        agree = va & vb & (a == b)
        gold[agree], src[agree] = a[agree], "agree"
        dis = va & vb & (a != b)
        use3 = dis & vc
        gold[use3], src[use3] = c[use3], "third"
        src[dis & ~vc] = "pending"
        rest = ~(va & vb)
        for i in np.flatnonzero(rest):
            vals = [x for x in (a[i], b[i], c[i]) if not np.isnan(x)]
            if len(vals) >= 2:
                s = sum(vals)
                if s * 2 != len(vals):
                    gold[i], src[i] = float(s * 2 > len(vals)), "majority"
        out[lab] = (gold, src)
    return out


def disagreement_queue(panel: Panel, main=("R1", "R2")) -> pd.DataFrame:
    """RECORD-LEVEL (local_only): cases with a binary disagreement between the main raters, with the labels in dispute."""
    ra, rb = main
    rows = []
    for lab in panel.labels:
        a, b = panel.bin[lab][ra], panel.bin[lab][rb]
        for i in np.flatnonzero(~np.isnan(a) & ~np.isnan(b) & (a != b)):
            rows.append({"case_id": panel.cases[i], "label": lab})
    d = pd.DataFrame(rows, columns=["case_id", "label"])
    if len(d):
        d = d.groupby("case_id")["label"].apply(lambda s: ";".join(s)).reset_index().rename(columns={"label": "labels_in_dispute"})
    return d


# ====================================================================================================== silver vs gold
def prevalence_block(gold: np.ndarray, w: np.ndarray, strata, *, B=2000, seed=0, k=SUPPRESS_BELOW) -> dict:
    ok = ~np.isnan(gold)
    n = int(ok.sum())
    npos = int((gold[ok] == 1).sum())
    if n < k or npos < k or (n - npos) < k:
        return {"n_gold_assessable": suppress_count(n, k), "n_gold_positive": suppress_count(npos, k), "prevalence_sample": SUPPRESSED,
                "prevalence_weighted": SUPPRESSED, "prevalence_weighted_ci95": SUPPRESSED}
    ix = boot_indices(len(gold), strata, B, seed)
    G, W = gold[ix], w[ix]
    okb = ~np.isnan(G)
    with np.errstate(divide="ignore", invalid="ignore"):
        reps = (W * (G == 1) * okb).sum(axis=1) / (W * okb).sum(axis=1)
    return {"n_gold_assessable": n, "n_gold_positive": npos, "prevalence_sample": _r(npos / n),
            "prevalence_weighted": _r(float(np.sum(w[ok] * (gold[ok] == 1)) / np.sum(w[ok]))), "prevalence_weighted_ci95": _ci(reps) or SUPPRESSED}


def _confusion(g, s, w):
    ok = ~(np.isnan(g) | np.isnan(s))
    tp = float(w[ok & (g == 1) & (s == 1)].sum()); fn = float(w[ok & (g == 1) & (s == 0)].sum())
    fp = float(w[ok & (g == 0) & (s == 1)].sum()); tn = float(w[ok & (g == 0) & (s == 0)].sum())
    return tp, fp, fn, tn


def _metrics(tp, fp, fn, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        tp, fp, fn, tn = (np.asarray(x, float) for x in (tp, fp, fn, tn))
        return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "ppv": tp / (tp + fp), "npv": tn / (tn + fn)}


def silver_vs_gold(gold: np.ndarray, silver: np.ndarray, w: np.ndarray, strata, *, B=2000, seed=0, k=SUPPRESS_BELOW) -> dict:
    """Sensitivity / specificity / PPV / NPV of the silver label against the consensus gold call.  Weighted point estimates (w =
    sampling weights, so they describe IU-large) with stratified bootstrap CIs, plus unweighted ones.  Each metric is shown only if both
    cells of its unweighted proportion are >= k."""
    ok = ~(np.isnan(gold) | np.isnan(silver))
    cell = {"tp": int((ok & (gold == 1) & (silver == 1)).sum()), "fn": int((ok & (gold == 1) & (silver == 0)).sum()),
            "fp": int((ok & (gold == 0) & (silver == 1)).sum()), "tn": int((ok & (gold == 0) & (silver == 0)).sum())}
    out = {"n_pairs": suppress_count(int(ok.sum()), k), "n_gold_positive": suppress_count(cell["tp"] + cell["fn"], k),
           "n_silver_positive": suppress_count(cell["tp"] + cell["fp"], k),
           "n_silver_not_assessable": suppress_count(int((~np.isnan(gold) & np.isnan(silver)).sum()), k)}
    pairs = {"sensitivity": (cell["tp"], cell["tp"] + cell["fn"]), "specificity": (cell["tn"], cell["tn"] + cell["fp"]),
             "ppv": (cell["tp"], cell["tp"] + cell["fp"]), "npv": (cell["tn"], cell["tn"] + cell["fn"])}
    ix = boot_indices(len(gold), strata, B, seed)
    G, S, W = gold[ix], silver[ix], w[ix]
    okb = ~(np.isnan(G) | np.isnan(S))
    bm = _metrics(*(((W * (okb & m)).sum(axis=1)) for m in ((G == 1) & (S == 1), (G == 0) & (S == 1), (G == 1) & (S == 0), (G == 0) & (S == 0))))
    pw = _metrics(*_confusion(gold, silver, w))
    pu = _metrics(*_confusion(gold, silver, np.ones(len(gold))))
    for m, (num, den) in pairs.items():
        if den < k or num < k or (den - num) < k:
            out[m] = {"weighted": SUPPRESSED, "weighted_ci95": SUPPRESSED, "unweighted": SUPPRESSED}
        else:
            out[m] = {"weighted": _r(float(pw[m])), "weighted_ci95": _ci(bm[m]) or SUPPRESSED, "unweighted": _r(float(pu[m]))}
    return out


# ====================================================================================================== quality checks
def consistency(long: pd.DataFrame, cf: pd.DataFrame, k=SUPPRESS_BELOW) -> dict:
    """Counts of internally inconsistent forms (primary etiology not rated probable/definite; 'unknown' with a positive label)."""
    st = long.assign(s=long["state"].str.lower()).pivot_table(index=["case_id", "rater"], columns="label", values="s", aggfunc="first")
    c = cf.set_index(["case_id", "rater"])
    j = st.join(c[["primary_etiology"]], how="inner") if "primary_etiology" in c else st.assign(primary_etiology=np.nan)
    n = len(j)
    bad_primary = unk_pos = 0
    for _, r in j.iterrows():
        p = r.get("primary_etiology")
        pos = [lab for lab in st.columns if str(r.get(lab)).lower() in POSITIVE]
        if p == "unknown":
            unk_pos += int(bool(pos))
        elif isinstance(p, str) and p in st.columns and str(r.get(p)).lower() not in POSITIVE:
            bad_primary += 1
    return {"n_forms_checked": suppress_count(n, k), "primary_etiology_not_positive": suppress_count(bad_primary, k),
            "unknown_primary_with_positive_label": suppress_count(unk_pos, k)}


def blinding_check(cf: pd.DataFrame, k=SUPPRESS_BELOW) -> dict:
    """Forms in which a rater reported EEG information in the packet (eeg_seen 1 minor / 2 major)."""
    if "eeg_seen" not in cf:
        return {"available": False}
    v = pd.to_numeric(cf["eeg_seen"], errors="coerce")
    n = int(v.notna().sum())
    return {"available": True, "n_forms": suppress_count(n, k), "minor": suppress_count(int((v == 1).sum()), k),
            "major": suppress_count(int((v == 2).sum()), k)}


def gate0(kappas_binary: dict, prevalence: dict, labels=None) -> dict:
    """Gate 0 (kappa >= 0.6 for >= 4 primary families and >= 3 families with >= 10% prevalence) from this module's numbers.
    ``kappas_binary[label]`` = kappa or the suppressed marker; ``prevalence[label]`` likewise.  Reuses the pilot's rule."""
    labels = list(labels or primary_endpoint_labels())
    ks = {l: {"kappa_weighted_linear": kappas_binary.get(l), "prevalence": prevalence.get(l)} for l in labels}
    return pilot.gate0_pilot_check(ks, labels)
