"""Synthetic Study 1 modelling data with a tunable planted EEG signal (tests and dry runs; no real data).

``signal`` scales how strongly selected qeeg./conn./morgoth./emb. columns shift with the TRUE label beyond what the
baseline columns carry; ``signal=0`` gives pure-noise EEG features (expected Delta ~ 0). ``site_shift`` adds
site-specific offsets to a few label-irrelevant EEG columns (to exercise the site leakage probe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..labels.ontology import ALL_LABELS
from .data import ModelData
from .inputs import embedding_frame

_PREV = {"E1": 0.25, "E2": 0.15, "E3": 0.15, "E4a": 0.12, "E4b": 0.20, "E5": 0.30, "E6": 0.25, "E7": 0.08}
SIGNAL_LABELS = ("E1", "E2", "E5", "E6", "E3")


def make_synthetic_study(n_per_site: int = 500, sites=("SITE_A", "SITE_B", "SITE_C"), signal: float = 1.0,
                         site_shift: float = 0.0, seed: int = 0, n_noise: int = 10, silver_flip: float = 0.08,
                         dev_frac: float = 0.25, eval_frac: float = 0.5, unassessable: float = 0.05,
                         channel_variation: bool = False, embeddings: bool = True,
                         morgoth: bool = True) -> ModelData:
    rng = np.random.default_rng(seed)
    labels = list(ALL_LABELS)
    K = len(labels)
    S = len(sites)
    n = S * n_per_site
    site_arr = np.repeat(np.array(sites), n_per_site)
    site_i = np.repeat(np.arange(S), n_per_site)

    # baseline: age, gcs, map, lactate (some missing)
    age = rng.normal(0, 1, n)
    gcs = rng.normal(0, 1, n)
    mapp = rng.normal(0, 1, n)
    lact = rng.normal(0, 1, n)
    B = np.column_stack([age, gcs, mapp, lact])
    W = rng.normal(0, 0.5, (4, K))
    logit0 = np.log(np.array([_PREV[k] for k in labels]) / (1 - np.array([_PREV[k] for k in labels])))
    p = 1 / (1 + np.exp(-(logit0 + B @ W)))
    y = (rng.random((n, K)) < p).astype(float)

    lact_obs = lact.copy()
    lact_obs[rng.random(n) < 0.08] = np.nan
    baseline = pd.DataFrame({"age": age, "gcs": gcs, "map": mapp, "lactate": lact_obs})

    # EEG: planted signal columns per label, plus noise
    cols = {}
    for k in SIGNAL_LABELS:
        j = labels.index(k)
        for q in range(3):
            cols[f"qeeg.{k}.sig{q}"] = signal * 0.9 * (y[:, j] - y[:, j].mean()) + rng.normal(0, 1, n)
        for q in range(2):
            cols[f"conn.coh.{k}.sig{q}"] = signal * 0.7 * (y[:, j] - y[:, j].mean()) + rng.normal(0, 1, n)
    for q in range(n_noise):
        c = rng.normal(0, 1, n)
        if q < 4:
            c = c + site_shift * (site_i - (S - 1) / 2)
        cols[f"qeeg.global.noise{q}"] = c
        cols[f"conn.coh.noise{q}"] = rng.normal(0, 1, n)
    eeg = pd.DataFrame(cols)
    sig_names = [c for c in cols if ".sig" in c]
    if morgoth:
        for i, k in enumerate(("E1", "E2", "E5")):
            j = labels.index(k)
            eeg[f"morgoth.{k}_burden"] = signal * 0.8 * (y[:, j] - y[:, j].mean()) + rng.normal(0, 1, n)
    if embeddings:
        proj = rng.normal(0, 1, (len(sig_names), 8))
        emb = eeg[sig_names].values @ proj / np.sqrt(len(sig_names)) + rng.normal(0, 1, (n, 8))
        eeg = pd.concat([eeg, embedding_frame(emb, "cbramod")], axis=1)

    # labels
    m_silver = rng.random((n, K)) > 0.05
    flip = rng.random((n, K)) < silver_flip
    y_silver = np.where(flip, 1 - y, y)
    m_gold = rng.random((n, K)) > unassessable
    role = np.empty(n, dtype=object)
    for s in range(S):
        ix = np.flatnonzero(site_i == s)
        r = rng.permutation(ix)
        nd, ne = int(dev_frac * len(ix)), int(eval_frac * len(ix))
        role[r[:nd]] = "dev"
        role[r[nd:nd + ne]] = "eval"
        role[r[nd + ne:]] = "none"
    sedated = y[:, labels.index("E4b")] > 0
    cov = {
        "duration_s": np.exp(rng.normal(np.log(1500), 0.4, n)),
        "n_channels": rng.choice([16, 19, 21], n) if channel_variation else np.full(n, 19),
        "severity": -gcs + rng.normal(0, 0.3, n),
        "sedated": sedated,
    }
    times = np.concatenate([rng.permutation(n_per_site).astype(float) for _ in range(S)])
    return ModelData(baseline, eeg, y_silver, m_silver, y, m_gold, role, site_arr, tuple(labels), times, cov)
