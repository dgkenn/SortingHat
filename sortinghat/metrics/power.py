"""Simulation-based power of the Study 1 primary endpoint (H1/H2) at a given evaluation-set size.

The SAP (docs/prereg_study1_sap.md, section 14) sizes the ~1,000-case evaluation set against
calibration-slope precision and, for H1/H2, only a normal-theory approximation with a *placeholder*
SD of d_i (``sample_size.power_h1_rule``). This module simulates the endpoint itself:

    d_i   = loss_i(EEG-augmented) - loss_i(baseline)      masked multi-label log loss (``loss.py``)
    H1/H2 = CI for Delta = mean(d_i) lies below 0   AND   Delta-hat < 0 at every one of 3 held-out sites.

Data-generating model (all of it is an assumption; see docs/research/evaluation_sample_size.md)
- K binary labels, one-factor Gaussian-copula dependence (loading ``label_loading``), marginal
  prevalences ``prevalences``; site-level logit shift of every prevalence (``site_prev_sd``).
- Each model emits one calibrated score per label: s = delta * y + z with z ~ N(0, 1), so AUROC =
  Phi(delta / sqrt(2)) and the calibrated logit is logit(pi) + delta * s - delta^2 / 2 (equal-variance
  binormal). Baseline AUROC per label in ``base_auc``; the EEG-augmented model has AUROC
  base + ``gain`` on every label (``gain_weights`` can concentrate it). The two models' noise terms
  correlate at ``noise_corr`` (the EEG model contains the baseline's information), which is what makes
  the paired d_i much tighter than an unpaired comparison.
- Between-site heterogeneity: at site s the EEG model's *true* AUROC gain is gain + u_s,
  u_s ~ N(0, tau_gain^2) (AUROC units, shared by all labels), while the model's probabilities were
  built for the nominal gain (so a weak site also gets over-confident EEG predictions, the transport
  failure the per-site condition is meant to catch).
- Each label is assessable with probability ``assess_prob`` (independent); patients with no assessable
  label drop out of Delta.

Three confidence intervals are evaluated on every simulated evaluation set:
``within_site``  the SAP's decision interval (stratified patient bootstrap). Its large-sample form is used
                 for speed: Delta-hat +/- 1.96 * sqrt(sum_s n_s^2 v_s / n_s) / n; ``test_metrics_power``
                 checks it against ``bootstrap.paired_bootstrap_delta``.
``site_t``       equal-weight mean of the 3 site Deltas with a t(2) interval (``bootstrap.site_mean_t_interval``).
                 Fully cluster-aware, and very conservative at S = 3.
``two_stage``    resample sites, then patients within sites (``bootstrap`` mode two_stage); the patient stage is
                 approximated by N(site mean, site SE^2) so that it vectorises.

Run:  python -m sortinghat.metrics.power            (markdown tables to stdout, ~2-4 minutes)
      python -m sortinghat.metrics.power --reps 100 (quick look)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from scipy import stats

from .loss import binary_log_loss

Z = 1.959963984540054
T975_DF2 = float(stats.t.ppf(0.975, 2))

PRIMARY_LABELS_6 = ("E1", "E2", "E4a", "E5", "E6", "E7")
PREVALENCES_6 = (0.25, 0.15, 0.10, 0.25, 0.20, 0.03)
BASE_AUC_6 = (0.72, 0.75, 0.72, 0.70, 0.70, 0.73)
GAINS = (0.00, 0.02, 0.04, 0.06)
N_GRID = (500, 1000, 1500, 2000, 3000)
N_GRID_EXT = (500, 1000, 1500, 2000, 3000, 4000, 6000, 8000)
CI_MODES = ("within_site", "site_t", "two_stage")


@dataclass(frozen=True)
class PowerConfig:
    labels: tuple[str, ...] = PRIMARY_LABELS_6
    prevalences: tuple[float, ...] = PREVALENCES_6
    base_auc: tuple[float, ...] = BASE_AUC_6
    gain: float = 0.04                       # true AUROC gain of the EEG model, every label (x gain_weights)
    gain_weights: tuple[float, ...] | None = None  # per-label multipliers on ``gain``; None = all 1
    tau_gain: float = 0.015                  # SD of the site-level random effect on the gain (AUROC units)
    site_prev_sd: float = 0.25               # SD of the site logit shift of every prevalence
    label_loading: float = 0.5               # one-factor loading; liability correlation = loading^2
    noise_corr: float = 0.6                  # correlation of baseline and EEG score noise
    assess_prob: float = 0.90                # P(label assessable), per patient and label
    n_sites: int = 3

    def __post_init__(self):
        k = len(self.labels)
        if not (len(self.prevalences) == len(self.base_auc) == k):
            raise ValueError("labels, prevalences and base_auc must have the same length")
        if self.gain_weights is not None and len(self.gain_weights) != k:
            raise ValueError("gain_weights must have one entry per label")

    @property
    def k(self) -> int:
        return len(self.labels)

    def gains(self) -> np.ndarray:
        w = np.ones(self.k) if self.gain_weights is None else np.asarray(self.gain_weights, float)
        return self.gain * w

    def drop_label(self, name: str) -> "PowerConfig":
        keep = [i for i, n in enumerate(self.labels) if n != name]
        gw = None if self.gain_weights is None else tuple(self.gain_weights[i] for i in keep)
        return replace(self, labels=tuple(self.labels[i] for i in keep),
                       prevalences=tuple(self.prevalences[i] for i in keep),
                       base_auc=tuple(self.base_auc[i] for i in keep), gain_weights=gw)


def delta_from_auc(auc) -> np.ndarray:
    """Binormal (equal unit variance) mean separation for a given AUROC: AUC = Phi(delta / sqrt 2)."""
    a = np.clip(np.asarray(auc, float), 0.5, 0.995)
    return np.sqrt(2.0) * stats.norm.ppf(a)


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def site_sizes(n_total: int, n_sites: int) -> list[int]:
    return [n_total // n_sites + (1 if s < n_total % n_sites else 0) for s in range(n_sites)]


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def simulate_site_d(cfg: PowerConfig, n: int, reps: int, rng: np.random.Generator,
                    u_gain: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Per-patient d_i for ``reps`` independent copies of one site of ``n`` patients.

    ``u_gain`` and ``eta`` are (reps,) site effects on the AUROC gain and the prevalence logit.
    Returns (reps, n) with NaN where a patient has no assessable label.
    """
    K = cfg.k
    pi = np.asarray(cfg.prevalences, float)
    lam = cfg.label_loading
    d_b = delta_from_auc(cfg.base_auc)                                     # (K,)
    gains = cfg.gains()
    d_e_nom = delta_from_auc(np.asarray(cfg.base_auc) + gains)             # (K,) the model's belief
    d_e_true = delta_from_auc(np.asarray(cfg.base_auc)[None, :] + gains[None, :] + u_gain[:, None])  # (R, K)

    f = rng.standard_normal((reps, n, 1))
    e = rng.standard_normal((reps, n, K))
    liab = lam * f + np.sqrt(1.0 - lam**2) * e
    pi_site = _expit(np.log(pi / (1 - pi))[None, :] + eta[:, None])        # (R, K)
    thr = stats.norm.ppf(1.0 - pi_site)[:, None, :]                        # (R, 1, K)
    y = (liab > thr).astype(float)

    zb = rng.standard_normal((reps, n, K))
    ze = cfg.noise_corr * zb + np.sqrt(1 - cfg.noise_corr**2) * rng.standard_normal((reps, n, K))
    s_b = d_b[None, None, :] * y + zb
    s_e = d_e_true[:, None, :] * y + ze

    lg = np.log(pi / (1 - pi))[None, None, :]                              # models know the global prevalence
    p_b = _expit(lg + d_b * s_b - d_b**2 / 2)
    p_e = _expit(lg + d_e_nom * s_e - d_e_nom**2 / 2)

    mask = rng.random((reps, n, K)) < cfg.assess_prob
    diff = np.where(mask, binary_log_loss(y, p_e) - binary_log_loss(y, p_b), 0.0)
    cnt = mask.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(cnt > 0, diff.sum(axis=2) / cnt, np.nan)
    return d


def simulate_site_stats(cfg: PowerConfig, n_total: int, reps: int, rng: np.random.Generator):
    """Per-rep, per-site (n, mean, var) of d_i. Arrays are (reps, S)."""
    S = cfg.n_sites
    u = rng.normal(0.0, cfg.tau_gain, (reps, S))
    eta = rng.normal(0.0, cfg.site_prev_sd, (reps, S))
    n_s = np.zeros((reps, S))
    m_s = np.zeros((reps, S))
    v_s = np.zeros((reps, S))
    for s, n in enumerate(site_sizes(n_total, S)):
        d = simulate_site_d(cfg, n, reps, rng, u[:, s], eta[:, s])
        ok = ~np.isnan(d)
        cnt = ok.sum(axis=1)
        dz = np.where(ok, d, 0.0)
        mean = dz.sum(axis=1) / cnt
        var = (np.where(ok, (d - mean[:, None]) ** 2, 0.0)).sum(axis=1) / np.maximum(cnt - 1, 1)
        n_s[:, s], m_s[:, s], v_s[:, s] = cnt, mean, var
    return n_s, m_s, v_s


# --------------------------------------------------------------------------
# Decision criteria
# --------------------------------------------------------------------------
def ci_upper_bounds(n_s, m_s, v_s, rng: np.random.Generator, n_boot: int = 1000) -> dict[str, np.ndarray]:
    """Upper limits of the three 95% CIs for Delta, each (reps,)."""
    n_tot = n_s.sum(axis=1)
    est = (n_s * m_s).sum(axis=1) / n_tot
    se_within = np.sqrt((n_s * v_s).sum(axis=1)) / n_tot       # sqrt(sum n_s^2 v_s / n_s) / n
    out = {"within_site": est + Z * se_within}

    S = m_s.shape[1]
    site_mean = m_s.mean(axis=1)
    out["site_t"] = site_mean + T975_DF2 * m_s.std(axis=1, ddof=1) / np.sqrt(S)

    reps = m_s.shape[0]
    j = rng.integers(0, S, (reps, n_boot, S))                      # stage 1: which sites
    r = np.arange(reps)[:, None, None]
    mj, se_j = m_s[r, j], np.sqrt(v_s / n_s)[r, j]
    draw = mj + se_j * rng.standard_normal(mj.shape)               # stage 2: patients within the drawn site
    w = n_s[r, j]
    boot = (w * draw).sum(axis=2) / w.sum(axis=2)                  # pooled, patient-weighted
    out["two_stage"] = np.quantile(boot, 0.975, axis=1)
    return out


def evaluate_rules(n_s, m_s, v_s, rng) -> dict[str, np.ndarray]:
    """Boolean (reps,) outcome of each criterion and of the full H1/H2 rule under each CI."""
    up = ci_upper_bounds(n_s, m_s, v_s, rng)
    every = np.all(m_s < 0, axis=1)
    out = {"every_site": every}
    for mode in CI_MODES:
        out[f"ci_{mode}"] = up[mode] < 0
        out[f"rule_{mode}"] = (up[mode] < 0) & every
    out["est"] = (n_s * m_s).sum(axis=1) / n_s.sum(axis=1)
    return out


def _cell_seed(seed: int, n_total: int, key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, int(n_total), int(key)]))


def power_cell(cfg: PowerConfig, n_total: int, reps: int = 400, seed: int = 20261007, key: int = 0) -> dict:
    """Monte Carlo power at one (config, N). Returns proportions and MC standard errors."""
    rng = _cell_seed(seed, n_total, key)
    chunk = max(1, int(2.5e6 // (max(n_total // cfg.n_sites, 1) * cfg.k)))
    acc: dict[str, list] = {}
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        n_s, m_s, v_s = simulate_site_stats(cfg, n_total, r, rng)
        for k, v in evaluate_rules(n_s, m_s, v_s, rng).items():
            acc.setdefault(k, []).append(v)
        done += r
    res = {k: np.concatenate(v) for k, v in acc.items()}
    out = {"n_total": n_total, "reps": reps, "mean_delta_hat": float(res["est"].mean())}
    for k, v in res.items():
        if v.dtype == bool:
            p = float(v.mean())
            out[k] = p
            out[k + "_se"] = float(np.sqrt(p * (1 - p) / reps))
    return out


def power_table(cfg: PowerConfig, gains: Sequence[float] = GAINS, n_grid: Sequence[int] = N_GRID,
                reps: int = 400, seed: int = 20261007) -> dict[float, list[dict]]:
    """{gain: [cell for each N]}; common random numbers across gains (same seed per N)."""
    return {g: [power_cell(replace(cfg, gain=g), n, reps, seed, key=0) for n in n_grid] for g in gains}


def min_n_for_power(ns: Sequence[int], powers: Sequence[float], target: float = 0.80) -> float:
    """Smallest N reaching ``target`` by linear interpolation on the grid; inf if never reached.

    Power is forced monotone non-decreasing first (removes Monte Carlo wiggle).
    """
    ns = np.asarray(ns, float)
    p = np.maximum.accumulate(np.asarray(powers, float))
    if p[0] >= target:
        return float(ns[0])
    for i in range(1, len(ns)):
        if p[i] >= target:
            return float(ns[i - 1] + (target - p[i - 1]) / (p[i] - p[i - 1]) * (ns[i] - ns[i - 1]))
    return float("inf")


def true_delta(cfg: PowerConfig, n_total: int = 300_000, seed: int = 1) -> dict:
    """Population Delta (tau = 0, no prevalence shift) and SD(d_i) from one very large sample."""
    c = replace(cfg, tau_gain=0.0, site_prev_sd=0.0)
    rng = np.random.default_rng(seed)
    d = simulate_site_d(c, n_total, 1, rng, np.zeros(1), np.zeros(1))[0]
    d = d[~np.isnan(d)]
    return {"delta": float(d.mean()), "sd_d": float(d.std(ddof=1))}


def expected_positives(cfg: PowerConfig, n_total: int) -> dict[str, float]:
    """Expected gold-positive count per label among assessable patients."""
    return {n: p * cfg.assess_prob * n_total for n, p in zip(cfg.labels, cfg.prevalences)}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _fmt_n(x: float) -> str:
    return ">8,000" if not np.isfinite(x) else f"{int(round(x, -1)):,}"


def format_power_table(tab: dict[float, list[dict]], key: str) -> str:
    ns = [c["n_total"] for c in next(iter(tab.values()))]
    h = "| Gain (AUROC) | " + " | ".join(f"N={n:,}" for n in ns) + " |\n|---|" + "---|" * len(ns) + "\n"
    return h + "".join(f"| +{g:.2f} | " + " | ".join(f"{c[key]:.2f}" for c in cells) + " |\n"
                       for g, cells in tab.items())


def format_loss_table(tab: dict[float, list[dict]]) -> str:
    """Lost-to-every-site table: CI-only power, full rule, and share of CI power lost."""
    ns = [c["n_total"] for c in next(iter(tab.values()))]
    h = "| Gain | " + " | ".join(f"N={n:,}" for n in ns) + " |\n|---|" + "---|" * len(ns) + "\n"
    rows = ""
    for g, cells in tab.items():
        if g == 0:
            continue
        rows += f"| +{g:.2f} | " + " | ".join(
            f"{c['ci_within_site']:.2f} -> {c['rule_within_site']:.2f} ({_share(c):.0%})" for c in cells) + " |\n"
    return h + rows


def _share(c: dict) -> float:
    ci = c["ci_within_site"]
    return (ci - c["rule_within_site"]) / ci if ci > 0 else float("nan")


def format_min_n(tab: dict[float, list[dict]], modes=("within_site", "site_t", "two_stage")) -> str:
    h = ("| Gain (AUROC) | " + " | ".join(f"Min N, rule with {m} CI" for m in modes)
         + " | Min N, CI only (within_site) |\n|---|" + "---|" * (len(modes) + 1) + "\n")
    rows = ""
    for g, cells in tab.items():
        if g == 0:
            continue
        ns = [c["n_total"] for c in cells]
        rows += f"| +{g:.2f} | " + " | ".join(
            _fmt_n(min_n_for_power(ns, [c[f'rule_{m}'] for c in cells])) for m in modes) + " | " + _fmt_n(
            min_n_for_power(ns, [c["ci_within_site"] for c in cells])) + " |\n"
    return h + rows


def render(reps: int = 400, seed: int = 20261007, cfg: PowerConfig | None = None) -> str:
    cfg = cfg or PowerConfig()
    parts = []
    td = {g: true_delta(replace(cfg, gain=g)) for g in GAINS}
    parts += ["### Table P0. Population Delta and SD(d_i) implied by each AUROC gain (tau = 0)", "",
              "| Gain (AUROC) | Delta (masked mean log loss) | SD(d_i) |", "|---|---|---|"]
    parts += [f"| +{g:.2f} | {v['delta']:+.4f} | {v['sd_d']:.3f} |" for g, v in td.items()]
    parts.append("")
    tab = power_table(cfg, GAINS, N_GRID_EXT, reps, seed)
    in_grid = {g: [c for c in cells if c["n_total"] in N_GRID] for g, cells in tab.items()}
    parts += [f"### Table P1. Power of the full H1/H2 rule (within-site CI < 0 AND every site < 0), {reps} reps", "",
              format_power_table(in_grid, "rule_within_site"),
              "### Table P2. CI criterion alone (within-site CI upper < 0)", "",
              format_power_table(in_grid, "ci_within_site"),
              "### Table P3. Every-site criterion alone (all three site Deltas < 0)", "",
              format_power_table(in_grid, "every_site"),
              "### Table P4. Power lost to the every-site requirement: CI-only -> full rule (share of CI power lost)", "",
              format_loss_table(in_grid),
              "### Table P5. Full rule with the cluster-aware CIs (site_t, t with 2 df)", "",
              format_power_table(in_grid, "rule_site_t"),
              "### Table P6. Full rule with the two-stage bootstrap CI", "",
              format_power_table(in_grid, "rule_two_stage"),
              "### Table P7. Minimum N for 80% power (linear interpolation on N = 500 ... 8,000)", "",
              format_min_n(tab)]
    return "\n".join(parts)


def sensitivity_min_n(cfg: PowerConfig, reps: int, seed: int, scenarios: dict[str, PowerConfig]) -> str:
    rows = "| Scenario | " + " | ".join(f"Min N, gain +{g:.2f}" for g in GAINS[1:]) + " |\n|---|" + "---|" * 3 + "\n"
    for name, c in scenarios.items():
        tab = power_table(c, GAINS[1:], N_GRID_EXT, reps, seed)
        rows += f"| {name} | " + " | ".join(
            _fmt_n(min_n_for_power([x['n_total'] for x in cells], [x['rule_within_site'] for x in cells]))
            for cells in tab.values()) + " |\n"
    return rows


def default_scenarios() -> dict[str, PowerConfig]:
    base = PowerConfig()
    return {
        "Base (6 labels, tau 0.015)": base,
        "No E7 (5 labels, SAP default)": base.drop_label("E7"),
        "No between-site heterogeneity (tau 0)": replace(base, tau_gain=0.0),
        "Strong heterogeneity (tau 0.03)": replace(base, tau_gain=0.03),
        "Gain concentrated in E1, E2, E4a (x2; others 0)": replace(base, gain_weights=(2, 2, 2, 0, 0, 0)),
        "Weak label correlation (loading 0.2)": replace(base, label_loading=0.2),
        "EEG model noise nearly independent of baseline (rho 0.2)": replace(base, noise_corr=0.2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--sensitivity", action="store_true", help="also print the min-N sensitivity table")
    ap.add_argument("--json", type=str, default=None, help="write the raw grid to this path")
    a = ap.parse_args(argv)
    print(render(a.reps, a.seed))
    if a.sensitivity:
        print("\n### Table P8. Minimum N for 80% power (full rule, within-site CI), sensitivity scenarios\n")
        print(sensitivity_min_n(PowerConfig(), a.reps, a.seed, default_scenarios()))
    if a.json:
        tab = power_table(PowerConfig(), GAINS, N_GRID_EXT, a.reps, a.seed)
        with open(a.json, "w") as fh:
            json.dump({str(g): cells for g, cells in tab.items()}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
