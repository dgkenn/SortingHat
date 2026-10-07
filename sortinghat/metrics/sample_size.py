"""External-validation sample size for binary labels (Riley et al. 2021 approach).

Implements the precision-based criteria of Riley RD et al., Stat Med 2021;40:4230-4251
(PMID 34031906), as summarised in Riley et al., BMJ 2024;384:e074821 (PMID 38253388):

1. O/E:   SE(ln(O/E)) ~= sqrt((1 - phi) / (n * phi)),  phi = outcome proportion.
2. Slope: Var(slope-hat) = [I^-1]_22 / n, where I = E[ w * (1, LP)^T (1, LP) ], w = p(1 - p),
          evaluated at the assumed truth (calibration slope 1, intercept 0).
3. C:     Newcombe / Hanley-McNeil SE of the C-statistic.

Linear-predictor (LP) distribution: Riley et al. recommend taking it from the development study;
we have none for these labels, so we use their documented last resort: LP | y ~ N(mu_y, s^2), common
variance, with s = sqrt(2) * Phi^-1(C) and means mu_{0,1} = logit(phi) -/+ s^2 / 2, which gives a
well-calibrated model (slope 1, intercept 0) with the stated C-statistic and prevalence. Real LP
distributions in the evaluation set will differ; rerun after the Phase 0 pilot.

Targets are 95% CI widths = 2 * 1.96 * SE. N is the number of patients *assessable for that label*.

Run:  python -m sortinghat.metrics.sample_size            (markdown tables to stdout)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy import stats

Z = 1.959963984540054
PREVALENCES = (0.05, 0.10, 0.20)
C_STATS = (0.70, 0.75, 0.80)


def se_from_width(width: float) -> float:
    return width / (2 * Z)


# ---- criterion 1: O/E ------------------------------------------------------
def n_for_oe(prevalence: float, ci_width: float) -> float:
    """N so that the 95% CI width of O/E (about 1) is ``ci_width`` (SE on the ln scale)."""
    se = se_from_width(ci_width)
    return (1 - prevalence) / (prevalence * se**2)


def oe_width_at_n(prevalence: float, n: float) -> float:
    return 2 * Z * np.sqrt((1 - prevalence) / (n * prevalence))


# ---- criterion 2: calibration slope ---------------------------------------
def binormal_lp_params(prevalence: float, c_stat: float) -> tuple[float, float, float]:
    """(mu0, mu1, s) of the equal-variance binormal LP giving slope 1, intercept 0, C = c_stat."""
    s = np.sqrt(2.0) * stats.norm.ppf(c_stat)
    lg = np.log(prevalence / (1 - prevalence))
    return lg - s**2 / 2, lg + s**2 / 2, s


def slope_information(prevalence: float, c_stat: float, n_grid: int = 20001) -> np.ndarray:
    """Per-observation 2x2 Fisher information for (intercept, slope) at slope 1, intercept 0."""
    mu0, mu1, s = binormal_lp_params(prevalence, c_stat)
    lo, hi = min(mu0, mu1) - 9 * s, max(mu0, mu1) + 9 * s
    x = np.linspace(lo, hi, n_grid)
    dens = (1 - prevalence) * stats.norm.pdf(x, mu0, s) + prevalence * stats.norm.pdf(x, mu1, s)
    pr = 1.0 / (1.0 + np.exp(-x))
    w = pr * (1 - pr) * dens
    I = np.array([[np.trapezoid(w, x), np.trapezoid(w * x, x)], [np.trapezoid(w * x, x), np.trapezoid(w * x * x, x)]])
    return I


def slope_var_per_n(prevalence: float, c_stat: float) -> float:
    """n * Var(slope-hat) under the assumed truth."""
    return float(np.linalg.inv(slope_information(prevalence, c_stat))[1, 1])


def n_for_slope(prevalence: float, c_stat: float, ci_width: float) -> float:
    return slope_var_per_n(prevalence, c_stat) / se_from_width(ci_width) ** 2


def slope_width_at_n(prevalence: float, c_stat: float, n: float) -> float:
    return 2 * Z * np.sqrt(slope_var_per_n(prevalence, c_stat) / n)


# ---- criterion 3: C-statistic ---------------------------------------------
def c_stat_se(c: float, n: float, prevalence: float) -> float:
    """Newcombe (= Hanley-McNeil) SE of the C-statistic."""
    n1, n0 = n * prevalence, n * (1 - prevalence)
    v = c * (1 - c) * (1 + (n1 - 1) * (1 - c) / (2 - c) + (n0 - 1) * c / (1 + c)) / (n1 * n0)
    return float(np.sqrt(v))


def n_for_cstat(prevalence: float, c_stat: float, ci_width: float, n_max: float = 1e7) -> float:
    target = se_from_width(ci_width)
    lo, hi = 10.0 / prevalence, n_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if c_stat_se(c_stat, mid, prevalence) > target:
            lo = mid
        else:
            hi = mid
    return hi


# ---- primary-endpoint precision (placeholder SDs; replace with pilot value) ----
def delta_halfwidth_at_n(sd_d: float, n: float) -> float:
    """Expected 95% CI half-width of Delta = mean(d_i) for a within-site patient-level CI."""
    return Z * sd_d / np.sqrt(n)


def n_for_delta_halfwidth(sd_d: float, halfwidth: float) -> float:
    return (Z * sd_d / halfwidth) ** 2


@dataclass(frozen=True)
class Row:
    prevalence: float
    c_stat: float
    n_oe: float
    n_slope: float
    n_cstat: float

    @property
    def n_required(self) -> float:
        return max(self.n_oe, self.n_slope, self.n_cstat)

    @property
    def binding(self) -> str:
        v = {"O/E": self.n_oe, "slope": self.n_slope, "C": self.n_cstat}
        return max(v, key=v.get)


def build_table(oe_width: float = 0.2, slope_width: float = 0.2, c_width: float = 0.1,
                prevalences=PREVALENCES, c_stats=C_STATS) -> list[Row]:
    return [
        Row(phi, c, n_for_oe(phi, oe_width), n_for_slope(phi, c, slope_width), n_for_cstat(phi, c, c_width))
        for phi in prevalences for c in c_stats
    ]


def _ceil(x: float) -> int:
    return int(np.ceil(x))


def format_main_table(rows: list[Row], oe_width: float, slope_width: float, c_width: float) -> str:
    h = (f"| Prevalence | C | N: O/E (CI width {oe_width:g}) | N: slope (CI width {slope_width:g}) "
         f"| N: C-stat (CI width {c_width:g}) | N required | Events at N | Binding |\n"
         "|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        n = _ceil(r.n_required)
        body += (f"| {r.prevalence:.0%} | {r.c_stat:.2f} | {_ceil(r.n_oe):,} | {_ceil(r.n_slope):,} | "
                 f"{_ceil(r.n_cstat):,} | {n:,} | {_ceil(n * r.prevalence):,} | {r.binding} |\n")
    return h + body


def format_precision_table(ns=(1000, 800, 500), prevalences=PREVALENCES, c_stats=C_STATS) -> str:
    cols = " | ".join(f"slope width N={n}" for n in ns)
    cols2 = " | ".join(f"O/E width N={n}" for n in ns)
    h = f"| Prevalence | C | {cols} | {cols2} |\n|" + "---|" * (2 + 2 * len(ns)) + "\n"
    body = ""
    for phi in prevalences:
        for c in c_stats:
            sl = " | ".join(f"{slope_width_at_n(phi, c, n):.2f}" for n in ns)
            oe = " | ".join(f"{oe_width_at_n(phi, n):.2f}" for n in ns)
            body += f"| {phi:.0%} | {c:.2f} | {sl} | {oe} |\n"
    return h + body


def format_delta_table(sds=(0.10, 0.20, 0.30), ns=(1000, 800, 500)) -> str:
    h = "| SD of d_i | " + " | ".join(f"half-width N={n}" for n in ns) + " |\n|" + "---|" * (1 + len(ns)) + "\n"
    body = "".join(f"| {sd:.2f} | " + " | ".join(f"{delta_halfwidth_at_n(sd, n):.3f}" for n in ns) + " |\n" for sd in sds)
    return h + body


def render(oe_width: float = 0.2, slope_width: float = 0.2, c_width: float = 0.1) -> str:
    rows = build_table(oe_width, slope_width, c_width)
    wide = build_table(oe_width * 2, 0.3, c_width)
    parts = [
        f"### Table S1. Minimum N (assessable patients per label), primary targets",
        "",
        format_main_table(rows, oe_width, slope_width, c_width),
        "### Table S2. Relaxed targets (O/E width %g, slope width 0.3)" % (oe_width * 2),
        "",
        format_main_table(wide, oe_width * 2, 0.3, c_width),
        "### Table S3. Expected 95% CI width at fixed N (assuming calibration slope 1, intercept 0)",
        "",
        format_precision_table(),
        "### Table S4. Expected 95% CI half-width for primary-endpoint Delta (placeholder SDs of d_i)",
        "",
        format_delta_table(),
    ]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oe-width", type=float, default=0.2)
    ap.add_argument("--slope-width", type=float, default=0.2)
    ap.add_argument("--c-width", type=float, default=0.1)
    a = ap.parse_args(argv)
    print(render(a.oe_width, a.slope_width, a.c_width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
