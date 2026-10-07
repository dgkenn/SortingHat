# Evaluation-set size for Study 1: power of the primary endpoint versus calibration precision

Status: analysis note for the biostatistician, written before any pilot data exist. Every input below is an assumption to be replaced by pilot measurements. Nothing here changes the SAP (`docs/prereg_study1_sap.md`); it supplies the missing H1/H2 power calculation and a recommendation for the SAP section 14 decision.

Reproduce: `python -m sortinghat.metrics.power --reps 400 --sensitivity --null --levels` (about 6.5 minutes; seeded, so the tables below regenerate exactly). Code: `sortinghat/metrics/power.py`; tests: `tests/test_metrics_power.py`.

## 1. Why this note exists

SAP section 14 shows that per-label calibration-slope CI width 0.2 needs 2,671 to 15,739 assessable patients, so the planned ~1,000-case evaluation set cannot deliver it. But the primary endpoint is Delta, the masked multi-label log-loss difference (H1/H2: Delta CI below 0 and Delta point estimate below 0 at every one of 3 held-out sites), and the SAP only gives a normal-theory power approximation for it with a placeholder SD(d_i) = 0.20 (Table S5). This note simulates the endpoint itself.

## 2. Simulation design (all assumptions)

| Element | Setting |
|---|---|
| Labels | E1, E2, E4a, E5, E6, E7 (six, as requested). Prevalence 0.25, 0.15, 0.10, 0.25, 0.20, 0.03 |
| Label dependence | One-factor Gaussian copula, loading 0.5 (liability correlation 0.25) |
| Baseline model | Calibrated, binormal scores; AUROC 0.72, 0.75, 0.72, 0.70, 0.70, 0.73 |
| EEG-augmented model | AUROC = baseline + gain on every label, gain in {0, 0.02, 0.04, 0.06}; its score noise correlates 0.6 with the baseline's |
| Sites | 3, equal size (N/3). Site-level random effect on the EEG gain, SD tau = 0.015 AUROC (shared by all labels); the model's probabilities assume the nominal gain, so a weak site also receives over-confident EEG probabilities. Site prevalence shift SD 0.25 on the logit scale |
| Assessability | Each label assessable with probability 0.90; patients with no assessable label drop out |
| Endpoint | d_i from the same masked-mean `loss.binary_log_loss` used by the SAP code; Delta = mean(d_i) |
| H1/H2 rule | CI upper bound < 0 **and** all three site Delta point estimates < 0 |
| CIs evaluated | `within_site` (SAP decision interval; its large-sample form, validated against `bootstrap.paired_bootstrap_delta` in the tests), `site_t` (mean of 3 site Deltas, t with 2 df), `two_stage` (sites then patients) |
| Replications | 400 per cell (Monte Carlo SE up to 0.025); seeded; common random numbers across effect sizes |

Table P0 translates the AUROC gains into the log-loss scale (large-sample, tau = 0):

| Gain (AUROC) | Delta | SD(d_i) |
|---|---|---|
| +0.00 | -0.0001 | 0.107 |
| +0.02 | -0.0086 | 0.113 |
| +0.04 | -0.0180 | 0.119 |
| +0.06 | -0.0285 | 0.127 |

SD(d_i) comes out at 0.11 to 0.13, about 60% of the SAP's placeholder 0.20, because the EEG and baseline predictions are strongly paired. This is the least certain input (it depends on `noise_corr`); the pilot must measure it.

## 3. Results

### 3.1 Power of the full H1/H2 rule (within-site CI below 0 and every site below 0)

| Gain (AUROC) | N=500 | N=1,000 | N=1,500 | N=2,000 | N=3,000 |
|---|---|---|---|---|---|
| +0.00 | 0.06 | 0.05 | 0.09 | 0.06 | 0.10 |
| +0.02 | 0.33 | 0.51 | 0.53 | 0.57 | 0.65 |
| +0.04 | 0.78 | 0.89 | 0.94 | 0.94 | 0.96 |
| +0.06 | 0.96 | 0.99 | 0.99 | 0.99 | 0.99 |

The +0.00 row is a rejection rate with between-site heterogeneity present, not a clean type I check; see 3.4.

### 3.2 Share of power lost to the every-site requirement

CI-only power, then the full rule, then the share of CI-only power lost:

| Gain | N=500 | N=1,000 | N=1,500 | N=2,000 | N=3,000 |
|---|---|---|---|---|---|
| +0.02 | 0.42 -> 0.33 (24%) | 0.65 -> 0.51 (21%) | 0.71 -> 0.53 (25%) | 0.76 -> 0.57 (24%) | 0.83 -> 0.65 (22%) |
| +0.04 | 0.87 -> 0.78 (11%) | 0.97 -> 0.89 (9%) | 0.99 -> 0.94 (6%) | 0.99 -> 0.94 (4%) | 1.00 -> 0.96 (4%) |
| +0.06 | 0.99 -> 0.96 (3%) | 1.00 -> 0.99 (1%) | 1.00 -> 0.99 (1%) | 1.00 -> 0.99 (1%) | 1.00 -> 0.99 (1%) |

At the effect size the study is most likely to be powered for (+0.04), the every-site requirement costs 8 percentage points at N = 1,000 and 4 at N = 3,000. At a marginal effect (+0.02) it costs about a fifth to a quarter of the CI-only power at every N, and that loss does not shrink with N because it is driven by site heterogeneity, not sampling noise.

### 3.3 Minimum N for 80% power (linear interpolation; grid extended to N = 8,000)

| Gain (AUROC) | Full rule, within-site CI | Full rule, two-stage CI | Full rule, site_t CI (2 df) | CI criterion only |
|---|---|---|---|---|
| +0.02 | not reached by 8,000 | not reached | not reached | about 2,580 |
| +0.04 | about 610 | about 850 | not reached | 500 or fewer |
| +0.06 | 500 or fewer | 500 or fewer | not reached | 500 or fewer |

Sensitivity of the within-site-CI rule (minimum N for 80%):

| Scenario | +0.02 | +0.04 | +0.06 |
|---|---|---|---|
| Base (6 labels, tau 0.015) | not reached | 610 | 500 or fewer |
| 5 labels, no E7 (SAP default primary set) | not reached | 570 | 500 or fewer |
| No between-site heterogeneity (tau 0) | 1,610 | 500 or fewer | 500 or fewer |
| Strong heterogeneity (tau 0.03) | not reached | not reached | 500 or fewer |
| Gain only on E1, E2, E4a (twice the size) | not reached | 500 or fewer | 500 or fewer |
| Weak label correlation (loading 0.2) | not reached | 630 | 500 or fewer |
| EEG noise nearly independent of baseline (rho 0.2) | not reached | 1,000 | 500 or fewer |

Two things follow. First, power is set by the effect size and by between-site heterogeneity, not by N, once N is about 1,000. For a +0.02 gain with tau = 0.015 the rule is capped near (P(a site's gain > 0))^3 = 0.91^3 = 0.75 however many patients are read; going from 1,000 to 3,000 patients moves power only from 0.51 to 0.65. Second, the cluster-aware t(2) interval (3 sites, 2 df, critical value 4.30) is so wide that the rule almost never passes (power 0.40 at +0.04, N = 1,000 and 0.58 at +0.06); no affordable N repairs that, so the SAP's choice of the within-site interval as the decision interval is what makes H1/H2 passable at all.

### 3.4 Type I error and the within-site interval

Rejection rate at a true gain of 0 (nominal 0.025 for the CI criterion; 1,000 reps):

| tau (AUROC) | N | CI only: within-site | CI only: site_t | CI only: two-stage | Full rule: within-site | Full rule: site_t | Full rule: two-stage |
|---|---|---|---|---|---|---|---|
| 0 | 1,000 | 0.022 | 0.025 | 0.017 | 0.022 | 0.025 | 0.017 |
| 0 | 3,000 | 0.028 | 0.024 | 0.012 | 0.024 | 0.024 | 0.012 |
| 0.015 | 1,000 | 0.102 | 0.033 | 0.043 | 0.071 | 0.033 | 0.043 |
| 0.015 | 3,000 | 0.171 | 0.020 | 0.050 | 0.080 | 0.020 | 0.050 |
| 0.03 | 1,000 | 0.205 | 0.029 | 0.081 | 0.109 | 0.029 | 0.078 |
| 0.03 | 3,000 | 0.310 | 0.015 | 0.083 | 0.103 | 0.015 | 0.080 |

Without heterogeneity every interval is near nominal. With heterogeneity, the within-site interval alone rejects 10 to 31% of the time at a pooled effect of zero, and rises with N, because it only reflects patient sampling at the three observed sites. The every-site condition halves that (to 7 to 11% for the full rule) but does not restore 2.5%. This is the SAP's own stated limitation (section 7.2) quantified: the H1/H2 rule is valid for "these three sites", not for a population of sites, and a larger N makes the gap worse, not better. The two-stage interval is much closer to nominal (4 to 8%) at a modest power cost (3.3). The SAP already co-reports all intervals; this supports keeping that, and stating in the Results that a pass on the within-site rule at larger N is not stronger evidence of site transportability.

## 4. Calibration versus the primary endpoint

Expected 95% CI width of the calibration slope at the simulation's prevalences (C = 0.72, calibrated model, 90% assessable; `sample_size.slope_width_at_n`):

| N (total) | E1 / E5 (0.25) | E6 (0.20) | E2 (0.15) | E4a (0.10) | E7 (0.03) |
|---|---|---|---|---|---|
| 1,000 | 0.41 | 0.44 | 0.49 | 0.57 | 0.96 |
| 1,500 | 0.34 | 0.36 | 0.40 | 0.46 | 0.78 |
| 2,000 | 0.29 | 0.31 | 0.35 | 0.40 | 0.68 |
| 3,000 | 0.24 | 0.26 | 0.28 | 0.33 | 0.55 |

Total N needed for slope width 0.2: 4,291 (E1, E5), 4,903 (E6), 5,956 (E2), 8,081 (E4a), 22,876 (E7); for width 0.3: 1,907, 2,179, 2,647, 3,591, 10,167. These agree with SAP Table S1/S2 (2,671 to 15,739 assessable). No affordable N meets 0.2 for all six labels; even N = 3,000 leaves 0.24 to 0.55.

## 5. Recommendation

**Evaluation-set size.** Keep about **1,000 consecutive gold cases (at least ~330 per site, each with at least one assessable primary label) as the committed design**; treat **1,500** as an optional reserve, not a requirement. Reasoning from 3.1 to 3.3:

- At N = 1,000 the full H1/H2 rule has power 0.89 for a +0.04 AUROC gain (80% power from about 610) and 0.99 for +0.06. These are the effect sizes at which a clinical increment would be worth announcing.
- Raising N to 1,500 adds 5 points at +0.04 (0.94) and 2 points at +0.02 (0.53); raising it to 3,000 adds 7 and 14 points. The marginal power per extra review is low because the binding constraint is the three-site structure and its heterogeneity.
- A +0.02 gain is not reliably detectable at any feasible N under the every-site rule (about 0.5 to 0.65), and that is the rule working as intended: a gain that small, with realistic site-to-site variation, will be unfavorable at one of three sites a quarter of the time. The SAP should say in advance that a miss at +0.02 is an expected outcome, not a design failure.
- Dropping below about 800 starts to cost power (0.78 at N = 500 for +0.04).
- Adding sites would raise power far more than adding patients; the SAP is fixed at 3 held-out sites, so the lever is not available here, but it is the answer if a funder asks what to buy.

**E7.** At 3% prevalence the E7 eligibility rule (at least 100 gold positives, at least 2 sites) needs about 3,700 consecutive cases (3,333 if every patient were assessable) and is unreachable at 1,000 (about 27 expected positives). Plan E7 as exploratory through the enriched set, as the SAP already allows; the primary endpoint is then the five-label version, which is slightly better powered here (570 versus 610 minimum N at +0.04). E4a at 10% gives about 90 expected positives at N = 1,000, just under the 100-event guidance, so expect it to be borderline in the slope and per-label results.

**Calibration slope: precision-reported secondary, not a gate.** The SAP (section 8) already keeps calibration out of H1 to H6 and out of the gates; this note supports making that explicit and adding to the SAP: (a) report per-label slope and O/E with bootstrap CIs and calibration plots at Study 1; (b) state beforehand the expected slope CI width at the achieved N (0.41 to 0.57 for E1 to E6 at N = 1,000; E7 uninformative); (c) forbid claims of "well calibrated" or slope near 1 from a CI of that width, and forbid using a calibration criterion to pass or fail a model, a site or a gate; (d) restrict slope reporting to labels with at least 100 expected positives (E1, E2, E5, E6), and treat E4a and E7 as O/E only; (e) leave a pooled or label-stacked slope as an option for the biostatistician, if the pilot shows a shared calibration distortion across labels. A gate on calibration slope would need 4,300 to 8,100 cases for the main five labels, an evaluation-only adjudication load of roughly 1,600 to 4,600 physician-hours at 10 to 15 minutes per review, to protect a property the primary claim does not rest on.

## 6. Adjudication hours

Basis: the plan's 10 to 15 minutes per review with pre-assembled packets; evaluation cases dual-read plus a 25% third reader (2.25 reviews per case); development 400 single-read plus 20% double-read (480 reviews); enriched set 300 cases at the same 2.25 reviews per case (675 reviews). Fixed non-evaluation work is 1,155 reviews (193 to 289 hours). This reproduces the plan's 3,405 reviews and 570 to 850 hours at N = 1,000.

| Evaluation N | Evaluation reviews | Evaluation hours | Whole study reviews | Whole study hours | Change versus N = 1,000 |
|---|---|---|---|---|---|
| 500 | 1,125 | 188 to 281 | 2,280 | 380 to 570 | -188 to -281 |
| 1,000 (plan) | 2,250 | 375 to 563 | 3,405 | 568 to 851 | 0 |
| 1,500 | 3,375 | 563 to 844 | 4,530 | 755 to 1,133 | +188 to +281 |
| 2,000 | 4,500 | 750 to 1,125 | 5,655 | 943 to 1,414 | +375 to +563 |
| 3,000 | 6,750 | 1,125 to 1,688 | 7,905 | 1,318 to 1,976 | +750 to +1,125 |
| 3,700 (E7 eligible) | 8,325 | 1,388 to 2,081 | 9,480 | 1,580 to 2,370 | +1,013 to +1,519 |
| 4,300 (slope width 0.2 for E1/E5 only) | 9,675 | 1,613 to 2,419 | 10,830 | 1,805 to 2,708 | +1,238 to +1,856 |
| 15,739 (SAP worst case) | 35,413 | 5,902 to 8,853 | 36,568 | 6,095 to 9,142 | +5,527 to +8,290 |

Net effect of the recommendation: **no change to the plan's 570 to 850 hours.** Reading calibration as a precision-reported secondary avoids a 1,000 to 8,000 hour increase that no team in the plan could staff, and the optional 1,500-case reserve costs 188 to 281 additional hours. Cheaper options that were not simulated and are not recommended without biostatistician sign-off: single-reading a random subset of the evaluation set (reduces dual-review labor but breaks the SAP's design), or lowering the third-reader rate (assumed 25%; the pilot measures the real rate).

## 7. Limits of this analysis

- Synthetic and binormal: scores are Gaussian given the label, models are calibrated up to the site effects, the EEG gain is shared across labels (or concentrated, in one scenario). Real model errors are heavier-tailed and correlated across labels; real SD(d_i) may differ from 0.11 to 0.13.
- The between-site effect size (tau = 0.015 AUROC) is a guess with no data behind it; results in 3.3 and 3.4 depend strongly on it. The pilot cannot estimate it from one site, so the SAP should either commit to a conservative value or accept that power at +0.02 and +0.04 is conditional on it.
- The within-site CI is evaluated in its large-sample form; the tests confirm it agrees with the percentile bootstrap to within a fraction of the interval half-width. The two-stage interval approximates the patient stage by a normal draw around each site's mean.
- Monte Carlo SE is up to 0.025 per cell (400 reps); minimum-N figures from the grid are rounded to 10 and are interpolations.
- Equal site sizes; independent 90% assessability; no covariate shift between development and evaluation sites; the baseline is treated as fixed and correctly calibrated (in practice a baseline fit on development data carries its own estimation error, which would shrink the observed gain).
- Run after the Phase 0 pilot with measured prevalences, assessable fractions, the realized baseline AUROCs and SD(d_i); `PowerConfig` takes each of these directly.

## 8. CI level for the within-site interval (follow-up)

The full rule's null rejection (7 to 11% at tau 0.015 to 0.03, section 3.4) is too high for a preregistered primary. `power.level_scan` re-evaluates the SAP rule (within-site CI upper bound below 0 and every site below 0) at three two-sided CI levels on shared simulated studies (2,000 reps per cell, same settings as section 2, true gain 0 for the null rows, tau as stated).

| N | CI level | Null rejection, tau 0.015 | Null rejection, tau 0.03 | Power +0.04 | Power +0.06 |
|---|---|---|---|---|---|
| 1,000 | 95.0% | 0.062 | 0.108 | 0.89 | 0.99 |
| 1,000 | 97.5% | 0.045 | 0.097 | 0.88 | 0.99 |
| 1,000 | 99.0% | 0.031 | 0.086 | 0.86 | 0.99 |
| 1,500 | 95.0% | 0.069 | 0.106 | 0.92 | 0.99 |
| 1,500 | 97.5% | 0.055 | 0.102 | 0.92 | 0.99 |
| 1,500 | 99.0% | 0.037 | 0.093 | 0.91 | 0.99 |

- Only the **99% interval** keeps null rejection at or below 5% at both N for tau = 0.015 (3.1% at N = 1,000, 3.7% at N = 1,500). 97.5% passes at N = 1,000 (4.5%) but fails at N = 1,500 (5.5%); 95% fails at both (6.2%, 6.9%).
- The power cost is small: at +0.04, 0.89 to 0.86 at N = 1,000 and 0.92 to 0.91 at N = 1,500; at +0.06 there is none (0.99).
- At tau = 0.03 no level reaches 5% (8.6 to 10.8%). The interval cannot fix between-site variance it does not see; the co-reported two-stage interval (section 3.4: 4 to 8%) is the protection, and the Results must say so. The recommended level is therefore a mitigation under the tau = 0.015 assumption, not a guarantee.
- Implementation: `bootstrap.paired_bootstrap_delta(..., mode="within_site", alpha=0.01)`; B = 10,000 as already planned so the 0.5th and 99.5th percentiles are stable.
