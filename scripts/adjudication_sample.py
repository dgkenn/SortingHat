#!/usr/bin/env python3
"""Gold adjudication pilot, step 1: draw the IU-large sample (D-157).   Aggregate-only stdout; the sample file is for the study team.

Draws N (default 200) patients from the IU-large stratum (D-155) of the BROAD analysed cohort (sites S0001 / S0002), stratified by site and
by SILVER-LABEL PATTERN to enrich the rarer etiologies (sortinghat/adjudication/sampling.py; targets in configs/adjudication.yaml), records
the Horvitz-Thompson sampling weights so a prevalence can be re-weighted to IU-large, and adds a 10% duplicate set (intra-rater
reliability) and a few calibration cases.  The cohort is assembled with the SAME code the feasibility analysis uses
(scripts/run_silver_feasibility.assemble + attach_omop), so the rows, labels and IU-large rule are the ones the silver analysis scored.

    # synthetic / local mirror (what the tests run)
    python3 scripts/adjudication_sample.py --cohort ... --features ... --silver ... --baselines ... --omop-source none --sample-out <tmp>/local_only/adj/sample.csv
    # real (reads only the local_only inputs; no S3; aggregate-only output)
    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/adjudication_sample.py --max-memory-gb 6

Outputs
    out/local_only/adjudication/sample.csv   RECORD-LEVEL (mode 0600, directory 0700; never printed or read by an agent): sample_id, person
        ids, site, t0, silver pattern and labels, stratum, weights, role (sample / duplicate / calibration).  Keep it away from raters
        (the pattern is the silver label).
    out/adjudication/sample_summary.json     aggregates only (n < 11 shown as "<11"; complementary suppression).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import run_silver_feasibility as rsf  # noqa: E402
from sortinghat.adjudication import aggregates as agg  # noqa: E402
from sortinghat.adjudication import sampling  # noqa: E402
from sortinghat.adjudication.private_io import require_local_only, write_private_csv  # noqa: E402
from sortinghat.metrics import intended_use as iu  # noqa: E402
from sortinghat.safe_output import SUPPRESSED, safe_print, safe_quantiles, safe_write_json, suppress_count  # noqa: E402

BANNER = "GOLD ADJUDICATION PILOT (D-157) - sampler; aggregate-only; the sample file is record-level and local_only"
DEFAULT_CONFIG = HERE.parent / "configs" / "adjudication.yaml"
DEFAULT_SAMPLE = "out/local_only/adjudication/sample.csv"
DEFAULT_SUMMARY = "out/adjudication/sample_summary.json"


def build_parser():
    ap = rsf.build_parser()
    ap.description = __doc__
    ap.set_defaults(cohort_def="broad", out="out/adjudication")
    ap.add_argument("--adj-config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--sample-out", default=DEFAULT_SAMPLE, help="record-level sample (must be under local_only/)")
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY, help="aggregate JSON (must NOT be under local_only/)")
    ap.add_argument("--sample-n", type=int, default=None, help="sample size (default: the config)")
    ap.add_argument("--sample-seed", type=int, default=None, help="sampling seed (default: the config)")
    return ap


def eligible_frame(A, a, priority) -> tuple[pd.DataFrame, dict]:
    """IU-large rows of the analysed set as the sampling frame (one row per patient) and the aggregate description of the rule applied."""
    sed = iu.sedation_features(A.baseline) if len(A.baseline.columns) else None
    om = {c: A.omop[c].to_numpy(bool) for c in A.omop.columns} if A.omop is not None and len(A.omop.columns) else None
    res = iu.intended_use_strata(A.hours, sed, A.covariates["duration_s"], iu.ceeg_task_flag(A.frame), om)
    mask = res["masks"][iu.IU_LARGE]
    if mask is None:
        raise SystemExit(f"IU-large is not available: {res['unavailable'].get('both') or res['unavailable']}")
    info = {"definition": iu.DEFINITIONS[iu.IU_LARGE], "components_applied": res["applied"][iu.IU_LARGE],
            "missing_conditions": res["missing"], "omop": A.omop_info, "visit_start_source": A.hours_src}
    Y, _h = rsf.load_silver(a.silver)
    Ys = Y.reindex(A.frame["person_id"].to_numpy()).reset_index(drop=True)
    fr = A.frame
    f = pd.DataFrame({"person_id": fr["person_id"].astype("int64").to_numpy(), "site": fr["SiteID"].astype(str).to_numpy(),
                      "t0": pd.to_datetime(fr["t0"]).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()})
    if "person_id_source" in fr:
        f.insert(1, "person_id_source", pd.to_numeric(fr["person_id_source"], errors="coerce").to_numpy())
    if "encounter_start" in fr:
        f["encounter_start"] = pd.to_datetime(fr["encounter_start"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    f["hours_since_visit_start"] = np.asarray(A.hours, float)
    f["age_years"] = pd.to_numeric(fr["age_years"], errors="coerce").to_numpy() if "age_years" in fr else np.nan
    f["sex_male"] = A.baseline["demo__sex_male"].to_numpy(float) if "demo__sex_male" in A.baseline else np.nan
    for lab in sampling.LABELS:
        f[f"silver_{lab}"] = Ys[lab].to_numpy(float) if lab in Ys else np.nan
    f["pattern"] = sampling.assign_pattern(Ys, tuple(priority))
    f = f[np.asarray(mask, bool)].drop_duplicates("person_id").reset_index(drop=True)
    return f, info


def summarize(sample: pd.DataFrame, frame: pd.DataFrame, labels: dict, patterns: list, info: dict, cfg: dict, n: int, seed: int,
              analysed: dict | None = None) -> dict:
    """Aggregates of the design, suppressed (n < 11) with complementary suppression within a site."""
    sites = sorted(frame["site"].unique())
    s = sample[sample["role"] == sampling.ROLE_SAMPLE]
    pop = {st: {p: int(((frame["site"] == st) & (frame["pattern"] == p)).sum()) for p in patterns} for st in sites}
    got = {st: {p: int(((s["site"] == st) & (s["pattern"] == p)).sum()) for p in patterns} for st in sites}
    table = {}
    for st in sites:
        vp, vn = agg.suppress_row(pop[st]), agg.suppress_row(got[st])
        table[labels[st]] = {p: {"eligible": vp[p], "sampled": vn[p],
                                 "weight": (round(pop[st][p] / got[st][p], 2) if vp[p] != SUPPRESSED and vn[p] != SUPPRESSED and got[st][p] else SUPPRESSED)}
                             for p in patterns}
    pooled_pop, pooled_n = agg.pooled_row(pop), agg.pooled_row(got)
    w = s["weight"].to_numpy(float)
    return {
        "banner": BANNER, "stratum": cfg["stratum"], "cohort_def": cfg["cohort"]["cohort_def"], "iu_large_rule": info,
        "design": {"n_requested": n, "seed": int(seed), "site_allocation": cfg["sampling"]["site_allocation"],
                   "priority": list(cfg["sampling"]["priority"]), "target_shares": cfg["sampling"]["shares"]},
        **({"analysed_broad_cohort": {"pooled": suppress_count(sum(analysed.values())), **{labels[st]: suppress_count(analysed.get(st, 0)) for st in sites}},
            "iu_large_share_of_analysed": agg.pooled_share_cells({labels[st]: (int((frame["site"] == st).sum()), int(analysed.get(st, 0))) for st in sites})}
           if analysed else {}),
        "eligible_iu_large": {"pooled": suppress_count(len(frame)), **{labels[st]: suppress_count(int((frame["site"] == st).sum())) for st in sites}},
        "sampled": {"pooled": suppress_count(len(s)), **{labels[st]: suppress_count(int((s["site"] == st).sum())) for st in sites}},
        "duplicates": suppress_count(int((sample["role"] == sampling.ROLE_DUPLICATE).sum())),
        "calibration": suppress_count(int((sample["role"] == sampling.ROLE_CALIBRATION).sum())),
        "by_site_and_silver_pattern": table,
        "by_silver_pattern_pooled": {p: {"eligible": pooled_pop[p], "sampled": pooled_n[p]} for p in patterns},
        "weights": {"effective_sample_size": round(sampling.kish_ess(w), 1) if len(w) >= 11 else SUPPRESSED,
                    "design_effect": round(len(w) / sampling.kish_ess(w), 3) if len(w) >= 11 else SUPPRESSED, **safe_quantiles(w)},
        "checks": sampling.check_weights(sample, frame),
        "strata_with_population_but_no_sample": suppress_count(int(sum(1 for st in sites for p in patterns if pop[st][p] > 0 and got[st][p] == 0))),
        "reading": ("Sampling weights w = eligible / sampled per site x silver-pattern stratum: sum(w) over the sample = the IU-large "
                    "population, so a gold prevalence is re-weighted with them. Strata are defined by the SILVER label pattern, which is the "
                    "design's enrichment device, not a gold-label property; silver-vs-gold agreement must use the weights.")}


def run(a) -> int:
    cfg = yaml.safe_load(Path(a.adj_config).read_text())
    sp = cfg["sampling"]
    n = a.sample_n if a.sample_n is not None else int(sp["n"])
    seed = a.sample_seed if a.sample_seed is not None else int(sp["seed"])
    sample_out = require_local_only(a.sample_out, "--sample-out")
    summary_out = Path(a.summary_out)
    if "local_only" in summary_out.resolve().parts:
        raise SystemExit("--summary-out holds aggregates and must NOT be under local_only/")
    if a.sites == list(rsf.bb.STUDY_SITES) and cfg["cohort"].get("sites"):
        a.sites = list(cfg["cohort"]["sites"])
    A = rsf.assemble(a)
    rsf.attach_omop(A, a)
    frame, info = eligible_frame(A, a, sp["priority"])
    if len(frame) < 11:
        raise SystemExit("fewer than 11 IU-large patients: nothing can be sampled or reported")
    patterns = list(sp["priority"]) + [sampling.ALL_NEGATIVE]
    sample = sampling.draw_sample(frame, n=n, seed=seed, shares=sp["shares"], site_mode=sp["site_allocation"],
                                  dup_frac=float(sp["duplicate_fraction"]), n_calibration=int(sp["n_calibration"]))
    write_private_csv(sample_out, sample)
    sites = sorted(frame["site"].unique())
    labels = {s: A.site_labels.get(s, f"site_{i + 1}") for i, s in enumerate(sites)}
    known = {str(p) for p in frame["person_id"]} | set(sample["sample_id"])
    rep = summarize(sample, frame, labels, patterns, info, cfg, n, seed, A.frame["SiteID"].astype(str).value_counts().to_dict())
    safe_write_json(summary_out, rep, known)
    safe_print(BANNER, known_ids=known)
    sh = rep["iu_large_share_of_analysed"]["all_sites"]
    safe_print(f"analysed broad cohort n={rep['analysed_broad_cohort']['pooled']}; IU-large share {sh['share']}", known_ids=known)
    safe_print(f"IU-large eligible n={rep['eligible_iu_large']['pooled']} sampled n={rep['sampled']['pooled']} duplicates={rep['duplicates']} "
               f"calibration={rep['calibration']} (rule components: {', '.join(info['components_applied'])})", known_ids=known)
    if info["missing_conditions"]:
        safe_print("  IU-large defined WITHOUT: " + "; ".join(sorted(info["missing_conditions"])), known_ids=known)
    for site, row in rep["by_site_and_silver_pattern"].items():
        safe_print(f"  {site}: " + "  ".join(f"{p}: {c['sampled']}/{c['eligible']}" + (f" (w {c['weight']})" if c["weight"] != SUPPRESSED else "")
                                                for p, c in row.items()), known_ids=known)
    w = rep["weights"]
    safe_print(f"  weights: ESS {w['effective_sample_size']} (design effect {w['design_effect']}), median {w.get('q50')}; checks {rep['checks']}", known_ids=known)
    safe_print(f"Sample file (record-level, mode 0600, not printed): {sample_out}; aggregates: {summary_out}", known_ids=known)
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from sortinghat.cohort.memguard import run_guarded
    return run_guarded(lambda: run(a), a.max_memory_gb, "adjudication sampler")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"adjudication_sample failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
