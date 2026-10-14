#!/usr/bin/env python3
"""Gold adjudication pilot, step 4: agreement and silver-vs-gold analysis (D-157).   Aggregate-only output.

Reads the rater exports (flat REDCap CSV, one row per case x rater; state fields as codes or names), the id map and the sample file (weights,
strata, silver labels; all under local_only/) and reports, for every etiology label of configs/adjudication_crf.yaml:
  inter-rater   Cohen's kappa (binary: probable/definite = positive) with a stratified case-bootstrap 95% CI, PABAK, prevalence and bias
                indices, % agreement; the same re-weighted to IU-large with the sampling weights; linear-weighted kappa on the 4 states;
                Fleiss' kappa; categorical kappa for the primary etiology
  intra-rater   original vs duplicate packet, pooled and per rater
  gold          consensus (R1/R2 agree -> that call, else the third rater) prevalence, sample and re-weighted to IU-large
  silver vs gold  sensitivity, specificity, PPV, NPV of the silver label against the consensus gold, weighted by the sampling weights
  Gate 0        the plan's pilot criteria (kappa >= 0.6 in >= 4 primary families; >= 3 families with >= 10% prevalence), evaluated on the
                enriched sample prevalence (as the plan words it) AND on the re-weighted IU-large prevalence
  minutes per case, forms reporting EEG information (blinding check), internal-consistency counts
Every count < 11 is shown "<11" and a statistic whose cells are < 11 is suppressed (--min-cell may be LOWERED only by a human outside an
agent session).  The disagreement queue for the third rater is record-level and goes to a local_only file only.

    python3 scripts/adjudication_analyze.py --ratings <r1.csv> <r2.csv> [<r3.csv>] --id-map out/local_only/adjudication/id_map.csv \\
        --sample out/local_only/adjudication/sample.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sortinghat.adjudication import agreement as ag  # noqa: E402
from sortinghat.adjudication import crf  # noqa: E402
from sortinghat.adjudication.private_io import require_local_only, write_private_csv  # noqa: E402
from sortinghat.labels import pilot  # noqa: E402
from sortinghat.labels.ontology import primary_endpoint_labels  # noqa: E402
from sortinghat.safe_output import SUPPRESSED, SUPPRESS_BELOW, safe_print, safe_write_json, safe_write_text, suppress_count  # noqa: E402

BANNER = "GOLD ADJUDICATION PILOT (D-157) - agreement analysis; aggregate-only"
SILVER_LABELS = ("E1", "E2", "E4a", "E5", "E6", "E7")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", nargs="+", required=True, help="REDCap export CSV(s), one per rater project or one combined (local_only)")
    ap.add_argument("--id-map", default="out/local_only/adjudication/id_map.csv")
    ap.add_argument("--sample", default="out/local_only/adjudication/sample.csv")
    ap.add_argument("--crf", default=str(crf.DEFAULT_PATH))
    ap.add_argument("--adj-config", default=str(HERE.parent / "configs" / "adjudication.yaml"))
    ap.add_argument("--out", default="out/adjudication", help="aggregate report directory (must NOT be under local_only/)")
    ap.add_argument("--queue-out", default="out/local_only/adjudication/third_rater_queue.csv", help="record-level (local_only)")
    ap.add_argument("--main-raters", nargs=2, default=["R1", "R2"])
    ap.add_argument("--third-rater", default="R3")
    ap.add_argument("--fleiss-raters", nargs="+", default=None, help="raters for Fleiss' kappa (default: the two main raters)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--min-cell", type=int, default=SUPPRESS_BELOW, help="small-cell threshold (default 11; only a human outside an agent session may lower it)")
    return ap


def load_exports(paths) -> pd.DataFrame:
    frames = [pd.read_csv(require_local_only(p, "--ratings"), dtype=str) for p in paths]
    return pd.concat(frames, ignore_index=True)


def amended_primary_block(per_label: dict, labels, kappa_min: float = 0.6, prev_min: float = 0.10) -> dict:
    """SAP Amendment A1 primary labels: per-label kappa and re-weighted gold prevalence against the plan's two Gate 0 thresholds.  Arithmetic
    only: how Gate 0's 'at least 4 families' reads on a 3-label set is open item [OP-A1]."""
    out = {}
    for lab in labels:
        b, g = per_label[lab]["inter_rater"], per_label[lab]["gold"]
        kap, prev = b["kappa"], g["prevalence_weighted"]
        out[lab] = {"kappa": kap, "kappa_ci95": b["kappa_ci95"], "prevalence_iu_large": prev, "prevalence_sample": g["prevalence_sample"],
                    "kappa_meets_threshold": bool(kap >= kappa_min) if isinstance(kap, float) else None,
                    "prevalence_meets_threshold": bool(prev >= prev_min) if isinstance(prev, float) else None}
    return {"labels": out, "kappa_threshold": kappa_min, "prevalence_threshold": prev_min,
            "n_kappa_meets": sum(v["kappa_meets_threshold"] is True for v in out.values()),
            "n_prevalence_meets": sum(v["prevalence_meets_threshold"] is True for v in out.values()),
            "n_labels": len(out), "note": "SAP Amendment A1 primary labels; no pass/fail is called here ([OP-A1])."}


def analyze(export: pd.DataFrame, id_map: pd.DataFrame, sample: pd.DataFrame, cfg: dict, *, main=("R1", "R2"), third="R3",
            fleiss_raters=None, B=2000, seed=0, k=SUPPRESS_BELOW, amended=("E1", "E2", "E5"), kappa_min=0.6, prev_min=0.10
            ) -> tuple[dict, pd.DataFrame]:
    """(aggregate report, third-rater queue [RECORD-LEVEL]).  Pure function of its inputs."""
    labels = crf.label_codes(cfg)
    long_all = crf.long_ratings(export, cfg)
    cf_all = crf.case_fields(export, cfg)
    im = id_map.set_index("case_id")
    known = set(im.index)
    unmatched = int((~long_all["case_id"].isin(known)).sum() + (~cf_all["case_id"].isin(known)).sum())
    long_all = long_all[long_all["case_id"].isin(known)]
    cf_all = cf_all[cf_all["case_id"].isin(known)]
    role = im["role"]
    prim_cases = set(role[role == "sample"].index)
    long = long_all[long_all["case_id"].isin(prim_cases)]
    cf = cf_all[cf_all["case_id"].isin(prim_cases)]
    smp = sample.set_index("sample_id")
    case_w = im.loc[im.index.isin(prim_cases), "sample_id"].map(smp["weight"]).astype(float)
    case_stratum = im.loc[im.index.isin(prim_cases), "sample_id"].map(smp["stratum"]).astype(str)
    raters = sorted(set(long["rater"]))
    rep: dict = {"banner": BANNER, "n_rater_forms": suppress_count(len(cf_all), k), "raters_present": raters,
                 "unmatched_rows_dropped": suppress_count(unmatched, k), "min_cell": k,
                 "design_note": ("Cases are an enriched sample of IU-large (strata = site x silver pattern). Unweighted statistics describe the "
                                 "sample; 'weighted' ones are re-weighted to IU-large with the sampling weights.")}
    if not set(main) <= set(raters):
        rep["error"] = f"ratings from both main raters {list(main)} are required"
        return rep, pd.DataFrame(columns=["case_id", "labels_in_dispute"])
    panel = ag.Panel(long, labels, [r for r in (*main, third) if r in raters])
    w = case_w.reindex(panel.cases).fillna(1.0).to_numpy(float)
    strata = case_stratum.reindex(panel.cases).fillna("").to_numpy(object)
    pm = ag.Panel(long, labels, list(main))
    inter = ag.interrater(pm, *main, weights=None, strata=strata_for(pm, case_stratum), B=B, seed=seed, k=k)
    interw = ag.interrater(pm, *main, weights=weights_for(pm, case_w), strata=strata_for(pm, case_stratum), B=B, seed=seed, k=k)
    fr = list(fleiss_raters or main)
    pf = ag.Panel(long, labels, fr)
    fl = ag.fleiss_report(pf, fr, strata=strata_for(pf, case_stratum), B=max(B // 2, 200), seed=seed, k=k)
    pairs = ag.duplicate_pairs(long_all, id_map)
    intra = ag.intrarater(long_all, pairs, labels, [r for r in main if r in raters], seed=seed, B=B, k=k)
    cons = ag.consensus(panel, main, third)
    sm = smp.reindex(im.loc[panel.cases, "sample_id"].to_numpy()) if len(panel.cases) else smp.iloc[0:0]
    per_label = {}
    for lab in labels:
        gold, src_ = cons[lab]
        entry = {"inter_rater": inter[lab]["binary"], "inter_rater_weighted_to_iu_large": interw[lab]["binary"],
                 "inter_rater_ordinal": inter[lab]["ordinal"], "fleiss": fl[lab],
                 "consensus_sources": {s: suppress_count(int((src_ == s).sum()), k) for s in ("agree", "third", "majority", "pending", "missing")},
                 "gold": ag.prevalence_block(gold, w, strata, B=B, seed=seed, k=k)}
        if lab in SILVER_LABELS and f"silver_{lab}" in sm:
            silver = pd.to_numeric(sm[f"silver_{lab}"], errors="coerce").to_numpy(float)
            entry["silver_vs_gold"] = ag.silver_vs_gold(gold, silver, w, strata, B=B, seed=seed, k=k)
        per_label[lab] = entry
    rep["labels"] = per_label
    cats = [c["code"] for c in cfg["primary_etiology_choices"]]
    rep["primary_etiology"] = ag.primary_agreement(cf, main[0], main[1], cats, strata_by_case=case_stratum.to_dict(), B=B, seed=seed, k=k)
    rep["intra_rater"] = {"n_duplicate_pairs": suppress_count(len(pairs), k), **intra}
    # ---- Gate 0 (plan): kappa >= 0.6 in >= 4 primary families and >= 3 families with >= 10% prevalence
    prim = [l for l in primary_endpoint_labels() if l in labels]
    ratings_for_gate = long.rename(columns={"rater": "reviewer"})[["case_id", "label", "reviewer", "state"]]
    pk = pilot.label_kappas(ratings_for_gate, prim, reviewers=tuple(main), min_n=k)
    rep["gate0"] = {"on_enriched_sample": pilot.gate0_pilot_check(pk, prim),
                    "on_iu_large_weighted_prevalence": pilot.gate0_pilot_check(
                        {l: {**pk.get(l, {}), "prevalence": per_label[l]["gold"].get("prevalence_weighted")} for l in prim}, prim),
                    "note": ("The plan's prevalence criterion is evaluated on the enriched pilot sample, whose rare etiologies are oversampled by "
                             "design; the second block re-weights to IU-large. E7 is exploratory (D-004) and not in the primary families.")}
    rep["amended_primary_labels"] = amended_primary_block(per_label, [l for l in amended if l in per_label], kappa_min, prev_min)
    if "minutes_spent" in cf_all:
        mins = cf_all.assign(minutes=pd.to_numeric(cf_all["minutes_spent"], errors="coerce")).dropna(subset=["minutes"])
        rep["minutes_per_case"] = pilot.minutes_summary(mins.rename(columns={"rater": "reviewer"})[["case_id", "reviewer", "minutes"]])
    rep["consistency"] = ag.consistency(long, cf, k)
    rep["blinding_check"] = ag.blinding_check(cf_all, k)
    queue = ag.disagreement_queue(pm, main)
    rep["third_rater_queue"] = {"n_cases": suppress_count(len(queue), k), "note": "the case list is record-level: local_only file only"}
    return rep, queue


def weights_for(panel, case_w) -> np.ndarray:
    return case_w.reindex(panel.cases).fillna(1.0).to_numpy(float)


def strata_for(panel, case_stratum) -> np.ndarray:
    return case_stratum.reindex(panel.cases).fillna("").to_numpy(object)


def _f(x) -> str:
    return "<11" if x == SUPPRESSED or x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def render_markdown(rep: dict) -> str:
    L = [f"# {BANNER}", "", rep.get("design_note", ""), ""]
    if "error" in rep:
        return "\n".join(L + [rep["error"]]) + "\n"
    L += ["## Inter-rater agreement (binary: probable or definite = positive)", "",
          "| Label | pairs | positive either | kappa | 95% CI | PABAK | % agree | kappa (re-weighted) | Fleiss | weighted linear kappa |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, e in rep["labels"].items():
        b, w, o, fl = e["inter_rater"], e["inter_rater_weighted_to_iu_large"], e["inter_rater_ordinal"], e["fleiss"]
        ci = b["kappa_ci95"]
        L.append(f"| {lab} | {_f(b['n_pairs'])} | {_f(b['n_positive_either'])} | {_f(b['kappa'])} | "
                 f"{('[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']') if isinstance(ci, list) else '<11'} | {_f(b['pabak'])} | {_f(b['percent_agreement'])} | "
                 f"{_f(w['kappa'])} | {_f(fl['fleiss_kappa'])} | {_f(o.get('kappa_linear_weighted'))} |")
    pe = rep["primary_etiology"]
    L += ["", f"Primary etiology (10 categories): pairs {_f(pe.get('n_pairs'))}, kappa {_f(pe.get('kappa'))}, % agreement {_f(pe.get('percent_agreement'))}.", "",
          "## Gold prevalence and silver vs gold (weights re-weight to IU-large)", "",
          "| Label | gold positive n | prevalence (sample) | prevalence (IU-large) | silver sensitivity | specificity | PPV | NPV |", "|---|---|---|---|---|---|---|---|"]
    for lab, e in rep["labels"].items():
        g, s = e["gold"], e.get("silver_vs_gold")
        cell = lambda m: _f(s[m]["weighted"]) if s else "n/a"      # noqa: E731
        L.append(f"| {lab} | {_f(g['n_gold_positive'])} | {_f(g['prevalence_sample'])} | {_f(g['prevalence_weighted'])} | "
                 f"{cell('sensitivity')} | {cell('specificity')} | {cell('ppv')} | {cell('npv')} |")
    g0 = rep["gate0"]
    L += ["", "## Gate 0 (pilot)", "", f"- enriched sample: {g0['on_enriched_sample']}", f"- re-weighted to IU-large: {g0['on_iu_large_weighted_prevalence']}", f"- {g0['note']}"]
    am = rep["amended_primary_labels"]
    L += ["", "## SAP Amendment A1 primary labels (E1, E2, E5)", "", "| Label | kappa | 95% CI | gold prevalence (IU-large) | kappa >= " + str(am["kappa_threshold"]) + " | prevalence >= " + f"{am['prevalence_threshold']:.0%}" + " |", "|---|---|---|---|---|---|"]
    for lab, v in am["labels"].items():
        ci = v["kappa_ci95"]
        L.append(f"| {lab} | {_f(v['kappa'])} | {('[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']') if isinstance(ci, list) else '<11'} | {_f(v['prevalence_iu_large'])} | "
                 f"{v['kappa_meets_threshold']} | {v['prevalence_meets_threshold']} |")
    L += ["", f"{am['n_kappa_meets']} of {am['n_labels']} meet the kappa threshold, {am['n_prevalence_meets']} of {am['n_labels']} the prevalence threshold. {am['note']}"]
    L += ["", "## Other", "", f"- intra-rater duplicate pairs: {_f(rep['intra_rater']['n_duplicate_pairs'])}",
          f"- forms reporting EEG information in the packet: {rep['blinding_check']}",
          f"- internal consistency: {rep['consistency']}", f"- third-rater queue: {_f(rep['third_rater_queue']['n_cases'])} cases"]
    if "minutes_per_case" in rep:
        L.append(f"- minutes per review: {rep['minutes_per_case']['overall']}")
    return "\n".join(L) + "\n"


def run(a) -> int:
    k = int(a.min_cell)
    if k < SUPPRESS_BELOW and (os.environ.get("CLAUDECODE") == "1" or os.environ.get("SORTINGHAT_AGENT_SESSION") == "1"):
        raise SystemExit("--min-cell below 11 is refused inside an agent session (CLAUDE.md rule 3)")
    out = Path(a.out)
    if "local_only" in out.resolve().parts:
        raise SystemExit("--out holds aggregates and must NOT be under local_only/")
    require_local_only(a.queue_out, "--queue-out")
    cfg = crf.load_crf(a.crf)
    an = (yaml.safe_load(Path(a.adj_config).read_text()) or {}).get("analysis", {})
    export = load_exports(a.ratings)
    id_map = pd.read_csv(require_local_only(a.id_map, "--id-map"), dtype={"case_id": str, "sample_id": str, "duplicate_of_case": str})
    id_map["duplicate_of_case"] = id_map["duplicate_of_case"].fillna("")
    sample = pd.read_csv(require_local_only(a.sample, "--sample"), dtype={"sample_id": str})
    rep, queue = analyze(export, id_map, sample, cfg, main=tuple(a.main_raters), third=a.third_rater, fleiss_raters=a.fleiss_raters,
                         B=a.boot, seed=a.seed, k=k, amended=tuple(an.get("amended_primary_labels", ("E1", "E2", "E5"))),
                         kappa_min=float(an.get("kappa_threshold", 0.6)), prev_min=float(an.get("prevalence_threshold", 0.10)))
    known = set(id_map["case_id"]) | set(id_map["sample_id"]) | {str(p) for p in id_map["person_id"]}
    safe_write_json(out / "agreement_report.json", rep, known)
    safe_write_text(out / "agreement_report.md", render_markdown(rep), known)
    if len(queue):
        write_private_csv(a.queue_out, queue)
    safe_print(BANNER, known_ids=known)
    if "error" in rep:
        safe_print(rep["error"], known_ids=known)
        return 2
    safe_print(f"forms={rep['n_rater_forms']} raters={','.join(rep['raters_present'])} third-rater queue={rep['third_rater_queue']['n_cases']} cases", known_ids=known)
    for lab, e in rep["labels"].items():
        b = e["inter_rater"]
        safe_print(f"  {lab}: pairs {_f(b['n_pairs'])} kappa {_f(b['kappa'])} PABAK {_f(b['pabak'])}; gold prevalence (IU-large) {_f(e['gold']['prevalence_weighted'])}", known_ids=known)
    safe_print(f"  Gate 0 (plan families, enriched sample): {rep['gate0']['on_enriched_sample']['pass']}; (IU-large weighted): {rep['gate0']['on_iu_large_weighted_prevalence']['pass']}", known_ids=known)
    am = rep["amended_primary_labels"]
    safe_print(f"  Amendment A1 labels: kappa >= {am['kappa_threshold']} in {am['n_kappa_meets']} of {am['n_labels']}, prevalence >= {am['prevalence_threshold']:.0%} in {am['n_prevalence_meets']} of {am['n_labels']}", known_ids=known)
    safe_print(f"Report: {out / 'agreement_report.md'} and {out / 'agreement_report.json'} (aggregate-only)", known_ids=known)
    return 0


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"adjudication_analyze failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
