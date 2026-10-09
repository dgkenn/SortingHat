#!/usr/bin/env bash
# Stage 2b: Baselines P/A/B/C (resumable), then -- once the final silver labels exist (marker out/logs/silver_final.ok) --
# the exploratory silver-label feasibility analysis. Success-gated: logs "stage2b: OK" only if every step exited 0.
set -u; cd "$(dirname "$0")/.."; STAGE=stage2b; . scripts/stage_lib.sh
step baselines 21600 out/logs/baselines.log $RUN scripts/build_baselines.py --s3 --cohort out/local_only/cohort_study1.csv --out out/local_only/baselines_AC.parquet || finish
say "waiting for out/logs/silver_final.ok"
until [ -f out/logs/silver_final.ok ]; do sleep 120; done
step feasibility 21600 out/logs/silver_feasibility.log $RUN scripts/run_silver_feasibility.py --s3 || finish
finish
