#!/usr/bin/env bash
# Stage 2 of the overnight run (D-143 exploratory silver-label feasibility). Waits for scripts/overnight.sh to finish,
# then builds Baselines A-C and runs the feasibility analysis. Aggregate-only outputs; record-level files in local_only.
set -u
cd "$(dirname "$0")/.."
LOG=out/logs/overnight.log
say() { echo "$(date -u +%FT%TZ) stage2: $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"

say "waiting for overnight.sh"
while pgrep -f "scripts/overnight.sh" > /dev/null; do sleep 120; done

say "baselines start"
timeout 21600 $RUN scripts/build_baselines.py --s3 --cohort out/local_only/cohort_study1.csv \
  --out out/local_only/baselines_AC.parquet > out/logs/baselines.log 2>&1
say "baselines exit=$?"

say "silver feasibility start"
timeout 21600 $RUN scripts/run_silver_feasibility.py --s3 > out/logs/silver_feasibility.log 2>&1
say "silver feasibility exit=$?"
say "stage2 done"
