#!/usr/bin/env bash
# Stage 2 of the overnight run (D-143 exploratory silver-label feasibility). Waits for scripts/overnight.sh to finish,
# then builds Baselines A-C and runs the feasibility analysis. Aggregate-only outputs; record-level files in local_only.
set -u
cd "$(dirname "$0")/.."
LOG=out/logs/overnight.log
say() { echo "$(date -u +%FT%TZ) stage2: $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"

say "waiting for overnight.sh"
until grep -q "silver labels exit=" out/logs/overnight.log 2>/dev/null && [ "$(grep -c "silver labels exit=" out/logs/overnight.log)" -gt "$(grep -c "stage2: silver gate passed" out/logs/overnight.log)" ]; do sleep 120; done
say "silver gate passed"

say "cohort rebuild start (adds encounter_start for the D-145 undifferentiated subgroup)"
timeout 7200 $RUN scripts/build_cohort.py --s3 --out out --debug-flow --max-memory-gb 5 > out/logs/build_cohort.log 2>&1
say "cohort rebuild exit=$?"

say "baselines start"
timeout 21600 $RUN scripts/build_baselines.py --s3 --cohort out/local_only/cohort_study1.csv \
  --out out/local_only/baselines_AC.parquet > out/logs/baselines.log 2>&1
say "baselines exit=$?"

say "silver feasibility start"
timeout 21600 $RUN scripts/run_silver_feasibility.py --s3 > out/logs/silver_feasibility.log 2>&1
say "silver feasibility exit=$?"
say "stage2 done"
