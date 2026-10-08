#!/usr/bin/env bash
# Finisher for tonight's run (replaces the tail of overnight.sh after it was edited while running).
# Waits for the in-flight silver-label job, warms the shared OMOP cache (D-149), then releases stage 2
# (which gates on the "silver labels exit=" log line) and runs the Phase 0a field audit from the cache.
set -u
cd "$(dirname "$0")/.."
LOG=out/logs/overnight.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"

say "finisher: waiting for the running silver-label job"
while pgrep -f "sortinghat.labels.extract --s3" > /dev/null; do sleep 60; done
# the in-flight job started 2026-10-08 19:07 UTC; a report written after that means it completed
if [ -n "$(find out/silver/silver_report.json -newermt '2026-10-08 19:07' 2>/dev/null)" ]; then status=0; else status=1; fi

say "omop cache warm-up start (shared by audit, cohort and baselines; resumable)"
timeout 21600 $RUN -m sortinghat.omop_cache warm --s3 >> out/logs/omop_cache_warm.log 2>&1
say "omop cache warm-up exit=$?"

if [ "$status" -ne 0 ]; then
  say "silver labels (re)start"
  timeout 21600 $RUN -m sortinghat.labels.extract --s3 --cohort out/local_only/cohort_study1.csv \
    --out out/silver --labels-out out/local_only/silver/silver_labels.csv >> out/logs/silver.log 2>&1
  status=$?
fi
say "silver labels exit=$status"

say "field audit start"
timeout 21600 $RUN -m sortinghat.audit.field_audit --s3 --out out/audit --workers 3 >> out/logs/field_audit.log 2>&1
say "field audit exit=$?"
say "driver done"
