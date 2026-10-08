#!/usr/bin/env bash
# Stage 4: after the frozen-representation rungs exist (stage 3b), rerun the exploratory silver-label feasibility analysis so
# the ladder includes cbramod_frozen and morgoth_findings (commercial-clean gap = delta(CBraMod) vs delta(MORGOTH)).
# Checkpointed (per scheme/fold/rung), so a relaunch resumes. Aggregate-only outputs.
set -u
cd "$(dirname "$0")/.."
LOG=out/logs/overnight.log
say() { echo "$(date -u +%FT%TZ) stage4: $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"

say "waiting for stage3b"
until grep -q "stage3b done" "$LOG" 2>/dev/null; do sleep 300; done
say "feasibility with frozen rungs start"
timeout 21600 $RUN scripts/run_silver_feasibility.py --s3 --out out/silver_feasibility_rungs >> out/logs/silver_feasibility_rungs.log 2>&1
say "feasibility with frozen rungs exit=$?"
say "stage4 done"
