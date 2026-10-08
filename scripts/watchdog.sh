#!/usr/bin/env bash
# In-container watchdog (no LLM involved): every 2 min, relaunch any overnight driver that died before logging completion.
# Started by scripts/resume_overnight.sh (SessionStart hook) and exits once every stage has finished.
cd "$(dirname "$0")/.." || exit 0
LOG=out/logs/overnight.log
while true; do
  scripts/resume_overnight.sh
  if grep -q "driver done" "$LOG" && grep -q "stage2 done" "$LOG" && grep -q "stage3b done" "$LOG"; then
    echo "$(date -u +%FT%TZ) watchdog: all stages done, exiting" >> "$LOG"; exit 0
  fi
  sleep 120
done
