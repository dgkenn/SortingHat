#!/usr/bin/env bash
# In-container watchdog (no LLM): every 2 min, relaunch any stage driver that stopped before logging "<stage>: OK".
cd "$(dirname "$0")/.." || exit 0
LOG=out/logs/overnight.log
while true; do
  scripts/resume_overnight.sh
  if grep -q "driver done" "$LOG" && grep -q "stage2b: OK" "$LOG" && grep -q "stage3c: OK" "$LOG" && grep -q "stage4b: OK" "$LOG" \
     && grep -q "stage3d: OK" "$LOG" && grep -q "stage4d: OK" "$LOG" && grep -q "stage4c: OK" "$LOG"; then
    echo "$(date -u +%FT%TZ) watchdog v3: all stages done" >> "$LOG"; exit 0
  fi
  sleep 120
done
