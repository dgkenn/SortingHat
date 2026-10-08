#!/usr/bin/env bash
# Idempotent relauncher for the overnight drivers after a container restart. Starts each driver only if it is not
# already running and has not logged its completion line. Safe to call any number of times (SessionStart hook, check-ins).
cd "$(dirname "$0")/.." || exit 0
LOG=out/logs/overnight.log
mkdir -p out/logs
start() {  # $1 script, $2 completion marker in the log
  if ps -eo args | grep -qE "^(/bin/)?bash scripts/$1\$"; then return; fi
  if grep -q "$2" "$LOG" 2>/dev/null; then return; fi
  (setsid nohup "scripts/$1" > /dev/null 2>&1 < /dev/null &)
  echo "$(date -u +%FT%TZ) resume: relaunched $1" >> "$LOG"
}
start overnight_finish.sh   "driver done"
start overnight_stage2.sh   "stage2 done"
start overnight_stage3b.sh  "stage3b done"
exit 0
