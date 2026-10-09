#!/usr/bin/env bash
# Idempotent relauncher (SessionStart hook, watchdog, check-ins). Starts each stage driver unless it is running, has logged
# success ("<stage>: OK"), or has failed 3 times (then it waits for a human/agent fix and logs "gave up" once).
cd "$(dirname "$0")/.." || exit 0
LOG=out/logs/overnight.log; mkdir -p out/logs
start() {  # $1 script  $2 stage name (as logged)  [$3 legacy success marker]
  ps -eo args | grep -qE "^(/bin/)?bash scripts/$1\$" && return
  grep -q "$2: OK" "$LOG" 2>/dev/null && return
  [ -n "${3:-}" ] && grep -q "$3" "$LOG" 2>/dev/null && return
  local fails; fails=$(grep -c "$2: FAILED" "$LOG" 2>/dev/null)
  if [ "${fails:-0}" -ge 3 ]; then
    grep -q "resume: $2 gave up" "$LOG" || echo "$(date -u +%FT%TZ) resume: $2 gave up after 3 failures" >> "$LOG"; return
  fi
  (setsid nohup "scripts/$1" > /dev/null 2>&1 < /dev/null &)
  echo "$(date -u +%FT%TZ) resume: relaunched $1" >> "$LOG"
}
start overnight_finish.sh finisher "driver done"
start stage2b.sh stage2b
start stage3c.sh stage3c
start stage4b.sh stage4b
start stage3d.sh stage3d
start stage4d.sh stage4d
start stage4c.sh stage4c
if ! ps -eo args | grep -qE "^(/bin/)?bash scripts/watchdog.sh\$" && ! grep -q "watchdog v3: all stages done" "$LOG" 2>/dev/null; then
  (setsid nohup scripts/watchdog.sh > /dev/null 2>&1 < /dev/null &)
fi
exit 0
