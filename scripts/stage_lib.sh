# Shared helpers for the overnight stage drivers: success-gated markers and bounded retries.
LOG=out/logs/overnight.log
say() { echo "$(date -u +%FT%TZ) $STAGE: $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"
FAILED=0
step() {  # step <name> <timeout_s> <logfile> <cmd...>; records failure, never marks success on error
  local name=$1 t=$2 lf=$3; shift 3
  say "$name start"
  timeout "$t" "$@" >> "$lf" 2>&1; local rc=$?
  say "$name exit=$rc"
  [ "$rc" -eq 0 ] || FAILED=1
  return $rc
}
finish() { if [ "$FAILED" -eq 0 ]; then say "OK"; else say "FAILED"; exit 1; fi; }
