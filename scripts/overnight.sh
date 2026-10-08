#!/usr/bin/env bash
# Overnight driver for the real-data Phase 0 / Study 1 steps (HUMAN-AUTHORISED RUN; aggregate-only outputs).
# Resumable: every step skips work already done, so the driver can be relaunched at any point.
#   1. streaming EEG feature extraction over the cohort key list (12 shards), then not_found retries
#   2. Phase 0a field audit (needs the network to itself, so it runs after extraction)
#   3. structured silver labels for the cohort
# Logs: out/logs/overnight.log (+ per-step logs). Record-level outputs stay under out/local_only/.
set -u
cd "$(dirname "$0")/.."
LOG=out/logs/overnight.log
mkdir -p out/logs
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"
EXTRACT="$RUN scripts/extract_eeg_features.py --input out/local_only/recording_keys.csv --out-dir out/local_only/features --of 12"

say "driver start"
# never run two writers on the same ledger: wait for any extraction already running
while pgrep -f extract_eeg_features.py > /dev/null; do sleep 60; done

progress() {
  python3 - <<'EOF'
import glob, csv, collections
seen = {}
for f in sorted(glob.glob('out/local_only/features/ledger-*.csv')):
    for r in csv.DictReader(open(f)):
        seen[r['recording_id']] = (r['status'], r['reason'])
c = collections.Counter(seen.values())
print(len(seen), dict((f"{k[0]}:{k[1]}", (v if v >= 11 else "<11")) for k, v in c.items()))
EOF
}

for pass in 1 2 3; do
  say "extraction pass $pass start: $(progress)"
  for k in $(seq 0 11); do
    ( $EXTRACT --shard "$k" >> "out/logs/overnight_extract_s$k.log" 2>&1
      $EXTRACT --shard "$k" --retry-reason not_found >> "out/logs/overnight_retry_s$k.log" 2>&1 ) &
  done
  wait
  say "extraction pass $pass end: $(progress)"
done

say "silver labels start"
timeout 21600 $RUN -m sortinghat.labels.extract --s3 --cohort out/local_only/cohort_study1.csv \
  --out out/silver --labels-out out/local_only/silver/silver_labels.csv > out/logs/silver.log 2>&1
say "silver labels exit=$?"
say "field audit start"
timeout 21600 $RUN -m sortinghat.audit.field_audit --s3 --out out/audit --workers 3 > out/logs/field_audit.log 2>&1
say "field audit exit=$?"

say "driver done"
