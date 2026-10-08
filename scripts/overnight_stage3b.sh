#!/usr/bin/env bash
# Stage 3b (replaces overnight_stage3.sh): frozen-representation rungs over the cohort recordings (CBraMod embeddings, MORGOTH findings), after stage 2.
# ONE extractor (scripts/extract_rungs.py) streams each recording once and runs CBraMod + MORGOTH on it; it is sharded and resumable, and writes the same outputs as extract_embeddings.py / extract_morgoth.py; record-level outputs stay under out/local_only/.
set -u
cd "$(dirname "$0")/.."
LOG=out/logs/overnight.log
say() { echo "$(date -u +%FT%TZ) stage3b: $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"
KEYS=out/local_only/recording_keys.csv

say "waiting for overnight.sh and overnight_stage2.sh"
while pgrep -f "scripts/overnight.sh" > /dev/null || pgrep -f "scripts/overnight_stage2.sh" > /dev/null; do sleep 120; done

say "rungs (cbramod + morgoth, one fetch per recording) start"
for k in 0 1 2 3; do
  ( $RUN scripts/extract_rungs.py --input "$KEYS" --embed-dir out/local_only/embeddings --morgoth-dir out/local_only/morgoth \
      --shard "$k" --of 4 --torch-threads 1 >> "out/logs/rungs_s$k.log" 2>&1 ) &
done
wait
say "rungs done"
say "stage3b done"
