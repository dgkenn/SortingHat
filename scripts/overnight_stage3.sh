#!/usr/bin/env bash
# Stage 3: frozen-representation rungs over the cohort recordings (CBraMod embeddings, MORGOTH findings), after stage 2.
# Both extractors are streaming, sharded and resumable; record-level outputs stay under out/local_only/.
set -u
cd "$(dirname "$0")/.."
LOG=out/logs/overnight.log
say() { echo "$(date -u +%FT%TZ) stage3: $*" >> "$LOG"; }
RUN="env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3"
KEYS=out/local_only/recording_keys.csv

say "waiting for overnight.sh and overnight_stage2.sh"
while pgrep -f "scripts/overnight.sh" > /dev/null || pgrep -f "scripts/overnight_stage2.sh" > /dev/null; do sleep 120; done

say "cbramod embeddings start"
for k in 0 1 2 3; do
  ( $RUN scripts/extract_embeddings.py --input "$KEYS" --out-dir out/local_only/embeddings --shard "$k" --of 4 \
      --torch-threads 1 >> "out/logs/embed_s$k.log" 2>&1 ) &
done
wait
say "cbramod embeddings done"

say "morgoth findings start"
for k in 0 1 2 3; do
  ( $RUN scripts/extract_morgoth.py features --input "$KEYS" --out-dir out/local_only/morgoth --shard "$k" --of 4 \
      >> "out/logs/morgoth_s$k.log" 2>&1 ) &
done
wait
say "morgoth findings done"
say "stage3 done"
