#!/usr/bin/env bash
# Stage 3c: single-pass CBraMod + MORGOTH extraction over the cohort recordings (4 shards, resumable). Success-gated.
set -u; cd "$(dirname "$0")/.."; STAGE=stage3c; . scripts/stage_lib.sh
KEYS=out/local_only/recording_keys_strict.csv   # D-150: strict cohort first (CPU-bound)
rm -f out/logs/stage3c_rc_*
for k in 0 1 2 3; do
  ( step "rungs shard $k" 21600 "out/logs/rungs_s$k.log" $RUN scripts/extract_rungs.py --input "$KEYS" \
      --embed-dir out/local_only/embeddings --morgoth-dir out/local_only/morgoth --shard "$k" --of 4 --torch-threads 1 --morgoth-heads normal,bs,spikes,slowing,spikeloc,iiic
    echo $? > "out/logs/stage3c_rc_$k" ) &
done
wait
for k in 0 1 2 3; do [ "$(cat out/logs/stage3c_rc_$k 2>/dev/null)" = "0" ] || FAILED=1; done
finish
