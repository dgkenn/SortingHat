#!/usr/bin/env bash
# Stage 3d: CBraMod + MORGOTH rungs for the BROAD cohort (4 shards, resumable). Same embed/morgoth dirs as 3c, so recordings
# already in the ledgers (strict cohort) are skipped and only new ones are processed. Success-gated.
set -u; cd "$(dirname "$0")/.."; STAGE=stage3d; . scripts/stage_lib.sh
KEYS=out/local_only/recording_keys_broad.csv   # scripts/make_recording_keys.py --cohort-def broad
rm -f out/logs/stage3d_rc_*
for k in 0 1 2 3; do
  ( step "rungs broad shard $k" 86400 "out/logs/rungs_broad_s$k.log" $RUN scripts/extract_rungs.py --input "$KEYS" \
      --embed-dir out/local_only/embeddings --morgoth-dir out/local_only/morgoth --shard "$k" --of 4 --torch-threads 1 --flush-every 2 --morgoth-heads normal,bs,spikes,slowing,spikeloc,iiic
    echo $? > "out/logs/stage3d_rc_$k" ) &
done
wait
for k in 0 1 2 3; do [ "$(cat out/logs/stage3d_rc_$k 2>/dev/null)" = "0" ] || FAILED=1; done
finish
