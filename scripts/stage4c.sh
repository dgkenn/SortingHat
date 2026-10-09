#!/usr/bin/env bash
# Stage 4c: silver feasibility on the BROAD cohort with the frozen CBraMod and MORGOTH rungs, after 3d and 4d and the lead releases reviewed code (out/logs/code_ready_v4.ok).
set -u; cd "$(dirname "$0")/.."; STAGE=stage4c; . scripts/stage_lib.sh
until grep -q "stage3d: OK" "$LOG" && grep -q "stage4d: OK" "$LOG" && [ -e out/logs/code_ready_v4.ok ]; do sleep 300; done
step feasibility_broad 43200 out/logs/silver_feasibility_broad.log $RUN scripts/run_silver_feasibility.py --s3 --cohort-def broad --out out/silver_feasibility_broad || finish
finish
