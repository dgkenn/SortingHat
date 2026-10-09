#!/usr/bin/env bash
# Stage 4b: rerun the feasibility analysis with the frozen CBraMod and MORGOTH rungs (commercial-clean gap), after 2b and 3c.
set -u; cd "$(dirname "$0")/.."; STAGE=stage4b; . scripts/stage_lib.sh
until grep -q "stage2b: OK" "$LOG" && grep -q "stage3c: OK" "$LOG"; do sleep 300; done
step feasibility_rungs 21600 out/logs/silver_feasibility_rungs.log $RUN scripts/run_silver_feasibility.py --s3 --out out/silver_feasibility_rungs || finish
finish
