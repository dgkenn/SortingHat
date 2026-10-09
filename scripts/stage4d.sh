#!/usr/bin/env bash
# Stage 4d: strict-cohort feasibility rerun (v3) once the reviewed code is released (out/logs/code_ready_v3.ok is created after code review).
set -u; cd "$(dirname "$0")/.."; STAGE=stage4d; . scripts/stage_lib.sh
until [ -e out/logs/code_ready_v3.ok ]; do sleep 120; done
step feasibility_strict_v3 21600 out/logs/silver_feasibility_v3.log $RUN scripts/run_silver_feasibility.py --s3 --out out/silver_feasibility_v3 || finish
finish
