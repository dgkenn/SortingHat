#!/usr/bin/env bash
# Run a HEEDB/BDSP job with the right AWS credential source.
# Ported from dgkenn/codex-playground- scripts/heedb_run.sh (research-program-continuation branch).
#
# HUMAN-RUN ONLY. Per CLAUDE.md rule 2, jobs that touch restricted HEEDB data run from a plain
# terminal or scheduler, never from inside an agent session. This wrapper refuses to start if it
# detects one (CLAUDECODE=1 or SORTINGHAT_AGENT_SESSION=1).
#
# WHY IT EXISTS. Some sandboxes (the Claude Code cloud container was the case that cost hours)
# export PLACEHOLDER AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY values for an outbound proxy.
# boto3 resolves static environment credentials BEFORE profile credentials, so every call silently
# authenticates as the stub and the BDSP access point answers 403 / InvalidAccessKeyId -- which
# reads exactly like expired credentials and is not (source catalogue rules 8 and 36).
# Removing the stub variables for the child process lets the normal chain find ~/.aws/credentials.
# On your own machine there is no stub; the wrapper is harmless there.
#
# AWS_CA_BUNDLE and HTTPS_PROXY are deliberately left alone: a proxy, if present, still carries
# the traffic.
#
# Optional environment (names only; this script never reads or prints a credential value):
#   HEEDB_AWS_PROFILE   AWS profile to use (exported to the child as AWS_PROFILE). The source
#                       project used a profile literally named "physionet".
#
# Usage:
#   scripts/heedb_run.sh python -m sortinghat.audit.field_audit --data <dir> --out <dir>
#   HEEDB_AWS_PROFILE=physionet scripts/heedb_run.sh python your_script.py
#
# Diagnose a 403 per credential source, never globally (source rule 8):
#   scripts/heedb_run.sh python -c "import boto3; print(boto3.Session().client('sts').get_caller_identity()['Arn'])"
#   (a 403 WITH the wrapper is a real credential problem; WITHOUT it, usually the stub collision)
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 64
fi

if [ "${CLAUDECODE:-}" = "1" ] || [ "${SORTINGHAT_AGENT_SESSION:-}" = "1" ]; then
    echo "heedb_run.sh: refusing to run inside an agent session (CLAUDE.md rule 2)." >&2
    echo "Run this from a plain terminal or scheduler." >&2
    exit 77
fi

if [ -n "${HEEDB_AWS_PROFILE:-}" ]; then
    export AWS_PROFILE="$HEEDB_AWS_PROFILE"
fi

# Drop the ambient AWS_* variables only when the key id provably is NOT a real AWS key id
# (real ones are 20 characters beginning AKIA or ASIA; the sandbox stub was 14 characters beginning
# "prox"). This is the same test as the source project's common/awsenv.py. Unlike the source
# wrapper (which unset them unconditionally), a genuine key exported under the standard names is
# left alone, so the wrapper cannot silently discard credentials you meant to use.
key="${AWS_ACCESS_KEY_ID:-}"
if [ -n "$key" ] && ! [[ "$key" =~ ^(AKIA|ASIA)[A-Z0-9]{16}$ ]]; then
    echo "heedb_run.sh: ambient AWS_ACCESS_KEY_ID is not an AWS key id (length ${#key}); dropping AWS_* for the child." >&2
    exec env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN "$@"
fi
exec "$@"
