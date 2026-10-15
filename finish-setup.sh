#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$DECIDER_ROOT"
mkdir -p logs
exec 9>logs/setup.lock
flock -n 9 || { echo "Setup is already running."; exit 0; }
trap 'echo "Setup failed. Check logs/base-download.log and logs/verification.log."' ERR
echo "Resuming local model setup at $(date -Is)."
"$DECIDER_ROOT/.venv/bin/python" "$DECIDER_ROOT/download-base.py" >> logs/base-download.log 2>&1
"$DECIDER_ROOT/verify-local.sh" >> logs/verification.log 2>&1
echo "Local model setup and verification finished at $(date -Is)."
