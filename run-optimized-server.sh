#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec "$DECIDER_ROOT/.venv/bin/python" "$DECIDER_ROOT/optimized-server.py" "$@"
