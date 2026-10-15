#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec "$DECIDER_ROOT/.venv/bin/strands-decider" serve "$DECIDER_MODEL" \
  --device cpu --host 127.0.0.1 --port "${DECIDER_PORT:-8099}" "$@"
