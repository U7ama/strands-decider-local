#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$DECIDER_ROOT/.venv/bin/strands-decider" ask "$DECIDER_MODEL" \
  --device cpu \
  --state "A classroom projector is offline. Its power indicator is on. The upstream switch reports 70 percent packet loss and other devices on that switch are unreachable." \
  --choice "Which specialist should investigate first?=network,system,power,human_review"
