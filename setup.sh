#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ ! -x "$DECIDER_ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.12 --managed-python "$DECIDER_ROOT/.venv"
fi
uv pip install --python "$DECIDER_ROOT/.venv/bin/python" \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$DECIDER_ROOT/.venv/bin/python" -r "$DECIDER_ROOT/requirements.txt"
"$DECIDER_ROOT/test-cli.sh"
