#!/usr/bin/env bash
# Source this file, or use one of the launch scripts.
DECIDER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$DECIDER_ROOT/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR="$DECIDER_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$DECIDER_ROOT/.python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export DECIDER_MODEL="${DECIDER_MODEL:-StrandsAgents/strands-decider-2B-hobson-v19}"
export DECIDER_URL="${DECIDER_URL:-http://127.0.0.1:8099}"
