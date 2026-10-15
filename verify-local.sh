#!/usr/bin/env bash
# Finish validation once the verified base-model cache file is available.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$DECIDER_ROOT/logs"
DECIDER_BLOB="$HF_HOME/hub/models--Qwen--Qwen3.5-2B-Base/blobs/928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d"
echo "Waiting for the verified base-model download."
for ((attempt=0; attempt<1440; attempt++)); do
  [[ -s "$DECIDER_BLOB" ]] && break
  sleep 10
done
[[ -s "$DECIDER_BLOB" ]] || { echo "Model download did not finish within four hours."; exit 1; }
export HF_HUB_OFFLINE=1
echo "Running standalone offline CPU inference."
"$DECIDER_ROOT/test-cli.sh" > "$DECIDER_ROOT/logs/cli-offline-test.log" 2>&1
echo "Starting the offline localhost API for validation."
"$DECIDER_ROOT/run-server.sh" > "$DECIDER_ROOT/logs/server-test.log" 2>&1 &
DECIDER_SERVER_PID=$!
cleanup() {
  kill "$DECIDER_SERVER_PID" 2>/dev/null || true
  wait "$DECIDER_SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
DECIDER_READY=false
for ((attempt=0; attempt<150; attempt++)); do
  kill -0 "$DECIDER_SERVER_PID" 2>/dev/null || { cat "$DECIDER_ROOT/logs/server-test.log"; exit 1; }
  if curl --fail --silent --max-time 2 "$DECIDER_URL/health" > "$DECIDER_ROOT/logs/health.json"; then
    DECIDER_READY=true
    break
  fi
  sleep 2
done
[[ "$DECIDER_READY" == true ]] || { echo "Local API did not become ready."; exit 1; }
"$DECIDER_ROOT/.venv/bin/python" "$DECIDER_ROOT/test-api.py" > "$DECIDER_ROOT/logs/api-test.log" 2>&1
cleanup
trap - EXIT
"$DECIDER_ROOT/.venv/bin/python" - "$DECIDER_ROOT" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
api = json.loads((root / 'logs/api-test.json').read_text())
result = {
    'status': 'verified',
    'verified_at': datetime.now(timezone.utc).isoformat(),
    'model': api['health']['model'],
    'device': api['health']['device'],
    'offline_cli_passed': True,
    'offline_http_api_passed': True,
    'api_latency_ms': api['response'].get('latency_ms'),
    'specialist': api['response']['answers']['specialist'],
    'api_stopped_after_verification': True,
}
(root / 'logs/setup-complete.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
echo "Verification passed. Start run-server.sh whenever you need the API."
