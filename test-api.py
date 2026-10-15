"""Exercise the local Decider API using synthetic DeviceOps evidence."""
import json
import math
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parent
base_url = os.environ.get("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/")
with urlopen(base_url + "/health", timeout=15) as response:
    health = json.load(response)
print("Health:", json.dumps(health))
assert health["status"] == "ok" and health["device"] == "cpu", health
payload = json.loads((root / "deviceops-request.json").read_text())
request = Request(base_url + "/v1/systemone", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
started = time.perf_counter()
with urlopen(request, timeout=300) as response:
    result = json.load(response)
elapsed = time.perf_counter() - started
assert "specialist" in result["answers"], result
answer = result["answers"]["specialist"]
options = payload["questions"]["specialist"]["criteria"]
assert answer["type"] == "choice" and answer["choice"] in options, answer
assert set(answer["probabilities"]) == set(options), answer
assert all(math.isfinite(p) and 0 <= p <= 1 for p in answer["probabilities"].values()), answer
assert math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-4), answer
assert 0 <= answer["confidence"] <= 1, answer
print(json.dumps(result, indent=2))
print(f"Client elapsed: {elapsed:.2f}s")
(root / "logs").mkdir(exist_ok=True)
(root / "logs" / "api-test.json").write_text(json.dumps({"health": health, "response": result,
                                                        "client_elapsed_seconds": elapsed}, indent=2) + "\n")
