# Local Strands Decider 2B

Isolated CPU installation for the ThinkPad T490. Official checkpoint:
`StrandsAgents/strands-decider-2B-hobson-v19`.

Everything is stored here: `.venv` (packages), `.python` (Python 3.12),
`.cache/huggingface` (downloaded weights), and `.cache/uv` (installer cache).
The launchers default to four CPU threads and do not modify DeviceOps.
The CPU backend expands the weights to float32 and needs roughly 7–8 GB RAM
for weights, plus runtime memory. Keep one model process running at a time.

To reinstall with `uv` available, run `./setup.sh`. Installed package versions
are recorded in `requirements.lock.txt`.

If a weight download stalls, `download-base.py` provides a resumable HTTP fallback
for the pinned Qwen base revision. It validates every byte range and the full
published SHA256 before adding the weights to the local cache:

```bash
source env.sh
.venv/bin/python download-base.py
```

`verify-local.sh` waits for the verified base-model file, then tests offline CLI
and HTTP inference and stops its test server. Results go to `logs/verification.log`.
Successful completion creates `logs/setup-complete.json`; until that file exists,
inference verification is pending. Download progress is in `logs/base-download.log`.

To resume downloading and then run the checks independently of the terminal:

```bash
nohup setsid bash ./finish-setup.sh > logs/background-setup.log 2>&1 < /dev/null &
```

This finite job prevents duplicate runs with a lock and stops its test server
after verification. Keep the laptop awake and connected while downloading.

## Start the API

```bash
./run-server.sh
```

The API binds to `http://127.0.0.1:8099`. Stop with Ctrl+C.
Use `DECIDER_PORT=8100 ./run-server.sh` to choose another port.
Send requests sequentially for local experiments.

The normal API launcher retains the original FP32 package server. Optional
`./run-optimized-server.sh` merges the trained LoRA adapter into the FP32 base
in memory and uses one interop thread. Downloaded weights are unchanged.
Merging adds startup work, but avoids repeated adapter calculations on requests.
Check `GET /local-runtime` on the optimized server for its mode and thread counts.
`./run-reference-server.sh` also starts the original package server without merging.

## CPU performance options

The T490 benchmark tried 1, 2, 4 and 8 threads in forward and reverse order.
Four threads performed best; eight were slower. `--threads 4` remains the default.

The in-process synthetic comparison measured:

| Mode | Short request median | 572-token request | Private memory after requests |
| --- | ---: | ---: | ---: |
| Original FP32 | 3.84 s | 15.04 s | 7.66 GiB |
| Optional merged FP32 | 3.67 s | 14.17 s | 7.60 GiB |
| Experimental dynamic INT8 | 2.09 s | 7.96 s | 3.77 GiB |

These timings vary with laptop load and temperature. FP32 merging produced the
same rounded outputs for six synthetic cases covering choice, noul and score.
It is a modest optimization, not a comprehensive DeviceOps accuracy evaluation.

A separate HTTP test using fresh processes measured:

| Mode | Startup | Short warm median | 572-token request | Idle private RAM after requests |
| --- | ---: | ---: | ---: | ---: |
| Original FP32 (default) | 7.54 s | 2.71 s | 15.35 s | 7.72 GiB |
| Optional merged FP32 | 17.56 s | 3.73 s | 14.41 s | 7.71 GiB |
| Experimental dynamic INT8 | 45.60 s | 1.48 s | 7.85 s | 3.96 GiB |

The mixed short-request results mean FP32 merging is not a consistent latency
improvement on this laptop. It remains optional; the normal launcher uses the
reference implementation. Long requests are still expensive in either FP32 mode.

INT8 kept the six selected choices, but changed a reported numeric value by up to
0.255, including changes to routing confidence. Its confidence and probabilities
must not be treated as retaining the original calibration. It quantizes the torso's
linear layers only; embeddings, normalization, convolutions and readout stay FP32.
Do not use it to authorize DeviceOps actions without domain evaluation.

To experiment with the faster, smaller mode, first stop any existing server:

```bash
./run-optimized-server.sh --mode experimental-int8
```

Its model identifier includes `-experimental-int8`. To try merging while retaining
FP32 calculations, use `./run-optimized-server.sh --mode merged-fp32` instead.
Both optimized modes merge
at each startup; INT8 additionally converts weights and therefore starts slower.
All servers retain RAM while idle. Stop with Ctrl+C to release model memory.
For responsive experiments, keep one server warm, use concise relevant evidence,
and avoid repeated inference on unchanged observations in future integration.

Reproduce the comparisons (each benchmark stops its temporary model afterward):

```bash
source env.sh
HF_HUB_OFFLINE=1 .venv/bin/python benchmark-optimizations.py
.venv/bin/python benchmark-local.py --quick --mode merged-fp32
.venv/bin/python benchmark-local.py --quick --mode experimental-int8
```

Results are in `logs/optimization-benchmark.json` and
`logs/resource-benchmark-<mode>.json`. The CLI wrappers still use original inference;
the options apply to `run-optimized-server.sh` only.

## Try DeviceOps routing

With the API running, in a second terminal:

```bash
.venv/bin/python test-api.py
```

Or use curl:

```bash
curl --fail http://127.0.0.1:8099/health
curl --fail http://127.0.0.1:8099/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @deviceops-request.json
```

A standalone CLI test loads the model separately:

```bash
./test-cli.sh
```

Any CLI command can be run through `./decider.sh`, which sets the local cache.
Do not start another model process while the server is running unless you intend
to load a second copy into memory.

## Offline use

The server defaults to cached-only startup after installation:

```bash
HF_HUB_OFFLINE=1 ./run-server.sh
```

Set `HF_HUB_OFFLINE=0` explicitly if you intentionally want the server to download
a different checkpoint. The standalone CLI keeps the official download behavior.

## Future DeviceOps integration

POST the JSON shape in `deviceops-request.json` to `/v1/systemone`.
The result contains choices and scores, not generated explanations.
This example only selects a diagnostic specialist; it does not control devices
or grant permission for disruptive actions. Retain existing DeviceOps approval
and execution checks.

## Sources

- https://github.com/strands-labs/strands-decider
- https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md
- https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19
- https://huggingface.co/docs/peft/v0.21.0/package_reference/lora
- https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.quantize_dynamic.html

## Repository contents

This repository contains local setup, launch, verification and CPU benchmark scripts plus a synthetic request fixture. Model weights, Python environments, caches, logs and machine-specific audit snapshots are excluded. The setup downloads the selected upstream package and checkpoint; their own licenses and notices remain applicable. The copied upstream CLI reference is kept local and is not redistributed here.

Publication checks on October 15, 2026 covered Python syntax, shell syntax, JSON parsing and selected-file credential patterns. Model loading, downloads, API inference and performance benchmarks were not rerun for publication. The measurements above are historical local results.
