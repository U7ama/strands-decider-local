"""Measure this laptop's offline server memory, CPU and serial request latency."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from urllib.request import Request, urlopen

import psutil

ROOT = Path(__file__).resolve().parent
PORT = 18109
URL = f"http://127.0.0.1:{PORT}"
GIB = 1024 ** 3
samples = []
stop_sampling = threading.Event()


def sample(process):
    process.cpu_percent()
    while not stop_sampling.wait(0.1):
        try:
            samples.append({"time": time.monotonic(),
                            "rss": process.memory_info().rss,
                            "cpu_percent": process.cpu_percent()})
        except psutil.NoSuchProcess:
            break


def snapshot(process):
    memory = process.memory_full_info()
    cpu = process.cpu_times()
    return {"time": time.monotonic(), "cpu_seconds": cpu.user + cpu.system,
            "rss_gib": memory.rss / GIB,
            "uss_gib": memory.uss / GIB,
            "pss_gib": memory.pss / GIB}


def interval(before, after):
    duration = after["time"] - before["time"]
    selected = [s for s in samples if before["time"] <= s["time"] <= after["time"]]
    cpu = 100 * (after["cpu_seconds"] - before["cpu_seconds"]) / duration
    return {"seconds": round(duration, 3),
            "average_process_cpu_percent": round(cpu, 1),
            "average_machine_cpu_percent": round(cpu / psutil.cpu_count(), 1),
            "peak_process_cpu_percent": round(max((s["cpu_percent"] for s in selected), default=0), 1),
            "peak_rss_gib": round(max((s["rss"] for s in selected), default=0) / GIB, 3),
            "ending_rss_gib": round(after["rss_gib"], 3),
            "ending_uss_gib": round(after["uss_gib"], 3),
            "ending_pss_gib": round(after["pss_gib"], 3)}


def ask(label, payload, process):
    before = snapshot(process)
    request = Request(URL + "/v1/systemone", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    wall = time.monotonic() - start
    after = snapshot(process)
    measurement = {"label": label, "client_seconds": round(wall, 3),
                   "server_ms": result["latency_ms"], "usage": result["usage"],
                   "answers": result["answers"],
                   "resources": interval(before, after)}
    print(json.dumps(measurement), flush=True)
    return measurement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reference", "merged-fp32", "experimental-int8"],
                        default="reference")
    parser.add_argument("--quick", action="store_true", help="Skip long/grouped cases, use 3s idle intervals")
    args = parser.parse_args()
    ROOT.joinpath("logs").mkdir(exist_ok=True)
    environment = dict(os.environ, DECIDER_PORT=str(PORT), HF_HUB_OFFLINE="1")
    swap_before = psutil.swap_memory().used
    available_before = psutil.virtual_memory().available
    suffix = f"-{args.mode}" if args.quick else ""
    server_log = (ROOT / f"logs/benchmark-server{suffix}.log").open("w")
    start = time.monotonic()
    command = [str(ROOT / "run-reference-server.sh")] if args.mode == "reference" else [
        str(ROOT / "run-optimized-server.sh"), "--mode", args.mode]
    server = subprocess.Popen(command, env=environment,
                              stdout=server_log, stderr=subprocess.STDOUT)
    process = psutil.Process(server.pid)
    sampler = threading.Thread(target=sample, args=(process,), daemon=True)
    sampler.start()
    try:
        deadline = start + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError("Benchmark server exited; see benchmark-server.log")
            try:
                with urlopen(URL + "/health", timeout=1) as response:
                    health = json.load(response)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Benchmark server failed to start")
                time.sleep(0.5)
        runtime = None
        if args.mode != "reference":
            with urlopen(URL + "/local-runtime", timeout=5) as response:
                runtime = json.load(response)
            assert runtime["mode"] == args.mode and runtime["threads"] == 4, runtime
            assert runtime["experimental"] == (args.mode == "experimental-int8"), runtime
        startup = {"seconds": round(time.monotonic() - start, 3),
                   "peak_rss_gib": round(max(s["rss"] for s in samples) / GIB, 3)}
        print("Startup: " + json.dumps(startup), flush=True)
        before = snapshot(process)
        time.sleep(3 if args.quick else 10)
        idle = interval(before, snapshot(process))
        print("Idle: " + json.dumps(idle), flush=True)
        base = json.loads((ROOT / "deviceops-request.json").read_text())
        results = [ask("short_first", base, process)]
        results.extend(ask(f"short_warm_{i}", base, process) for i in range(3))
        medium = deepcopy(base)
        medium["state"] += "\n" + "\n".join(
            f"Observation {i}: switch port {i} shows packet loss 70%, link up, projector unreachable, power on."
            for i in range(16))
        results.append(ask("medium_summary", medium, process))
        for result in results:
            for answer in result["answers"].values():
                if "probabilities" in answer:
                    probabilities = list(answer["probabilities"].values())
                    assert all(0 <= p <= 1 for p in probabilities), answer
                    assert abs(sum(probabilities) - 1) <= 0.001, answer
                if "confidence" in answer:
                    assert 0 <= answer["confidence"] <= 1, answer
        if not args.quick:
            long = deepcopy(base)
            long["state"] += "\n" + "\n".join(
                f"Observation {i}: switch port {i} shows packet loss 70%, link up, projector unreachable, power on."
                for i in range(64))
            results.append(ask("long_summary", long, process))
            grouped = deepcopy(base)
            grouped["questions"]["enough_evidence"] = {
                "type": "noul", "instructions": "Is there evidence of a shared network problem?"}
            grouped["questions"]["urgency"] = {
                "type": "score", "instructions": "Rate urgency for campus operations.",
                "criteria": ["Low: no users affected", "Medium: one room affected", "High: multiple devices affected"]}
            results.append(ask("three_grouped_questions", grouped, process))
        before = snapshot(process)
        time.sleep(3 if args.quick else 10)
        idle_after = interval(before, snapshot(process))
        report = {"logical_cpus": psutil.cpu_count(), "configured_threads": 4, "mode": args.mode,
                  "health": health, "runtime": runtime, "startup": startup, "idle_before": idle,
                  "requests": results, "idle_after": idle_after,
                  "short_warm_median_seconds": statistics.median(
                      r["client_seconds"] for r in results if r["label"].startswith("short_warm")),
                  "overall_peak_rss_gib": round(max(s["rss"] for s in samples) / GIB, 3),
                  "available_ram_before_gib": round(available_before / GIB, 3),
                  "available_ram_after_gib": round(psutil.virtual_memory().available / GIB, 3),
                  "swap_used_change_gib": round((psutil.swap_memory().used - swap_before) / GIB, 3),
                  "note": "Serial synthetic requests; machine percentage is process CPU divided by 8 logical CPUs."}
        (ROOT / f"logs/resource-benchmark{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Result: " + json.dumps(report), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stop_sampling.set()
        sampler.join(timeout=2)
        server_log.close()
