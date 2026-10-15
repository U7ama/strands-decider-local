"""Compare CPU threads, merged FP32 LoRA and experimental dynamic INT8.

All transformations are in memory; downloaded checkpoints remain unchanged.
Synthetic agreement checks are not a domain accuracy/calibration evaluation.
"""
from copy import deepcopy
import ctypes
import gc
import json
import os
from pathlib import Path
import statistics
import time

import psutil
import torch
from strands_decider.infer import load_engine
from strands_decider.schema import SystemOneRequest
from cpu_runtime import optimize

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "logs/optimization-benchmark.json"
process = psutil.Process()
report = {"thread_trials": [], "variants": {}, "note":
          "Serial synthetic local CPU tests; agreement does not establish accuracy or calibration."}


def save():
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


def trim():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def measure(engine, name, payload):
    request = SystemOneRequest.model_validate(payload)
    start = time.perf_counter()
    result = engine.evaluate(request).model_dump()
    seconds = time.perf_counter() - start
    row = {"case": name, "seconds": round(seconds, 4),
           "input_tokens": result["usage"]["input_tokens"], "answers": result["answers"]}
    print(json.dumps({"case": name, "seconds": row["seconds"]}), flush=True)
    return row


def compare(reference, candidate):
    max_delta = 0.0
    choices_same = 0
    choices_total = 0
    for old, new in zip(reference, candidate, strict=True):
        for name, answer in old["answers"].items():
            other = new["answers"][name]
            if answer["type"] == "choice":
                choices_total += 1
                choices_same += answer["choice"] == other["choice"]
            for field in ("confidence", "score", "noul"):
                if field in answer:
                    max_delta = max(max_delta, abs(answer[field] - other[field]))
            for key, value in answer.get("probabilities", {}).items():
                max_delta = max(max_delta, abs(value - other["probabilities"][key]))
    return {"choices_same": choices_same, "choices_total": choices_total,
            "max_reported_numeric_delta": round(max_delta, 6)}


def suite(engine, variant, cases, medium, reference=None):
    rows = [measure(engine, name, payload) for name, payload in cases]
    warm = [measure(engine, "short_warm", cases[0][1]) for _ in range(3)]
    medium_row = measure(engine, "medium_572_tokens", medium)
    trim()
    memory = process.memory_full_info()
    output = {"threads": torch.get_num_threads(), "cases": rows,
              "short_warm_seconds": [r["seconds"] for r in warm],
              "short_warm_median_seconds": statistics.median(r["seconds"] for r in warm),
              "medium": medium_row, "rss_gib": round(memory.rss / 1024**3, 3),
              "uss_gib": round(memory.uss / 1024**3, 3)}
    if reference is not None:
        output["agreement"] = compare(reference, rows)
    report["variants"][variant] = output
    save()
    print(json.dumps({"variant": variant, "median": output["short_warm_median_seconds"],
                      "uss_gib": output["uss_gib"], "agreement": output.get("agreement")}), flush=True)
    return rows


def main():
    torch.set_num_interop_threads(1)
    torch.set_num_threads(4)
    base = json.loads((ROOT / "deviceops-request.json").read_text())
    medium = deepcopy(base)
    medium["state"] += "\n" + "\n".join(
        f"Observation {i}: switch port {i} shows packet loss 70%, link up, projector unreachable, power on."
        for i in range(16))
    cases = [("network", base)]
    for name, state in (
        ("power", "The projector has no power light. The outlet has no voltage. Other network devices are healthy."),
        ("system", "Projector power and network are healthy. Its application repeatedly crashes with a software error."),
        ("uncertain", "A user reports that the projector is offline. No diagnostics or power checks are available."),
        ("conflicting", "A user says the projector is offline, but monitoring reports it is online. Power and network status are unknown."),
    ):
        item = deepcopy(base)
        item["state"] = state
        cases.append((name, item))
    grouped = deepcopy(base)
    grouped["questions"]["enough_evidence"] = {
        "type": "noul", "instructions": "Is there evidence of a shared network problem?"}
    grouped["questions"]["urgency"] = {
        "type": "score", "instructions": "Rate urgency for campus operations.",
        "criteria": ["Low: no users affected", "Medium: one room affected", "High: multiple devices affected"]}
    cases.append(("grouped_choice_noul_score", grouped))
    start = time.perf_counter()
    engine = load_engine(os.environ["DECIDER_MODEL"], device="cpu")
    report["load_seconds"] = round(time.perf_counter() - start, 3)
    print("Loaded FP32 engine", flush=True)
    measure(engine, "warmup", base)
    # Forward and reverse order reduce order bias from laptop thermal changes.
    for threads in (1, 2, 4, 8, 8, 4, 2, 1):
        torch.set_num_threads(threads)
        row = measure(engine, f"threads_{threads}", base)
        report["thread_trials"].append({"threads": threads, **row})
        save()
    medians = {n: statistics.median(r["seconds"] for r in report["thread_trials"] if r["threads"] == n)
               for n in (1, 2, 4, 8)}
    best = min(medians, key=medians.get)
    report["thread_medians_seconds"] = medians
    report["selected_threads"] = best
    torch.set_num_threads(best)
    reference = suite(engine, "reference_fp32", cases, medium)
    start = time.perf_counter()
    optimize(engine, "merged-fp32")
    report["merge_seconds"] = round(time.perf_counter() - start, 3)
    suite(engine, "merged_fp32", cases, medium, reference)
    try:
        torch.backends.quantized.engine = "fbgemm"
        start = time.perf_counter()
        # The torso is already merged; quantize only its linear layers.
        with torch.inference_mode():
            engine.model.torso = torch.ao.quantization.quantize_dynamic(
                engine.model.torso, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        report["quantize_seconds"] = round(time.perf_counter() - start, 3)
        report["quantized_linear_count"] = sum(
            isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in engine.model.torso.modules())
        suite(engine, "experimental_int8", cases, medium, reference)
    except Exception as exc:
        report["int8_error"] = f"{type(exc).__name__}: {exc}"
        print(report["int8_error"], flush=True)
    save()


if __name__ == "__main__":
    main()
