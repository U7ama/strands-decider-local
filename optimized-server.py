"""Serve the official API with explicit, reversible local CPU options."""
import argparse
import os

import torch
import uvicorn
from strands_decider.server import create_app, get_engine

from cpu_runtime import optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reference", "merged-fp32", "experimental-int8"],
                        default=os.getenv("DECIDER_CPU_MODE", "merged-fp32"))
    parser.add_argument("--threads", type=int, default=int(os.getenv("DECIDER_THREADS", "4")))
    parser.add_argument("--port", type=int, default=int(os.getenv("DECIDER_PORT", "8099")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--model-name")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = os.environ["DECIDER_MODEL"]
    name = args.model_name or checkpoint.rstrip("/").split("/")[-1]
    if args.mode == "experimental-int8":
        name += "-experimental-int8"
    app = create_app(checkpoint, device="cpu", use_prefix_cache=not args.no_prefix_cache,
                     model_name=name)
    optimize(get_engine(), args.mode)

    @app.get("/local-runtime")
    def runtime():
        return {"mode": args.mode, "threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "experimental": args.mode == "experimental-int8"}

    print(f"Local CPU mode={args.mode}, threads={args.threads}; checkpoint unchanged", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
