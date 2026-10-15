"""Optional local CPU transformations; never writes model checkpoints."""
import torch
import ctypes
import gc


def optimize(engine, mode):
    if mode not in {"reference", "merged-fp32", "experimental-int8"}:
        raise ValueError(f"Unknown CPU mode: {mode}")
    if mode == "reference":
        return
    # Keep the readout and its calibration in FP32. Merge after the official
    # engine's CPU upcast, so the LoRA delta is accumulated in FP32 as well.
    with torch.inference_mode():
        engine.model.torso = engine.model.torso.merge_and_unload(safe_merge=True)
        if mode == "experimental-int8":
            if "fbgemm" not in torch.backends.quantized.supported_engines:
                raise RuntimeError("This installation does not support FBGEMM INT8")
            torch.backends.quantized.engine = "fbgemm"
            engine.model.torso = torch.ao.quantization.quantize_dynamic(
                engine.model.torso, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    engine.model.eval()
    # Release allocator-held FP32 buffers after merging/quantizing on Linux.
    gc.collect()
    libc = ctypes.CDLL(None)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
