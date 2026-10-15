"""Resume the public base-model transfer in validated parallel HTTP ranges.

Fallback for connections where Hugging Face's Xet transfer stalls. The published
revision, size and SHA256 are pinned, and only the verified full file enters the
Hub cache. Run with the local Python environment after sourcing env.sh.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import re
import subprocess

REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
FILENAME = "model.safetensors-00001-of-00001.safetensors"
SIZE = 4548221488
SHA256 = "928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d"
URL = f"https://huggingface.co/Qwen/Qwen3.5-2B-Base/resolve/{REVISION}/{FILENAME}"
ROOT = Path(__file__).resolve().parent
PARTS = ROOT / ".cache" / "base-download"
PARTS.mkdir(parents=True, exist_ok=True)
CHUNK = 32 * 1024 * 1024
COUNT = (SIZE + CHUNK - 1) // CHUNK


def fetch(index):
    start, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
    path = PARTS / f"{index:04d}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return index, True
    temporary, headers = path.with_suffix(".tmp"), path.with_suffix(".headers")
    received = temporary.stat().st_size if temporary.exists() else 0
    tail = path.with_suffix(".tail")
    if tail.exists() and headers.exists():
        previous_ranges = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
                                     headers.read_text(), re.IGNORECASE)
        if (previous_ranges
                and tuple(map(int, previous_ranges[-1])) == (start + received, end, SIZE)
                and tail.stat().st_size <= end - start + 1 - received):
            with temporary.open("ab") as target, tail.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
            received = temporary.stat().st_size
        tail.unlink()
    if received > end - start + 1:
        raise RuntimeError(f"Oversized partial file for part {index}")
    if received == end - start + 1:
        temporary.replace(path)
        return index, True
    resume_start = start + received
    subprocess.run([
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
        "--max-time", "1800", "--range", f"{resume_start}-{end}",
        "--dump-header", str(headers), "--output", str(tail),
        URL + f"?download=true&part={index}&offset={resume_start}",
    ], check=True)
    ranges = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
                        headers.read_text(), re.IGNORECASE)
    if not ranges or tuple(map(int, ranges[-1])) != (resume_start, end, SIZE):
        raise RuntimeError(f"Invalid HTTP range for part {index}")
    if tail.stat().st_size != end - resume_start + 1:
        raise RuntimeError(f"Incomplete part {index}")
    with temporary.open("ab") as target, tail.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            target.write(block)
    tail.unlink()
    temporary.replace(path)
    return index, False


if __name__ == "__main__":
    hub = Path(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface")) / "hub"
    repo = hub / "models--Qwen--Qwen3.5-2B-Base"
    blob = repo / "blobs" / SHA256
    if blob.exists() and blob.stat().st_size == SIZE:
        digest = hashlib.sha256()
        with blob.open("rb") as source:
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() == SHA256:
            print(f"Verified SHA256 {SHA256}; base model already cached", flush=True)
            raise SystemExit(0)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, i) for i in range(COUNT)]):
            index, cached = future.result()
            done += 1
            print(f"Completed {done}/{COUNT} ranges ({100*done/COUNT:.1f}%), part {index}"
                  + (" cached" if cached else ""), flush=True)
    blob.parent.mkdir(parents=True, exist_ok=True)
    assembled = blob.with_suffix(".assembling")
    digest = hashlib.sha256()
    with assembled.open("wb") as target:
        for index in range(COUNT):
            with (PARTS / f"{index:04d}.part").open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
                    target.write(block)
    if assembled.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise RuntimeError("Base-model SHA256 verification failed; cache not updated")
    assembled.replace(blob)
    snapshot = repo / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    link = snapshot / FILENAME
    if not link.exists():
        link.symlink_to(Path("../../blobs") / SHA256)
    for path in PARTS.glob("*"):
        if path.is_file():
            path.unlink()
    print(f"Verified SHA256 {SHA256}; model cached at {link}", flush=True)
