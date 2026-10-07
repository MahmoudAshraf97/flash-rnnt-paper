"""Hashing, JSON output and subprocess helpers."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_lock():
    files = list((ROOT / "harness").glob("*.py"))
    files += [
        ROOT / p for p in ("pyproject.toml", "uv.lock", "harness/tolerances.json")
    ]
    files += [
        p
        for p in (ROOT / "dependencies").rglob("*")
        if p.is_file() and p.suffix in (".json", ".patch")
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_json(path, value, *, replace=False):
    """Write complete JSON; never overwrite an existing file unless replace is set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Path escapes artifact root: {relative}")
    return path


def verify_hashes(root, values):
    for relative, digest in values.items():
        require(
            sha(contained(root, relative)) == digest,
            f"Source/artifact changed: {relative}",
        )


def visible_tokens(count, value=None):
    if value is None:
        return [str(i) for i in range(count)]
    tokens = [t.strip() for t in value.split(",")]
    require(
        len(tokens) == count and all(tokens) and len(set(tokens)) == count,
        "CUDA_VISIBLE_DEVICES must identify each visible device exactly once",
    )
    return tokens


def cpu_partitions(cpus, workers, requested=4):
    cpus = sorted(set(cpus))
    require(
        workers > 0 and requested > 0 and len(cpus) >= workers,
        "Need at least one available CPU per GPU worker",
    )
    width = min(requested, len(cpus) // workers)
    return [cpus[i * width : (i + 1) * width] for i in range(workers)]


def run_process(command, env, log_path, timeout):
    """On timeout, kill the child's whole process group."""
    with Path(log_path).open("x") as log:
        child = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=ROOT,
        )
        try:
            return child.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return child.returncode, True
