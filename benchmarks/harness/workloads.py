"""Load the three paired-length workloads and compute their geometry."""

import json
import math
from pathlib import Path

NAMES = ("short", "medium", "long")


def load(path):
    """Read paired [T, U] lengths into per-axis lists with their lattice sizes."""
    data = json.loads(Path(path).read_text())
    if set(data) != {"batch", "hidden", "vocab", "dropout_p", "seed", "workloads"}:
        raise ValueError("Expected shared benchmark settings and workloads")
    if set(data["workloads"]) != set(NAMES):
        raise ValueError("Expected short, medium and long workloads")
    for key in ("batch", "hidden", "vocab"):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (
        type(data["seed"]) is not int
        or data["seed"] < 0
        or not 0 <= data["dropout_p"] < 1
    ):
        raise ValueError("Invalid seed or dropout probability")
    workloads = {}
    for name in NAMES:
        item = data["workloads"][name]
        pairs = item["lengths"]
        if len(pairs) != data["batch"] or any(
            not isinstance(p, list)
            or len(p) != 2
            or any(type(v) is not int for v in p)
            or p[0] <= 0
            or p[1] < 0
            for p in pairs
        ):
            raise ValueError(
                f"{name}: expected {data['batch']} integer [T, U] pairs with T > 0 and U >= 0"
            )
        duration = item["mean_duration_s"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{name}: invalid mean duration")
        workload = dict(
            times=[p[0] for p in pairs],
            lengths=[p[1] for p in pairs],
            mean_duration_s=duration,
        )
        sizes = geometry(workload)
        workloads[name] = dict(
            workload,
            valid_positions=sizes["valid"],
            padded_positions=sizes["padded"],
        )
    return dict(data, workloads=workloads)


def geometry(workload):
    times, lengths = workload["times"], workload["lengths"]
    return dict(
        valid=sum(t * (u + 1) for t, u in zip(times, lengths)),
        padded=len(times) * max(times) * (max(lengths) + 1),
    )
