"""CUDA setup and metadata for the uv-installed benchmark environment."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tomllib

from .common import ROOT, require, visible_tokens


def reference_environment():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pins = dict(
        item.split("==", 1)
        for item in project["project"]["dependencies"]
        if "==" in item
    )
    pins = {name.split("[")[0]: version for name, version in pins.items()}
    cuda = pins["torch"].split("+cu")[1]
    return dict(
        distributions=pins,
        nemo_commit=project["tool"]["uv"]["sources"]["nemo-toolkit"]["rev"],
        torch_cuda_runtime=f"{cuda[:-1]}.{cuda[-1]}",
    )


def setup(machine=None, *, cache=None, cuda_home=None):
    machine = dict(machine or {})
    nvcc = shutil.which("nvcc")
    toolkit = cuda_home or machine.get("cuda_home") or os.environ.get("CUDA_HOME")
    if toolkit is None and nvcc:
        toolkit = Path(nvcc).resolve().parent.parent
    require(
        toolkit is not None,
        "Install the CUDA toolkit and set CUDA_HOME or pass --cuda-home",
    )
    machine["cuda_home"] = str(Path(toolkit).resolve())
    machine["artifact_root"] = str(
        Path(
            cache or machine.get("artifact_root", ROOT.parent / "build/benchmarks")
        ).resolve()
    )
    machine.setdefault("threads_per_worker", 4)
    machine.setdefault("build_jobs", 2)
    os.environ["CUDA_HOME"] = machine["cuda_home"]
    os.environ["CUDA_PATH"] = machine["cuda_home"]
    os.environ["CUDACXX"] = str(Path(machine["cuda_home"]) / "bin/nvcc")
    os.environ["PATH"] = (
        str(Path(machine["cuda_home"]) / "bin") + os.pathsep + os.environ["PATH"]
    )
    os.environ["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT)
    return machine


def inspect(machine):
    import torch

    require(torch.cuda.is_available(), "CUDA is unavailable")
    nvcc = subprocess.check_output(
        [str(Path(machine["cuda_home"]) / "bin/nvcc"), "--version"], text=True
    )
    require(
        f"release {torch.version.cuda}," in nvcc,
        "nvcc/PyTorch CUDA major.minor mismatch",
    )
    devices = []
    for i, token in enumerate(
        visible_tokens(
            torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES")
        )
    ):
        p = torch.cuda.get_device_properties(i)
        devices.append(
            dict(
                token=token,
                name=p.name,
                uuid=str(p.uuid),
                capability=[p.major, p.minor],
                total_memory=p.total_memory,
            )
        )
    require(
        len({tuple(d["capability"]) for d in devices}) == 1,
        "Select GPUs with the same architecture",
    )
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        nvcc=nvcc,
        devices=devices,
        nemo_commit=reference_environment()["nemo_commit"],
        packages={
            p: importlib.metadata.version(p)
            for p in ("torch", "triton", "numba", "numba-cuda", "nemo-toolkit")
        },
    )


def environment_id(record):
    fields = {k: record[k] for k in ("python", "torch", "torch_cuda", "nvcc")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
