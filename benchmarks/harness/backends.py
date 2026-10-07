"""Fetch, patch and build the pinned native backends for the target GPU and PyTorch.

`python -m harness.backends --artifact-root PATH` fetches and verifies the
sources without CUDA.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, atomic_json, contained, require, sha, verify_hashes
from .runtime import environment_id

SPECS = ROOT / "dependencies"


def recipe():
    return json.loads((SPECS / "manifest.json").read_text())


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def checkout_path(artifact_root, name, spec):
    path = Path(artifact_root).resolve() / "sources" / f"{name}-{spec['commit']}"
    require(
        not path.is_relative_to(ROOT),
        "Use an artifact_root outside the benchmark source directory",
    )
    return path


def validate_checkout(path, spec):
    require(
        path.is_dir() and (path / ".git").is_dir(),
        f"Missing dependency checkout: {path}; run build",
    )
    require(
        git(path, "rev-parse", "HEAD") == spec["commit"],
        f"Dependency commit changed: {path}",
    )
    require(
        not git(path, "ls-files", "--others", "--exclude-standard"),
        f"Untracked files in dependency: {path}",
    )
    changed = set(git(path, "diff", "--name-only", "--no-renames", "HEAD").splitlines())
    require(
        changed == set(spec["patched_sha256"]), f"Unexpected dependency changes: {path}"
    )
    verify_hashes(path, spec["patched_sha256"])


def validate(artifact_root):
    paths = {}
    for name, spec in recipe().items():
        path = checkout_path(artifact_root, name, spec)
        validate_checkout(path, spec)
        paths[name] = path
    return paths


def prepare(artifact_root):
    paths = {}
    for name, spec in recipe().items():
        path = checkout_path(artifact_root, name, spec)
        # Never reset, repatch or overwrite an existing checkout.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q", str(path)], check=True)
            git(path, "remote", "add", "origin", spec["url"])
            git(
                path,
                "-c",
                "core.autocrlf=false",
                "fetch",
                "--depth=1",
                "origin",
                spec["commit"],
            )
            git(path, "-c", "core.autocrlf=false", "checkout", "--detach", "FETCH_HEAD")
            if spec["patch"]:
                patch = str(SPECS / spec["patch"])
                git(path, "apply", "--check", patch)
                git(path, "apply", patch)
        validate_checkout(path, spec)
        paths[name] = path
    return paths


def build(machine, environment):
    import torch
    from torch.utils.cpp_extension import load as compile_extension

    root = Path(machine["artifact_root"]) / "extensions"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        architecture = ".".join(map(str, environment["devices"][0]["capability"]))
        if (
            manifest["environment_id"] == environment_id(environment)
            and manifest["dependencies"] == recipe()
            and manifest["architecture"] == architecture
        ):
            load(machine, environment)
            return manifest
    paths = prepare(machine["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    spec = recipe()
    major, minor = environment["devices"][0]["capability"]
    architecture = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = architecture
    os.environ["MAX_JOBS"] = str(machine["build_jobs"])
    compiler = os.environ.get("CXX", "c++")
    manifest = dict(
        environment_id=environment_id(environment),
        architecture=architecture,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        binaries={},
        dependencies=spec,
    )
    manifest.update(
        host_compiler=compiler,
        host_compiler_version=subprocess.check_output(
            [compiler, "--version"], text=True
        ),
        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        build_environment={
            k: os.environ[k]
            for k in (
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
                "TORCH_CUDA_ARCH_LIST",
                "MAX_JOBS",
            )
            if k in os.environ
        },
    )
    for kind in ("joint", "loss"):
        name = f"transducer_{kind}_cuda"
        directory = root / kind
        directory.mkdir(exist_ok=True)
        base = paths["apex"] / "apex/contrib/csrc/transducer"
        module = compile_extension(
            name=name,
            sources=[
                str(base / f"transducer_{kind}{suffix}")
                for suffix in (".cpp", "_kernel.cu")
            ],
            extra_cflags=spec["apex"]["cflags"],
            extra_cuda_cflags=spec["apex"]["cuda_flags"],
            extra_include_paths=[str(paths["apex"] / "csrc")],
            build_directory=str(directory),
            with_cuda=True,
            verbose=True,
        )
        path = Path(module.__file__).resolve()
        manifest["binaries"][name] = dict(
            path=str(path.relative_to(root)), sha256=sha(path)
        )
    directory = root / "optimized"
    directory.mkdir(exist_ok=True)
    name = "_optimized_transducer"
    sources = [
        *[
            paths["optimized"] / "optimized_transducer/csrc" / p
            for p in ("transducer-loss.cc", "cpu.cc", "cuda.cu", "kernels.cu")
        ],
        paths["optimized"] / "optimized_transducer/python/csrc/optimized_transducer.cc",
    ]
    module = compile_extension(
        name=name,
        sources=list(map(str, sources)),
        extra_cflags=spec["optimized"]["cflags"],
        extra_cuda_cflags=spec["optimized"]["cuda_flags"],
        extra_include_paths=[str(paths["optimized"]), str(paths["moderngpu"] / "src")],
        build_directory=str(directory),
        with_cuda=True,
        verbose=True,
    )
    path = Path(module.__file__).resolve()
    manifest["binaries"][name] = dict(
        path=str(path.relative_to(root)), sha256=sha(path)
    )
    atomic_json(root / "manifest.json", manifest, replace=True)
    return manifest


def import_file(name, path):
    if name in sys.modules:
        require(
            Path(sys.modules[name].__file__).resolve() == path.resolve(),
            f"Conflicting module: {name}",
        )
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def load(machine, environment, family=None):
    root = Path(machine["artifact_root"]) / "extensions"
    manifest = json.loads((root / "manifest.json").read_text())
    require(
        manifest["environment_id"] == environment_id(environment),
        "Extension environment changed; rebuild",
    )
    require(
        manifest["dependencies"] == recipe(),
        "Native dependency pins or build flags changed; rebuild",
    )
    paths = validate(machine["artifact_root"])
    capability = ".".join(map(str, environment["devices"][0]["capability"]))
    require(
        manifest["architecture"] == capability,
        "Wrong extension architecture; rebuild on the target GPU",
    )
    for name, entry in manifest["binaries"].items():
        if family == "apex" and name == "_optimized_transducer":
            continue
        if family == "optimized" and name != "_optimized_transducer":
            continue
        path = contained(root, entry["path"])
        require(sha(path) == entry["sha256"], f"Extension binary changed: {name}")
        import_file(name, path)
    api = (
        import_file(
            "_benchmark_apex_transducer",
            paths["apex"] / "apex/contrib/transducer/transducer.py",
        )
        if family in (None, "apex")
        else None
    )
    opt = (
        import_file(
            "_benchmark_optimized_wrapper",
            paths["optimized"]
            / "optimized_transducer/python/optimized_transducer/transducer_loss.py",
        ).transducer_loss
        if family in (None, "optimized")
        else None
    )
    return api, opt, sha(root / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    for name, path in prepare(args.artifact_root).items():
        print(f"{name}: {path}")
