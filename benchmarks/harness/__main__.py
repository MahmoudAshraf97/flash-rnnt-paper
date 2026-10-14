"""Build the benchmark backends and run the comparison and ablation studies.

Records go to results/local/raw/ by default. A local run with matching environment,
settings, workloads and sources is resumed; any other is replaced.
"""

import argparse
import json
from pathlib import Path
import shutil

from .common import ROOT, sha
from .runtime import inspect, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run"))
    parser.add_argument(
        "--study", choices=("comparison", "ablation", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, default=ROOT.parent / "results/local/raw")
    parser.add_argument(
        "--workloads", type=Path, default=ROOT.parent / "data/workloads/workloads.json"
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    args = parser.parse_args()
    machine = setup(cache=args.cache, cuda_home=args.cuda_home)
    environment = inspect(machine)
    from .backends import build

    build(machine, environment)
    if args.command == "build":
        print("Benchmark backends ready.")
        return
    from .campaign import campaign, prepare, source_state

    studies = ("comparison", "ablation") if args.study == "all" else (args.study,)
    for study in studies:
        folder = args.output.resolve() / study
        path = folder / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if (
                not (folder / "coordinator.lock").exists()
                or manifest["machine"]
                != {k: machine[k] for k in ("threads_per_worker", "build_jobs")}
                or manifest["environment"] != environment
                or manifest["study"] != study
                or manifest["workload_sha256"] != sha(args.workloads)
                or manifest["source_state"] != source_state()
            ):
                print(f"Replacing {folder}: recorded with other settings", flush=True)
                shutil.rmtree(folder)
        if not path.exists():
            prepare(folder, args.workloads, study, machine, environment)
        campaign(folder, machine, args.workloads)


if __name__ == "__main__":
    main()
