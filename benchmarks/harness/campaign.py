"""Run the comparison and ablation with one isolated worker process per measurement."""

import argparse
import contextlib
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .common import atomic_json, cpu_partitions, require, run_process, sha

FAMILIES = ("flash_tiled", "flash_untiled", "apex", "numba", "optimized")
FAILURES = ("oom", "compile_error", "runtime_error", "timeout", "unsupported_length")
SUB_BATCHES = (1, 2, 4, 8, 16, 32, 64)
TILE_ROWS = (25000, 50000, 100000, 200000, 400000, 800000)


@contextlib.contextmanager
def device_lock(uuid):
    key = hashlib.sha256(uuid.encode()).hexdigest()
    # One lock file per physical GPU, shared by every checkout on the machine.
    with (Path(tempfile.gettempdir()) / f"rnnt-paper-gpu-{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def configure_torch(threads):
    import torch

    torch.cuda.set_device(0)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


# Polling the GPU for compute processes other than the worker; activity shorter
# than the polling interval can go unseen.


def query(arguments):
    return subprocess.check_output(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )


def processes(uuid):
    rows = []
    for line in query(["--query-compute-apps=gpu_uuid,pid,used_memory"]).splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"Unparseable GPU process telemetry: {line}")
        if fields[0].removeprefix("GPU-") == uuid.removeprefix("GPU-"):
            rows.append(dict(pid=int(fields[1]), memory_MiB=fields[2]))
    return rows


def idle_snapshot(uuid):
    rows = (
        query(
            [
                f"--id=GPU-{uuid.removeprefix('GPU-')}",
                "--query-gpu=memory.used,utilization.gpu",
            ]
        )
        .strip()
        .split(",")
    )
    if len(rows) != 2:
        raise RuntimeError("Missing GPU idle telemetry")
    return dict(memory_MiB=int(rows[0]), utilization_percent=int(rows[1]))


class Guard:
    def __init__(self, uuid, interval=0.5, monitor_cpus=None):
        self.uuid = uuid
        self.interval = interval
        self.pid = os.getpid()
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = None
        # nvidia-smi children inherit this CPU mask, keeping them off the timing core.
        self.monitor_cpus = None if monitor_cpus is None else tuple(monitor_cpus)

    def sample(self):
        try:
            rows = processes(self.uuid)
            self.samples.append(
                dict(
                    monotonic=time.monotonic(),
                    processes=rows,
                    foreign_pids=[r["pid"] for r in rows if r["pid"] != self.pid],
                )
            )
        except Exception as exc:
            self.errors.append(str(exc))

    def begin(self, require_idle=True):
        self.initial = idle_snapshot(self.uuid)
        self.sample()
        if (
            require_idle
            and (
                self.initial["memory_MiB"] > 1024
                or self.initial["utilization_percent"] != 0
            )
        ) or not self.clean():
            raise RuntimeError("GPU busy or used by another process before the worker")
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()

    def poll(self):
        if self.monitor_cpus is not None:
            try:
                os.sched_setaffinity(0, self.monitor_cpus)
            except Exception as exc:
                self.errors.append(f"Monitor affinity failed: {exc}")
                return
        while not self.stop.wait(self.interval):
            self.sample()

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=12)
        if self.thread is not None and self.thread.is_alive():
            self.errors.append("Monitor did not finish")
        self.sample()
        return self.record()

    def clean(self):
        return (
            bool(self.samples)
            and not self.errors
            and not any(s["foreign_pids"] for s in self.samples)
        )

    def record(self):
        return dict(
            status="pass" if self.clean() else "fail",
            own_pid=self.pid,
            initial=getattr(self, "initial", None),
            interval_seconds=self.interval,
            monitor_cpus=self.monitor_cpus,
            samples=self.samples,
            errors=self.errors,
        )


# Worker records, read for summaries and for resuming interrupted runs.


def identity(row):
    return tuple(row[k] for k in ("phase", "family", "workload", "config_id", "rep"))


def worker_records(folder):
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    current_digest = sha(manifest_path)
    entries = [
        p
        for p in sorted((folder / "workers").glob("*.json"))
        if ".contaminated." not in p.name
    ]
    seen = set()
    for path in entries:
        row = json.loads(path.read_text())
        require(
            row["manifest_sha256"] == current_digest,
            f"Worker record belongs to another manifest: {path}",
        )
        key = identity(row)
        require(key not in seen, f"Duplicate worker record: {key}")
        seen.add(key)
        yield row


def configurations():
    rows = [
        dict(
            id=f"flash_tiled_{tile_rows}",
            family="flash_tiled",
            backend="flash",
            schedule="full_batch",
            sub_batch=None,
            tile_rows=tile_rows,
            dropout_p=0.1,
        )
        for tile_rows in TILE_ROWS
    ]
    for family in FAMILIES[1:]:
        settings = [("full_batch", None)] + [
            (schedule, sub_batch)
            for schedule in ("retain", "release")
            for sub_batch in SUB_BATCHES
        ]
        for schedule, sub_batch in settings:
            config = dict(
                id=f"{family}_{schedule}" + (f"_{sub_batch}" if sub_batch else ""),
                family=family,
                backend="flash" if family == "flash_untiled" else family,
                schedule=schedule,
                sub_batch=sub_batch,
                tile_rows=None,
                dropout_p=0.1,
            )
            if family == "flash_untiled":
                config.update(
                    joint="specialized",
                    normalization="specialized",
                    tiled=False,
                    whole_graph=False,
                )
            rows.append(config)
    return rows


def study_configurations(study):
    if study == "ablation":
        from .ablation import configurations as variants

        return variants()
    return configurations()


def source_state():
    import nemo
    from .common import source_lock

    root = Path(nemo.__file__).parent
    files = (
        "collections/asr/losses/flash_rnnt.py",
        "collections/asr/parts/triton/rnnt_joint.py",
        "collections/asr/parts/triton/rnnt_loss.py",
        "collections/asr/modules/rnnt.py",
    )
    return dict(paper=source_lock(), nemo={p: sha(root / p) for p in files})


def prepare(folder, workload_path, study, machine, environment):
    from .workloads import load

    workloads = load(workload_path)
    require(
        (workloads["hidden"], workloads["vocab"], workloads["dropout_p"])
        == (640, 1024, 0.1),
        "This benchmark uses H640, V1024 and dropout 0.1",
    )
    manifest = dict(
        workloads,
        study=study,
        workload_file=str(workload_path.resolve()),
        workload_sha256=sha(workload_path),
        configurations=study_configurations(study),
        machine=machine,
        environment=environment,
        source_state=source_state(),
        timing=dict(warmup_seconds=3, measurement_seconds=10, min_samples=30),
        memory=dict(probes=3, oom_confirmations=2),
    )
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "manifest.json", manifest)


def verify(manifest):
    require(
        sha(manifest["workload_file"]) == manifest["workload_sha256"],
        "Workloads changed",
    )
    require(
        source_state() == manifest["source_state"],
        "Benchmark sources changed; use a new output directory",
    )
    for package, version in manifest["environment"]["packages"].items():
        require(
            importlib.metadata.version(package) == version,
            f"Package changed: {package}",
        )


def worker(args):
    startup_started = time.perf_counter()
    from .runtime import setup

    manifest = json.loads((args.folder / "manifest.json").read_text())
    setup(manifest["machine"])
    import torch
    from numba.cuda import config as numba_config

    numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    cpus = [int(v) for v in args.cpus.split(",")]
    os.sched_setaffinity(0, cpus)
    configure_torch(len(cpus))
    device = dict(
        uuid=str(torch.cuda.get_device_properties(0).uuid),
        name=torch.cuda.get_device_name(0),
    )
    environment = manifest["environment"]
    require(
        torch.__version__ == environment["torch"]
        and torch.version.cuda == environment["torch_cuda"],
        "PyTorch build differs from the run manifest",
    )
    capability = list(torch.cuda.get_device_capability(0))
    require(
        all(d["capability"] == capability for d in environment["devices"]),
        "GPU architecture differs from the run manifest",
    )
    record = dict(
        phase=args.phase,
        family=args.family,
        workload=args.workload,
        config_id=args.config_id,
        rep=args.rep,
        gpu=device,
        cpus=cpus,
        manifest_sha256=sha(args.folder / "manifest.json"),
        pid=os.getpid(),
    )
    guard = Guard(device["uuid"], monitor_cpus=cpus[1:] or cpus)
    with device_lock(device["uuid"]):
        try:
            torch.cuda.synchronize()
            deadline = time.monotonic() + 60
            settled = 0
            while settled < 2:
                state = idle_snapshot(device["uuid"])
                pids = processes(device["uuid"])
                clean = (
                    state["memory_MiB"] <= 1024
                    and state["utilization_percent"] == 0
                    and all(p["pid"] == os.getpid() for p in pids)
                )
                settled = settled + 1 if clean else 0
                require(time.monotonic() < deadline, "GPU still busy after 60 s")
                if settled < 2:
                    time.sleep(0.25)
            guard.begin(require_idle=True)
            api, opt = None, None
            if args.family in ("apex", "optimized"):
                from .backends import load

                api, opt, record["extension_manifest_sha256"] = load(
                    manifest["machine"], environment, family=args.family
                )
            if args.phase == "check":
                from .correctness import check

                record["check"] = check(args.family, api, opt, manifest["batch"])
                record["status"] = "pass"
            else:
                from .case import Case
                from .measurement import measure

                config = next(
                    c for c in manifest["configurations"] if c["id"] == args.config_id
                )
                record["config"] = config
                workload = manifest["workloads"][args.workload]
                if config["family"] in ("numba", "optimized"):
                    from numba import cuda

                    # Both losses launch U + 1 threads per block.
                    limit = int(cuda.get_current_device().MAX_THREADS_PER_BLOCK)
                    target_length = max(workload["lengths"])
                    if target_length + 1 > limit:
                        record.update(
                            status="unsupported_length",
                            target_length=target_length,
                            max_threads_per_block=limit,
                        )
                        return record
                case = Case(workload, config, api, opt)
                loss = case.step()
                record["initial_invariants"] = case.invariants(loss)
                require(
                    all(record["initial_invariants"].values()),
                    "Invariants failed on the first step",
                )
                del loss
                case.clear()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                record["startup_seconds"] = time.perf_counter() - startup_started
                os.sched_setaffinity(0, {cpus[0]})
                timing = manifest["timing"]
                record.update(
                    measure(
                        case,
                        warmup_seconds=timing["warmup_seconds"],
                        measurement_seconds=timing["measurement_seconds"],
                        min_samples=timing["min_samples"],
                        memory_probes=manifest["memory"]["probes"],
                        timing=args.phase == "measure",
                    )
                )
                record["status"] = "measured"
        except torch.OutOfMemoryError:
            record.update(
                status="oom" if args.phase != "check" else "runtime_error",
                error=traceback.format_exc(),
            )
        except Exception:
            error = traceback.format_exc()
            status = (
                "compile_error"
                if any(
                    s in error
                    for s in (
                        "BackendCompilerFailed",
                        "CompilationError",
                        "InductorError",
                        "RecompileLimitExceeded",
                    )
                )
                else "runtime_error"
            )
            record.update(status=status, error=error)
        finally:
            record["interference"] = guard.finish()
            if record["interference"]["status"] != "pass":
                record["status"] = "contaminated"
            atomic_json(args.output, record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("phase", "family", "workload", "config_id", "status")
            }
        ),
        flush=True,
    )
    require(
        record["status"] in ("pass", "measured", *FAILURES),
        f"Unexpected worker status: {record['status']}",
    )
    return record


def aggregate(rows, manifest):
    cells = []
    for workload in manifest["workloads"]:
        for config in manifest["configurations"]:
            failed_checks = [
                r
                for r in rows
                if r["phase"] == "check"
                and r["family"] == config["family"]
                and r["status"] != "pass"
            ]
            records = [
                r
                for r in rows
                if r["phase"] != "check"
                and r["workload"] == workload
                and r["config_id"] == config["id"]
            ]
            cells.append(
                aggregate_cell(records, workload, config, manifest, failed_checks)
            )
    return cells


def aggregate_cell(records, workload, config, manifest, failed_checks):
    cell = dict(workload=workload, config=config, status="incomplete")
    if failed_checks:
        return dict(cell, status="check_failed", failure_records=failed_checks)
    failed = [r for r in records if r["status"] in FAILURES and r["status"] != "oom"]
    if failed:
        return dict(cell, status=failed[0]["status"], failure_records=failed)
    oom = [r for r in records if r["status"] == "oom"]
    good = [r for r in records if r["status"] == "measured"]
    if oom:
        if good:
            return dict(
                cell, status="runtime_error", failure_kind="inconsistent_feasibility"
            )
        if len({r["pid"] for r in oom}) >= manifest["memory"]["oom_confirmations"]:
            return dict(cell, status="oom", oom_process_records=len(oom))
        return cell
    if not good:
        return cell
    require(
        len(good) == 1 and good[0]["phase"] == "measure", "Expected one timing record"
    )
    result = good[0]
    probes = result["memory"]["probes"]
    t = result["timing"]
    return dict(
        cell,
        status="complete",
        allocated_bytes=max(p["absolute_peak_allocated_bytes"] for p in probes),
        incremental_allocated_bytes=max(
            p["incremental_peak_allocated_bytes"] for p in probes
        ),
        reserved_bytes=max(p["peak_reserved_bytes"] for p in probes),
        median_ms=t["median_wall_ms"],
        samples_per_second=1000 * manifest["batch"] / t["median_wall_ms"],
        within_process_quarter_span=t["within_process_quarter_span"],
        timing_quality=t["quality"],
    )


def next_jobs(manifest, job, row):
    """A passed check queues its measurements; an OOM is confirmed in a fresh process."""
    if job["phase"] == "check":
        if row["status"] != "pass":
            return []
        return [
            dict(
                phase="measure",
                family=c["family"],
                workload=w,
                config_id=c["id"],
                rep=0,
            )
            for w in manifest["workloads"]
            for c in manifest["configurations"]
            if c["family"] == job["family"]
        ]
    if row["status"] == "oom" and job["rep"] == 0:
        return [
            dict(
                phase="memory",
                family=job["family"],
                workload=job["workload"],
                config_id=job["config_id"],
                rep=1,
            )
        ]
    return []


def campaign(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    verify(manifest)
    lock = (folder / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    devices = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).splitlines():
        index, uuid = [v.strip() for v in line.split(",")]
        if visible is None or index in visible.split(",") or uuid in visible.split(","):
            devices.append(dict(token=index, uuid=uuid))
    require(devices, "No visible GPUs")
    partitions = cpu_partitions(
        os.sched_getaffinity(0), len(devices), manifest["machine"]["threads_per_worker"]
    )
    condition = threading.Condition()
    pending = [
        dict(phase="check", family=family, workload=None, config_id=None, rep=0)
        for family in sorted({c["family"] for c in manifest["configurations"]})
    ]
    records = []
    active = set()
    failures = []

    def complete(job, row):
        records.append(row)
        require(
            row["status"] in ("pass", "measured", *FAILURES),
            f"Unexpected worker status: {row['status']}",
        )
        pending.extend(next_jobs(manifest, job, row))

    def output_for(job):
        stem = f"{job['phase']}_{job['family']}_{job['workload']}_{job['config_id']}_{job['rep']}"
        return folder / "workers" / (stem + ".json")

    # Replay finished jobs before scheduling new ones, so an interrupted run resumes.
    completed_records = {identity(row): row for row in worker_records(folder)}
    replayed = True
    while replayed:
        replayed = False
        for job in list(pending):
            key = identity(job)
            if key not in completed_records:
                continue
            pending.remove(job)
            complete(job, completed_records[key])
            replayed = True

    def execute(slot):
        device = devices[slot]
        while True:
            with condition:
                if failures or (not pending and not active):
                    return
            state = idle_snapshot(device["uuid"])
            if (
                state["memory_MiB"] > 1024
                or state["utilization_percent"] != 0
                or processes(device["uuid"])
            ):
                with condition:
                    condition.wait(5.0)
                continue
            with condition:
                if not pending:
                    condition.wait(2.0)
                    continue
                job = pending.pop(0)
                active.add(slot)
            output = output_for(job)
            output.parent.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "harness.campaign",
                "--folder",
                str(folder.resolve()),
                "--phase",
                job["phase"],
                "--family",
                job["family"],
                "--rep",
                str(job["rep"]),
                "--cpus",
                ",".join(map(str, partitions[slot])),
                "--output",
                str(output.resolve()),
            ]
            for key in ("workload", "config_id"):
                if job[key] is not None:
                    command += ["--" + key.replace("_", "-"), job[key]]
            try:
                log = output.with_suffix(f".{time.time_ns()}.log")
                rc, timeout = run_process(
                    command,
                    dict(os.environ, CUDA_VISIBLE_DEVICES=device["token"]),
                    log,
                    7200,
                )
                if timeout:
                    atomic_json(
                        output,
                        dict(
                            **job,
                            status="timeout",
                            gpu=device,
                            manifest_sha256=sha(folder / "manifest.json"),
                        ),
                    )
                    rc = 0
                require(output.exists(), "Worker wrote no record")
                row = json.loads(output.read_text())
                if row["status"] == "contaminated":
                    attempts = job.get("attempt", 0)
                    archived = output.with_name(
                        output.stem + f".contaminated.{time.time_ns()}.json"
                    )
                    output.rename(archived)
                    require(
                        attempts < 2,
                        f"GPU used by another process in three attempts: {archived}",
                    )
                    with condition:
                        active.discard(slot)
                        pending.append(dict(job, attempt=attempts + 1))
                        condition.notify_all()
                    continue
                require(rc == 0, f"Worker failed with status {row.get('status')}")
                with condition:
                    complete(job, row)
                    active.discard(slot)
                    condition.notify_all()
                print(
                    json.dumps(
                        dict(event="completed", gpu=slot, **job, status=row["status"])
                    ),
                    flush=True,
                )
            except Exception:
                with condition:
                    failures.append(traceback.format_exc())
                    active.discard(slot)
                    condition.notify_all()
                return

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(execute, range(len(devices))))
    verify(manifest)
    if failures:
        atomic_json(
            folder / f"stopped_{time.time_ns()}.json",
            dict(failures=failures, pending=pending),
        )
        raise RuntimeError(failures[0])
    cells = aggregate(records, manifest)
    require(
        all(c["status"] in ("complete", "check_failed", *FAILURES) for c in cells),
        "Incomplete grid",
    )
    if not (folder / "summary.json").exists():
        atomic_json(
            folder / "summary.json",
            dict(
                status="complete_with_failures"
                if any(c["status"] == "check_failed" for c in cells)
                else "complete",
                cells=cells,
                manifest_sha256=sha(folder / "manifest.json"),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="Run one benchmark worker")
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("check", "memory", "measure"), required=True
    )
    parser.add_argument("--family", choices=(*FAMILIES, "ablation"), required=True)
    parser.add_argument("--workload")
    parser.add_argument("--config-id")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    worker(parser.parse_args())


if __name__ == "__main__":
    main()
