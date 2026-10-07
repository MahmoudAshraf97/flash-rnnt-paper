"""Correctness checks run once per implementation before any measurement."""

import gc
from unittest.mock import patch

import torch

from . import ablation
from .case import Case
from .common import require
from .pipeline import Pipeline, forward_loss
from .reference import check_against_reference, compare, run


def to_fp32(case):
    case.args = tuple(
        x.float().detach().requires_grad_() if i < 2 else x
        for i, x in enumerate(case.args)
    )


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()


def check(family, api, opt, batch):
    """Small dropout-free cases against the float64 reference; sub-batches against
    the full batch."""
    from .campaign import configurations

    if family == "ablation":
        return check_ablation()
    base = next(c for c in configurations() if c["family"] == family)
    cases = []
    for dtype in (torch.float32, torch.float16):
        for lengths in ([3, 1, 0], [0, 0, 0]):
            workload = dict(times=[9, 6, 3], lengths=lengths)
            case = Case(workload, dict(base, tile_rows=13, dropout_p=0.0), api, opt)
            if dtype == torch.float32:
                to_fp32(case)
            cases.append(check_against_reference(case, workload, dtype))
            del case
            release_memory()
    require(
        all(c["status"] == "pass" for c in cases),
        f"{family}: reference check failed: {cases}",
    )

    schedules = (
        [("full_batch", None)]
        if family == "flash_tiled"
        else [
            ("full_batch", None),
            ("retain", batch // 2),
            ("release", batch // 2),
            ("release", 1),
        ]
    )
    workload = dict(
        times=([3, 2, 1] * batch)[:batch], lengths=([2, 1, 0] * batch)[:batch]
    )
    sub_batch_checks, expected = [], None
    for schedule, sub_batch in schedules:
        config = dict(base, dropout_p=0.0, schedule=schedule, sub_batch=sub_batch)
        case = Case(workload, config, api, opt)
        to_fp32(case)
        actual = run(case, torch.float32)
        if expected is None:
            expected = actual
        else:
            checks = compare(actual, expected, workload, torch.float32)
            require(
                all(c["pass"] for c in checks.values()),
                "Sub-batch gradients differ from the full batch",
            )
            sub_batch_checks.append(
                dict(schedule=schedule, sub_batch=sub_batch, checks=checks)
            )
        del case
        release_memory()
    return dict(status="pass", cases=cases, sub_batch_checks=sub_batch_checks)


def check_ablation():
    torch._dynamo.config.recompile_limit = 128
    cases = []
    for config in ablation.configurations():
        for dtype in (torch.float32, torch.float16):
            for lengths in ([3, 1, 0], [0, 0, 0]):
                row = check_variant(
                    config, dict(times=[9, 6, 3], lengths=lengths), dtype
                )
                require(
                    row["status"] == "pass",
                    f"{config['id']}: reference check failed: {row}",
                )
                cases.append(row)
                print(f"Checked {config['id']} {dtype} lengths={lengths}", flush=True)
                release_memory()
        if config["tiled"]:
            row = check_checkpoint_replay(config)
            require(
                row["status"] == "pass",
                f"{config['id']}: checkpoint replay failed: {row}",
            )
            cases.append(row)
        row = check_saved_tensors(config)
        require(
            row["status"] == "pass",
            f"{config['id']}: saved-tensor check failed: {row}",
        )
        cases.append(row)
        release_memory()
    flash_match = check_flash_match()
    require(
        flash_match["status"] == "pass",
        f"Tiled specialized variant differs from FlashRNNTLoss: {flash_match}",
    )
    return dict(status="pass", cases=cases, flash_match=flash_match)


def check_variant(config, workload, dtype):
    case = Case(workload, dict(config, tile_rows=13, dropout_p=0.0))
    if dtype == torch.float32:
        to_fp32(case)
    return dict(check_against_reference(case, workload, dtype), id=config["id"])


def check_checkpoint_replay(config):
    """Recomputing a checkpointed tile must reproduce its forward dropout masks."""
    workload = dict(times=[9, 6, 3], lengths=[3, 1, 0])
    case = Case(workload, dict(config, tile_rows=13))
    run(case, torch.float16)

    def direct(fn, *args, **kwargs):
        return fn(*args)

    torch.manual_seed(131071)
    actual = run(case, torch.float16)
    torch.manual_seed(131071)
    with patch.object(ablation, "checkpoint", direct):
        expected = run(case, torch.float16)
    checks = compare(actual, expected, workload, torch.float16)
    require(torch.equal(actual["loss"], expected["loss"]), "Replayed loss changed")
    return dict(
        id=config["id"],
        checks=checks,
        status="pass" if all(c["pass"] for c in checks.values()) else "fail",
    )


def check_saved_tensors(config):
    """Tiled variants keep no hidden states or logits for backward; memory is released."""
    workload = dict(times=[19, 13, 3], lengths=[6, 2, 0])
    case = Case(workload, dict(config, tile_rows=23))
    case.step()
    case.clear()
    gc.collect()
    torch.cuda.synchronize()
    saved = []

    def pack(t):
        saved.append(
            dict(
                shape=list(t.shape),
                dtype=str(t.dtype),
                storage=t.untyped_storage().data_ptr(),
                storage_bytes=t.untyped_storage().nbytes(),
            )
        )
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = forward_loss(case.pipeline, case.args, host_lengths=case.host_lengths)
    loss.backward()
    del loss
    case.clear()
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    case.step()
    case.clear()
    gc.collect()
    torch.cuda.synchronize()
    cleanup = torch.cuda.memory_allocated() == baseline
    # Several saved tensors can share one storage; count each storage once.
    unique = {r["storage"]: r["storage_bytes"] for r in saved}
    total = sum(t * (u + 1) for t, u in zip(workload["times"], workload["lengths"]))
    intermediates = {
        r["storage"]: r["storage_bytes"]
        for r in saved
        if len(r["shape"]) == 2
        and 0 < r["shape"][0] <= total
        and r["shape"][1] in (640, 1024)
    }
    lifetimes = (
        not intermediates
        if config["tiled"]
        else (
            bool(intermediates)
            if config["joint"] == config["normalization"] == "specialized"
            else True
        )
    )
    result = dict(
        id=config["id"],
        saved=saved,
        unique_saved_storage_bytes=sum(unique.values()),
        cleanup=cleanup,
        saved_joint_intermediate_bytes=sum(intermediates.values()),
        lifetimes=lifetimes,
    )
    result["status"] = "pass" if cleanup and lifetimes else "fail"
    return result


def check_flash_match():
    """The tiled specialized variant reproduces FlashRNNTLoss exactly."""
    workload = dict(times=[33, 17, 3], lengths=[9, 5, 0])
    config = dict(
        next(c for c in ablation.configurations() if c["id"] == "tiled_specialized"),
        tile_rows=53,
    )
    case = Case(workload, config)
    torch.manual_seed(524287)
    actual = run(case, torch.float16)
    torch.manual_seed(524287)
    case.pipeline = Pipeline(
        case.pipeline.joint,
        "flash",
        tile_rows=config["tile_rows"],
    )
    expected = run(case, torch.float16)
    checks = compare(actual, expected, workload, torch.float16)
    require(
        torch.equal(actual["loss"], expected["loss"]),
        "Tiled specialized loss differs from FlashRNNTLoss",
    )
    return dict(
        checks=checks,
        status="pass" if all(c["pass"] for c in checks.values()) else "fail",
    )
