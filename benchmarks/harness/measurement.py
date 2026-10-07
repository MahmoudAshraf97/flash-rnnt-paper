"""Measure steady-state latency and allocator peaks after warm-up."""

import math
import statistics
import time


def measure(
    case,
    *,
    warmup_seconds=3.0,
    measurement_seconds=10.0,
    min_samples=30,
    memory_probes=3,
    timing=True,
):
    """Time one configuration, reading allocator statistics outside the timed steps.

    The caller has already compiled, tuned and checked the first step. With
    timing=False (OOM confirmation) only memory is probed. The final invariants
    are taken from the last measured step.
    """
    import torch

    if min(warmup_seconds, measurement_seconds, min_samples, memory_probes) <= 0:
        raise ValueError("Positive measurement budgets required")
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin.record()
    end.record()
    end.synchronize()
    case.clear()
    torch.cuda.synchronize()
    cleanup_baseline = torch.cuda.memory_allocated()
    warmup = []
    started = time.perf_counter()
    if timing:
        while time.perf_counter() - started < warmup_seconds:
            case.clear()
            start = time.perf_counter()
            loss = case.step()
            end.record()
            end.synchronize()
            warmup.append((time.perf_counter() - start) * 1000)
            del loss
    warmed = time.perf_counter() - started
    samples, probes, elapsed = [], [], 0.0
    count = max(min_samples, memory_probes) if timing else memory_probes
    while len(samples) < count or (timing and elapsed < 1000 * measurement_seconds):
        case.clear()
        probe = len(probes) < memory_probes
        if probe:
            baseline = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        begin.record()
        loss = case.step()
        end.record()
        end.synchronize()
        stop = time.perf_counter()
        value = dict(
            wall_ms=(stop - start) * 1000, cuda_event_ms=begin.elapsed_time(end)
        )
        if not all(math.isfinite(v) and v > 0 for v in value.values()):
            raise ValueError("Invalid timing sample")
        samples.append(value)
        elapsed += value["wall_ms"]
        if probe:
            peak = torch.cuda.max_memory_allocated()
            probes.append(
                dict(
                    baseline_allocated_bytes=baseline,
                    baseline_reserved_bytes=reserved,
                    absolute_peak_allocated_bytes=peak,
                    incremental_peak_allocated_bytes=peak - baseline,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            )
        if len(samples) >= count and (
            not timing or elapsed >= 1000 * measurement_seconds
        ):
            invariants = case.invariants(loss)
        del loss
    case.clear()
    torch.cuda.synchronize()
    cleanup = dict(
        before_bytes=cleanup_baseline, after_bytes=torch.cuda.memory_allocated()
    )
    if cleanup["after_bytes"] != cleanup["before_bytes"]:
        raise RuntimeError(f"Allocation cleanup growth: {cleanup}")
    if not all(invariants.values()):
        raise RuntimeError(f"Final invariants failed: {invariants}")
    values = [s["wall_ms"] for s in samples]
    quarters = (
        [
            statistics.median(values[i * len(values) // 4 : (i + 1) * len(values) // 4])
            for i in range(4)
        ]
        if timing
        else []
    )
    drift = max(quarters) / min(quarters) - 1 if quarters else None
    measured = dict(
        samples=samples,
        median_wall_ms=statistics.median(values),
        warmup_seconds=warmed,
        warmup_ms=warmup,
        within_process_quarter_span=drift,
        quality=dict(
            status="timing_warning"
            if drift is not None and drift > 0.05
            else "stable_timing",
            timing_warnings=["Within-process timing drift >5%"]
            if drift is not None and drift > 0.05
            else [],
        ),
    )
    return dict(
        timing=measured if timing else None,
        memory=dict(
            probes=probes,
            final_invariants=invariants,
        ),
        cleanup=cleanup,
    )
