"""Float64 reference loss and the loss/gradient comparisons used by the checks."""

import copy
import json

import torch
from torch.nn import functional as F

from .common import ROOT


def diagonal_loss(logits, labels, blank):
    """RNN-T loss of one [T, U+1, V] lattice by a float64 diagonal recurrence.

    Independent of the benchmarked kernels. Only states with t < T are built.
    """
    tmax, states, _ = logits.shape
    umax = states - 1
    if tmax < 1 or len(labels) != umax:
        raise ValueError("invalid lattice or label count")
    scores = logits.double().log_softmax(-1)
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    blank_scores = scores[..., blank]
    label_scores = (
        scores[:, :umax]
        .gather(-1, labels[None, :, None].expand(tmax, -1, 1))
        .squeeze(-1)
    )
    previous = scores.new_zeros(1)
    previous_lo = 0
    for diagonal in range(1, tmax + umax):
        lo, hi = max(0, diagonal - umax), min(tmax - 1, diagonal)
        t = torch.arange(lo, hi + 1, device=logits.device)
        u = diagonal - t
        blank_index = (t - 1 - previous_lo).clamp(0, previous.numel() - 1)
        blank_term = previous[blank_index] + blank_scores[(t - 1).clamp_min(0), u]
        blank_term = torch.where(t > 0, blank_term, -torch.inf)
        if umax:
            label_index = (t - previous_lo).clamp(0, previous.numel() - 1)
            label_term = previous[label_index] + label_scores[t, (u - 1).clamp_min(0)]
            label_term = torch.where(u > 0, label_term, -torch.inf)
            previous = torch.logaddexp(blank_term, label_term)
        else:
            previous = blank_term
        previous_lo = lo
    return -previous[0] - scores[tmax - 1, umax, blank]


def gradient_check(actual, expected, limits):
    if actual is None or actual.shape != expected.shape:
        return {"pass": False, "reason": "missing gradient or shape mismatch"}
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"pass": False, "reason": "nonfinite gradient/reference"}
    if not b.numel():
        return {"pass": True, "elements": 0}
    error = a - b
    norm = b.norm().item()
    rms = b.square().mean().sqrt().item()
    maximum = error.abs().max().item()
    bound = limits["absolute_floor"] + limits["outlier_rms"] * rms
    relative = error.norm().item() / norm if norm else None
    norm_ok = (
        (relative <= limits["relative_l2"])
        if norm
        else maximum <= limits["absolute_floor"]
    )
    outlier_ok = maximum <= bound
    return {
        "pass": bool(norm_ok and outlier_ok),
        "elements": b.numel(),
        "reference_norm": norm,
        "reference_rms": rms,
        "relative_l2": relative,
        "max_abs": maximum,
        "outlier_bound": bound,
        "outlier_count": int((error.abs() > bound).sum()),
        "norm_pass": bool(norm_ok),
        "outlier_pass": bool(outlier_ok),
    }


def loss_check(actual, expected, path_lengths, per_transition_limit):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    paths = torch.as_tensor(path_lengths, dtype=torch.float64)
    if a.shape != b.shape or a.shape != paths.shape or not bool((paths > 0).all()):
        return {"pass": False, "reason": "loss/path shape mismatch"}
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"pass": False, "reason": "nonfinite loss/reference"}
    error = (a - b).abs()
    scaled = error / paths
    return {
        "pass": bool((scaled <= per_transition_limit).all()),
        "max_abs": error.max().item(),
        "max_error_per_transition": scaled.max().item(),
        "limit": per_transition_limit,
    }


def padding_check(gradient, lengths, extra=0):
    if gradient is None or gradient.ndim != 3 or gradient.shape[0] != len(lengths):
        return False
    return all(
        bool(torch.isfinite(gradient[b, n + extra :]).all())
        and torch.count_nonzero(gradient[b, n + extra :]).item() == 0
        for b, n in enumerate(lengths)
    )


def compare(actual, expected, workload, dtype):
    limits = json.loads((ROOT / "harness/tolerances.json").read_text())[
        str(dtype).split(".")[-1]
    ]
    path_mean = sum(
        t + u for t, u in zip(workload["times"], workload["lengths"])
    ) / len(workload["times"])
    result = {
        "loss": loss_check(
            actual["loss"], expected["loss"], path_mean, limits["loss_per_transition"]
        )
    }
    result.update(
        {
            n: gradient_check(g, expected["gradients"][n], limits)
            for n, g in actual["gradients"].items()
        }
    )
    return result


def run(case, dtype):
    """One step of the case; returns its loss and gradients in float64."""
    from .scheduler import scheduled_backward

    case.clear()
    if dtype == torch.float16:
        loss = case.step()
    else:

        def fp32(pipeline, args, host_lengths=None):
            return pipeline(*args, host_lengths=host_lengths)

        if case.config["schedule"] == "full_batch":
            loss = fp32(case.pipeline, case.args, case.host_lengths)
            loss.backward()
            loss = loss.detach()
        else:
            loss = scheduled_backward(
                case.pipeline,
                case.args,
                sub_batch_size=case.config["sub_batch"],
                schedule=case.config["schedule"],
                loss_forward=fp32,
            )
    return dict(
        loss=loss.detach().cpu().double(),
        gradients={
            n: g.detach().cpu().double()
            for n, g in {
                "encoder_input": case.args[0].grad,
                "predictor_input": case.args[1].grad,
                **{n: q.grad for n, q in case.pipeline.joint.named_parameters()},
            }.items()
        },
    )


def reference_run(case, dtype):
    """The same step without dropout, with the loss computed by diagonal_loss."""
    joint = copy.deepcopy(case.pipeline.joint).float()
    e, p = [x.detach().clone().requires_grad_() for x in case.args[:2]]
    targets = case.args[2]
    times, lengths = case.host_lengths
    params = {n: q.to(dtype) for n, q in joint.named_parameters()}

    def linear(x, module, prefix):
        if isinstance(module, torch.nn.Identity):
            return x
        return F.linear(x, params[prefix + ".weight"], params[prefix + ".bias"])

    # Project per sub-batch, as the scheduler does.
    sub_batch = case.config["sub_batch"] or len(times)
    losses = []
    for start in range(0, len(times), sub_batch):
        stop = min(start + sub_batch, len(times))
        tm = max(times[start:stop])
        um = max(lengths[start:stop])
        pe = linear(e[start:stop, :tm], joint.enc, "enc")
        pp = linear(p[start:stop, : um + 1], joint.pred, "pred")
        for b in range(start, stop):
            t, u = times[b], lengths[b]
            a, c = pe[b - start, :t, None], pp[b - start, None, : u + 1]
            # Flash and Apex add in FP32; the PyTorch joint adds in the input precision.
            fused = case.config["backend"] in ("flash", "apex")
            hidden = torch.relu(a.float() + c.float()) if fused else torch.relu(a + c)
            idx = len(joint.joint_net) - 1
            logits = linear(hidden.to(dtype), joint.joint_net[-1], f"joint_net.{idx}")
            losses.append(diagonal_loss(logits.cpu(), targets[b, :u].cpu(), 1023))
    loss = torch.stack(losses).mean()
    loss.backward()
    return dict(
        loss=loss.detach().cpu().double(),
        gradients={
            n: g.detach().cpu().double()
            for n, g in {
                "encoder_input": e.grad,
                "predictor_input": p.grad,
                **{n: q.grad for n, q in joint.named_parameters()},
            }.items()
        },
    )


def check_against_reference(case, workload, dtype):
    """Compare a dropout-free case with reference_run; padding gradients must be zero."""
    actual = run(case, dtype)
    expected = reference_run(case, dtype)
    checks = compare(actual, expected, workload, dtype)
    padding = padding_check(
        actual["gradients"]["encoder_input"], workload["times"]
    ) and padding_check(actual["gradients"]["predictor_input"], workload["lengths"], 1)
    ok = padding and all(c["pass"] for c in checks.values())
    return dict(
        dtype=str(dtype),
        fixture=workload,
        checks=checks,
        padding=padding,
        status="pass" if ok else "fail",
    )
