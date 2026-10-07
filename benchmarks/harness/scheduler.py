"""Retain/release sub-batch scheduling with shared compute parameters."""

from contextlib import nullcontext
import torch
from harness.pipeline import forward_loss, shared_compute_parameters


def scheduled_backward(
    pipeline,
    args,
    *,
    sub_batch_size,
    schedule,
    loss_forward=forward_loss,
):
    scope = (
        shared_compute_parameters(pipeline)
        if loss_forward is forward_loss
        else nullcontext()
    )
    with scope:
        return _scheduled_backward(
            pipeline,
            args,
            sub_batch_size=sub_batch_size,
            schedule=schedule,
            loss_forward=loss_forward,
        )


def _scheduled_backward(pipeline, args, *, sub_batch_size, schedule, loss_forward):
    """Accumulate parameter gradients per sub-batch and return state gradients once."""
    e, p, y, tl, ul = args
    batch = e.shape[0]
    times, lengths = tl.tolist(), ul.tolist()
    # Sub-batches cover disjoint utterances, so each state row's gradient is copied
    # once into an FP32 buffer. Joint parameter gradients accumulate in FP32.
    de = torch.zeros_like(e, dtype=torch.float32)
    dp = torch.zeros_like(p, dtype=torch.float32)
    pending, means = [], []

    def collect(ee, pp, start, stop, tmax, umax):
        if ee.grad is None or pp.grad is None:
            raise RuntimeError("Missing state-boundary gradient")
        de[start:stop, :tmax].copy_(ee.grad)
        dp[start:stop, : umax + 1].copy_(pp.grad)

    for start in range(0, batch, sub_batch_size):
        stop = min(start + sub_batch_size, batch)
        tmax, umax = max(times[start:stop]), max(lengths[start:stop])
        ee = e[start:stop, :tmax].detach().contiguous().requires_grad_()
        pp = p[start:stop, : umax + 1].detach().contiguous().requires_grad_()
        sub_batch_args = (
            ee,
            pp,
            y[start:stop, :umax].contiguous(),
            tl[start:stop].contiguous(),
            ul[start:stop].contiguous(),
        )
        mean = loss_forward(
            pipeline,
            sub_batch_args,
            host_lengths=(times[start:stop], lengths[start:stop]),
        )
        if mean.ndim != 0 or mean.dtype != torch.float32:
            raise RuntimeError("Loss must be an FP32 scalar mean")
        weighted = mean * ((stop - start) / batch)
        means.append(weighted.detach())
        if schedule == "release":
            weighted.backward()
            collect(ee, pp, start, stop, tmax, umax)
        else:
            pending.append((weighted, ee, pp, start, stop, tmax, umax))
        del mean, weighted, ee, pp, sub_batch_args
    if schedule == "retain":
        total = torch.stack([item[0] for item in pending]).sum()
        total.backward()
        for _, ee, pp, start, stop, tmax, umax in pending:
            collect(ee, pp, start, stop, tmax, umax)
        del _, ee, pp, total
        pending.clear()
    # One joint autograd invocation preserves shared encoder/predictor ancestors.
    torch.autograd.backward((e, p), (de.to(e.dtype), dp.to(p.dtype)))
    return torch.stack(means).sum()
