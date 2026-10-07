"""Backend forward paths and differentiable FP16 parameter casts."""

from contextlib import contextmanager
import torch
from torch.nn import functional as F


class Pipeline(torch.nn.Module):
    def __init__(
        self,
        joint,
        backend,
        *,
        apex=None,
        optimized=None,
        tile_rows=None,
    ):
        super().__init__()
        self.joint = joint
        self.backend = backend
        self.apex, self.optimized = apex, optimized
        self.tile_rows = tile_rows

    def forward(self, e, p, y, tl, ul, host_lengths=None):
        from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba
        from nemo.collections.asr.losses.flash_rnnt import FlashRNNTLoss

        blank = self.joint.joint_net[-1].out_features - 1
        # Every backend takes its dropout probability from the joint, as Flash does.
        dropout_p = self.joint.dropout
        if self.backend == "flash":
            losses = FlashRNNTLoss(blank=blank, max_joint_rows=self.tile_rows)(
                self.joint, e, p, y, tl, ul
            )
        else:
            times, lengths = (
                (tl.tolist(), ul.tolist()) if host_lengths is None else host_lengths
            )
            ti, ui = tl.int().contiguous(), ul.int().contiguous()
            pe, pp = self.joint.enc(e), self.joint.pred(p)
            if self.backend == "apex":
                offsets = (ti.long() * (ui.long() + 1)).cumsum(0)
                joint_op = self.apex.TransducerJoint(
                    pack_output=True,
                    relu=True,
                    dropout=dropout_p > 0.0,
                    dropout_prob=dropout_p,
                    opt=1,
                    fwd_tile_size=4,
                )
                joint_op.train(self.joint.training)
                hidden = joint_op(
                    pe,
                    pp,
                    ti,
                    ui + 1,
                    offsets,
                    sum(t * (u + 1) for t, u in zip(times, lengths)),
                )
            elif self.backend == "optimized":
                # Optimized Transducer takes packed logits.
                hidden = torch.relu(
                    torch.cat(
                        [
                            (pe[b, :t, None] + pp[b, None, : u + 1]).flatten(0, 1)
                            for b, (t, u) in enumerate(zip(times, lengths))
                        ]
                    )
                )
            else:
                hidden = torch.relu(pe[:, :, None] + pp[:, None, :])
            if self.backend != "apex" and dropout_p > 0.0:
                hidden = F.dropout(hidden, p=dropout_p, training=self.joint.training)
            logits = self.joint.joint_net[-1](hidden)
            if self.backend == "apex":
                loss_fn = self.apex.TransducerLoss(
                    fuse_softmax_backward=True, opt=1, packed_input=True
                )
                # Apex computes log-softmax in FP32 from FP16 logits under autocast.
                with torch.autocast("cuda", dtype=torch.float16):
                    losses = loss_fn(
                        logits, y.int().contiguous(), ti, ui, blank, offsets, max(times)
                    )
            elif self.backend == "numba":
                losses = RNNTLossNumba(blank=blank, reduction="none")(
                    logits.contiguous(), y, tl, ul
                )
            else:
                incoming = logits.float().contiguous()
                del logits
                loss = self.optimized(
                    incoming,
                    y.int().contiguous(),
                    ti,
                    ui,
                    blank=blank,
                    from_log_softmax=False,
                    one_sym_per_frame=False,
                    reduction="mean",
                )
                # Optimized Transducer returns the scalar batch mean.
                return loss
        # A contiguous per-example upstream vector is required by Apex.
        return (losses.float() / e.shape[0]).sum()


class _SharedComputeCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, storage):
        ctx.master_dtype = master.dtype
        return storage

    @staticmethod
    def backward(ctx, gradient):
        # Each sub-batch converts its gradient before FP32 accumulation.
        return gradient.to(ctx.master_dtype), None


@contextmanager
def shared_compute_parameters(pipeline):
    """Share FP16 values within one effective batch, not across optimizer steps."""
    if hasattr(pipeline, "_compute_parameter_storage"):
        raise RuntimeError("Nested compute-parameter storage scope")
    pipeline._compute_parameter_storage = {
        name: parameter.detach().to(torch.float16)
        for name, parameter in pipeline.named_parameters()
    }
    try:
        yield
    finally:
        del pipeline._compute_parameter_storage


def forward_loss(pipeline, args, *, host_lengths=None):
    """FP32 master parameters; explicit compute casts remain differentiable."""
    storage = getattr(pipeline, "_compute_parameter_storage", None)
    params = {
        n: q.to(torch.float16)
        if storage is None
        else _SharedComputeCast.apply(q, storage[n])
        for n, q in pipeline.named_parameters()
    }
    with torch.autocast("cuda", enabled=False):
        return torch.func.functional_call(
            pipeline, params, args, dict(host_lengths=host_lengths)
        )
