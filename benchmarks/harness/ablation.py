"""Compiled and specialized joint/normalization variants, with and without tiling."""

from functools import lru_cache

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

COMPONENTS = {
    "compiled": ("compiled", "compiled"),
    "specialized_joint": ("specialized", "compiled"),
    "specialized_normalization": ("compiled", "specialized"),
    "specialized": ("specialized", "specialized"),
}


def configurations():
    rows = [
        dict(
            id=f"{'tiled' if tiled else 'untiled'}_{name}",
            joint=joint,
            normalization=normalization,
            tiled=tiled,
            whole_graph=False,
        )
        for tiled in (False, True)
        for name, (joint, normalization) in COMPONENTS.items()
    ]
    rows += [
        dict(rows[0], id="untiled_whole_graph", whole_graph=True),
        dict(rows[4], id="tiled_whole_graph", whole_graph=True),
    ]
    return [
        dict(
            c,
            family="ablation",
            backend="flash",
            schedule="full_batch",
            sub_batch=None,
            tile_rows=200000,
            dropout_p=0.1,
        )
        for c in rows
    ]


def coordinates(offsets, states, start, count):
    index = (torch.arange(count, device=offsets.device) + start).clamp_max(
        offsets[-1] - 1
    )
    batch = torch.bucketize(index, offsets[1:], right=True)
    local = index - offsets[batch]
    return batch, local // states[batch], local % states[batch]


def torch_joint(encoder, predictor, offsets, states, start, rows, dropout_p, dtype):
    batch, t, u = coordinates(offsets, states, start, rows)
    hidden = torch.relu(encoder[batch, t] + predictor[batch, u])
    return F.dropout(hidden, p=dropout_p, training=True).to(dtype)


def torch_normalization(logits, targets, offsets, states, start, blank):
    batch, _, u = coordinates(offsets, states, start, logits.shape[0])
    scores = logits.float()
    lse = torch.logsumexp(scores, -1)
    if targets.shape[1]:
        labels = targets[batch, u.clamp_max(targets.shape[1] - 1)]
        target = scores.gather(1, labels[:, None]).squeeze(1) - lse
        target = torch.where(u < states[batch] - 1, target, 0.0)
    else:
        target = torch.zeros_like(lse)
    return target, scores[:, blank] - lse


def torch_scores(
    e, p, w, bias, targets, offsets, states, start, rows, dropout_p, blank
):
    hidden = torch_joint(e, p, offsets, states, start, rows, dropout_p, w.dtype)
    return torch_normalization(
        F.linear(hidden, w, bias), targets, offsets, states, start, blank
    )


@lru_cache(None)
def compiled(name):
    fn = dict(
        joint=torch_joint,
        normalization=torch_normalization,
        scores=torch_scores,
    )[name]
    # Keep the GEMM shapes unchanged. Inductor's optional padding search
    # otherwise allocates full-size candidate matrices during compilation.
    return torch.compile(
        fn, fullgraph=True, dynamic=False, options={"shape_padding": False}
    )


class Pipeline(torch.nn.Module):
    def __init__(self, joint, config):
        super().__init__()
        self.joint, self.config = joint, config

    def forward(self, e, p, targets, tl, ul, host_lengths=None):
        from nemo.collections.asr.losses.flash_rnnt import _validate_joint
        from nemo.collections.asr.parts.triton.rnnt_joint import (
            lattice_layout,
            packed_scatter,
        )
        from nemo.collections.asr.parts.triton.rnnt_loss import rnnt_loss_triton

        c = self.config
        blank = self.joint.joint_net[-1].out_features - 1
        _validate_joint(self.joint, blank)
        pe, pp = self.joint.project_encoder(e), self.joint.project_prednet(p)
        # Draw the prediction-network mask as FlashRNNTLoss does, consuming the same random numbers.
        mask = (
            torch.rand(len(e), pp.shape[1], device=e.device) > self.joint.masking_prob
        )
        offsets, states, total = lattice_layout(tl, ul, e.shape[1], p.shape[1])
        lengths, targets = tl.int(), targets.contiguous()
        count = total
        if c["tiled"]:
            tiles = (total + c["tile_rows"] - 1) // c["tile_rows"]
            count = (total + tiles - 1) // tiles
        shape = (len(e), e.shape[1], p.shape[1])
        a = torch.zeros(
            shape[0] * shape[1] * shape[2], device=e.device, dtype=torch.float32
        )
        b = torch.zeros_like(a)
        weight, bias = self.joint.joint_net[-1].weight, self.joint.joint_net[-1].bias
        if c["joint"] == "compiled":
            pe, pp = pe.float(), pp.float()
        # GPU scalar starts avoid compiling a new graph for each Python tile offset.
        starts = (
            torch.arange(0, total, count, device=e.device)
            if "compiled" in (c["joint"], c["normalization"])
            else None
        )
        prob = self.joint.dropout if self.joint.training else 0.0
        for tile, start in enumerate(range(0, total, count)):
            rows = min(count, total - start)
            seed = (
                torch.randint(0, 2**31 - 1, (1,), device=e.device, dtype=torch.int32)
                if prob
                else lengths
            )
            gpu_start = starts[tile] if starts is not None else None
            args = (
                pe,
                pp,
                weight,
                bias,
                targets,
                offsets,
                states,
                lengths,
                seed,
                mask,
                start,
                rows,
                prob,
                blank,
                gpu_start,
            )
            if c["tiled"]:
                at, bt = checkpoint(
                    self.score,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=c["joint"] == "compiled",
                )
            else:
                at, bt = self.score(*args)
            a, b = packed_scatter(at, bt, a, b, offsets, states, start, shape)
        losses = rnnt_loss_triton(a.view(shape)[..., :-1], b.view(shape), tl, ul, 0.0)
        return (losses.float() / len(e)).sum()

    def score(
        self,
        e,
        p,
        w,
        bias,
        targets,
        offsets,
        states,
        lengths,
        seed,
        mask,
        start,
        rows,
        prob,
        blank,
        gpu_start,
    ):
        from nemo.collections.asr.parts.triton.rnnt_joint import (
            _PackedJoint,
            _PackedLogProbs,
        )

        c = self.config
        if c["whole_graph"]:
            return compiled("scores")(
                e, p, w, bias, targets, offsets, states, gpu_start, rows, prob, blank
            )
        if c["joint"] == "specialized":
            hidden = _PackedJoint.apply(
                e, p, offsets, states, lengths, seed, mask, start, rows, "relu", prob
            )
        else:
            hidden = compiled("joint")(
                e, p, offsets, states, gpu_start, rows, prob, w.dtype
            )
        logits = F.linear(hidden, w, bias)
        if c["normalization"] == "specialized":
            return _PackedLogProbs.apply(
                logits, targets, offsets, states, start, blank, 0.0, None
            )
        return compiled("normalization")(
            logits, targets, offsets, states, gpu_start, blank
        )
