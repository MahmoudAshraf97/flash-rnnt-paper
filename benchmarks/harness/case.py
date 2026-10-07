"""One benchmark case: joint parameters, projected inputs and a loss implementation."""

import torch

from .ablation import Pipeline as ComponentPipeline
from .pipeline import Pipeline, forward_loss
from .scheduler import scheduled_backward


def make_fixture(times, lengths, seed, dtype):
    from nemo.collections.asr.modules.rnnt import RNNTJoint

    torch.manual_seed(seed)
    model = (
        RNNTJoint(
            jointnet={
                "encoder_hidden": 512,
                "pred_hidden": 640,
                "joint_hidden": 640,
                "activation": "relu",
                "dropout": 0.0,
            },
            num_classes=1023,
            log_softmax=False,
        )
        .cuda()
        .to(dtype)
    )
    # Use default initialization and natural inputs, including ReLU boundaries.
    encoder = torch.randn(len(times), max(times), 512, device="cuda", dtype=dtype)
    predictor = torch.randn(
        len(times), max(lengths) + 1, 640, device="cuda", dtype=dtype
    )
    targets = torch.randint(0, 1023, (len(times), max(lengths)), device="cuda")
    return (
        model,
        encoder,
        predictor,
        targets,
        torch.tensor(times, device="cuda"),
        torch.tensor(lengths, device="cuda"),
    )


def project_inputs(case):
    """Replace the input projections by their FP16 outputs, computed once as leaves.

    The timed region starts at the projected states, so the projection layers
    become identities and their parameters leave the measured graph.
    """
    joint = case.pipeline.joint
    e, p, *rest = case.args
    with torch.no_grad():
        pe = joint.project_encoder(e.float()).half()
        pp = joint.project_prednet(p.float()).half()
    joint.enc = torch.nn.Identity()
    joint.pred = torch.nn.Identity()
    case.args = (pe.requires_grad_(), pp.requires_grad_(), *rest)


class Case:
    def __init__(self, workload, config, api=None, opt=None, seed=324161):
        joint, e, p, y, tl, ul = make_fixture(
            workload["times"], workload["lengths"], seed, torch.float32
        )
        self.args = (
            e.half().detach().requires_grad_(),
            p.half().detach().requires_grad_(),
            y,
            tl,
            ul,
        )
        joint.dropout = config["dropout_p"]
        joint.train()
        if config["family"] in ("flash_untiled", "ablation"):
            self.pipeline = ComponentPipeline(joint, config)
        else:
            self.pipeline = Pipeline(
                joint,
                config["backend"],
                apex=api,
                optimized=opt,
                tile_rows=config["tile_rows"],
            )
        project_inputs(self)
        self.config = config
        self.host_lengths = (list(workload["times"]), list(workload["lengths"]))

    def step(self):
        c = self.config
        if c["schedule"] != "full_batch":
            return scheduled_backward(
                self.pipeline,
                self.args,
                sub_batch_size=c["sub_batch"],
                schedule=c["schedule"],
            )
        loss = forward_loss(self.pipeline, self.args, host_lengths=self.host_lengths)
        loss.backward()
        return loss.detach()

    def clear(self):
        self.args[0].grad = self.args[1].grad = None
        self.pipeline.zero_grad(set_to_none=True)

    def invariants(self, loss):
        e, p = self.args[:2]
        parameters = list(self.pipeline.parameters())
        gradients = [e.grad, p.grad, *(q.grad for q in parameters)]
        return dict(
            finite=bool(torch.isfinite(loss).all())
            and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients),
            fp32_parameters=all(q.dtype == torch.float32 for q in parameters),
            fp32_parameter_gradients=all(
                q.grad is not None and q.grad.dtype == torch.float32 for q in parameters
            ),
            fp16_states=e.dtype == p.dtype == torch.float16,
            fp16_state_gradients=e.grad is not None
            and p.grad is not None
            and e.grad.dtype == p.grad.dtype == torch.float16,
            loss_fp32=loss.dtype == torch.float32,
            dropout=self.pipeline.joint.training
            and self.pipeline.joint.dropout == self.config["dropout_p"],
        )
