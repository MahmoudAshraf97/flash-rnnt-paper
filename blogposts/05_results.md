## What changed in practice

Together, these changes make the full model cheaper to train. Switching our
NeMo training setup to tiled Flash reduced training-step time by **22%** and
peak GPU memory by **46%**. We measured this with a 114-million-parameter speech
recognition model on eight H100 GPUs, keeping the model, data and batch size
the same.

Projecting those short timing runs to 30,000 training steps gives **about
4 hours 11 minutes with Flash instead of 5 hours 23 minutes**—roughly
72 minutes saved. At the same hourly GPU rate, that means **22% less compute
cost** for those steps. Across repeated training runs, the same GPU budget
can support more experiments.
