## What this unlocks

Memory limits affect which experiments a team can run. Reducing the working
space required by the joint network and loss leaves more room for the rest of
the model, larger batches, or longer training examples. Which option is useful
depends on the training setup; the important change is that fewer of those
choices are dictated by temporary tensors that do not need to coexist.

The [paper](https://arxiv.org/abs/2610.23409) explains the implementation and experiments in
detail. The [repository](https://github.com/MahmoudAshraf97/flash-rnnt-paper)
includes the benchmark code and measurement records for reproducing the
comparisons and exploring the tradeoff on other hardware.
