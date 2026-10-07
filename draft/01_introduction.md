# 1. Introduction

RNN-T remains a practical choice for streaming speech recognition, with recent
systems demonstrating accurate recognition under real-time constraints
[1](https://doi.org/10.1109/ICASSP55912.2026.11460802),
[2](https://arxiv.org/abs/2609.14991).
Training an RNN-T requires marginalizing over alignments between an input sequence and a target
transcript [3](https://arxiv.org/abs/1211.3711). Its joint network
combines encoder and prediction network states at every valid position in the
alignment lattice, producing a distribution over the vocabulary at each position.
For a nonlinear joint, the resulting hidden states and vocabulary logits can
occupy substantially more memory than the scalar quantities needed by the
forward–backward recurrence [4](https://doi.org/10.1109/ASRU46091.2019.9003906),
[5](https://doi.org/10.1109/ICASSP49357.2023.10095544).
This makes the joint network and loss computation natural targets
for memory optimization.

Existing approaches reduce this memory cost by packing valid alignment positions
[4](https://doi.org/10.1109/ASRU46091.2019.9003906)
and processing utterances individually [5](https://doi.org/10.1109/ICASSP49357.2023.10095544).
However, avoiding padded positions and processing utterances separately do not,
by themselves, bound the large intermediate tensors produced within a single
utterance. This motivates finer-grained control over their storage and lifetime.

We introduce Flash RNN-T, which divides joint network computation into groups
of alignment positions, including within a single utterance. For each group,
it computes the blank and target transition log probabilities needed by the
forward–backward recurrence, while discarding the much larger hidden states and
vocabulary logits.
To compute gradients without retaining these intermediates, we use activation
checkpointing [6](https://arxiv.org/abs/1604.06174) to recompute
them one group at a time during backpropagation, trading additional computation
for control over their memory footprint. All valid alignment positions
and the full vocabulary are still evaluated, preserving the RNN-T objective.

Flash RNN-T combines this memory management strategy with specialized GPU kernels
for joint network and loss computation. We evaluate the resulting throughput–memory
tradeoff against existing implementations and examine how kernel specialization
and the choice to retain or recompute intermediate tensors contribute to it.
On the evaluated medium and long workloads, Flash RNN-T substantially reduces
peak allocated memory while maintaining competitive joint-and-loss throughput
against existing implementations.
