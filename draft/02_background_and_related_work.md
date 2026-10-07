# 2. Background and related work

## 2.1 Joint network and alignment lattice

For an utterance with $T$ encoder positions and $U$ target labels, the joint
network is evaluated at $T(U+1)$ alignment positions, including the empty
target prefix. Let $e_t$ and $p_u$ denote the encoder and prediction network
states projected to the joint hidden dimension, where $u$ counts labels
already emitted. The nonlinear joint [7](https://doi.org/10.1109/ICASSP.2013.6638947) computes

$$
h_{t,u}=\operatorname{Dropout}\!\left(\phi(e_t+p_u)\right),
\qquad
z_{t,u}=Wh_{t,u}+b,
$$

where $\phi$ is the activation function, $h_{t,u}$ is the hidden state, and
$W,b$ project it to logits $z_{t,u}$ over the $V$ vocabulary symbols,
including blank.

Although normalization requires all $V$ logits at each position, the RNN-T
forward–backward recurrence needs only the blank and valid next-target-label
log probabilities: $T(U+1)$ blank log probabilities and $TU$ target log
probabilities per utterance.
Conventional batched training retains the much larger joint intermediates for
gradient computation, so their memory cost accumulates across the batch
[5](https://doi.org/10.1109/ICASSP49357.2023.10095544).

## 2.2 Memory-efficient transducer computation

Li et al. [4](https://doi.org/10.1109/ASRU46091.2019.9003906) reduce memory consumption by packing valid alignment positions
rather than padding every utterance to the batch’s maximum lengths, removing
padded entries from joint hidden states and vocabulary logits. They compute
logit gradients directly, avoiding intermediate probability gradients, and
overwrite the probability tensor with those logit gradients to further reduce
storage.

GPU implementations also reduce intermediate storage and memory traffic.
Warp-transducer [8](https://github.com/HawkAaron/warp-transducer) retains the raw logits and one normalization value per
alignment position, computing blank and target log probabilities directly
within the forward–backward recurrence instead of storing a full probability
tensor.
Apex [9](https://github.com/NVIDIA/apex) supports packing and merged softmax/loss differentiation, while also
fusing joint addition, ReLU and dropout to avoid separate tensors for
intermediate sums and activations and reduce memory traffic.

Kuang et al. [10](https://doi.org/10.21437/Interspeech.2022-10340) instead reduce the number of alignment positions evaluated by the
expensive nonlinear joint. Their lightweight linear joint computes compact blank
and target log probabilities. These guide the selection of a narrow target range
at each encoder step, restricting the nonlinear joint’s computation and
intermediate storage to that range.
This reduces memory and computation by restricting the alignment sum rather
than changing how the full sum is evaluated.

Without restricting alignments, Braun et al. [5](https://doi.org/10.1109/ICASSP49357.2023.10095544) reduce the number of utterances
whose joint intermediates coexist in memory. They compute joint and loss
gradients utterance by utterance, accumulate gradients, and release utterance-local
tensors before propagating through the batched encoders. Consequently, the
large joint intermediates scale with an individual utterance rather than the
entire batch.

Flash RNN-T builds on packing, compact transition log probabilities and
checkpointing to control joint intermediate memory within individual utterances. We contribute
a GPU implementation that couples this finer-grained memory control with
specialized kernels, and quantify its throughput–memory tradeoffs against
existing implementations.
