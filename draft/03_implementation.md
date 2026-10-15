# 3. Implementation

## 3.1 Tiled joint computation and checkpointing

Tiled Flash RNN-T evaluates the joint network in small groups of alignment positions
called tiles. For each tile, it retains the compact blank and target transition
log probabilities while discarding the larger hidden states and logits, which
are recomputed during backpropagation.

For a batch of $B$ utterances, let $T_b$ and $U_b$ denote the encoder and target
lengths of utterance $b$. We pack the $N=\sum_{b=1}^{B}T_b(U_b+1)$ valid alignment
positions into approximately balanced tiles of at most $R$ positions, which may
span utterance boundaries (Figure 1).

![Seventeen valid alignment positions from two utterances are packed consecutively and divided into tiles of six, six and five positions. The middle tile crosses the utterance boundary.](figures/tiling/packing.svg)

*Figure 1. Packing two utterances into tiles with an illustrative limit of $R=6$
alignment positions.*

Once all tiles have produced their transition log probabilities, the
forward–backward recurrence computes the loss and its gradients with respect to
them. Because these gradients depend on all valid alignments of each utterance,
joint backpropagation must wait until this computation is complete. Each tile is
then recomputed and differentiated in turn, releasing its hidden states and
logits before processing the next tile (Figure 2).

![Projected states produce hidden states, logits, transition log probabilities and loss. Forward values flow downward and gradients flow upward. Hidden states and logits inside the dashed enclosure are discarded during forward and recomputed during backward; projected states and transition log probabilities are checkpoint inputs and outputs.](figures/checkpoint/execution.svg)

*Figure 2. Dashed enclosure marks intermediates discarded after each tile's
forward computation and recomputed during backward. Outputs from all tiles feed
the loss.*

The projected encoder and prediction network states remain outside the
checkpoint and are reused during recomputation. A saved seed per tile reproduces
the original dropout mask without retaining it. Tile recomputation runs inside
autograd backward, without changing the training loop.

## 3.2 Specialized joint and loss kernels

Within each tile, the joint kernel gathers the projected states for valid
alignment positions and fuses addition, activation and dropout, avoiding
intermediate tensors between these operations. Vocabulary projection uses
PyTorch’s matrix multiplication separately. During backward, the joint kernel
fuses the dropout and activation derivatives with reductions along the encoder
and target axes, producing encoder and prediction network gradients from one
read of the hidden-state gradients.

The normalization kernel reads the full vocabulary logits but returns only the
blank and target log probabilities. During backward, it computes logit gradients
directly from the gradients of these two log probabilities, overwriting the
tile’s logits rather than allocating a separate gradient tensor.

Copies between packed tiles and lattice tensors calculate their coordinates to
avoid storing an index for every position. The kernel identifies the utterance
from the packed boundaries and subtracts its starting offset, then obtains the
encoder and target coordinates through integer division and remainder by
$U_b+1$. This mapping places log probabilities into the lattice during forward
and gathers their gradients into tile order during backward.

## 3.3 Storage and recomputation

Let $H$ be the joint hidden dimension and $V$ the vocabulary size. Retaining
hidden states and logits for all $N$ packed positions requires
$O(N(H+V))$ storage. Checkpointing limits these tile-local tensors to
$O(\min(N,R)(H+V))$. The compact log probabilities and forward–backward buffers
remain rectangular, requiring $O(BT_{\max}(U_{\max}+1))$ storage, where
$T_{\max},U_{\max}$ are the maximum encoder and target lengths in the batch.
Projected states, parameters, gradients and metadata add further storage, so
$R$ controls the large joint intermediates rather than total device memory.

This reduction requires additional joint computation during backward, including
the vocabulary projection. Smaller tiles reduce peak workspace but increase the
number of kernel launches and partial gradient accumulations, creating a
throughput–memory tradeoff.
