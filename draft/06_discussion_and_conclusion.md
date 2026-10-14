# 6. Discussion and conclusion

Flash RNN-T improves the measured throughput–memory tradeoff of joint-and-loss
computation while preserving the full alignment space and vocabulary. On medium
and long, tiled and untiled Flash together form the entire measured frontier on
both GPUs. On short, the selected untiled Flash configurations substantially
reduce memory relative to Apex at modestly lower throughput on H100.

The ablation explains how specialization and checkpointing contribute to this
tradeoff. Specialized normalization improves throughput over compiled PyTorch,
while the specialized joint further reduces memory with a GPU-dependent
throughput effect. Checkpointing trades additional computation for lower memory,
making the long workload feasible even with compiled components.

Using untiled Flash does not require retaining intermediates for the entire
batch. Sub-batch release makes it feasible on the long workload and faster
than tiled Flash, although it requires more memory and per-sub-batch backward
calls. Tiling instead limits joint intermediate storage to one tile and
preserves a single loss backward call, making memory reduction available without
changing the training loop.

Evaluation covers three corpus-derived batches on two GPU architectures
with one model configuration. Measurements cover joint-and-loss computation,
not end-to-end training throughput. Lattice storage still grows with sequence
length, although large joint intermediates are bounded by the tile limit.
Within this scope, Flash RNN-T offers lower-memory tiled execution and
higher-throughput untiled execution, both evaluating the full alignment
space and vocabulary.
