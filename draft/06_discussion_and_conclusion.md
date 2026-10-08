# 6. Discussion and conclusion

Flash RNN-T improves the speed and memory efficiency of joint-and-loss
computation while preserving the full alignment space and vocabulary. Tiled
Flash reduces memory at throughput comparable to the fastest evaluated
baseline, while untiled Flash achieves the highest measured throughput among
the evaluated implementations on all three workloads.

The ablation explains the complementary roles of specialization and
checkpointing. Specialized normalization provides most of the throughput
improvement over compiled PyTorch, while checkpointing makes the long
workload feasible even with compiled components. Recomputation has a
measurable cost, but the specialized implementation remains competitive
with external implementations despite that additional work.

Using untiled Flash does not require retaining intermediates for the entire
batch. Sub-batch release makes it feasible on the long workload and faster
than tiled Flash, although it requires more memory and per-sub-batch backward
calls. Tiling instead limits joint intermediate storage to one tile and
preserves a single loss backward call, making memory reduction available without
changing the training loop.

Evaluation is limited to three corpus-derived batches on one GPU architecture
and one model configuration. Measurements cover joint-and-loss computation,
not end-to-end training throughput. Lattice storage still grows with sequence
length, although large joint intermediates are bounded by the tile limit.
Within this scope, Flash RNN-T offers lower-memory tiled execution and
higher-throughput untiled execution, both evaluating the full alignment
space and vocabulary.
