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

The preferred execution strategy depends on the training schedule. Sub-batch
release avoids recomputation and, on medium, improves both throughput and
memory over the selected tiled configuration. Tiling instead controls
intermediate storage within an utterance and preserves a single loss
backward call, making its memory savings available without moving gradient
accumulation into the training loop.

Evaluation is limited to three corpus-derived batches on one GPU architecture
and one model configuration. Measurements cover joint-and-loss computation,
not end-to-end training throughput. Lattice storage still grows with sequence
length, although large joint intermediates are bounded by the tile limit.
Within this scope, Flash RNN-T offers lower-memory tiled execution and
higher-throughput untiled execution, both evaluating the full alignment
space and vocabulary.
