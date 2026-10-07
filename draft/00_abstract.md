# Abstract

The recurrent neural network transducer (RNN-T) remains a practical choice for
streaming speech recognition, but training requires large intermediate tensors.
We present Flash RNN-T, a joint and loss implementation using specialized kernels
over the full alignment space and vocabulary. Its tiled variant recomputes
intermediates for groups of alignment positions during backward.
In joint-and-loss benchmarks with one backward call per batch, tiled Flash reduces
peak allocated memory 4.1-7.6-fold on short and medium workloads versus the fastest
evaluated baseline, with throughput 2.1% lower to 3.7% higher. We select each
implementation's lowest-memory configuration within 5% of its peak measured
throughput. On the long workload, tiled Flash fits within 32 GB while all tested
baseline configurations using the same backward schedule run out of memory.
Its untiled variant avoids recomputation and achieves the highest measured
throughput among evaluated implementations on all three workloads.
