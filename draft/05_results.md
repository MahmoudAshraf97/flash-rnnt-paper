# 5. Results

## 5.1 Comparison with existing implementations

On H100, tiled and untiled Flash together form the entire measured
throughput–memory frontier on medium and long under both schedules (Figure 3).
Apex has the highest peak throughput among external implementations on all
three workloads.

Short presents a tradeoff. Using the 5% selection rule, untiled Flash uses 60%
less memory than Apex under retain with 7.8% lower throughput, and 73% less
memory under release with 6.6% lower throughput.

On medium, the selected tiled Flash configuration reaches 2,839 samples/s at
1.84 GiB, reducing memory ninefold relative to Apex retain at 16.3% lower
throughput. Apex release narrows this memory advantage to a factor of 3.7,
with a similar throughput gap. Untiled Flash instead improves both metrics
over Apex, providing 24–25% higher throughput with 49–56% less memory across
the two schedules.

On long, every tested retain configuration of untiled Flash and the external
implementations runs out of memory even with 80 GB, while all six tiled Flash
configurations fit. The selected tiled configuration reaches 163.6 samples/s
at 4.26 GiB, using fourfold less memory than Apex release at 12.3% lower
throughput. Apex release requires a backward call for each sub-batch, whereas
tiled Flash preserves one backward call per batch. When release scheduling
is used, untiled Flash achieves 39% higher throughput than Apex with 26%
less memory.

![Throughput and memory for the B128 projected-input comparison](figures/frontiers/frontiers.svg)

*Throughput–memory tradeoffs under retain and release scheduling. Points
correspond to tile limits for tiled Flash and sub-batch sizes for the
other implementations.*

## 5.2 Component implementation ablation

We replace compiled PyTorch [17](https://doi.org/10.1145/3620665.3640366) joint and normalization components individually
and together, under untiled and tiled execution (Table 2). Compilation specializes
to each benchmark shape, so the comparison measures steady-state performance
rather than variable-length training overhead. All variants share packing and
the wavefront recurrence; untiled variants process the full batch without
sub-batching.

| Components | Short | Medium | Long |
|---|---:|---:|---:|
| **Untiled** | | | |
| Compiled | 22743 / 2.80 | 3177 / 21.97 | OOM |
| Specialized joint | 21741 / 2.58 | 3205 / 20.44 | OOM |
| Specialized norm. | 26564 / 1.65 | 3878 / 11.80 | OOM |
| Both specialized | 25333 / 1.44 | 3852 / 10.27 | OOM |
| **Tiled** | | | |
| Compiled | 17524 / 1.50 | 2448 / 2.16 | 124.2 / 3.49 |
| Specialized joint | 16954 / 1.35 | 2357 / 1.89 | 121.7 / 2.63 |
| Specialized norm. | 19879 / 0.86 | 2834 / 1.29 | 141.4 / 2.58 |
| Both specialized | 19135 / 0.71 | 2711 / 1.02 | 137.8 / 1.72 |

*Throughput (samples/s) / peak allocated memory (GiB). Tiled variants use a
tile limit of 200k alignment positions.*

Replacing compiled normalization improves tiled throughput by 13–16%.
Adding the specialized joint reduces peak memory by a further 17–33%.
Its throughput effect depends on the GPU, decreasing by 2.5–4.3% on H100
but increasing on RTX 5090. Together, the specialized components remain
9–11% faster than compiled components while using 51–53% less memory.

Even when the joint, vocabulary projection and normalization are compiled
together, tiled Flash achieves 8–9% higher throughput with
42–45% less memory.

## 5.3 Recomputation and release

With the same specialized kernels, tiled checkpointing reduces peak memory
approximately twofold on short and tenfold on medium, at 24.5% and 29.6% lower
throughput, respectively. On long, all full-batch untiled variants run out of
memory, while all tiled variants fit, including those using compiled PyTorch
components.

On medium, switching untiled Flash from retain to release with 16-utterance
sub-batches reduces peak memory from 7.89 to 1.70 GiB while maintaining throughput.
On long, release with four-utterance sub-batches makes untiled execution feasible
at 243.0 samples/s and 7.23 GiB, while every tested retain configuration runs
out of memory.

The RTX 5090 experiments show the same medium- and long-workload frontier
advantage, with nearly identical peak memory for matched configurations.
Relative throughput differs across GPUs, however, as illustrated by the
specialized joint’s opposite effects in §5.2. Complete RTX 5090 results are
available in the benchmark repository.
