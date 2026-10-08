# 5. Results

## 5.1 Comparison with existing implementations

Apex has the highest peak throughput among the external implementations on
all three workloads (Figure 3). Using the 5% selection rule, tiled Flash reaches 12,072 samples/s
at 0.38 GiB on short, compared with Apex retain at 11,759 samples/s and
1.93 GiB. Apex release reduces its peak to 0.95 GiB at 11,855 samples/s.

On medium, the selected tiled Flash configuration reaches 1,508 samples/s at
1.02 GiB. Apex retain reaches 1,503 samples/s at 14.02 GiB, giving tiled Flash
13.8-fold lower peak memory and 0.3% higher throughput. Apex release reduces
its peak to 2.04 GiB at 1,490 samples/s, narrowing tiled Flash's memory advantage
to a factor of 2.00.

On long, every tested retain configuration of untiled Flash and the external
implementations runs out of memory, while all six tiled Flash configurations
fit. The selected tiled Flash configuration reaches 82.6 samples/s at 2.58 GiB,
versus Apex release at 81.4 samples/s and 8.95 GiB. Tiled Flash therefore uses
3.5-fold less memory at 1.4% higher throughput.

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
| Compiled | 13232 / 2.80 | 1704 / 21.97 | OOM |
| Specialized joint | 13581 / 2.58 | 1784 / 20.44 | OOM |
| Specialized norm. | 15467 / 1.65 | 1956 / 11.80 | OOM |
| Both specialized | 15943 / 1.44 | 2062 / 10.27 | OOM |
| **Tiled** | | | |
| Compiled | 10216 / 1.50 | 1305 / 2.16 | 64.3 / 3.49 |
| Specialized joint | 10613 / 1.35 | 1346 / 1.89 | 68.5 / 2.63 |
| Specialized norm. | 11482 / 0.86 | 1426 / 1.29 | 73.0 / 2.58 |
| Both specialized | 11861 / 0.71 | 1490 / 1.02 | 75.5 / 1.72 |

*Throughput (samples/s) / peak allocated memory (GiB). Tiled variants use a
tile limit of 200k alignment positions.*

Replacing compiled normalization provides most of the throughput improvement.
In tiled variants with specialized normalization already enabled, the specialized
joint adds 3.3–4.5% throughput while reducing peak memory by 17–33%. Together, the specialized
components improve tiled throughput by 14–17% and reduce memory by 51–53%
relative to compiled components.

Even when the joint, vocabulary projection and normalization are compiled
together, tiled Flash achieves 13–16% higher throughput with
42–45% less memory.

## 5.3 Recomputation and release

With the same specialized kernels, tiled checkpointing reduces peak memory
approximately twofold on short and tenfold on medium, at 25.6% and 27.7% lower
throughput, respectively. On long, all full-batch untiled variants run out of
memory, while all tiled variants fit, including those using compiled PyTorch
components.

On medium, switching untiled Flash from retain to release with 16-utterance
sub-batches reduces peak memory from 7.89 to 1.70 GiB, with 2.0% lower throughput.
On long, release makes untiled execution feasible at 115.8 samples/s and
7.23 GiB, while every tested retain configuration runs out of memory.
