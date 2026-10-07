# 5. Results

## 5.1 Comparison with existing implementations

Apex has the highest peak throughput among the external implementations on
all three workloads (Figure 3). Using the 5% selection rule, tiled Flash reaches 11,060 samples/s
at 0.54 GiB on short, compared with Apex retain at 11,302 samples/s and
2.23 GiB. Apex release reduces its peak to 1.00 GiB at 11,337 samples/s.

On medium, the selected tiled Flash configuration reaches 1,544 samples/s at
1.85 GiB. Apex retain reaches 1,488 samples/s at 14.07 GiB, giving tiled Flash
7.6-fold lower peak memory and 3.7% higher throughput. Apex release reduces
its peak to 2.07 GiB at 1,496 samples/s, narrowing tiled Flash's memory advantage
to a factor of 1.12.

On long, every tested retain configuration of untiled Flash and the external
implementations runs out of memory, while all six tiled Flash configurations
fit. The selected tiled Flash configuration reaches 93.3 samples/s at 2.49 GiB,
versus Apex release at 94.6 samples/s and 7.96 GiB. Tiled Flash therefore uses
3.2-fold less memory at 1.4% lower throughput.

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
| Compiled | 12568 / 2.91 | 1701 / 22.08 | OOM |
| Specialized joint | 12956 / 2.69 | 1772 / 20.55 | OOM |
| Specialized norm. | 14561 / 1.71 | 1940 / 11.86 | OOM |
| Both specialized | 15143 / 1.49 | 2045 / 10.32 | OOM |
| **Tiled** | | | |
| Compiled | 9821 / 1.56 | 1281 / 2.17 | 76.5 / 3.36 |
| Specialized joint | 9985 / 1.41 | 1324 / 1.90 | 79.2 / 2.55 |
| Specialized norm. | 10977 / 0.89 | 1462 / 1.30 | 85.1 / 2.46 |
| Both specialized | 11295 / 0.74 | 1485 / 1.02 | 88.5 / 1.64 |

*Throughput (samples/s) / peak allocated memory (GiB). Tiled variants use a
tile limit of 200k alignment positions.*

Replacing compiled normalization provides most of the throughput improvement.
In tiled variants with specialized normalization already enabled, the specialized
joint adds 1.6–4.0% throughput while reducing peak memory by 17–33%. Together, the specialized
components improve tiled throughput by 15–16% and reduce memory by 51–53%
relative to compiled components.

Even when the joint, vocabulary projection and normalization are compiled
together, tiled Flash achieves approximately 15% higher throughput with
42–45% less memory.

## 5.3 Recomputation and release

With the same specialized kernels, tiled checkpointing reduces peak memory
approximately twofold on short and tenfold on medium, at 25.4% and 27.4% lower
throughput, respectively. On long, all full-batch untiled variants run out of
memory, while all tiled variants fit, including those using compiled PyTorch
components.

Untiled Flash can instead use sub-batch release, reaching 2,022 samples/s at
1.68 GiB on medium and improving both throughput and memory over the selected
tiled configuration. On long, it reaches 134.3 samples/s at 6.08 GiB, offering
higher throughput at greater memory cost.
