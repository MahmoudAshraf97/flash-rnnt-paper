# 4. Evaluation methodology

We benchmark joint and loss forward and backward from projected encoder and
prediction network states, including vocabulary projection and gradients with
respect to joint parameters and projected states.

## 4.1 Workloads

We select short and medium workloads from the English subset of Granary
[11](https://doi.org/10.21437/Interspeech.2025-540)
and a long workload from the linked long-form training split of TED-LIUM 3
[12](https://doi.org/10.1109/ICASSP48485.2024.10446286)
to evaluate performance across realistic sequence lengths (Table 1). We use 2D bucketing
by audio duration and transcript length, with 30 duration buckets and two
transcript-length sub-buckets each
[13](https://doi.org/10.1109/ASRU65441.2025.11434640).
Within each duration band, we select the batch closest to the band’s median
duration, alignment-position count and padding fraction.

All experiments use $B=128$, $H=640$, a SentencePiece BPE [14](https://doi.org/10.18653/v1/D18-2012) vocabulary of $V=1024$
including blank, 80 ms encoder steps, and joint dropout of 0.1. We evaluate
primarily on NVIDIA H100 GPUs with 80 GB memory and repeat the experiments on
RTX 5090 GPUs with 32 GB to assess hardware dependence.

| Workload | Mean duration (s) | Encoder length $T$ | Transcript length $U$ | Valid alignment positions |
|---|---:|---:|---:|---:|
| Short | 4.92 | 47–69 | 29–50 | 282,687 |
| Medium | 18.72 | 227–244 | 40–96 | 2,300,131 |
| Long | 69.62 | 851–885 | 325–458 | 41,122,726 |

*Representative workload batches.*

## 4.2 Comparators and execution controls

We compare tiled and untiled Flash RNN-T with Apex [9](https://github.com/NVIDIA/apex),
NeMo Numba [15](https://github.com/NVIDIA/NeMo) and Optimized
Transducer [16](https://github.com/csukuangfj/optimized_transducer), an implementation
of Li et al.’s loss optimizations [4](https://doi.org/10.1109/ASRU46091.2019.9003906).
The untiled variant omits tile checkpointing.

Joint inputs and computation use FP16; parameters, parameter gradients,
normalization and the forward–backward recurrence use FP32. The normalization
exception is Numba’s FP16 shared-memory reduction intermediates; its denominators
remain FP32. Optimized Transducer additionally converts logits to FP32 to reuse
their buffer for FP32 gradients, with conversion costs included in timing and
memory.

Tiled Flash’s tile limit doubles from 25k to 800k alignment positions. Other
implementations process the full batch or sub-batches of 1–64 utterances, doubling
the sub-batch size at each step. Retain keeps sub-batch graphs until a combined
backward call. Release applies Braun et al.’s sample-wise computation
[5](https://doi.org/10.1109/ICASSP49357.2023.10095544) to sub-batches, differentiating
each immediately to accumulate parameter and projected-state gradients while
freeing intermediates.
Both compute the full-batch mean loss. Tiled Flash uses tile checkpointing in both
panels. Apex uses joint tile size 4 and joint/loss optimization setting 1.

## 4.3 Measurement and numerical scope

We report throughput as batch size divided by median latency and memory as
PyTorch’s peak allocated memory, including inputs and parameters. Measurements
exclude compilation, autotuning and warm-up. Summary comparisons select each
implementation’s lowest-memory configuration achieving at least 95% of its peak
measured throughput, preserving near-peak speed while reducing memory. We validate
losses and gradients against reference calculations; the benchmark repository
documents the measurement protocol and correctness tests.
