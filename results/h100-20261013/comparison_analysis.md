# H100 versus RTX 5090

Same workloads, configurations and measurement settings. These are different host systems as well as different GPUs; throughput ratios are system-level comparisons.

Eight H100 80 GB GPUs, one independent worker per GPU; Intel Xeon Platinum 8481C host, NVIDIA driver 580.178.04, CUDA 13.0. Source commit `7e24452dfd56a3ff93b7ad36e28a4ebd76aa2036`. The VM was acquired at a 0.5 advisor score with explicit approval. No preemption, extra repetitions or measurement-setting changes were needed.

## Comparison

All 198 cells accounted for. Statuses: {'complete': 159, 'oom': 39}. 6 previously OOM configurations now fit.

Correctness checks, source/workload hashes, locked package versions, measurement settings and interference records verified. OOM classifications require two process records.

Newly fitting configurations

- long: `flash_untiled_release_32`
- long: `apex_release_8`
- long: `apex_release_16`
- long: `numba_release_16`
- long: `optimized_release_8`
- long: `optimized_release_16`

Timing warnings: 1. Clean measurements are preserved, including slow or drifting measurements.

- `medium/flash_tiled_25000`: within-process quarter span 5.88% (warning threshold 5%).

For identical configurations that fit both GPUs, the median memory ratio is 1.000; the range is 1.000–1.014. Changes in the selected point’s memory therefore mainly reflect selecting a different tile or sub-batch size, not greater memory consumption for the same configuration.

### Peak throughput

Peaks are taken over all tested configurations and schedules, separately from the 5% selection rule.

| Workload | Implementation | 5090 samples/s | H100 samples/s | H100 / 5090 |
|---|---|---:|---:|---:|
| short | Flash (tiled) | 12120.1 | 20128.4 | 1.66× |
| short | Flash (untiled) | 15651.6 | 25661.7 | 1.64× |
| short | Apex | 12359.9 | 26919.1 | 2.18× |
| short | Numba | 5433.3 | 9521.3 | 1.75× |
| short | Optimized Transducer | 4485.1 | 5330.8 | 1.19× |
| medium | Flash (tiled) | 1573.2 | 2887.5 | 1.84× |
| medium | Flash (untiled) | 2082.4 | 4343.9 | 2.09× |
| medium | Apex | 1564.4 | 3499.1 | 2.24× |
| medium | Numba | 801.7 | 1466.0 | 1.83× |
| medium | Optimized Transducer | 662.7 | 864.1 | 1.30× |
| long | Flash (tiled) | 84.1 | 163.6 | 1.95× |
| long | Flash (untiled) | 118.8 | 262.2 | 2.21× |
| long | Apex | 84.2 | 194.6 | 2.31× |
| long | Numba | 48.7 | 89.7 | 1.84× |
| long | Optimized Transducer | 37.4 | 36.9 | 0.99× |

Apex leads peak throughput on short; untiled Flash leads on medium and long. The 5090 claim that untiled Flash leads on every workload does not extend to H100. Tiled Flash remains the only implementation to fit long under retain, but its peak throughput is below Apex’s on all three workloads when release is available.

The unchanged 5% rule selects larger Flash tiles on H100: 800k, 400k and 800k rows for short, medium and long, versus 50k, 200k and 400k on the 5090. The full curves show the trade-off; the selected point is not the minimum-memory configuration overall.

### Lowest memory within 5% of each implementation’s peak

| Panel | Workload | Implementation | 5090 configuration | H100 configuration | 5090 samples/s | H100 samples/s | 5090 GiB | H100 GiB |
|---|---|---|---|---|---:|---:|---:|---:|
| retain | short | Flash (tiled) | Tile 50,000 rows | Tile 800,000 rows | 12072.1 | 20128.4 | 0.38 | 1.30 |
| retain | short | Flash (untiled) | Sub-batch 64 | Sub-batch 64 | 15225.4 | 24830.9 | 1.30 | 1.31 |
| retain | short | Apex | Sub-batch 16 | Full batch | 11758.8 | 26919.1 | 1.93 | 3.29 |
| retain | short | Numba | Sub-batch 64 | Full batch | 5433.3 | 9521.3 | 2.64 | 3.15 |
| retain | short | Optimized Transducer | Full batch | Sub-batch 32 | 4485.1 | 4957.6 | 3.09 | 2.40 |
| retain | medium | Flash (tiled) | Tile 200,000 rows | Tile 400,000 rows | 1508.0 | 2839.2 | 1.02 | 1.84 |
| retain | medium | Flash (untiled) | Sub-batch 16 | Sub-batch 32 | 2033.5 | 4230.2 | 7.89 | 8.36 |
| retain | medium | Apex | Sub-batch 8 | Sub-batch 32 | 1503.2 | 3392.4 | 14.02 | 16.48 |
| retain | medium | Numba | Sub-batch 16 | Sub-batch 32 | 769.1 | 1407.9 | 15.47 | 16.33 |
| retain | medium | Optimized Transducer | Sub-batch 8 | Sub-batch 4 | 650.9 | 828.3 | 16.47 | 16.26 |
| retain | long | Flash (tiled) | Tile 400,000 rows | Tile 800,000 rows | 82.6 | 163.6 | 2.58 | 4.26 |
| retain | long | Flash (untiled) | OOM | OOM | — | — | — | — |
| retain | long | Apex | OOM | OOM | — | — | — | — |
| retain | long | Numba | OOM | OOM | — | — | — | — |
| retain | long | Optimized Transducer | OOM | OOM | — | — | — | — |
| release | short | Flash (tiled) | Tile 50,000 rows | Tile 800,000 rows | 12072.1 | 20128.4 | 0.38 | 1.30 |
| release | short | Flash (untiled) | Sub-batch 64 | Sub-batch 64 | 15287.6 | 25142.7 | 0.87 | 0.87 |
| release | short | Apex | Sub-batch 32 | Full batch | 11854.7 | 26919.1 | 0.95 | 3.29 |
| release | short | Numba | Sub-batch 32 | Sub-batch 64 | 5160.7 | 9065.3 | 0.93 | 1.65 |
| release | short | Optimized Transducer | Full batch | Sub-batch 32 | 4485.1 | 5330.8 | 3.09 | 0.98 |
| release | medium | Flash (tiled) | Tile 200,000 rows | Tile 400,000 rows | 1508.0 | 2839.2 | 1.02 | 1.84 |
| release | medium | Flash (untiled) | Sub-batch 16 | Sub-batch 32 | 1992.6 | 4225.6 | 1.70 | 3.02 |
| release | medium | Apex | Sub-batch 8 | Sub-batch 32 | 1490.4 | 3395.2 | 2.04 | 6.87 |
| release | medium | Numba | Sub-batch 32 | Sub-batch 32 | 779.3 | 1411.9 | 5.38 | 5.38 |
| release | medium | Optimized Transducer | Sub-batch 8 | Sub-batch 4 | 654.1 | 828.2 | 1.94 | 1.20 |
| release | long | Flash (tiled) | Tile 400,000 rows | Tile 800,000 rows | 82.6 | 163.6 | 2.58 | 4.26 |
| release | long | Flash (untiled) | Sub-batch 4 | Sub-batch 8 | 115.8 | 258.6 | 7.23 | 12.55 |
| release | long | Apex | Sub-batch 2 | Sub-batch 4 | 81.4 | 186.5 | 8.95 | 17.05 |
| release | long | Numba | Sub-batch 2 | Sub-batch 4 | 47.4 | 87.5 | 6.17 | 11.67 |
| release | long | Optimized Transducer | Sub-batch 1 | Sub-batch 2 | 35.6 | 35.2 | 4.95 | 8.43 |

## Ablation

All 30 cells accounted for. Statuses: {'complete': 25, 'oom': 5}. 0 previously OOM configurations now fit.

Correctness checks, source/workload hashes, locked package versions, measurement settings and interference records verified. OOM classifications require two process records.

Timing warnings: 0. Clean measurements are preserved, including slow or drifting measurements.


For identical configurations that fit both GPUs, the median memory ratio is 1.000; the range is 1.000–1.002. Changes in the selected point’s memory therefore mainly reflect selecting a different tile or sub-batch size, not greater memory consumption for the same configuration.

### Component effects on H100

Tiled variants all use 200k rows; these comparisons do not tune tile size per variant.

| Workload | Both specialized vs compiled: throughput | Peak memory reduction | Joint added to specialized normalization: throughput | Peak memory reduction |
|---|---:|---:|---:|---:|
| short | +9.2% | 52.4% | -3.7% | 16.9% |
| medium | +10.7% | 52.9% | -4.3% | 21.2% |
| long | +11.0% | 50.6% | -2.5% | 33.2% |

Specialized normalization supplies the throughput gain. On H100 the specialized joint further reduces memory but lowers tiled throughput by 2.5–4.3%; it is not an additional speedup. Combined specialization still improves tiled throughput by 9.2–11.0% and lowers memory by 50.6–52.9% relative to the separately compiled components. The corresponding combined gain on the 5090 was 14.2–17.3%, with an additional 3.3–4.5% speed gain from the specialized joint. All five untiled long ablation variants remain OOM.

## Regeneration

From the repository root, leaving the 5090 results and manuscript unchanged:

```bash
uv run python -m analysis_tools.summarize_results --comparison results/h100-20261013/raw/comparison --ablation results/h100-20261013/raw/ablation --output results/h100-20261013
uv run python -m analysis_tools.plot_frontiers --source results/h100-20261013/raw/comparison --output results/h100-20261013/figures
```

The per-cell comparison CSVs pair identical workload/configuration IDs across the two campaigns. No unresolved failures or contaminated measurements remain. The single timing warning is listed above.
