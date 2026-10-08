# Results

NVIDIA GeForce RTX 5090; B128, H640, V1024, dropout 0.1. Projected-input joint-and-loss forward/backward.

All tested configurations, including OOMs: [comparison.csv](comparison.csv) (198 rows) and [ablation.csv](ablation.csv) (30 rows).

## Comparison

Each entry selects the lowest-memory configuration within 5% of that implementation’s peak throughput in the indicated schedule panel. Tiled Flash and full-batch measurements are available in both panels.

### Retain

| Workload | Implementation | Selected configuration | Samples/s | Peak GiB |
|---|---|---|---:|---:|
| Short | Flash (tiled) | Tile 50,000 rows | 12,072 | 0.38 |
| Short | Flash (untiled) | Sub-batch 64 | 15,225 | 1.30 |
| Short | Apex | Sub-batch 16 | 11,759 | 1.93 |
| Short | Numba | Sub-batch 64 | 5,433 | 2.64 |
| Short | Optimized Transducer | Full batch | 4,485 | 3.09 |
| Medium | Flash (tiled) | Tile 200,000 rows | 1,508 | 1.02 |
| Medium | Flash (untiled) | Sub-batch 16 | 2,034 | 7.89 |
| Medium | Apex | Sub-batch 8 | 1,503 | 14.02 |
| Medium | Numba | Sub-batch 16 | 769 | 15.47 |
| Medium | Optimized Transducer | Sub-batch 8 | 651 | 16.47 |
| Long | Flash (tiled) | Tile 400,000 rows | 82.6 | 2.58 |
| Long | Flash (untiled) | All OOM | — | — |
| Long | Apex | All OOM | — | — |
| Long | Numba | All OOM | — | — |
| Long | Optimized Transducer | All OOM | — | — |

### Release

| Workload | Implementation | Selected configuration | Samples/s | Peak GiB |
|---|---|---|---:|---:|
| Short | Flash (tiled) | Tile 50,000 rows | 12,072 | 0.38 |
| Short | Flash (untiled) | Sub-batch 64 | 15,288 | 0.87 |
| Short | Apex | Sub-batch 32 | 11,855 | 0.95 |
| Short | Numba | Sub-batch 32 | 5,161 | 0.93 |
| Short | Optimized Transducer | Full batch | 4,485 | 3.09 |
| Medium | Flash (tiled) | Tile 200,000 rows | 1,508 | 1.02 |
| Medium | Flash (untiled) | Sub-batch 16 | 1,993 | 1.70 |
| Medium | Apex | Sub-batch 8 | 1,490 | 2.04 |
| Medium | Numba | Sub-batch 32 | 779 | 5.38 |
| Medium | Optimized Transducer | Sub-batch 8 | 654 | 1.94 |
| Long | Flash (tiled) | Tile 400,000 rows | 82.6 | 2.58 |
| Long | Flash (untiled) | Sub-batch 4 | 115.8 | 7.23 |
| Long | Apex | Sub-batch 2 | 81.4 | 8.95 |
| Long | Numba | Sub-batch 2 | 47.4 | 6.17 |
| Long | Optimized Transducer | Sub-batch 1 | 35.6 | 4.95 |

## Ablation

Samples/s / peak GiB. Tiled variants use 200,000 rows per tile; untiled variants use the full batch without sub-batching.

| Execution | Components | Short | Medium | Long |
|---|---|---:|---:|---:|
| Untiled | Compiled | 13,232 / 2.80 | 1,704 / 21.97 | OOM |
| Untiled | Specialized joint | 13,581 / 2.58 | 1,784 / 20.44 | OOM |
| Untiled | Specialized normalization | 15,467 / 1.65 | 1,956 / 11.80 | OOM |
| Untiled | Both specialized | 15,943 / 1.44 | 2,062 / 10.27 | OOM |
| Tiled | Compiled | 10,216 / 1.50 | 1,305 / 2.16 | 64.3 / 3.49 |
| Tiled | Specialized joint | 10,613 / 1.35 | 1,346 / 1.89 | 68.5 / 2.63 |
| Tiled | Specialized normalization | 11,482 / 0.86 | 1,426 / 1.29 | 73.0 / 2.58 |
| Tiled | Both specialized | 11,861 / 0.71 | 1,490 / 1.02 | 75.5 / 1.72 |
| Untiled | Whole-graph compilation | 12,993 / 2.26 | 1,710 / 17.60 | OOM |
| Tiled | Whole-graph compilation | 10,254 / 1.23 | 1,315 / 1.80 | 65.9 / 3.11 |

## Files and regeneration

CSV columns: workload, implementation or components, execution settings, throughput, peak allocated memory and status. `full batch` means one backward pass without sub-batching; tile rows apply only to tiled Flash.

Run `make results` to check the records and regenerate these tables and CSVs. Manifests, summaries and worker records, including timing warnings, are under `raw/comparison/` and `raw/ablation/`.
