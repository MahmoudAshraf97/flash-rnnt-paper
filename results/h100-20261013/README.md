# Results

NVIDIA H100 80GB HBM3; B128, H640, V1024, dropout 0.1. Projected-input joint-and-loss forward/backward.

All tested configurations, including OOMs: [comparison.csv](comparison.csv) (198 rows) and [ablation.csv](ablation.csv) (30 rows).

## Comparison

Each entry selects the lowest-memory configuration within 5% of that implementation’s peak throughput in the indicated schedule panel. Tiled Flash and full-batch measurements are available in both panels.

### Retain

| Workload | Implementation | Selected configuration | Samples/s | Peak GiB |
|---|---|---|---:|---:|
| Short | Flash (tiled) | Tile 800,000 rows | 20,128 | 1.30 |
| Short | Flash (untiled) | Sub-batch 64 | 24,831 | 1.31 |
| Short | Apex | Full batch | 26,919 | 3.29 |
| Short | Numba | Full batch | 9,521 | 3.15 |
| Short | Optimized Transducer | Sub-batch 32 | 4,958 | 2.40 |
| Medium | Flash (tiled) | Tile 400,000 rows | 2,839 | 1.84 |
| Medium | Flash (untiled) | Sub-batch 32 | 4,230 | 8.36 |
| Medium | Apex | Sub-batch 32 | 3,392 | 16.48 |
| Medium | Numba | Sub-batch 32 | 1,408 | 16.33 |
| Medium | Optimized Transducer | Sub-batch 4 | 828 | 16.26 |
| Long | Flash (tiled) | Tile 800,000 rows | 163.6 | 4.26 |
| Long | Flash (untiled) | All OOM | — | — |
| Long | Apex | All OOM | — | — |
| Long | Numba | All OOM | — | — |
| Long | Optimized Transducer | All OOM | — | — |

### Release

| Workload | Implementation | Selected configuration | Samples/s | Peak GiB |
|---|---|---|---:|---:|
| Short | Flash (tiled) | Tile 800,000 rows | 20,128 | 1.30 |
| Short | Flash (untiled) | Sub-batch 64 | 25,143 | 0.87 |
| Short | Apex | Full batch | 26,919 | 3.29 |
| Short | Numba | Sub-batch 64 | 9,065 | 1.65 |
| Short | Optimized Transducer | Sub-batch 32 | 5,331 | 0.98 |
| Medium | Flash (tiled) | Tile 400,000 rows | 2,839 | 1.84 |
| Medium | Flash (untiled) | Sub-batch 32 | 4,226 | 3.02 |
| Medium | Apex | Sub-batch 32 | 3,395 | 6.87 |
| Medium | Numba | Sub-batch 32 | 1,412 | 5.38 |
| Medium | Optimized Transducer | Sub-batch 4 | 828 | 1.20 |
| Long | Flash (tiled) | Tile 800,000 rows | 163.6 | 4.26 |
| Long | Flash (untiled) | Sub-batch 8 | 258.6 | 12.55 |
| Long | Apex | Sub-batch 4 | 186.5 | 17.05 |
| Long | Numba | Sub-batch 4 | 87.5 | 11.67 |
| Long | Optimized Transducer | Sub-batch 2 | 35.2 | 8.43 |

## Ablation

Samples/s / peak GiB. Tiled variants use 200,000 rows per tile; untiled variants use the full batch without sub-batching.

| Execution | Components | Short | Medium | Long |
|---|---|---:|---:|---:|
| Untiled | Compiled | 22,743 / 2.80 | 3,177 / 21.97 | OOM |
| Untiled | Specialized joint | 21,741 / 2.58 | 3,205 / 20.44 | OOM |
| Untiled | Specialized normalization | 26,564 / 1.65 | 3,878 / 11.80 | OOM |
| Untiled | Both specialized | 25,333 / 1.44 | 3,852 / 10.27 | OOM |
| Tiled | Compiled | 17,524 / 1.50 | 2,448 / 2.16 | 124.2 / 3.49 |
| Tiled | Specialized joint | 16,954 / 1.35 | 2,357 / 1.89 | 121.7 / 2.63 |
| Tiled | Specialized normalization | 19,879 / 0.86 | 2,834 / 1.29 | 141.4 / 2.58 |
| Tiled | Both specialized | 19,135 / 0.71 | 2,711 / 1.02 | 137.8 / 1.72 |
| Untiled | Whole-graph compilation | 22,524 / 2.26 | 3,230 / 17.60 | OOM |
| Tiled | Whole-graph compilation | 17,784 / 1.23 | 2,504 / 1.80 | 127.0 / 3.11 |

## Files and regeneration

CSV columns: workload, implementation or components, execution settings, throughput, peak allocated memory and status. `full batch` means one backward pass without sub-batching; tile rows apply only to tiled Flash.

Run `make results RESULTS=<campaign-folder>` from the repository root to check the records and regenerate these tables and CSVs. Manifests, summaries and worker records, including timing warnings, are under `raw/comparison/` and `raw/ablation/`.
