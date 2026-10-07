# Joint-and-loss benchmarks

From a fresh clone, run `make benchmarks` in the repository root, or in this
directory:

```bash
uv run --locked python -m harness run
```

This installs the locked Python dependencies (including NeMo), fetches and
patches the native backends, builds them, and runs the comparison and ablation.

Requirements: Linux x86-64, Git, [uv](https://docs.astral.sh/uv/), a C++ compiler,
an NVIDIA GPU and the CUDA 13.0 development toolkit. Set `CUDA_HOME` if
`nvcc` is not on PATH. Use `CUDA_VISIBLE_DEVICES` to select GPUs with the same
architecture.

To prepare the backends without starting measurements:

```bash
uv run --locked python -m harness build
```

Dependency checkouts and native builds are reused under `../build/benchmarks/`.
Use `--cache PATH` to change this location. Dependency commits, patches and build
flags are in [dependencies/](dependencies/README.md).

## Measurements

The run writes `../results/raw/comparison/` and `../results/raw/ablation/`.
A study already there is resumed if it was recorded on this machine with the
same workloads and sources, and replaced otherwise; repeat the command to
resume an interrupted run. Use `--output PATH` to write elsewhere and
`--study comparison` or `--study ablation` to run just one study.

The comparison has 198 cells. The ablation has 30: five variants (compiled,
specialized joint, specialized normalization, both specialized, whole-graph
compilation), each tiled and untiled, across three workloads. The supplied
[workloads](../data/workloads/workloads.json) fix B128, H640, V1024, dropout 0.1
and paired lengths. Use `--workloads PATH` to supply a regenerated workload
file; see [data/README.md](../data/README.md).

Measurements start at projected FP16 inputs and include joint/loss forward and
backward. Input projections, encoder/prediction networks, optimizer and DDP
are excluded. Correctness checks precede timing. Each fitting cell uses a
3-second warm-up, at least 10 seconds and 30 measured iterations, and three
memory probes in the same process. Each worker compiles and autotunes on its
own workload before warm-up. OOMs are confirmed in a fresh process. One worker
runs per idle GPU.

Each study folder holds its manifest, worker records and summary. From the
repository root, `make results figures` turns them into the result tables,
CSVs and the paper figure. The paper reports RTX 5090 measurements; other GPUs
need their own build and measurements.
