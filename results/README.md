# Benchmark results

- [H100](h100-20261013/README.md): primary paper results.
- [RTX 5090](rtx5090/README.md): cross-GPU comparison.

Each campaign contains comparison and ablation CSVs, readable tables, and
the underlying manifests, summaries and worker records in `raw/`.

Regenerate a campaign's tables and CSVs from the repository root:

```bash
make results RESULTS=results/h100-20261013
make results RESULTS=results/rtx5090
```

New benchmark runs write to `results/local/`, which is ignored by Git.
`make results` summarizes that local run without changing the published results.
