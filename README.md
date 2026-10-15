# Flash RNN-T

Paper sources, benchmark harness and measurement records for
*Flash RNN-T: Memory-Efficient Joint Network and Transducer Loss Computation*.

## Build the paper

Use uv, Python 3.12 and a TeX Live installation with pdfLaTeX, BibTeX, PGF/TikZ
and the `dejavu` fonts (for the figures).
No GPU or speech dataset is needed to build the paper.

```bash
uv sync --locked
make paper
make check
```

The PDF is written to `build/paper/main.pdf`. `make paper TEX=tectonic` typesets
with Tectonic, but pdfLaTeX must still be on `PATH`: Matplotlib's pgf backend
drives it for the frontier figure's text metrics. Set `PYTHON=/path/to/python`
when needed.

Edit the [Markdown sections](draft/README.md) and
[bibliography](draft/references.bib), not the generated LaTeX. The generator
checks citation numbering and includes the official Interspeech class and
bibliography style without modifying their formatting.

### arXiv

```bash
make arxiv
```

This builds the same manuscript using the standard `article` class, writes
`build/arxiv/main.pdf`, and packages `dist/flash-rnnt-arxiv.zip` for submission.
Use TeX Live 2025 or newer locally; select TeX Live 2025 and pdfLaTeX on arXiv.
The archive contains `main.tex`, the bibliography and its style, the
`latexml.sty` compatibility shim, and the three figures as TikZ and pgf sources.
arXiv's HTML renders these as inline SVG with
selectable text that follows its dark mode, which embedded PDF figures cannot.
The archive needs no Python, external fonts, network access or figure conversion
to compile. It deliberately excludes the repository, benchmark records and
standalone diagram sources.

The arXiv target uses standard author metadata and natural float placement. Its two-column, 10-point, Times-style layout uses the
Interspeech text area (170 × 235 mm) and column gap (10 mm). Method figures and
tables occupy one column; the frontier figure spans both. Title styling and
pagination may differ from the conference PDF. Both targets share the Markdown
sections, bibliography and figures.

To check the actual upload independently, extract the archive into an empty
directory and run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
latexmlc --dest=html/main.html main.tex
```

LaTeXML is only needed for the HTML check, not submission or PDF compilation.
Inspect both outputs, including author details, equations, table headers,
figures, captions and links. Local conversion does not guarantee identical
output from arXiv's deployed converter.

## Run the benchmarks

With an NVIDIA GPU, CUDA 13.0 toolkit and C++ compiler installed:

```bash
cd benchmarks
uv run --locked python -m harness run
```

The command installs the locked dependencies, builds the native backends and
runs both studies. See [benchmark instructions](benchmarks/README.md) for GPU selection,
outputs and resuming runs.

## Repository contents

| Directory | Contents |
|---|---|
| [draft/](draft/README.md) | Manuscript, bibliography and the three current figures |
| [latex/](latex/README.md) | Official Interspeech 2026 class and BibTeX style |
| `analysis_tools/` | Paper generation, layout checks and figure rendering |
| [benchmarks/](benchmarks/README.md) | Benchmark harness, dependency pins, patches and build instructions |
| [data/](data/README.md) | Paired workload lengths, selection instructions and tokenizer |
| [results/](results/README.md) | Separate H100 and RTX 5090 tables, CSVs and raw evidence |

## Reproduce the paper

The paper builds above use the published H100 results. To repeat the experiments
and summarize a separate local run:

```bash
make build-tokenizer   # data/tokenizer/tokenizer.model from public transcripts (~180 MB download)
make workloads         # data/workloads/workloads.json from Granary and TED-LIUM metadata (hours, CPU)
make benchmarks        # results/local/raw/ measurements (GPU)
make results           # results/local/ tables and CSVs from the local run
```

`make benchmarks` needs an NVIDIA GPU, the CUDA 13.0 toolkit and a C++ compiler;
see [benchmarks/README.md](benchmarks/README.md). It replaces the records in
`results/local/raw/` unless they belong to a local run with matching environment,
settings, workloads and sources, in which case it resumes them. The numbers
quoted in `draft/*.md` use the H100 campaign in
`results/h100-20261013/`, with `results/rtx5090/` preserving the RTX 5090 cross-check.
The paper comparison figure defaults to the H100 campaign. To plot a local
rerun separately, use `uv run --locked python -m analysis_tools.plot_frontiers
--source results/local/raw/comparison --output build/figures/local`.
The implementation is pinned to NeMo commit
`c0188115c167f027b14ad550c968e9ff83ba8f7f`.

The two method diagrams are TikZ sources in `draft/figures/tiling/` and
`draft/figures/checkpoint/`: a `*-body.tex` picture that the paper inputs
directly and a `standalone` wrapper for previews. `make method-figures` compiles
the wrappers with `$(TEX)`, exports SVGs beside the PDFs and PNG previews under
`build/figures/`. The benchmark plot is generated by Matplotlib as PDF, SVG and
PNG plus a `.pgf` export for the paper, which needs pdfLaTeX on `PATH` for text
metrics. Both paper targets generate the frontier figure and input the method
diagrams directly; they do not build standalone diagram previews. Use
`make figures` to generate all standalone figures and previews.

## License and source archive

Original Python code is licensed under [Apache-2.0](LICENSE).
See [THIRD_PARTY.md](THIRD_PARTY.md) for material outside that grant.

Local archives, build products and caches are excluded from version control.
After committing the source, `make source-archive` exports the committed tree
to `dist/flash-rnnt-paper-source.zip`; uncommitted changes are not included.
That repository archive is not the arXiv submission archive; use `make arxiv`
for the latter.
