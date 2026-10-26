# GTC poster

    uv sync --group poster
    uv run --group poster python -m poster.render

Run from the repository root. Writes `build/poster/poster.pdf` (43 × 24 in, vector,
fonts embedded, the file to submit), `poster.png` (85 dpi preview) and
`layout-check.json` (measured block positions).

- `render.py` holds the copy, layout, diagrams, charts and checks. Every quoted ratio
  is recomputed from `results/h100-20261013/raw/comparison/summary.json` and the
  training summary under `results/local/`, and asserted against the paper's rounded
  values, so a data change that moves a number fails the build instead of printing
  stale text. The render also asserts the one-inch margin, column containment, no
  overlaps and no colons in the copy.
- `layout_engine.py` measures each block and stacks it; `test_layout.py` is its
  regression suite (`uv run --group poster python -m unittest poster.test_layout`).
- `fonts/` vendors Carlito (Calibri metrics, OFL) so the render is reproducible.
- `assets/` holds the Tarteel logo; the repository QR code is generated and decode-checked.

The training table reads `results/local/blog-encoder-comparison-20261018/summary.json`,
which is not in a fresh clone.
