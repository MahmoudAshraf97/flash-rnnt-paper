# Flash RNN-T blog post

Working title: **Flash RNN-T: Making Room to Train Speech Recognition**

Audience: general readers and technical readers without a research background.
The prose is under review. Section 5 now uses end-to-end training measurements;
absolute dollar estimates await a chosen GPU rate.

## Sections

1. [Opening](01_opening.md) — establish the training-memory constraint and introduce Flash.
2. [Why the memory grows](02_why_the_memory_grows.md) — use a recording and its transcript to explain alignment and the growing grid of calculations.
3. [Compute it, use it, make room](03_tiled_recomputation.md) — explain what we retain, what we discard, and why recreating temporary data saves memory at a computational cost.
4. [Making each operation more efficient](04_specialized_operations.md) — explain specialization and why untiled Flash is useful when memory permits.
5. [What changed in practice](05_results.md) — report training-step time and memory savings, then project the time and relative compute cost of 30,000 steps.
6. [What this makes possible](06_conclusion.md) — explain the practical implications and link to the paper and code.

## Editorial direction

Lead with the training constraint, introduce Flash early, then explain the
problem and solution with increasing technical depth. Use connected prose and
concrete examples; introduce technical terms only when they help the explanation.
Keep the distinction between joint-and-loss benchmarks and full training clear.
Do not imply inference improvements or measured recognition-quality gains.
Keep planning notes here rather than in reader-facing section files.

## Training-result evidence

Section 5 uses the optimized-encoder pair from the controlled comparison in
`results/local/blog-encoder-comparison-20261018/summary.json`, not the earlier
B128 comparison or the historical-encoder control. Both implementations use
eight H100 80GB GPUs, the same 114,225,408-parameter FastConformer RNN-T,
initial weights, training data and optimizer, B64 per GPU and accumulation two
(global batch 1,024). The baseline is NeMo's Numba-based RNN-T loss; Flash is
tiled at 200,000 rows. Both already include the encoder optimizations.

Each run warms up for 200 optimizer steps and measures at least 120 seconds.
The comparison uses the same 152 measured steps across all ranks. Native versus
Flash: 0.646116 versus 0.502735 seconds per step; 49.71 versus 26.63 GiB peak
allocated memory per GPU. These give 22.19% less time and 46.42% less memory.
The 30,000-step projections are 5.3843 versus 4.1895 hours, a 71.69-minute saving.

These short runs measure data loading, forward, backward, DDP and optimizer
work, excluding setup, validation and checkpoint writing. They are not completed
30,000-step runs or measurements of time to convergence. Raw records, checks,
timing variability and rollback details remain in that local evidence directory;
private-data artifacts are not publication assets.

At a fixed per-GPU hourly rate, projected compute cost is elapsed hours × eight
GPUs × rate, so its relative reduction is also 22.19%. Leave absolute dollar
amounts unset until the rate is chosen. Do not mix this pair's memory savings
with the different-microbatch B128 comparison.

## Publication preparation

Replace the relative paper links with its public URL before publishing.

Section 2 links to [the alignment animation](animations/alignment.html), a
self-contained HTML/SVG file with no external dependencies. Open it in a browser
to preview; host it alongside the post and embed it with an iframe for publication.
It loops every four seconds and supports light/dark themes, pause and reduced motion.
Other illustrations are not selected yet.

## Style references

- [Tarteel: From Page to Screen](https://tarteel.ai/blog/from-page-to-screen-rethinking-quran-rendering-for-the-digital-age/) — reader context, constraints, engineering decisions.
- [Tarteel: Research & Development](https://tarteel.ai/blog/tag/research-development/) — publication voice and audience.
- [Baseten: Introducing NVIDIA Nemotron 3.5 ASR Streaming](https://www.baseten.co/blog/introducing-nvidia-nemotron-35-asr-streaming/) — establish the benefit early.
- [Baseten: What makes a high-quality NVFP4 quantization?](https://www.baseten.co/blog/what-makes-a-high-quality-nvfp4-quantization/) — build intuition through examples before technical distinctions.
- [Baseten: Agentic kernels in production](https://www.baseten.co/blog/agentic-kernels-in-production/) — connect each optimization to its effect.
