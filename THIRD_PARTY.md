# Third-party material

Native dependencies are fetched externally at the pins in
[dependencies/manifest.json](benchmarks/dependencies/manifest.json); their source
trees are not distributed with the paper. License notices for the included
patches and LaTeX template are retained:

- [Optimized Transducer license](benchmarks/dependencies/licenses/optimized-transducer.txt).
- [ModernGPU license](benchmarks/dependencies/licenses/moderngpu.txt).
- [Interspeech kit notice](latex/README.txt) and notices in
  [Interspeech.cls](latex/Interspeech.cls) and [IEEEtran.bst](latex/IEEEtran.bst).

Apex is fetched without source modifications and retains its upstream license
in the checkout. NeMo is also an external dependency. The native checkout and
patch checks are implemented in `benchmarks/harness/backends.py`.

Original Python code is licensed under [Apache-2.0](LICENSE). This grant does not
cover the manuscript, dataset-derived tokenizer, or third-party material, which
retain their respective rights. Tokenizer training sources and reproduction
details are recorded in [data/tokenizer/provenance.json](data/tokenizer/provenance.json).
