# Interspeech format

`Interspeech.cls` and `IEEEtran.bst` are unchanged copies from the
[official Interspeech 2026 kit](https://www.overleaf.com/latex/templates/interspeech-paper-kit/svqkgcpdbxfg).
`README.txt` is the kit's original notice.

The build copies the class and style to `build/paper/`, generates `main.tex`
from the Markdown manuscript, and resolves the numbered references using
BibTeX. `make check` checks source freshness, citations, embedded fonts,
overflow, the abstract limit, four main-content pages and six total pages.

Original file SHA-256 hashes:

- Class: `9bd5333079885b7834c811813f6c75de0ece1e65216028efb080d33f22a7ca5c`
- Style: `7083eece0f76d6a7a84416a3709c313837e345c0a25f9ee78ec9b9f40be457bc`

Author metadata currently appears in the generator's preamble. Filling it in
can change pagination; rebuild and check afterwards.
