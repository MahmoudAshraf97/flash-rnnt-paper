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

`latexml.sty` is a stand-in for LaTeXML's style file: it defines `\iflatexml`
as false under pdfLaTeX, while LaTeXML uses its own binding and sets it true.
The preambles use it to skip the figures' TeX font loads under LaTeXML, which
cannot resolve them and renders picture text in the page font regardless.

Author metadata appears in the generator's preamble. Rebuild and check after
changing it, as this can affect pagination.

`make arxiv` uses the same manuscript with the standard `article` class instead
of this conference class. It retains `IEEEtran.bst` to preserve the references.
The self-contained submission archive is `dist/flash-rnnt-arxiv.zip`; see the
[arXiv build instructions](../README.md#arxiv).
