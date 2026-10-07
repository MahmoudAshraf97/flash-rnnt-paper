"""Typeset all manuscript sections and references using the official 2026 kit.

Markdown and draft/references.bib remain authoritative. Numbered Markdown links
are resolved to BibTeX keys by URL; their displayed numbers are checked against
first-citation order. Compile main.tex with Tectonic or pdfLaTeX plus BibTeX.
"""

import hashlib
import json
import os
import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "draft"
OUT = ROOT / "build/paper"
TEMPLATE = ROOT / "latex"
SOURCES = [
    "00_abstract.md",
    "01_introduction.md",
    "02_background_and_related_work.md",
    "03_implementation.md",
    "04_evaluation_methodology.md",
    "05_results.md",
    "06_discussion_and_conclusion.md",
]
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
FIGURE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)\s+\*(.*?)\*", re.S)
TABLE = re.compile(r"((?:^\|[^\n]*\|[ \t]*\n)+)(?:\s*\*([^*]+)\*)?", re.M)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bibliography_urls(text):
    """Read the key and URL from our braced, one-entry-per-block BibTeX file."""
    entries = re.findall(r"^@\w+\{([^,]+),(.*?)^\}", text, re.M | re.S)
    urls = {}
    keys = set()
    for key, body in entries:
        match = re.search(r"^\s*url\s*=\s*\{([^}]+)\}", body, re.M)
        if not match or key in keys or match[1] in urls:
            raise ValueError(f"Missing URL or duplicate bibliography entry: {key}")
        keys.add(key)
        urls[match[1]] = key
    if not urls:
        raise ValueError("Empty bibliography")
    return urls


class Renderer:
    def __init__(self, urls):
        self.urls = urls
        self.citations = {}
        self.figure_count = 0
        self.table_count = 0
        self.assets = {}

    def cite(self, match):
        label, url = match.groups()
        if url not in self.urls:
            raise ValueError(f"Citation has no bibliography entry: {url}")
        key = self.urls[url]
        number = self.citations.setdefault(key, len(self.citations) + 1)
        if label != str(number):
            raise ValueError(f"Citation {key} is [{label}]; expected [{number}]")
        return rf"\cite{{{key}}}"

    def convert(self, source):
        blocks = []

        def save(content):
            token = f"BLOCKTOKEN{len(blocks)}END"
            blocks.append((token, content))
            return "\n" + token + "\n"

        def figure(match):
            path, caption = match.groups()
            self.figure_count += 1
            number = self.figure_count
            pdf = (DRAFT / path).with_suffix(".pdf")
            if not pdf.is_file():
                raise FileNotFoundError(pdf)
            self.assets[str(pdf.relative_to(DRAFT))] = sha256(pdf)
            wide = "frontiers" in path
            env, width = (
                ("figure*", r"\textwidth") if wide else ("figure", r"\columnwidth")
            )
            caption = re.sub(r"^Figure \d+\.\s*", "", caption)
            return save(
                f"\\begin{{{env}}}[!t]\n\\centering\n"
                + rf"\includegraphics[width={width}]{{{os.path.relpath(pdf, OUT)}}}"
                + "\n\\caption{"
                + self.convert(caption).strip()
                + "}\n"
                + rf"\label{{fig:{number}}}"
                + f"\n\\end{{{env}}}\n"
            )

        def table(match):
            self.table_count += 1
            number = self.table_count
            lines, caption = match.groups()
            if not caption:
                raise ValueError(f"Table {number} needs a caption")
            rows = [
                [cell.strip() for cell in line.strip().strip("|").split("|")]
                for line in lines.strip().splitlines()
            ]
            # Keep the ablation table after its introductory paragraph.
            placement = "H" if rows[0][0] == "Components" else "!t"
            body = [
                " & ".join(self.convert(cell).strip() for cell in row) + r" \\"
                for row in [rows[0], *rows[2:]]
            ]
            headers = []
            for cell in rows[0]:
                lines = textwrap.wrap(
                    cell, width=10, break_long_words=False, break_on_hyphens=False
                )
                headers.append(
                    r"\shortstack{"
                    + r"\\".join(self.convert(line).strip() for line in lines)
                    + "}"
                )
            body[0] = " & ".join(headers) + r" \\"
            return save(
                f"\\begin{{table}}[{placement}]\n\\centering\n"
                + "\\caption{"
                + self.convert(caption).strip()
                + "}\n"
                + rf"\label{{tab:{number}}}"
                + "\n\\small\n"
                + "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{"
                + "l"
                + "r" * (len(rows[0]) - 1)
                + "}\n\\hline\n"
                + body[0]
                + "\n\\hline\n"
                + "\n".join(body[1:])
                + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
            )

        source = FIGURE.sub(figure, source)
        source = TABLE.sub(table, source)
        if "[TODO:" in source:
            raise ValueError(
                "Resolve or explicitly exclude manuscript TODOs before building"
            )
        source = re.sub(r"^## \d+\.\d+ (.+)$", r"\\subsection{\1}", source, flags=re.M)
        source = re.sub(r"^# \d+\. (.+)$", r"\\section{\1}", source, flags=re.M)
        source = re.sub(r"^# Abstract\s*", "", source)
        source = LINK.sub(self.cite, source)
        source = re.sub(
            r"\\cite\{([^}]+)\}(?:,\s*\\cite\{([^}]+)\})+",
            lambda m: r"\cite{"
            + ",".join(re.findall(r"\\cite\{([^}]+)\}", m[0]))
            + "}",
            source,
        )
        source = re.sub(r"\bFigure (\d+)\b", r"Figure~\\ref{fig:\1}", source)
        source = re.sub(r"\bTable (\d+)\b", r"Table~\\ref{tab:\1}", source)
        source = re.sub(
            r"\$\$(.*?)\$\$", r"\\begin{equation}\1\\end{equation}", source, flags=re.S
        )
        source = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", source)
        source = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", source)
        source = source.replace("%", r"\%")
        for old, new in {
            "–": "--",
            "—": "---",
            "’": "'",
            "“": "``",
            "”": "''",
            "§": r"\S~",
        }.items():
            source = source.replace(old, new)
        for token, content in blocks:
            source = source.replace(token, content)
        return source.strip() + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("Interspeech.cls", "IEEEtran.bst"):
        shutil.copyfile(TEMPLATE / name, OUT / name)
    shutil.copyfile(DRAFT / "references.bib", OUT / "references.bib")
    renderer = Renderer(bibliography_urls((DRAFT / "references.bib").read_text()))
    hashes = {name: sha256(DRAFT / name) for name in [*SOURCES, "references.bib"]}
    parts = [renderer.convert((DRAFT / name).read_text()) for name in SOURCES]
    unused = set(renderer.urls.values()) - set(renderer.citations)
    if unused:
        raise ValueError(f"Uncited bibliography entries: {sorted(unused)}")
    abstract = " ".join(
        (DRAFT / SOURCES[0]).read_text().removeprefix("# Abstract").split()
    )
    preamble = r"""% Generated by analysis_tools/generate_paper.py. Do not edit.
% Official 2026 class and IEEEtran style, unchanged. Author details are placeholders.
\PassOptionsToPackage{safe}{tipa}
\documentclass[cameraready]{Interspeech}
% Preserve the class's Times fonts under XeTeX/Tectonic as well as pdfLaTeX.
\usepackage[OT1]{fontenc}
\usepackage{float}
\title{Flash RNN-T: Memory-Efficient Joint and Loss Computation}
\author{Author names}{pending}
\address{Affiliations pending}
\email{}
\keywords{speech recognition, RNN-T, GPU kernels, memory efficiency}
\begin{document}
\maketitle
\begin{abstract}
"""
    document = preamble + parts[0] + "\\end{abstract}\n\n" + "\n".join(parts[1:])
    document += r"""
\label{end:main}
\clearpage
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""
    (OUT / "main.tex").write_text(document)
    manifest = {
        "source_sha256": hashes,
        "asset_sha256": renderer.assets,
        "generator_sha256": sha256(Path(__file__)),
        "abstract_characters_including_spaces": len(abstract),
        "abstract_limit": 1000,
        "abstract_ascii_only": abstract.isascii(),
        "class_sha256": sha256(OUT / "Interspeech.cls"),
        "bibliography_style_sha256": sha256(OUT / "IEEEtran.bst"),
        "citations": renderer.citations,
        "figures": renderer.figure_count,
        "tables": renderer.table_count,
        "placeholders": ["author names", "affiliations", "email", "index terms"],
        "pending_author_content": [
            "generative AI use disclosure",
            "acknowledgments, if applicable",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {OUT / 'main.tex'}")
    print(
        f"Abstract: {len(abstract)} characters; limit 1000; ASCII only: {abstract.isascii()}"
    )
    print(
        f"References: {len(renderer.citations)}; figures: {renderer.figure_count}; tables: {renderer.table_count}"
    )


if __name__ == "__main__":
    main()
