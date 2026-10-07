"""Check the compiled paper's layout and references; optionally render every page.

Requires PyMuPDF. Run after the generator and LaTeX/BibTeX compilation.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "draft"
OUT = ROOT / "build/paper"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    aux = (OUT / "main.aux").read_text()
    log = (OUT / "main.log").read_text(errors="replace")
    biblog = (OUT / "main.blg").read_text()
    doc = fitz.open(OUT / "main.pdf")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    stale = [
        name
        for name, value in manifest["source_sha256"].items()
        if digest(DRAFT / name) != value
    ]
    stale += [
        name
        for name, value in manifest["asset_sha256"].items()
        if digest(DRAFT / name) != value
    ]
    if (
        digest(ROOT / "analysis_tools/generate_paper.py")
        != manifest["generator_sha256"]
    ):
        stale.append("analysis_tools/generate_paper.py")
    main_page = int(re.search(r"\\newlabel\{end:main\}\{\{[^}]*\}\{(\d+)\}", aux)[1])
    reference_page = next(
        i + 1 for i, p in enumerate(doc) if re.search(r"7\.\s+References", p.get_text())
    )
    compiled_citations = {
        key: int(number)
        for key, number in re.findall(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux)
    }
    float_pages = {
        key: int(page)
        for key, page in re.findall(
            r"\\newlabel\{((?:fig|tab):[^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux
        )
    }
    font_records = {}
    pages = []
    for i, page in enumerate(doc):
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip()
        ]
        pages.append(
            {
                "page": i + 1,
                "width_pt": round(page.rect.width, 2),
                "height_pt": round(page.rect.height, 2),
                "lowest_text_y_pt": round(max(span["bbox"][3] for span in spans), 2),
            }
        )
        for xref, ext, kind, name, *_ in page.get_fonts(full=True):
            font_records[xref] = {
                "name": name,
                "type": kind,
                "embedded": bool(doc.extract_font(xref)[3]),
            }
        if args.render:
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(
                OUT / f"page-{i + 1}.png"
            )
    errors = [
        line
        for line in log.splitlines()
        if any(
            term in line
            for term in (
                "Overfull",
                "Missing character",
                "undefined",
                "multiply defined",
            )
        )
    ]
    errors += [
        line
        for line in biblog.splitlines()
        if "Warning" in line or "error" in line.lower()
    ]
    checks = {
        "sources_current": not stale,
        "class_unmodified": digest(OUT / "Interspeech.cls")
        == digest(ROOT / "latex/Interspeech.cls"),
        "bibliography_style_unmodified": digest(OUT / "IEEEtran.bst")
        == digest(ROOT / "latex/IEEEtran.bst"),
        "six_page_limit": len(doc) <= 6,
        "four_page_main_content_limit": main_page <= 4
        and max(float_pages.values()) <= 4,
        "all_citations_resolved_in_order": compiled_citations == manifest["citations"],
        "all_floats_present": len(float_pages)
        == manifest["figures"] + manifest["tables"],
        "a4_pages": all(
            abs(p.rect.width - 595.28) < 1 and abs(p.rect.height - 841.89) < 1
            for p in doc
        ),
        "all_fonts_embedded": all(f["embedded"] for f in font_records.values()),
        "no_type3_fonts": all(f["type"] != "Type3" for f in font_records.values()),
        "no_layout_or_reference_errors": not errors,
        "abstract_within_limit": manifest["abstract_characters_including_spaces"]
        <= 1000,
        "abstract_ascii_only": manifest["abstract_ascii_only"],
    }
    report = {
        "pdf_sha256": digest(OUT / "main.pdf"),
        "pages_total": len(doc),
        "main_content_ends_on_page": main_page,
        "references_start_on_page": reference_page,
        "bibliography_entries": len(compiled_citations),
        "abstract_characters": manifest["abstract_characters_including_spaces"],
        "checks": checks,
        "stale_inputs": stale,
        "errors": errors,
        "float_pages": float_pages,
        "page_geometry": pages,
        "fonts": list(font_records.values()),
        "author_placeholders": manifest["placeholders"],
        "pending_author_content": manifest["pending_author_content"],
    }
    (OUT / "layout_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "pages_total",
                    "main_content_ends_on_page",
                    "references_start_on_page",
                    "bibliography_entries",
                    "abstract_characters",
                    "checks",
                    "errors",
                )
            },
            indent=2,
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
