"""Export SVG and PNG previews of the compiled TikZ PDFs."""

import argparse
from pathlib import Path

import fitz


def export(pdf):
    with fitz.open(pdf) as document:
        if len(document) != 1:
            raise ValueError(f"Expected a single-page figure: {pdf}")
        page = document[0]
        # Outline SVG text so previews do not depend on the viewer's fonts.
        pdf.with_suffix(".svg").write_text(page.get_svg_image(text_as_path=True))
        preview = (
            Path(__file__).resolve().parents[1]
            / "build/figures"
            / pdf.parent.name
            / pdf.with_suffix(".png").name
        )
        preview.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(dpi=220, alpha=False).save(preview)
    print(pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="+")
    for pdf in parser.parse_args().pdf:
        export(pdf)


if __name__ == "__main__":
    main()
