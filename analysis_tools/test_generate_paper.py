"""Shared manuscript rendering and self-contained arXiv figure inclusion."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .generate_paper import Renderer


class RendererTest(unittest.TestCase):
    def test_whole_figure_reference_is_linked(self):
        for target in ("interspeech", "arxiv"):
            text = Renderer({}, target).convert("See Figure 1, Figure 2 and Figure 3.")
            for number in (1, 2, 3):
                self.assertIn(
                    rf"\hyperref[fig:{number}]{{Figure~\ref*{{fig:{number}}}}}",
                    text,
                )

    def test_section_links_and_emphasis(self):
        source = "# 5. Results\n\n## 5.2 Ablation\n\nSee §5.2 for *details*."
        for target in ("interspeech", "arxiv"):
            text = Renderer({}, target).convert(source)
            self.assertIn(r"\section{Results}\label{sec:5}", text)
            self.assertIn(r"\subsection{Ablation}\label{sec:5.2}", text)
            self.assertIn(r"\S~\ref{sec:5.2}", text)
            self.assertIn(r"\emph{details}", text)

    def test_figures_are_input_as_copied_tikz_source(self):
        source = (
            "![Positions divided into tiles](figures/tiling/packing.svg)\n\n"
            "*Packing and tiling.*"
        )
        with tempfile.TemporaryDirectory() as folder:
            draft = Path(folder) / "draft"
            figure = draft / "figures/tiling/packing-body.tex"
            figure.parent.mkdir(parents=True)
            figure.write_bytes(b"figure-copy-test")
            output = Path(folder) / "output"
            with patch("analysis_tools.generate_paper.DRAFT", draft):
                text = Renderer({}, "arxiv", output).convert(source)
            self.assertIn(r"\begin{figure}[tbp]", text)
            self.assertIn(r"\input{figures/tiling/packing-body.tex}", text)
            self.assertIn(
                "\\iflatexml\n"
                "\\tikzset{every picture/.append style={scale=1.3,transform shape}}\n"
                "\\fi\n\\input{figures/tiling/packing-body.tex}",
                text,
            )
            self.assertNotIn("../", text)
            self.assertEqual(
                (output / "figures/tiling/packing-body.tex").read_bytes(),
                figure.read_bytes(),
            )

            frontier = draft / "figures/frontiers/frontiers.pgf"
            frontier.parent.mkdir(parents=True)
            frontier.write_bytes(b"frontier-copy-test")
            with patch("analysis_tools.generate_paper.DRAFT", draft):
                text = Renderer({}, "arxiv", output).convert(
                    "![Throughput versus memory](figures/frontiers/frontiers.png)\n\n"
                    "*Comparison.*"
                )
            self.assertIn(r"\begin{figure*}[!t]", text)
            self.assertIn(r"\input{figures/frontiers/frontiers.pgf}", text)
            self.assertNotIn(r"\tikzset", text)

    def test_arxiv_uses_natural_table_placement(self):
        source = "| Components | Short |\n|---|---:|\n| Compiled | 12 |\n\n*Results.*"
        self.assertIn(r"\begin{table}[H]", Renderer({}).convert(source))
        self.assertIn(r"\begin{table}[tbp]", Renderer({}, "arxiv").convert(source))


if __name__ == "__main__":
    unittest.main()
