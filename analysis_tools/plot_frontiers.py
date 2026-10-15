"""Plot every fitting configuration in the three B128 workloads."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .summarize_results import IMPLEMENTATIONS as LABELS, WORKLOADS, in_panel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/h100-20261013/raw/comparison"
OUT = ROOT / "draft/figures/frontiers"
FAMILIES = ("flash_tiled", "flash_untiled", "apex", "optimized", "numba")
STYLES = {
    "apex": ("#ed8b23", "o"),
    "numba": ("#2c9a58", "s"),
    "optimized": ("#bd4b69", "D"),
    "flash_tiled": ("#1678b8", "^"),
    "flash_untiled": ("#8856a7", "v"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plotted_points(cells, batch):
    """Every fitting configuration, with full-batch runs repeated in both panels."""
    result = []
    for row in cells:
        if row["status"] != "complete":
            continue
        assert row["median_ms"] > 0 and row["allocated_bytes"] > 0
        speed = batch * 1000 / row["median_ms"]
        assert abs(speed - row["samples_per_second"]) < 1e-6 * speed
        for schedule in ("retain", "release"):
            if in_panel(row, schedule):
                result.append(dict(row, panel_schedule=schedule))
    return result


def inline_pgf(path):
    """Prepare the pgf export for \\input into the paper and arXiv's HTML.

    The pgf backend writes dash lengths in TeX points but means PostScript
    points (line widths are already converted). Each text gets a LaTeX size
    command, which LaTeXML honours where it ignores \\fontsize, followed by a
    DejaVu Serif font primitive at the exact size for pdfLaTeX. The font loads
    are skipped under LaTeXML (\\iflatexml from the paper's preamble), which has
    no metrics for them and renders picture text in the page font anyway.
    """
    text = path.read_text()

    def dash(m):
        lengths = re.sub(
            r"([0-9.]+)pt", lambda n: f"{float(n[1]) * 72.27 / 72:f}pt", m[1]
        )
        return f"\\pgfsetdash{{{lengths}}}{{{float(m[2]) * 72.27 / 72:f}pt}}"

    text = re.sub(r"\\pgfsetdash\{((?:\{[0-9.]+pt\})+)\}\{([0-9.]+)pt\}", dash, text)
    fonts = {}

    def font(m):
        size, bold = float(m[1]), bool(m[2])
        name = fonts.setdefault((size, bold), "\\pgfdejavu" + "ABCDEFGHIJ"[len(fonts)])
        command = (
            "\\scriptsize"
            if size <= 7.5
            else "\\footnotesize"
            if size <= 8
            else "\\small"
            if size <= 9
            else "\\normalsize"
        )
        return command + ("\\bfseries" if bold else "") + name

    text = re.sub(
        r"\\fontfamily\{\\familydefault\}\\fontsize\{([0-9.]+)\}\{[0-9.]+\}"
        r"(\\bfseries)?\\selectfont",
        font,
        text,
    )
    definitions = (
        "\\iflatexml\n"
        + "".join(f"\\let{name}\\relax%\n" for name in fonts.values())
        + "\\else\n"
        + "".join(
            f"\\font{name}=DejaVuSerif{'-Bold' if bold else ''}-tlf-t1 at {size:g}bp%\n"
            for (size, bold), name in fonts.items()
        )
        + "\\fi\n"
    )
    assert fonts and text.count("\\begingroup%\n") == 1
    path.write_text(text.replace("\\begingroup%\n", "\\begingroup%\n" + definitions))


def main(source=SOURCE, out=OUT):
    generated = (
        ROOT / "build/figures/frontiers"
        if out.resolve() == (ROOT / "draft/figures/frontiers").resolve()
        else out
    )
    generated.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

    summary_path, manifest_path = source / "summary.json", source / "manifest.json"
    summary, manifest = [
        json.loads(p.read_text()) for p in (summary_path, manifest_path)
    ]
    assert sha(manifest_path) == summary["manifest_sha256"]
    batch = manifest["batch"]
    cells = summary["cells"]
    assert len(cells) == len(WORKLOADS) * len(manifest["configurations"])
    assert len({(r["workload"], r["config"]["id"]) for r in cells}) == len(cells)
    points = plotted_points(cells, batch)

    out.mkdir(parents=True, exist_ok=True)
    width, height = 170, 106
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 240,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{DejaVuSerif}",
        }
    ):
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        handles = [
            Line2D(
                [],
                [],
                color=STYLES[b][0],
                marker=STYLES[b][1],
                ls="--" if b == "flash_tiled" else "-",
                lw=1.2,
                markersize=3.5,
                label=LABELS[b],
            )
            for b in FAMILIES
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.996),
            ncol=5,
            frameon=False,
            handlelength=1.6,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0,
        )
        for col, schedule in enumerate(("retain", "release")):
            fig.text(
                (46 + 90 * col) / width,
                94 / height,
                schedule.title(),
                ha="center",
                fontweight="bold",
                fontsize=10,
            )
            for row_index, name in enumerate(WORKLOADS):
                ax = fig.add_axes(
                    [
                        (14 + col * 90) / width,
                        (67 - row_index * 28) / height,
                        64 / width,
                        22 / height,
                    ]
                )
                paired = [r for r in points if r["workload"] == name]
                xs = [r["allocated_bytes"] / 2**30 for r in paired]
                ymax = max(r["samples_per_second"] for r in paired) * 1.22
                absent = []
                for family in FAMILIES:
                    selected = [
                        r
                        for r in paired
                        if r["panel_schedule"] == schedule
                        and r["config"]["family"] == family
                    ]
                    if not selected:
                        outcomes = {
                            r["status"]
                            for r in cells
                            if r["workload"] == name
                            and r["config"]["family"] == family
                            and in_panel(r, schedule)
                        }
                        assert outcomes == {"oom"}, (name, schedule, family, outcomes)
                        absent.append(LABELS[family].replace(" Transducer", ""))
                        continue
                    # Connect configurations in order of tile size or sub-batch size,
                    # with the full batch last; dominated points stay on the line.
                    selected.sort(
                        key=lambda r: (
                            r["config"]["tile_rows"]
                            or r["config"]["sub_batch"]
                            or batch
                        )
                    )
                    color, marker = STYLES[family]
                    x = [r["allocated_bytes"] / 2**30 for r in selected]
                    y = [r["samples_per_second"] for r in selected]
                    ax.plot(
                        x,
                        y,
                        color=color,
                        marker=marker,
                        markersize=3,
                        lw=1.15,
                        ls="--" if family == "flash_tiled" else "-",
                    )
                low, high = min(xs) / 1.12, max(xs) * 1.12
                ax.set_xscale("log", base=2)
                ax.set(xlim=(low, high), ylim=(0, ymax))
                ticks = [
                    2.0**exponent
                    for exponent in range(-12, 10)
                    if low <= 2.0**exponent <= high
                ]
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10])
                )
                ax.yaxis.set_major_formatter(
                    FuncFormatter(
                        lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"
                    )
                )
                ax.tick_params(length=2.5, width=0.6, pad=2)
                ax.grid(alpha=0.20, lw=0.5)
                ax.set_axisbelow(True)
                ax.text(
                    0.03,
                    0.94,
                    f"{name.title()} · {manifest['workloads'][name]['mean_duration_s']:.1f} s",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                )
                if absent:
                    ax.text(
                        0.97,
                        0.08,
                        ", ".join(absent) + "\nAll tested configs OOM",
                        transform=ax.transAxes,
                        fontsize=7.5,
                        va="bottom",
                        ha="right",
                    )
            fig.text(
                (46 + 90 * col) / width,
                0.5 / height,
                "Peak allocated memory (GiB, log scale)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.text(
            1.5 / width,
            55 / height,
            "Throughput (samples/s)",
            rotation=90,
            ha="left",
            va="center",
            fontsize=9,
        )
        for extension in ("png", "pdf", "svg"):
            fig.savefig(
                (generated if extension == "png" else out) / f"frontiers.{extension}"
            )
        fig.savefig(out / "frontiers.pgf", transparent=True)
        inline_pgf(out / "frontiers.pgf")
        plt.close(fig)

    print(f"{out / 'frontiers.pdf'}: {len(points)} plotted points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve())
