"""Render the GTC poster: python -m poster.render (from the repository root, poster dependency group).

Writes build/poster/poster.pdf (vector, fonts embedded), poster.png (85 dpi preview) and
layout-check.json. Every quoted ratio is recomputed from the benchmark summaries and
asserted against the paper's rounded values; the layout is checked for overlap, column
containment and the one-inch margin before anything is written.
"""

import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from analysis_tools.plot_frontiers import FAMILIES, LABELS, plotted_points
from analysis_tools.summarize_results import in_panel, select_within_peak
from poster.layout_engine import Block, Column, fit_axes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/poster"
REPOSITORY = "https://github.com/MahmoudAshraf97/flash-rnnt-paper"

# Page: 43 x 24 in with a one-inch margin, three tinted panels inset by PAD so they stay inside it.
W, H = 43, 24
LEFT, RIGHT, GUTTER = 1, 42, .6
TOP, BOTTOM = 3.75, 22.1
PANEL, PAD = "#f1f6f8", .2
COL_WIDTHS = tuple((RIGHT-LEFT-2*PAD-2*GUTTER)*share for share in (.20, .26, .54))
COL_X = (LEFT+PAD, LEFT+PAD+COL_WIDTHS[0]+GUTTER, RIGHT-PAD-COL_WIDTHS[2])
NODE_HEIGHT = 1.22

# Type: Carlito (Calibri metrics, OFL, vendored in poster/fonts). One body size for every paragraph.
FONT = "Carlito"
for _file in (ROOT / "poster/fonts").glob("Carlito-*.ttf"):
    font_manager.fontManager.addfont(str(_file))
BODY = 32

# Colour: navy header, one teal accent, grey ink. The chart palette was validated all-pairs
# for colour-vision deficiency; its floor-band pair is covered by distinct markers.
INK, MUTED, RULE = "#203047", "#596a7c", "#dce3ea"
NAVY, ACCENT, TINT = "#020d23", "#0f7a9c", "#e8f3f6"
TILE_1, TILE_2 = TINT, "#b9dbe4"
STYLES = {                         # family: colour, marker, line style, line width, marker size
    "flash_tiled": ("#0f7a9c", "^", "--", 4.5, 10),
    "flash_untiled": ("#6250d6", "v", "-", 4.5, 10),
    "apex": ("#e34948", "o", "-", 3.2, 8),
    "optimized": ("#c98500", "D", "-", 3.2, 8),
    "numba": ("#008300", "s", "-", 3.2, 8),
}


def paragraph(fig, x, y, width, copy, size=BODY, weight="normal", color=INK):
    """Wrap to the physical column width using the poster face; coordinates are inches from top-left."""
    prop = FontProperties(family=FONT, size=size, weight=weight)
    lines, line = [], ""
    for word in copy.split():
        trial = f"{line} {word}".strip()
        if line and fig.canvas.get_renderer().get_text_width_height_descent(trial, prop, False)[0] > width*fig.dpi:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    return fig.text(x/W, 1-y/H, "\n".join(lines), fontsize=size, weight=weight, color=color, va="top", linespacing=1.28)


def label(fig, x, y, copy, size=27, **kwargs):
    return fig.text(x/W, 1-y/H, copy, fontsize=size, va="top", color=kwargs.pop("color", INK), **kwargs)


def rule(fig, x, y, width, color=RULE, lw=1):
    fig.add_artist(Line2D([x/W, (x+width)/W], [1-y/H]*2, transform=fig.transFigure, color=color, lw=lw))


def axes_at(fig, x, y, width, height):
    return fig.add_axes([x/W, 1-(y+height)/H, width/W, height/H])


def spread(column, names):
    """Give the column's spare height to the gaps before the named blocks, in equal shares."""
    share = (column.bottom-column.top-column.natural_height)/len(names)
    assert share >= 0, (column.x, share)
    column.entries = [(block, gap+share if block.name in names else gap) for block, gap in column.entries]


def bar(fig, width, title):
    ax = axes_at(fig, 0, 0, width, .62)
    ax.set(xlim=(0, width), ylim=(0, .62))
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), width, .62, facecolor=ACCENT, edgecolor="none"))
    ax.text(.22, .31, title, fontsize=31, weight="bold", color="white", va="center")


def equation(fig, width):
    ax = axes_at(fig, 0, 0, width, .8)
    ax.set(xlim=(0, width), ylim=(0, .8))
    ax.axis("off")
    ax.text(width/2, .4, r"$O(N(H+V))\ \longrightarrow\ O(R(H+V))$", fontsize=32, ha="center", va="center")


def problem_diagram(fig, width):
    height, node = 4.3, 1.0
    ax = axes_at(fig, 0, 0, width, height)
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    for x, y, w, copy, fill, edge in [
        (0, 3.3, width*.46, "Speech\nT frames", "#eef0f2", MUTED),
        (width*.54, 3.3, width*.46, "Transcript\nU tokens", "#eef0f2", MUTED),
        (width*.2, 1.65, width*.6, "Joint network", "white", INK),
        (0, 0, width, "B × T × U × (H + V)\nsaved for backward", TINT, ACCENT),
    ]:
        ax.add_patch(Rectangle((x, y), w, node, facecolor=fill, edgecolor=edge, lw=1.6))
        ax.text(x+w/2, y+node/2, copy, ha="center", va="center", fontsize=25, linespacing=1.2)
    for start, end in [((width*.23, 3.25), (width*.4, 2.7)), ((width*.77, 3.25), (width*.6, 2.7)),
                       ((width*.5, 1.6), (width*.5, 1.05))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=24, lw=2, color=INK))


def method_diagram(fig, width):
    """One hue in two steps for the utterances; the teal tint marks what is kept across tiles."""
    ax = axes_at(fig, 0, 0, width, 4.5)
    ax.set(xlim=(.2, 12.8), ylim=(.7, 4.55))
    ax.axis("off")
    for i in range(18):
        ax.add_patch(Rectangle((.2+i*.7, 3.52), .7, .45, facecolor=TILE_1 if i < 8 else TILE_2,
                               edgecolor=ACCENT, lw=1))
    ax.text(3, 4.25, "Utterance 1", ha="center", fontsize=26)
    ax.text(9.3, 4.25, "Utterance 2", ha="center", fontsize=26)
    for i in range(3):
        left, right = .2+i*4.2, .2+(i+1)*4.2
        ax.plot([left+.04, left+.04, right-.04, right-.04], [3.44, 3.32, 3.32, 3.44], color=MUTED, lw=2)
        ax.text((left+right)/2, 3.0, f"Tile {i+1}", color=MUTED, ha="center", fontsize=26)

    def box(x, y, w, copy, fill, edge, style="-"):
        h = NODE_HEIGHT*3.85/4.5
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=1.6, ls=style))
        ax.text(x+w/2, y+h/2, copy, ha="center", va="center", fontsize=27)

    box(.2, 1.30, 4.25, "Hidden states +\nvocabulary scores", "white", MUTED, (0, (4, 3)))
    box(5.6, 1.30, 4.15, "Two transition\nlog probabilities", TINT, ACCENT)
    box(10.7, 1.30, 2.1, "Loss", "white", INK)
    for start, end in [(4.45, 5.6), (9.75, 10.7)]:
        ax.add_patch(FancyArrowPatch((start+.05, 1.82), (end-.05, 1.82),
                                     arrowstyle="->", mutation_scale=22, lw=2, color=INK))
    ax.text(2.325, .87, "Discard after forward", color=MUTED, ha="center", fontsize=25)
    ax.text(7.675, .87, "Keep across tiles", color=ACCENT, ha="center", fontsize=25)


def training_table(fig, width, data):
    """Table 1 with a reduction column, so the reader never has to compute the percentage."""
    columns = (0, width*.52, width*.74, width-.01)
    for x, head in zip(columns, ("", "NeMo / Numba", "Tiled Flash", "Reduction")):
        label(fig, x, 0, head, 26, ha="left" if x == 0 else "right", color=MUTED)
    rule(fig, 0, .43, width)
    rows = [("seconds_per_optimizer_step", "flash_time_reduction_fraction", "Step time (s)", 3),
            ("max_rank_peak_allocated_gib", "flash_peak_memory_reduction_fraction", "Peak GPU memory (GiB)", 2)]
    for row, (key, cut, name, digits) in enumerate(rows):
        y = .62+row*.56
        label(fig, columns[0], y, name, 30)
        for x, implementation in zip(columns[1:3], ("numba", "flash")):
            label(fig, x, y, f"{data['implementations'][implementation][key]:.{digits}f}", 33, ha="right",
                  weight="bold" if implementation == "flash" else "normal")
        label(fig, columns[3], y, f"{100*data[cut]:.0f}%", 33, ha="right", weight="bold", color=ACCENT)
    rule(fig, 0, 1.74, width)


def comparison(fig, height):
    """Figure 1: six throughput-versus-memory panels, every fitting H100 configuration plotted."""
    source = ROOT / "results/h100-20261013/raw/comparison"
    summary = json.loads((source / "summary.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    points = plotted_points(summary["cells"], manifest["batch"])
    handles = [Line2D([], [], color=STYLES[f][0], marker=STYLES[f][1], ls=STYLES[f][2], lw=STYLES[f][3],
                      markersize=STYLES[f][4], label=LABELS[f]) for f in FAMILIES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1), ncol=5, frameon=False,
               fontsize=21, handlelength=1.4, columnspacing=.8, borderaxespad=0)
    plotted = 0
    for col, workload in enumerate(("short", "medium", "long")):
        pool = [p for p in points if p["workload"] == workload]
        xs = [p["allocated_bytes"]/2**30 for p in pool]
        lo, hi = min(xs)/1.15, max(xs)*1.15
        ymax = max(p["samples_per_second"] for p in pool)*1.12
        panel_width = (COL_WIDTHS[2]-2*.35)/3
        x = col*(panel_width+.35)
        duration = manifest["workloads"][workload]["mean_duration_s"]
        label(fig, x+panel_width/2, .7, f"{workload.title()} · {duration:.1f} s", 25, ha="center", weight="bold")
        panel_height = (height-2.6)/2
        for row, schedule in enumerate(("retain", "release")):
            top = 1.65+row*(panel_height+.35)
            ax = axes_at(fig, x, top, panel_width, panel_height)
            absent = []
            for f in FAMILIES:
                subset = [p for p in pool if p["panel_schedule"] == schedule and p["config"]["family"] == f]
                if not subset:
                    states = {p["status"] for p in summary["cells"] if p["workload"] == workload
                              and p["config"]["family"] == f and in_panel(p, schedule)}
                    assert states == {"oom"}, (workload, schedule, f, states)
                    absent.append(f)
                    continue
                subset.sort(key=lambda p: p["config"]["tile_rows"] or p["config"]["sub_batch"] or manifest["batch"])
                color, marker, style, line_width, marker_size = STYLES[f]
                ax.plot([p["allocated_bytes"]/2**30 for p in subset], [p["samples_per_second"] for p in subset],
                        color=color, marker=marker, ms=marker_size, lw=line_width, ls=style,
                        zorder=4 if f.startswith("flash") else 3)
                plotted += len(subset)
            ax.set(xscale="log", xlim=(lo, hi), ylim=(0, ymax))
            ax.xaxis.set_major_locator(FixedLocator([2.**e for e in range(-5, 9) if lo <= 2.**e <= hi]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"))
            ax.grid(axis="y", alpha=.16, lw=1)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#94a1ae")
            ax.tick_params(labelsize=21, length=4, pad=7)
            if col == 0:
                ax.set_ylabel(f"{schedule.title()}\nSamples/s", fontsize=23, labelpad=12)
            if absent:
                ax.text(.94, .90, "Others: OOM", ha="right", va="top", transform=ax.transAxes, fontsize=21, color=INK)
            fit_axes(ax, x, top, panel_width, panel_height)
            if workload == "medium" and schedule == "retain":
                chosen = [select_within_peak([p for p in summary["cells"] if p["workload"] == workload
                                              and p["config"]["family"] == f and in_panel(p, schedule)], .05)
                          for f in ("flash_tiled", "apex")]
                flash, apex = chosen
                ratio = apex["allocated_bytes"]/flash["allocated_bytes"]
                penalty = 100*(1-flash["samples_per_second"]/apex["samples_per_second"])
                assert round(ratio) == 9 and round(penalty) == 16
                label(fig, x+panel_width/2, 1.13, f"{ratio:.0f}× less memory · {penalty:.0f}% lower throughput",
                      19, ha="center", color=INK)
                for point in chosen:
                    ax.plot(point["allocated_bytes"]/2**30, point["samples_per_second"],
                            marker="o", ms=16, mfc="none", mec=INK, mew=1.6, zorder=5)
    label(fig, COL_WIDTHS[2]/2, height-.45, "Peak allocated memory (GiB, log scale)", 24, ha="center")
    assert plotted == len(points)
    print(f"Included all {plotted} fitting panel points.")


def comparisons():
    """Selected-configuration ratios quoted in the text, asserted against the paper's rounded numbers."""
    cells = json.loads((ROOT / "results/h100-20261013/raw/comparison/summary.json").read_text())["cells"]

    def pick(workload, schedule, family):
        return select_within_peak([c for c in cells if c["workload"] == workload
                                   and c["config"]["family"] == family and in_panel(c, schedule)], .05)

    def versus(workload, schedule, family, baseline="apex"):
        ours, theirs = pick(workload, schedule, family), pick(workload, schedule, baseline)
        return theirs["allocated_bytes"]/ours["allocated_bytes"], ours["samples_per_second"]/theirs["samples_per_second"]-1

    out = {"medium_retain": versus("medium", "retain", "flash_tiled"),
           "long_release": versus("long", "release", "flash_tiled"),
           "medium_untiled": versus("medium", "release", "flash_untiled"),
           "long_untiled": versus("long", "release", "flash_untiled"),
           "apex_medium_gib": pick("medium", "retain", "apex")["allocated_bytes"]/2**30}
    for family in ("apex", "numba", "optimized", "flash_untiled"):
        states = {c["status"] for c in cells if c["workload"] == "long" and c["config"]["family"] == family
                  and c["config"]["schedule"] in ("retain", "full_batch")}
        assert states == {"oom"}, (family, states)
    assert (round(out["medium_retain"][0]), round(-100*out["medium_retain"][1])) == (9, 16)
    assert (round(out["long_release"][0]), round(-100*out["long_release"][1])) == (4, 12)
    assert (round(100*out["medium_untiled"][1]), round(100*out["long_untiled"][1])) == (24, 39)
    assert (round(100*(1-1/out["medium_untiled"][0])), round(100*(1-1/out["long_untiled"][0]))) == (56, 26)
    assert round(out["apex_medium_gib"]) == 16
    return out


def branding(fig):
    import cairosvg
    import numpy as np
    import qrcode
    import zxingcpp
    from PIL import Image

    logo = Image.open(BytesIO(cairosvg.svg2png(url=str(ROOT / "poster/assets/tarteel-logo.svg"), output_width=1200)))
    ax = axes_at(fig, 1.35, 1.4, 2.5, 2.5*24/39)
    ax.imshow(logo)
    ax.axis("off")

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=20, border=3)
    qr.add_data(REPOSITORY)
    qr.make(fit=True)
    bitmap = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    assert zxingcpp.read_barcode(np.asarray(bitmap)).text == REPOSITORY
    ax = axes_at(fig, 40.05, 1.15, 1.7, 1.7)
    ax.imshow(bitmap, interpolation="nearest")
    ax.axis("off")
    label(fig, 41.9, 2.92, "paper · code · results", 16, ha="right", color="white")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": FONT, "font.size": 26, "axes.facecolor": "none",
                         "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
                         "mathtext.bf": f"{FONT}:bold", "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, H), facecolor="white")
    numbers = comparisons()
    training = json.loads((ROOT / "results/local/blog-encoder-comparison-20261018/summary.json").read_text())["encoder_variants"]["optimized"]
    memory_cut, time_cut = (round(100*training[k]) for k in ("flash_peak_memory_reduction_fraction", "flash_time_reduction_fraction"))
    assert (memory_cut, time_cut) == (46, 22)

    for x, width in zip(COL_X, COL_WIDTHS):
        fig.add_artist(Rectangle(((x-PAD)/W, 1-(BOTTOM+PAD)/H), (width+2*PAD)/W, (BOTTOM-TOP+2*PAD)/H,
                                 transform=fig.transFigure, facecolor=PANEL, edgecolor="none", zorder=-1))
    fig.add_artist(Rectangle((LEFT/W, 1-3.3/H), (RIGHT-LEFT)/W, 2.3/H,
                             transform=fig.transFigure, facecolor=NAVY, edgecolor="none", zorder=-1))
    title = label(fig, W/2, 1.32, "Flash RNN-T: Memory-Efficient Joint Network and Transducer Loss Computation",
                  48, weight="bold", ha="center", color="white")
    label(fig, W/2, 2.32, "Mahmoud Ashraf   ·   Tarteel AI   ·   mahmoud.ashraf@tarteel.ai", 27, ha="center", color="white")
    branding(fig)

    columns = [Column(fig, x, width, TOP, BOTTOM) for x, width in zip(COL_X, COL_WIDTHS)]

    def text(column, name, copy, size=BODY, bold=False, gap=.3, color=INK):
        return column.add(Block.build(fig, name, lambda: paragraph(
            fig, 0, 0, column.width, copy, size, "bold" if bold else "normal", color)), gap)

    def heading(column, name, copy):
        return column.add(Block.build(fig, name, lambda: bar(fig, column.width, copy)), .45)

    intro, method, results = columns
    heading(intro, "intro-bar", "Introduction")
    text(intro, "intro-background",
         "RNN-T is one of the standard losses for streaming speech recognition. Its joint network produces a hidden state "
         "and vocabulary logits for every pairing of an audio frame with a transcript token [1], and these "
         "intermediates grow as B × T × U × (H + V). When they are kept for backward they can exceed the memory "
         "of the GPU even though the model itself fits.")
    intro.add(Block.build(fig, "problem-diagram", lambda: problem_diagram(fig, intro.width)), .4)
    text(intro, "intro-measured",
         f"On 19-second audio we measured the joint and loss step of NVIDIA's Apex [4] at {numbers['apex_medium_gib']:.0f} GiB "
         "with batch 128, and on 70-second audio none of Apex, NeMo/Numba [7], Optimized Transducer [8] or our own "
         "untiled kernels fit an 80 GB H100 with one backward call per batch. The usual remedy is to split the batch "
         "and call backward after each sub-batch [2], which frees the memory but changes how the training loop "
         "schedules backward. In this work we reduce the memory of the joint and loss without changing the RNN-T "
         "objective and without sub-batch backward.", gap=.4)
    spread(intro, ["problem-diagram", "intro-measured"])
    intro.arrange()

    heading(method, "method-bar", "Method")
    text(method, "kernels-heading", "Two kernels, two numbers per position", 30, True, gap=.15)
    text(method, "kernels-body",
         "At each (frame, token) position the loss reads only two numbers, log P(blank) and log P(next token), "
         "but both need a log-sum-exp over all V logits. Our normalization kernel computes that log-sum-exp and "
         "writes just these two log probabilities, so the full log-softmax output is never stored, and in backward "
         "it overwrites the logits with their gradients instead of allocating another tensor. A second kernel fuses "
         "the joint's addition, activation and dropout so that nothing is written between them, which removes a "
         "further 17 to 33% of tiled peak memory.")
    text(method, "tiles-heading", "Tiled recomputation", 30, True, gap=.35)
    text(method, "tiles-body",
         "Untiled Flash saves the hidden states and logits for backward. In tiled Flash we discard them after each "
         "tile's forward pass and recompute them tile by tile during backward [3], so these large intermediates "
         "need storage for only one tile at a time while the batch still runs as a single backward call.")
    method.add(Block.build(fig, "method-diagram", lambda: method_diagram(fig, method.width)), .3)
    method.add(Block.build(fig, "storage-equation", lambda: equation(fig, method.width)), .15)
    text(method, "storage-definition",
         "Upper bound on joint-intermediate storage for N valid alignment positions, R positions per tile, hidden "
         "size H and vocabulary V. Every alignment and every vocabulary entry is still evaluated, and the loss's "
         "own alignment buffers still grow with sequence length.", BODY, gap=.15)
    spread(method, ["method-diagram", "storage-equation", "storage-definition"])
    method.arrange()

    ratio, penalty = numbers["medium_retain"]
    long_ratio, long_penalty = numbers["long_release"]
    heading(results, "results-bar", "Results")
    text(results, "training-prose",
         "Table 1 gives the end-to-end effect of training a 114M-parameter FastConformer RNN-T on eight H100s with "
         "tiled Flash in place of NeMo's Numba loss [7], with the same model, data order and global batch of 1,024.")
    results.add(Block.build(fig, "training-table", lambda: training_table(fig, results.width, training)), .35)
    text(results, "table-caption",
         "Table 1. Time per optimizer step and maximum per-GPU peak allocated memory over 152 matched steps after "
         "200 warm-up steps, BF16, including data loading, DDP and optimizer.", 20, gap=.15, color=MUTED)
    figure_caption = Block.build(fig, "figure-caption", lambda: paragraph(
        fig, 0, 0, results.width,
        "Figure 1. Joint and loss throughput against peak allocated memory on one H100 80 GB, batch 128, joint "
        "size 640, vocabulary 1,024, FP16 joint with FP32 parameters, timed from projected inputs through loss "
        "backward. Retain keeps intermediates until one backward per batch, release runs backward after every "
        "sub-batch [2], and tiled Flash uses one backward in both panels. Circled points are the lowest-memory "
        "configurations within 5% of each implementation's peak throughput, which the quoted ratios compare.", 20, "normal", MUTED))
    findings = Block.build(fig, "findings", lambda: paragraph(
        fig, 0, 0, results.width,
        f"Under retain, tiled Flash needs {ratio:.0f}× less memory than Apex on medium audio at {-100*penalty:.0f}% "
        "lower throughput, and on long audio it is the only implementation that fits. Against Apex under release, "
        f"tiled Flash still uses {long_ratio:.0f}× less memory at {-100*long_penalty:.0f}% lower throughput, and our "
        f"untiled kernels are {100*numbers['medium_untiled'][1]:.0f}% faster with "
        f"{100*(1-1/numbers['medium_untiled'][0]):.0f}% less memory on medium audio and "
        f"{100*numbers['long_untiled'][1]:.0f}% faster with {100*(1-1/numbers['long_untiled'][0]):.0f}% less on long."))
    conclusion_bar = Block.build(fig, "conclusion-bar", lambda: bar(fig, results.width, "Conclusion"))
    conclusion = Block.build(fig, "conclusion", lambda: paragraph(
        fig, 0, 0, results.width,
        "We have fused the joint network and loss into kernels that store only the two transition probabilities, "
        "and added tile-wise recomputation so that the large joint intermediates are bounded by the tile size "
        f"rather than the sequence length. In the training run of Table 1 this reduced peak memory by {memory_cut}% "
        f"and step time by {time_cut}%. The loss's own alignment buffers still grow with sequence length, and "
        "bounding them is our next step."))
    fixed = sum(b.height for b in [figure_caption, findings, conclusion_bar, conclusion])+.15+.4+.5+.25
    chart_height = BOTTOM-results.top-results.natural_height-fixed-.45
    assert chart_height >= 5.8, chart_height
    results.add(Block.build(fig, "comparison-plots", lambda: comparison(fig, chart_height)), .35)
    results.add(figure_caption, .15)
    results.add(findings, .4)
    results.add(conclusion_bar, .5)
    results.add(conclusion, .25)
    results.arrange(flexible_before="comparison-plots")

    paragraph(fig, LEFT, 22.5, RIGHT-LEFT,
              "[1] Graves, Sequence Transduction, 2012.  [2] Braun et al., Sample-Wise Computation, ICASSP 2023.  "
              "[3] Chen et al., Sublinear Memory Cost, 2016.  [4] NVIDIA Apex.  [5] Koluguri et al., Granary, "
              "Interspeech 2025.  [6] Fox et al., Long-Form Speech Recognition, ICASSP 2024.  [7] NVIDIA NeMo.  "
              "[8] Kuang, Optimized Transducer.", 18, color=MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_box = title.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    assert title_box.x0 > 4.2 and title_box.x1 < 39.7, (title_box.x0, title_box.x1)
    report = {str(i+1): column.validate() for i, column in enumerate(columns)}
    for artist in [*fig.texts, *fig.axes, *fig.legends]:
        box = artist.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        assert box.x0 >= LEFT-.01 and box.x1 <= RIGHT+.01, (artist, "horizontal margin")
        assert box.y0 >= 1-.01 and box.y1 <= H-1+.01, (artist, "vertical margin")
    copy = " ".join(t.get_text() for t in fig.texts)
    assert ":" not in copy.replace("Flash RNN-T:", ""), "colon in poster copy"
    report["column_bottoms_inches"] = [H-min(b.bounds.y0 for b, _ in c.entries) for c in columns]
    (OUT / "layout-check.json").write_text(json.dumps(report, indent=2)+"\n")
    fig.savefig(OUT / "poster.png", dpi=85, facecolor="white")
    fig.savefig(OUT / "poster.pdf", dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT / "poster.pdf", [round(b, 2) for b in report["column_bottoms_inches"]])


if __name__ == "__main__":
    sys.exit(main())
