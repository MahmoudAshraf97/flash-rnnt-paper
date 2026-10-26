"""Measured vertical flow for the poster; dimensions are in physical inches."""

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox


@dataclass
class Block:
    name: str
    fig: object
    artists: list

    @classmethod
    def build(cls, fig, name, draw):
        before = set(fig.get_children())
        draw()
        return cls(name, fig, [a for a in fig.get_children() if a not in before])

    @property
    def bounds(self):
        renderer = self.fig.canvas.get_renderer()
        boxes = [a.get_tightbbox(renderer) if isinstance(a, Axes)
                 else a.get_window_extent(renderer) for a in self.artists]
        return Bbox.union(boxes).transformed(self.fig.dpi_scale_trans.inverted())

    @property
    def height(self):
        return self.bounds.height

    def place(self, x, top):
        """Translate as a group, without scaling fonts, nodes, or artwork."""
        box = self.bounds
        w, h = self.fig.get_size_inches()
        dx, dy = (x-box.x0)/w, (h-top-box.y1)/h
        for artist in self.artists:
            if isinstance(artist, Axes):
                p = artist.get_position()
                artist.set_position([p.x0+dx, p.y0+dy, p.width, p.height])
            elif isinstance(artist, Text):
                px, py = artist.get_position()
                artist.set_position((px+dx, py+dy))
            elif isinstance(artist, Legend):
                p = artist.get_bbox_to_anchor().transformed(self.fig.transFigure.inverted())
                artist.set_bbox_to_anchor((p.x0+dx, p.y0+dy, p.width, p.height))
            elif isinstance(artist, Line2D):
                px, py = artist.get_data()
                artist.set_data([v+dx for v in px], [v+dy for v in py])
            else:
                raise TypeError(type(artist))


class Column:
    def __init__(self, fig, x, width, top, bottom):
        self.fig, self.x, self.width = fig, x, width
        self.top, self.bottom = top, bottom
        self.entries = []

    def add(self, block, gap=.35):
        self.entries.append((block, gap if self.entries else 0))
        return block

    @property
    def natural_height(self):
        return sum(b.height+gap for b, gap in self.entries)

    def arrange(self, flexible_before=None):
        spare = self.bottom-self.top-self.natural_height
        if spare < -.01:
            raise ValueError(f"Column at {self.x:g} overflows by {-spare:.2f} inches")
        y = self.top
        for block, gap in self.entries:
            y += gap
            if block.name == flexible_before:
                y += max(0, spare)
            block.place(self.x, y)
            y += block.height

    def validate(self):
        h = self.fig.get_figheight()
        previous_bottom = h-self.top
        report = []
        for block, _ in self.entries:
            b = block.bounds
            assert b.x0 >= self.x-.01, (block.name, "left overflow")
            assert b.x1 <= self.x+self.width+.01, (block.name, "right overflow", b.width, self.width)
            assert b.y1 <= previous_bottom+.01, (block.name, "overlap")
            assert b.y0 >= h-self.bottom-.01, (block.name, "bottom overflow")
            report.append({"name": block.name, "top": round(h-b.y1, 3),
                           "bottom": round(h-b.y0, 3), "width": round(b.width, 3)})
            previous_bottom = b.y0
        return report


def fit_axes(ax, x, top, width, height):
    """Fit the complete plot, including all labels, inside an allotted cell."""
    fig = ax.figure
    fw, fh = fig.get_size_inches()
    # One point of clearance absorbs Agg/PDF font-metric rounding differences.
    inset = 1/72
    x, top, width, height = x+inset, top+inset, width-2*inset, height-2*inset
    target = Bbox.from_bounds(x, fh-top-height, width, height)
    for _ in range(5):
        inner = ax.get_position().transformed(fig.transFigure).transformed(fig.dpi_scale_trans.inverted())
        outer = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        left, right = inner.x0-outer.x0, outer.x1-inner.x1
        bottom, upper = inner.y0-outer.y0, outer.y1-inner.y1
        ax.set_position([(target.x0+left)/fw, (target.y0+bottom)/fh,
                         (width-left-right)/fw, (height-bottom-upper)/fh])
    outer = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    assert outer.x0 >= target.x0-.01 and outer.x1 <= target.x1+.01
    assert outer.y0 >= target.y0-.01 and outer.y1 <= target.y1+.01
