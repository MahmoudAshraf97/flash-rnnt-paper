"""Run with python -m unittest poster.test_layout."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from poster.layout_engine import Block, Column, fit_axes


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.fig = plt.figure(figsize=(12, 12))

    def tearDown(self):
        plt.close(self.fig)

    def text(self, name, copy):
        return Block.build(self.fig, name, lambda: self.fig.text(
            0, 1, copy, fontsize=20, va="top"))

    def test_stack_uses_actual_text_height(self):
        column = Column(self.fig, 1, 5, 1, 10)
        first = column.add(self.text("first", "One line\nTwo lines\nThree lines"))
        second = column.add(self.text("second", "Next block"), gap=.4)
        column.arrange()
        self.assertAlmostEqual(first.bounds.y0-second.bounds.y1, .4)
        column.validate()
        first.artists[0].set_text("Shorter")
        column.arrange()
        self.assertAlmostEqual(first.bounds.y0-second.bounds.y1, .4)
        column.validate()

    def test_flexible_space_does_not_scale_diagram(self):
        column = Column(self.fig, 1, 5, 1, 10)
        column.add(self.text("heading", "Heading"))

        def diagram():
            ax = self.fig.add_axes([0, .75, 5/12, 2/12])
            ax.set(xlim=(0, 5), ylim=(0, 2))
            ax.axis("off")
            ax.add_patch(Rectangle((0, 0), 5, 1.22))

        block = column.add(Block.build(self.fig, "diagram", diagram))
        closing = column.add(self.text("closing", "Closing paragraph"))
        column.arrange(flexible_before="diagram")
        node = block.artists[0].patches[0]
        self.assertAlmostEqual(node.get_window_extent().height/self.fig.dpi, 1.22)
        self.assertAlmostEqual(closing.bounds.y0, 2)
        column.validate()

    def test_plot_labels_stay_in_cell(self):
        ax = self.fig.add_axes([.1, .1, .5, .5])
        ax.plot([.1, 1, 10], [0, 20000, 30000])
        ax.set(xscale="log", ylabel="Retain\nSamples/s")
        ax.tick_params(labelsize=20)
        fit_axes(ax, 2, 2, 6, 5)
        box = ax.get_tightbbox(self.fig.canvas.get_renderer()).transformed(
            self.fig.dpi_scale_trans.inverted())
        self.assertGreaterEqual(box.x0, 2-.01)
        self.assertLessEqual(box.x1, 8+.01)
        self.assertGreaterEqual(box.y0, 5-.01)
        self.assertLessEqual(box.y1, 10+.01)

    def test_vertical_overflow_is_not_silently_compressed(self):
        column = Column(self.fig, 1, 5, 1, 1.1)
        column.add(self.text("too tall", "Line one\nLine two"))
        with self.assertRaises(ValueError):
            column.arrange()

    def test_width_overflow_is_detected(self):
        column = Column(self.fig, 1, .1, 1, 10)
        column.add(self.text("too wide", "This does not fit"))
        column.arrange()
        with self.assertRaises(AssertionError):
            column.validate()


if __name__ == "__main__":
    unittest.main()
