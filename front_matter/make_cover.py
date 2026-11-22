"""Generate the cover banner (cover.png) of the book.

Machine Learning with Python: From Linear Algebra to Deep Learning
Copyright (c) 2026 Mario Medone - MIT License (code).

The banner shows a matrix of numbers (linear algebra) on the left and a
layered neural network (deep learning) on the right, joined by an arrow.

Usage:
    python make_cover.py

Requirements: numpy, matplotlib.
The image is saved as cover.png in the same folder as this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# ------------------------------------------------------------------ style
BACKGROUND = "#0F1B2D"   # dark navy
MATRIX_LIGHT = "#1B3A57"  # low end of the matrix colormap
MATRIX_DARK = "#5FC8D8"   # high end of the matrix colormap (teal)
EDGE_COLOR = "#8FA3BA"    # network connections
NODE_COLOR = "#F2A65A"    # network nodes (warm accent)
ARROW_COLOR = "#E6EDF5"

WIDTH_IN, HEIGHT_IN, DPI = 8.0, 2.8, 200  # 1600 x 560 pixels
OUTPUT = Path(__file__).resolve().parent / "cover.png"

rng = np.random.default_rng(42)  # fixed seed: the image is reproducible

fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
fig.patch.set_facecolor(BACKGROUND)
ax.set_facecolor(BACKGROUND)
ax.set_xlim(0, 16)
ax.set_ylim(0, 5.6)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# ------------------------------------------------- left: a matrix of numbers
n = 7
cmap = LinearSegmentedColormap.from_list("matrix", [MATRIX_LIGHT, MATRIX_DARK])
values = rng.normal(size=(n, n))
values = values + 1.5 * np.eye(n)  # a visible diagonal structure
norm_values = (values - values.min()) / (values.max() - values.min())

cell, gap = 0.52, 0.06
x0, y0 = 1.3, (5.6 - n * cell) / 2
for i in range(n):
    for j in range(n):
        ax.add_patch(
            plt.Rectangle(
                (x0 + j * cell, y0 + (n - 1 - i) * cell),
                cell - gap,
                cell - gap,
                facecolor=cmap(norm_values[i, j]),
                edgecolor="none",
            )
        )
# matrix brackets
bx_left, bx_right = x0 - 0.18, x0 + n * cell - gap + 0.18
by_bottom, by_top = y0 - 0.1, y0 + n * cell - gap + 0.1
for bx, direction in ((bx_left, 1), (bx_right, -1)):
    ax.plot([bx + 0.12 * direction, bx, bx, bx + 0.12 * direction],
            [by_top, by_top, by_bottom, by_bottom],
            color=ARROW_COLOR, linewidth=1.6, solid_capstyle="round")

# --------------------------------------------------------- centre: the arrow
arrow = FancyArrowPatch(
    (5.6, 2.8), (7.7, 2.8),
    arrowstyle="-|>", mutation_scale=22,
    color=ARROW_COLOR, linewidth=2.2,
)
ax.add_patch(arrow)

# ------------------------------------------------ right: a neural network
layers = [4, 6, 6, 3]
layer_x = np.linspace(9.0, 14.8, len(layers))
positions = []
for x, size in zip(layer_x, layers):
    ys = np.linspace(0.9, 4.7, size) if size > 1 else np.array([2.8])
    # centre each layer vertically
    ys = ys - ys.mean() + 2.8
    positions.append([(x, y) for y in ys])

for left, right in zip(positions[:-1], positions[1:]):
    for (xa, ya) in left:
        for (xb, yb) in right:
            ax.plot([xa, xb], [ya, yb], color=EDGE_COLOR,
                    linewidth=0.6, alpha=0.35, zorder=1)

for layer in positions:
    for (x, y) in layer:
        ax.scatter(x, y, s=95, color=NODE_COLOR, edgecolor=BACKGROUND,
                   linewidth=1.6, zorder=3)

fig.savefig(OUTPUT, dpi=DPI, facecolor=BACKGROUND)
print(f"Saved {OUTPUT}")
