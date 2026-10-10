#!/usr/bin/env python3
# ENV: shared
"""
How samples give the uncertainty, as simply as possible: a histogram of 20 000 samples from one
1D Gaussian posterior, with the central 68, 95 and 99.7 % of the samples shaded (the 1, 2 and
3 sigma regions; brackets under the axis give the cumulative regions). The bounds are the sample
percentiles, not drawn by hand. In orange, the point-estimate view: theta-hat (the MAP) and the
Gaussian of width sigma from the curvature, on the histogram's normalisation.

Output (this directory): trust_1d.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, KW, KW2, skip_if_built, slide_style

OUT = "trust_1d.svg"
skip_if_built(HERE, OUT)
slide_style(scale=1.4)
import matplotlib.pyplot as plt

rng = np.random.default_rng(20261012)
x = rng.standard_normal(20_000)
LEVELS = ((99.73, 0.22, "99.7 %", r"$3\sigma$"), (95.45, 0.45, "95 %", r"$2\sigma$"),
          (68.27, 0.85, "68 %", r"$1\sigma$"))

fig, ax = plt.subplots(figsize=(9.0, 3.4))
bins = np.linspace(-4, 4, 81)
h, e = np.histogram(x, bins=bins, density=True)
c, w = 0.5 * (e[1:] + e[:-1]), e[1] - e[0]
ax.bar(c, h, width=w, color=KW2, alpha=0.08, lw=0)
for q, a, lab, sig in LEVELS:
    lo, hi = np.percentile(x, [50 - q / 2, 50 + q / 2])
    m = (c >= lo) & (c <= hi)
    ax.bar(c[m], h[m], width=w, color=KW2, alpha=a, lw=0)
# the point estimate: theta-hat (the MAP) and the Gaussian of width sigma from the curvature,
# on the histogram's normalisation (density), to compare with the samples
xs = np.linspace(-4, 4, 400)
ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), color=KW, lw=1.8, ls="--")
ax.axvline(0, ymax=0.98, color=KW, lw=1.6)

# brackets under the axis: each percentage is the cumulative region within +-n sigma
from matplotlib.transforms import blended_transform_factory
tr = blended_transform_factory(ax.transData, ax.transAxes)
for n, yb, lab in ((1, -0.27, "68 %"), (2, -0.38, "95 %"), (3, -0.49, "99.7 %")):
    lo, hi = np.percentile(x, [50 - LEVELS[3 - n][0] / 2, 50 + LEVELS[3 - n][0] / 2])
    ax.plot([lo, lo, hi, hi], [yb + 0.04, yb, yb, yb + 0.04], color=KW2, lw=1.8,
            transform=tr, clip_on=False, solid_capstyle="butt")
    ax.text(hi + 0.12, yb, lab, color=KW2, fontsize=15, fontweight="bold", va="center",
            transform=tr, clip_on=False)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3],
              [r"$-3\sigma$", r"$-2\sigma$", r"$-1\sigma$", r"$\hat\theta$", r"$1\sigma$",
               r"$2\sigma$", r"$3\sigma$"])
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.45)
ax.set_yticks([])
ax.set_xlabel(r"$\theta$", fontsize=24, ha="left", va="center")
ax.xaxis.set_label_coords(1.01, 0.0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(top=False, right=False)
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
