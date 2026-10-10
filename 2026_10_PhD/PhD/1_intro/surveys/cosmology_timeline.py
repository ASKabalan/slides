#!/usr/bin/env python3
# ENV: shared
"""
From Euclid to Hubble: the timeline across the top of the "Modern observational cosmology" slide.

A flat arrow, gold on the left to indigo on the right (the palette of hubble_diagram.py and
flrw_curv.svg on the same slide), broken (//) after Euclid. One dot per name, names alternating
above and below the line. Not to scale in time.

Three states with one fixed frame, so they overlay 1:1 in the slide's r-stack, each ringing (big
circle, bold name, coloured year) the names the slide is talking about:

  cosmology_timeline_einstein.svg   on entry: Einstein
  cosmology_timeline_fl.svg         with the FLRW geometry (click 1): Friedmann and Lemaitre
  cosmology_timeline_hubble.svg     with the Hubble block (click 2 onwards): Hubble

Output (this directory): the three SVGs above
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

STATES = {"einstein": {"Einstein"}, "fl": {"Friedmann", "Lemaître"}, "hubble": {"Hubble"}}
OUTS = [f"cosmology_timeline_{k}.svg" for k in STATES]
skip_if_built(HERE, *OUTS)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GOLD, INDIGO = "#b8791d", "#4a3fb0"
TEXT, MUTED, HAIR = "#2b2b33", "#7a7a86", "#b9b6c2"
RAMP = LinearSegmentedColormap.from_list("timeline", [GOLD, "#8a5a6e", INDIGO])

# name, year label, side (+1 above the line, -1 below)
EVENTS = [("Euclid", "~300 BCE", 1), ("Copernicus", "1543", -1), ("Kepler", "1609", 1),
          ("Galileo", "1610", -1), ("Newton", "1687", 1), ("Riemann", "1854", -1),
          ("Einstein", "1915", 1), ("Slipher", "1917", -1), ("Friedmann", "1922", 1),
          ("Lemaître", "1927", -1), ("Hubble", "1929", 1)]
X = np.array([0.0, 1.45] + [1.45 + 0.8 * k for k in range(1, len(EVENTS) - 1)])
X_BREAK = 0.72
X_END = X[-1] + 0.85
NAME_PT, YEAR_PT, RING_PT = 20, 13.5, 22     # well under the slide title

plt.rcParams.update({"font.family": "Lato", "svg.fonttype": "path"})


def colour(x):
    return RAMP(np.clip(x / X[-1], 0, 1))


def draw(ringed, out):
    # a fixed frame (no tight bounding box): the three states must register exactly
    fig = plt.figure(figsize=(14, 2.25))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-0.55, X_END + 0.15)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # the line: dashes, each in the colour of its position
    for a in np.arange(-0.35, X_END - 0.2, 0.34):
        b = min(a + 0.3, X_END - 0.2)
        if a < X_BREAK + 0.12 and b > X_BREAK - 0.12:
            continue                                      # room for the break mark
        ax.plot([a, b], [0, 0], color=colour(0.5 * (a + b)), lw=5, solid_capstyle="butt", zorder=1)
    ax.fill([X_END - 0.22, X_END - 0.22, X_END], [-0.2, 0.2, 0], color=INDIGO, lw=0, zorder=1)
    for dx in (-0.045, 0.045):                            # the // break
        ax.plot([X_BREAK + dx - 0.05, X_BREAK + dx + 0.05], [-0.16, 0.16], color=HAIR, lw=2.4,
                solid_capstyle="round", zorder=2)

    for (name, year, side), x in zip(EVENTS, X):
        on = name in ringed
        c = colour(x)
        if on:
            ax.scatter([x], [0], s=700, facecolor="none", edgecolor=c, alpha=0.55, lw=2.2, zorder=3)
        ax.scatter([x], [0], s=200 if on else 120, color=c, edgecolor="white", lw=1.8, zorder=4)
        ax.annotate("", xy=(x, 0), xytext=(0, 21 * side), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=HAIR, lw=1.4, shrinkA=0, shrinkB=10))
        ax.annotate(year, xy=(x, 0), xytext=(0, 24 * side), textcoords="offset points",
                    ha="center", va="bottom" if side > 0 else "top", fontsize=YEAR_PT,
                    color=c if on else MUTED, fontweight="bold" if on else "normal")
        ax.annotate(name, xy=(x, 0), xytext=(0, 41 * side), textcoords="offset points",
                    ha="center", va="bottom" if side > 0 else "top",
                    fontsize=RING_PT if on else NAME_PT, color=TEXT,
                    fontweight="bold" if on else "medium")

    fig.savefig(HERE / out, transparent=True)
    plt.close(fig)
    print(f"wrote {out}")


for (key, ringed), out in zip(STATES.items(), OUTS):
    draw(ringed, out)
