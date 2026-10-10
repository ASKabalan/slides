#!/usr/bin/env python3
# ENV: shared
"""
A decade of upper limits on the tensor-to-scalar ratio, and where the next
experiments aim.

Numbers as compiled by Tristram et al. (2022), arXiv:2112.07961, Table 3 and
Figure 6; the forecast sensitivities are the published targets of the Simons
Observatory (Ade et al. 2019, arXiv:1808.07445) and LiteBIRD (LiteBIRD
Collaboration 2023, arXiv:2202.02773).

The bar chart is the slide's argument: every factor of two costs a generation
of experiments, and the next factor of ten has to come from the analysis as
much as from the instrument.

Output (this directory): r_constraints.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, skip_if_built, slide_style

OUT = "r_constraints.svg"
skip_if_built(HERE, OUT)

# label, 95 % upper limit on r
LIMITS = [
    ("Planck PR1 (2013)", 0.11),
    ("PR1 + BK (2015)", 0.12),
    ("PR2 + BK (2016)", 0.09),
    ("BK15 (2018)", 0.07),
    ("PR3 + BK15 (2019)", 0.065),
    ("Planck PR4 (2021)", 0.056),
    ("PR4 + BK15 (2021)", 0.044),
    ("BK18 (2021)", 0.036),
    ("PR4 + BK18 (2022)", 0.032),
]
# name, target sigma(r)
TARGETS = [("Simons Observatory Nominal", 0.003), ("LiteBIRD", 0.001)]

slide_style(scale=1.2)
import matplotlib.pyplot as plt

labels = [n for n, _ in LIMITS]
values = np.array([v for _, v in LIMITS])
# rows top to bottom: the limits, one gap row for the separator, the forecasts
n_lim, n_fc = len(values), len(TARGETS)
y = n_fc + 1 + np.arange(n_lim)[::-1]
y_fc = np.arange(n_fc)[::-1]
y_sep = n_fc

fig, ax = plt.subplots(figsize=(7.4, 4.6))

colors = ["#9FB4CE"] * len(values)
colors[-1] = KW  # the tightest published limit (Tristram et al. 2022)
ax.barh(y, values, height=0.68, color=colors, edgecolor="none")
for yi, v, c in zip(y, values, colors):
    ax.text(v + 0.0025, yi, f"$r<{v:g}$", va="center", fontsize=12.5,
            color=KW if c == KW else INK,
            fontweight="bold" if c == KW else "normal")

# Forecasts, greyed out below a dashed line.
FC = "#C9C9C9"
ax.axhline(y_sep, color=GREY, ls="--", lw=1.2)
ax.text(0.146, y_sep + 0.12, "forecasts", color=GREY, fontsize=12,
        fontstyle="italic", ha="right", va="bottom")
for yi, (name, sig) in zip(y_fc, TARGETS):
    ax.barh(yi, sig, height=0.68, color=FC, edgecolor="none")
    ax.text(sig + 0.0025, yi, f"$\\sigma(r)\\sim{sig:g}$", va="center",
            fontsize=12.5, color=GREY)

ax.set_yticks(np.concatenate([y, y_fc]))
ax.set_yticklabels(labels + [n for n, _ in TARGETS], fontsize=12.5)
for t in ax.get_yticklabels()[n_lim:]:
    t.set_color(GREY)
ax.set_xlim(0, 0.148)
ax.set_ylim(-0.6, y[0] + 0.6)
ax.set_xlabel("95 % upper limit on the tensor-to-scalar ratio $r$")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(top=False, right=False)

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
