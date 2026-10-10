#!/usr/bin/env python3
# ENV: shared
"""
Prior times likelihood gives the posterior, in one dimension, for the Bayesian inference slide:
prior (grey, dotted), likelihood (navy, dashed), posterior (purple) with its 68 % credible
interval shaded.

Also writes the outline badge: the frequentist sampling distribution beside the same posterior,
with no text or axes.

Outputs (this directory):
  bayes_posterior.svg     the slide figure
  mle_vs_bayes_icon.svg   the outline badge (2_outline/rail.tex)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, KW2, skip_if_built, slide_style

OUT = "bayes_posterior.svg"
ICON = "mle_vs_bayes_icon.svg"
skip_if_built(HERE, OUT, ICON)

slide_style(scale=1.2)
import matplotlib.pyplot as plt
from scipy.stats import norm

BLUE = "#25406B"
fig, axr = plt.subplots(figsize=(5.6, 4.6))

# ---------------------------------------------------------------- frequentist (icon only)
x = np.linspace(-4, 4, 800)
pdf = norm.pdf(x)
theta_true = -0.55
band = np.abs(x) <= 1

# ------------------------------------------------------------------- Bayesian
prior = norm.pdf(x, loc=0.4, scale=1.6)
like = norm.pdf(x, loc=-0.7, scale=0.8)
post = prior * like
post /= np.trapezoid(post, x)

order = np.argsort(post)[::-1]
mass = np.cumsum(post[order]) * (x[1] - x[0])
level = post[order][np.searchsorted(mass, 0.68)]
inside = post >= level

axr.fill_between(x[inside], 0, post[inside], color=KW2, alpha=0.20, lw=0)
axr.plot(x, prior / prior.max() * 0.22, color=GREY, lw=2.0, ls=":")
axr.plot(x, like / like.max() * 0.42, color=BLUE, lw=2.0, ls="--")
axr.plot(x, post, color=KW2, lw=2.8)
axr.text(2.1, 0.20, "prior", color=GREY, fontsize=12.5)
axr.text(-3.5, 0.30, "likelihood", color=BLUE, fontsize=12.5)
axr.text(-0.62, 0.60, "posterior", color=KW2, fontsize=13.5, ha="center",
         fontweight="bold")
axr.text(-0.62, 0.085, "68 %", color=KW2, ha="center", fontsize=13)
axr.set_xlabel(r"parameter $\theta$")
axr.set_ylim(0, 0.70)
axr.set_yticks([])

for ax in (axr,):
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(left=False, top=False, right=False)

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")

# ------------------------------------------------------------------ the icon
# Same curves and bands, stripped of every label, tick and spine but a baseline.
fig, (il, ir) = plt.subplots(1, 2, figsize=(4.0, 1.9))
il.fill_between(x[band], 0, pdf[band], color=BLUE, alpha=0.25, lw=0)
il.plot(x, pdf, color=BLUE, lw=3.0)
il.axvline(0.0, color=BLUE, lw=2.2)
il.axvline(theta_true, color=KW, lw=2.2, ls="--")
il.set_ylim(0, 0.46)
ir.fill_between(x[inside], 0, post[inside], color=KW2, alpha=0.22, lw=0)
ir.plot(x, prior / prior.max() * 0.22, color=GREY, lw=2.4, ls=":")
ir.plot(x, like / like.max() * 0.42, color=BLUE, lw=2.4, ls="--")
ir.plot(x, post, color=KW2, lw=3.2)
ir.set_ylim(0, 0.62)
for ax in (il, ir):
    ax.set_axis_off()
    ax.axhline(0, color=INK, lw=1.4)
fig.subplots_adjust(wspace=0.12)
fig.savefig(HERE / ICON, transparent=True)
print(f"wrote {ICON}")
