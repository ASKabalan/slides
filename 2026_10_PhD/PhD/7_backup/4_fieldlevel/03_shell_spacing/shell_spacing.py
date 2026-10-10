#!/usr/bin/env python3
# ENV: jax-fli
"""
The shell-spacing laws side by side, for the backup slide on shell spacing: the slide version of the
thesis figure chap6/shell_spacing.pdf (These_wassim/figures/chap6/shell_spacing.py).

An illustrative eight-shell lightcone out to 1000 Mpc/h (a 2000 Mpc/h box at 512^3), Planck18, with the
edges from jax-fli's resolve_geometry for five laws: uniform in comoving distance, in the scale factor
a, in the growth factor D_1, equal volume, and equal volume with the outer shells floored at 70 Mpc/h.
Top, the shells to scale around the observer, coloured by index. Bottom, per shell: its width, its
volume and the particles it receives (nbar = 512^3 / 2000^3), the error bar being the particles within
one rms linear displacement of either edge. No simulation is run.

Output (this directory): shell_spacing.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, skip_if_built, slide_style

OUT = "shell_spacing.svg"
skip_if_built(HERE, OUT)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax_cosmo as jc
from jax_fli import resolve_geometry

MESH, BOX, NB, FLOOR = 512, 2000.0, 8, 70.0
R_MAX = BOX / 2
cosmo = jc.Planck18()


def edges(spacing, min_width=1.0):
    _, c, w = resolve_geometry(cosmo, R_MAX, nb_shells=NB, shell_spacing=spacing, min_width=min_width)
    w = np.asarray(w)[np.argsort(np.asarray(c))]
    return np.concatenate([[0.0], np.cumsum(w)])


LAWS = [("comoving\ndistance", edges("comoving")), ("scale\nfactor $a$", edges("a")),
        ("growth\nfactor $D_1$", edges("growth")), ("equal\nvolume", edges("equal_vol")),
        (f"equal volume\nfloor {FLOOR:.0f}", edges("equal_vol", FLOOR))]
NBAR = MESH**3 / BOX**3


def counts(e):
    near, far = e[:-1], e[1:]
    mean = NBAR * 4 / 3 * np.pi * (far**3 - near**3)
    a_mid = np.atleast_1d(np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(0.5 * (near + far)))))
    k = jnp.logspace(-4, 1.3, 4000)
    sig = np.array([float(jnp.sqrt(jnp.trapezoid(jc.power.linear_matter_power(cosmo, k, a=a), k) / (6 * np.pi**2)))
                    for a in a_mid])
    return mean, NBAR * sig * 4 * np.pi * (near**2 + far**2)


slide_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["savefig.bbox"] = None
LAW_COLOURS = ["#2E2E2E", "#C2560A", "#2AA198", "#521463", "#3B6FB6"]
MARKERS = ["o", "s", "^", "D", "v"]
shell_colours = plt.cm.turbo(np.linspace(0.08, 0.92, NB))

fig = plt.figure(figsize=(10.4, 5.6))
for j, (title, e) in enumerate(LAWS):
    ax = fig.add_axes([0.035 + j * 0.193, 0.56, 0.17, 0.34])
    for i in range(NB - 1, -1, -1):
        ax.add_patch(Circle((0, 0), e[i + 1], fc=shell_colours[i], ec="black", lw=0.5))
    ax.set_xlim(-1.05 * R_MAX, 1.05 * R_MAX)
    ax.set_ylim(-1.05 * R_MAX, 1.05 * R_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, color=LAW_COLOURS[j], linespacing=1.1, pad=4)

idx = np.arange(NB)
bw = 0.8 / len(LAWS)
axes = [fig.add_axes([0.06 + c * 0.325, 0.1, 0.26, 0.36]) for c in range(3)]
for j, (title, e) in enumerate(LAWS):
    off = (j - (len(LAWS) - 1) / 2) * bw
    axes[0].bar(idx + off, np.diff(e), bw, color=LAW_COLOURS[j])
    axes[1].bar(idx + off, 4 / 3 * np.pi * np.diff(e**3) / 1e9, bw, color=LAW_COLOURS[j])
    mean, leak = counts(e)
    axes[2].errorbar(idx + off, mean, yerr=leak, fmt=MARKERS[j], ms=5, lw=1.2, capsize=2,
                     color=LAW_COLOURS[j], mec="black", mew=0.4)
axes[2].set_yscale("log")
for ax, lab in zip(axes, ("width  [Mpc/$h$]", r"volume  [$10^9\,($Mpc/$h)^3$]", "particles per shell")):
    ax.set_title(lab, fontsize=13, color=INK, pad=4)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx], fontsize=11)
    ax.set_xlabel("shell", labelpad=1)
    ax.grid(alpha=0.25, axis="y")
fig.savefig(HERE / OUT)
for (title, e) in LAWS:
    print(title.replace("\n", " "), "widths", np.diff(e).round(0))
print(f"wrote {OUT}")
