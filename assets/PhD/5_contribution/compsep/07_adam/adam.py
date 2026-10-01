#!/usr/bin/env python3
# ENV: furax-cs
"""
Adaptive-moment minimisers on the noisy spectral likelihood.

adam_path.gif    Adam on the same landscape the previous slide ends on: the faint high-latitude
                 patch at the nominal LiteBIRD noise, (beta_d, T_d) at beta_s fixed. The surface
                 is the real negative log-likelihood (newton_data.py), interpolated by a bicubic
                 spline; the optimiser works in coordinates normalised to [0, 1] across the
                 grid, as AdaTopK does. Loops, with a still (.png) of the last frame.
adabelief.svg    Adam and AdaBelief on a quadratic likelihood whose gradient is measured with
                 noise, weak along the first parameter and strong along the second (the same
                 noise draws for both). Adam steps by m / sqrt(v) and follows the noisy
                 gradients; AdaBelief steps by m / sqrt(s), s the spread of the gradient about its
                 running mean, and shrinks the steps along the noisy direction. An illustration
                 in the spirit of Zhuang et al. (2020), Fig. 3: the real landscape above is too
                 smooth for the two to differ.
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "06_newton"))
from _common import GREY, INK, KW, KW2, skip_if_built, slide_style

OUTS = ["adam_path.gif", "adam_path.png", "adabelief.svg"]
skip_if_built(HERE, *OUTS)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RectBivariateSpline

from newton_data import START, load

D = load()
BG = (250, 247, 240)
BD, TD = D["bd"], D["td"]
B, T = np.meshgrid(BD, TD, indexing="ij")
grid = D["grids"][-1]                                    # nominal LiteBIRD noise
PURPLES = ["#3E1654", "#5E2C7A", "#7F4A9E", "#A57DBE", "#CDB5DC", "#E9DDF0", "#F6F1F8"]
DELTA = [0, 2, 8, 30, 100, 300, 1000, 1e9]

# ---------------------------------------------------------------- Adam in normalised coordinates
U = np.linspace(0, 1, BD.size)
V = np.linspace(0, 1, TD.size)
spl = RectBivariateSpline(U, V, grid)
lo, span = np.array([BD[0], TD[0]]), np.array([BD[-1] - BD[0], TD[-1] - TD[0]])
to_phys = lambda u: lo + span * u
u = (START - lo) / span
m, v = np.zeros(2), np.zeros(2)
lr, b1, b2, eps = 0.012, 0.9, 0.999, 1e-8
path = [to_phys(u)]
for t in range(1, 700):
    g = np.array([spl(*u, dx=1)[0, 0], spl(*u, dy=1)[0, 0]])
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    u = np.clip(u - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps), 0, 1)
    path.append(to_phys(u))
path = np.array(path)
kmin = np.unravel_index(np.argmin(grid), grid.shape)
print("Adam end", path[-1], "grid minimum", BD[kmin[0]], TD[kmin[1]])

slide_style(scale=1.35)


def grab(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")


def bowl(ax):
    g = grid - grid.min()
    ax.contourf(B, T, g, levels=DELTA, colors=PURPLES, alpha=0.95)
    ax.contour(B, T, g, levels=DELTA[1:4], colors=[KW2], linewidths=0.8, alpha=0.5)
    ax.plot(BD[kmin[0]], TD[kmin[1]], "o", color="white", ms=9, mec=INK, mew=1.4)
    ax.set_xlim(BD[0], BD[-1])
    ax.set_ylim(TD[0], TD[-1])
    ax.set_xlabel(r"dust index $\beta_d$")
    ax.set_ylabel(r"dust temperature $T_d$ [K]")


frames, dur = [], []
ks = list(range(1, 60, 2)) + list(range(60, len(path), 12)) + [len(path)]
for k in ks:
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=95)
    bowl(ax)
    ax.plot(path[:k, 0], path[:k, 1], color=KW, lw=1.8, alpha=0.85)
    ax.plot(*path[k - 1], "o", color=KW, ms=13, mec="white", mew=2, zorder=5)
    frames.append(grab(fig))
    dur.append(90)
dur[0], dur[-1] = 700, 2600
frames[-1].save(HERE / "adam_path.png")
rgb = [Image.alpha_composite(Image.new("RGBA", f.size, BG + (255,)), f).convert("RGB") for f in frames]
pal = rgb[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
gif = []
for f, r in zip(frames, rgb):
    q = r.quantize(palette=pal, dither=Image.Dither.NONE)
    q.paste(255, mask=f.getchannel("A").point(lambda a: 255 if a < 40 else 0))
    gif.append(q)
gif[0].save(HERE / "adam_path.gif", save_all=True, append_images=gif[1:], duration=dur, loop=0,
            transparency=255, disposal=2, optimize=True)
print(f"wrote adam_path.gif ({(HERE / 'adam_path.gif').stat().st_size / 1e6:.1f} MB)")

# ---------------------------------------------------------------- AdaBelief, in one picture
AX, AY = 1.0, 0.4                        # widths of the quadratic likelihood
SIG = np.array([0.5, 8.0])               # gradient noise: steady first parameter, noisy second


def descend(kind, lr, n=100, b1=0.9, b2=0.999, eps=1e-8, seed=9):
    rng = np.random.default_rng(seed)    # the same noise draws for both optimisers
    u, m, v, out = np.array([-1.8, 0.8]), np.zeros(2), np.zeros(2), []
    for t in range(1, n + 1):
        out.append(u.copy())
        g = u / np.array([AX, AY]) ** 2 + rng.normal(size=2) * SIG
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g if kind == "adam" else (g - m) ** 2 + eps)
        u = u - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return np.array(out)


pa, pb = descend("adam", 0.06), descend("adabelief", 0.02)
X, Y = np.meshgrid(np.linspace(-2.1, 0.7, 300), np.linspace(-1.0, 1.0, 220))
Z = 0.5 * ((X / AX) ** 2 + (Y / AY) ** 2)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.contourf(X, Y, Z, levels=[0, 0.02, 0.08, 0.2, 0.4, 0.8, 1.6, 1e9], colors=PURPLES, alpha=0.95)
ax.plot(pa[:, 0], pa[:, 1], "o-", color=GREY, lw=1.8, ms=3.5, alpha=0.9)
ax.plot(pb[:, 0], pb[:, 1], "o-", color=KW, lw=2.2, ms=3.5)
ax.plot(0, 0, "o", color="white", ms=10, mec=INK, mew=1.4, zorder=5)
ax.plot(*pa[0], "o", color=INK, ms=7, zorder=5)
ax.text(-1.25, -0.72, "Adam\nbold steps on noisy gradients", color=GREY, fontsize=14, ha="left",
        va="center", fontweight="bold")
ax.text(-1.2, 0.66, "AdaBelief\ndistrusts the noisy direction", color=KW, fontsize=14, ha="left",
        va="center", fontweight="bold")
ax.set_xlim(-2.1, 0.7)
ax.set_ylim(-1.0, 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$\beta_1$ (steady gradient)")
ax.set_ylabel(r"$\beta_2$ (noisy gradient)")
fig.savefig(HERE / "adabelief.svg", transparent=True)
print("wrote adabelief.svg")
