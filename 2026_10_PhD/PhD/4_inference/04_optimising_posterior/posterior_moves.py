#!/usr/bin/env python3
# ENV: shared
"""
Optimising and sampling one posterior, and the uncertainty each gives.

The posterior is a correlated 2D Gaussian (sigma_1 = 1, sigma_2 = 1.4, rho = 0.65, as in the
thesis figure chap4/bayesian_2d_posterior). Every output shares one square panel and one axis
range, so they can replace each other on the slide.

minimise.gif  a point following gradient-descent steps from far away to the mode
sample.gif    a Metropolis-Hastings random walk: accepted moves joined by a thin line, the chain's
              samples accumulating as dots

The GIFs loop on a transparent background, with a still of the last frame beside each (.png).
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, KW, KW2, skip_if_built

OUTS = ["minimise.gif", "sample.gif", "minimise.png", "sample.png"]
skip_if_built(HERE, *OUTS)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans"})

BG = (250, 247, 240)              # slide colour, for pre-blending the anti-aliased edges
LIM = 5.3                         # the panel spans [-LIM, LIM] on both axes
S1, S2, RHO = 1.0, 1.4, 0.65
COV = np.array([[S1 ** 2, RHO * S1 * S2], [RHO * S1 * S2, S2 ** 2]])
PREC = np.linalg.inv(COV)         # the Hessian of -ln p


def logp(p):
    return -0.5 * p @ PREC @ p


def grad(p):
    return -PREC @ p


# ---------------------------------------------------------------- panel
X, Y = np.meshgrid(np.linspace(-LIM, LIM, 400), np.linspace(-LIM, LIM, 400))
XY = np.stack([X, Y], axis=-1)
P = np.exp(-0.5 * np.einsum("...i,ij,...j->...", XY, PREC, XY))
LEVELS = [np.exp(-0.5 * k ** 2) for k in (3, 2, 1)] + [1.0]     # the 3, 2, 1 sigma ellipses


def panel(filled=True):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    if filled:
        ax.contourf(X, Y, P, levels=LEVELS, colors=["#E9DDF0", "#CDB5DC", "#A57DBE"], alpha=0.95)
        ax.contour(X, Y, P, levels=LEVELS[:-1], colors=[KW2], linewidths=1.0, alpha=0.6)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def grab(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")


def to_gif(frames, name, duration):
    """Looping GIF: alpha thresholded to 1 bit after pre-blending the edges onto BG."""
    frames[-1].save(HERE / name.replace(".gif", ".png"))
    out = []
    for f in frames:
        bg = Image.new("RGBA", f.size, BG + (255,))
        rgb = Image.alpha_composite(bg, f).convert("RGB")
        out.append((rgb, f.getchannel("A")))
    pal = out[-1][0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = []
    for rgb, a in out:
        q = rgb.quantize(palette=pal, dither=Image.Dither.NONE)
        q.paste(255, mask=a.point(lambda v: 255 if v < 40 else 0))
        gif.append(q)
    gif[0].save(HERE / name, save_all=True, append_images=gif[1:], duration=duration, loop=0,
                transparency=255, disposal=2, optimize=True)
    print(f"wrote {name} ({(HERE / name).stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


# ---------------------------------------------------------------- minimise (gradient descent)
p = np.array([4.2, 0.6])
lr = 0.12
path = [p.copy()]
for _ in range(600):
    p = p + lr * grad(p)                           # descend -ln p
    path.append(p.copy())
path = np.array(path)
stop = int(np.argmax(np.linalg.norm(path, axis=1) < 0.02)) or len(path)

frames = []
for k in list(range(1, stop, 2)) + [stop] * 14:                 # hold on the mode
    fig, ax = panel()
    ax.plot(path[:k, 0], path[:k, 1], color=KW, lw=1.8, alpha=0.8)
    ax.plot(*path[k - 1], "o", color=KW, ms=11, mec="white", mew=1.8)
    frames.append(grab(fig))
to_gif(frames, "minimise.gif", 70)

# ---------------------------------------------------------------- sample (MH)
rng = np.random.default_rng(3)
q = np.array([-3.0, 3.2])
chain = [q.copy()]
for _ in range(2000):
    prop = q + rng.normal(scale=0.6, size=2)
    if np.log(rng.uniform()) < logp(prop) - logp(q):
        q = prop
    chain.append(q.copy())
chain = np.array(chain)

frames = []
for k in np.unique(np.geomspace(2, len(chain), 70).astype(int)):
    fig, ax = panel()
    ax.plot(chain[max(0, k - 25):k, 0], chain[max(0, k - 25):k, 1], color=KW, lw=1.0, alpha=0.7)
    ax.plot(chain[:k, 0], chain[:k, 1], ".", color=KW, ms=3.5, alpha=0.75)
    ax.plot(*chain[k - 1], "o", color=KW, ms=9, mec="white", mew=1.6)
    frames.append(grab(fig))
frames += [frames[-1]] * 12
to_gif(frames, "sample.gif", 90)
