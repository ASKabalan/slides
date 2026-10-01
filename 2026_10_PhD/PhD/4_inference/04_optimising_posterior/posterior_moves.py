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
laplace.svg   the mode and the 1, 2, 3 sigma ellipses of C = H^-1, the inverse Hessian of
              -ln L at the mode
hpd.svg       direct samples and the 68.3, 95.4, 99.7 % highest-posterior-density regions,
              estimated from a kernel density of the samples (the thesis recipe)

The GIFs loop on a transparent background, with a still of the last frame beside each (.png).
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import INK, KW, KW2, skip_if_built

OUTS = ["minimise.gif", "sample.gif", "minimise.png", "sample.png", "laplace.svg", "hpd.svg"]
skip_if_built(HERE, *OUTS)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde

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

# ---------------------------------------------------------------- laplace (inverse Hessian)
C = np.linalg.inv(PREC)                          # = COV for a Gaussian
L = np.linalg.cholesky(C)
t = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(t), np.sin(t)])
fig, ax = panel(filled=False)
for n, alpha, lw in ((3, 0.45, 1.8), (2, 0.7, 2.2), (1, 1.0, 2.6)):
    e = n * (L @ circle)
    ax.plot(e[0], e[1], color=KW, lw=lw, alpha=alpha)
    # label each ellipse where it crosses the major axis, where the three are furthest apart
    w, V = np.linalg.eigh(C)
    lab = n * np.sqrt(w[-1]) * V[:, -1]
    ax.text(lab[0] + 0.15, lab[1], f"{n}σ", color=KW, fontsize=14, fontweight="bold",
            ha="left", va="center")
ax.plot(0, 0, "o", color=KW, ms=11, mec="white", mew=1.8)
fig.savefig(HERE / "laplace.svg", transparent=True)
plt.close(fig)
print("wrote laplace.svg")

# ---------------------------------------------------------------- HPD regions from samples
draws = np.random.default_rng(42).multivariate_normal([0, 0], COV, size=8000)
kde = gaussian_kde(draws.T)
dens = kde(draws.T)
masses = (0.997, 0.954, 0.683)
levels = np.percentile(dens, [100 * (1 - m_) for m_ in masses])
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
fig, ax = panel(filled=False)
ax.scatter(draws[:, 0], draws[:, 1], s=2, color=KW2, alpha=0.12, rasterized=True, zorder=1)
ax.contourf(X, Y, Z, levels=[levels[0], levels[1], levels[2], Z.max()],
            colors=["#E9DDF0", "#CDB5DC", "#A57DBE"], alpha=0.6, zorder=2)
cs = ax.contour(X, Y, Z, levels=levels, colors=[INK], linewidths=1.1, zorder=3)
ax.clabel(cs, fmt={levels[0]: "99.7 %", levels[1]: "95.4 %", levels[2]: "68.3 %"},
          inline=True, fontsize=11)
fig.savefig(HERE / "hpd.svg", transparent=True, dpi=200)
plt.close(fig)
print("wrote hpd.svg")
