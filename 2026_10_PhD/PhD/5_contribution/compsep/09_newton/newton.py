#!/usr/bin/env python3
# ENV: furax-cs
"""
Newton's method on the real spectral likelihood, and what noise does to its Hessian.

newton_bowl.gif      the negative log-likelihood of one faint high-latitude patch against
                     (beta_d, T_d), noise-free; a ball takes damped Newton steps, each step
                     drawn with the ellipses of the local quadratic model it jumps to
noise_left.gif       the same landscape as the noise rises from zero to the LiteBIRD depth, with
                     a slider, and the eigenvalues of the Hessian of the high-latitude region cut
                     into 10 patches (30 parameters): the smallest shrinks and turns negative
snr_map.png          signal-to-noise of the foreground polarisation per pixel at 140 GHz

Data: newton_data.py (cached in ../.cache/newton_data.npz). Every GIF loops; a .png beside each
holds its last frame for print.
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _common import GREY, INK, KW, KW2, skip_if_built, slide_style
from _compsep import MASKED

OUTS = ["newton_bowl.gif", "noise_left.gif", "snr_map.png"]
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from PIL import Image

from newton_data import load

D = load()
BG = (250, 247, 240)
BD, TD = D["bd"], D["td"]
B, T = np.meshgrid(BD, TD, indexing="ij")
PURPLES = ["#3E1654", "#5E2C7A", "#7F4A9E", "#A57DBE", "#CDB5DC", "#E9DDF0", "#F6F1F8"]
DELTA = [0, 2, 8, 30, 100, 300, 1000, 1e9]


def grab(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")


def to_gif(frames, path, durations):
    frames[-1].save(path.with_suffix(".png"))
    rgb = [Image.alpha_composite(Image.new("RGBA", f.size, BG + (255,)), f).convert("RGB") for f in frames]
    pal = rgb[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = []
    for f, r in zip(frames, rgb):
        q = r.quantize(palette=pal, dither=Image.Dither.NONE)
        q.paste(255, mask=f.getchannel("A").point(lambda a: 255 if a < 40 else 0))
        gif.append(q)
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=durations, loop=0,
                transparency=255, disposal=2, optimize=True)
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def bowl(ax, grid):
    g = grid - grid.min()
    ax.contourf(B, T, g, levels=DELTA, colors=PURPLES, alpha=0.95)
    ax.contour(B, T, g, levels=DELTA[1:4], colors=[KW2], linewidths=0.8, alpha=0.5)
    ax.set_xlim(BD[0], BD[-1])
    ax.set_ylim(TD[0], TD[-1])
    ax.set_xlabel(r"dust index $\beta_d$")
    ax.set_ylabel(r"dust temperature $T_d$ [K]")


slide_style(scale=1.35)

# ---------------------------------------------------------------- Newton on the bowl
path, models = D["path"], D["models"]
t = np.linspace(0, 2 * np.pi, 200)


def model_ellipses(ax, m, alpha):
    centre, H = m[:2], m[2:6].reshape(2, 2)
    w, V = np.linalg.eigh(H)
    for c in (2.0, 8.0):
        e = centre[:, None] + np.sqrt(2 * c) * V @ (np.stack([np.cos(t), np.sin(t)]) / np.sqrt(w)[:, None])
        ax.plot(e[0], e[1], color=KW, lw=2.0, ls="--", alpha=alpha)


frames, dur = [], []
nstep = len(models)
for k in range(nstep):
    for phase in range(7):
        fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=95)
        bowl(ax, D["grids"][0])
        ax.plot(path[: k + 1, 0], path[: k + 1, 1], color=KW, lw=1.8, alpha=0.8)
        model_ellipses(ax, models[k], alpha=min(1.0, 0.35 + phase * 0.25))
        s = min(1.0, max(0.0, (phase - 2) / 4))
        p = path[k] + s * (path[k + 1] - path[k])
        if phase >= 3:
            ax.annotate("", xy=path[k + 1], xytext=path[k],
                        arrowprops=dict(arrowstyle="->", color=KW, lw=1.6, alpha=0.7))
        ax.plot(*p, "o", color=KW, ms=13, mec="white", mew=2, zorder=5)
        frames.append(grab(fig))
        dur.append(120 if phase not in (0, 6) else 450)
    if np.linalg.norm(path[k + 1] - path[k]) < 2e-3:
        break
fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=95)
bowl(ax, D["grids"][0])
ax.plot(path[:, 0], path[:, 1], color=KW, lw=1.8, alpha=0.8)
ax.plot(*path[-1], "o", color=KW, ms=13, mec="white", mew=2, zorder=5)
frames.append(grab(fig))
dur.append(2400)
to_gif(frames, HERE / "newton_bowl.gif", dur)

# ---------------------------------------------------------------- the noise slider (left)
noise, eig = D["noise"], D["eig30"]
left = []


def slider(ax, r):
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()
    ax.plot([0, 1], [0, 0], color=GREY, lw=5, solid_capstyle="round", alpha=0.45)
    ax.plot([0, r], [0, 0], color=KW, lw=5, solid_capstyle="round")
    ax.plot([r], [0], "o", color=KW, ms=14, mec="white", mew=2)
    ax.text(0, -0.95, "no noise", ha="left", va="center", fontsize=12, color=GREY)
    ax.text(1, -0.95, "LiteBIRD", ha="right", va="center", fontsize=12, color=GREY)


for i, r in enumerate(noise):
    fig = plt.figure(figsize=(6.0, 7.6), dpi=110)
    ax = fig.add_axes([0.15, 0.56, 0.8, 0.41])
    bowl(ax, D["grids"][i])
    k = np.unravel_index(np.argmin(D["grids"][i]), D["grids"][i].shape)
    ax.plot(BD[k[0]], TD[k[1]], "o", color="white", ms=9, mec=INK, mew=1.4)
    axe = fig.add_axes([0.15, 0.14, 0.8, 0.28])
    ev = np.sort(eig[i])
    axe.bar(np.arange(ev.size), ev, color=[KW2 if v > 0 else "#C0392B" for v in ev], width=0.8)
    axe.set_yscale("symlog", linthresh=30)
    axe.set_ylim(-1.5e3, 5e6)
    axe.axhline(0, color=INK, lw=0.8)
    axe.set_xticks([])
    axe.set_yticks([-1e3, 0, 1e3, 1e6])
    axe.set_yticklabels([r"$-10^3$", "0", r"$10^3$", r"$10^6$"], fontsize=11)
    axe.tick_params(right=False, top=False)
    lam = ev.min()
    fig.text(0.15, 0.44, "eigenvalues of the Hessian (30 parameters)", fontsize=14, color=INK)
    if lam <= 0:
        axe.text(0.04, 0.95, "not positive\ndefinite", fontsize=12, color="#C0392B", ha="left",
                 va="top", fontweight="bold", transform=axe.transAxes)
    axs = fig.add_axes([0.12, 0.01, 0.78, 0.08])
    slider(axs, r)
    left.append(grab(fig))

dur = [900] * len(noise)
dur[0], dur[-1] = 1500, 2600
to_gif(left, HERE / "noise_left.gif", dur)

# ---------------------------------------------------------------- SNR at 140 GHz
snr = np.array(D["snr_fg"], dtype=float)
snr[~D["mask"].astype(bool)] = hp.UNSEEN
fig = plt.figure(figsize=(6.4, 4.0), dpi=150)
lsnr = np.where(snr > 0, np.log10(np.clip(snr, 0.3, 300)), hp.UNSEEN)
hp.mollview(lsnr, fig=fig.number, title="", cbar=False, cmap="viridis", min=np.log10(0.3),
            max=np.log10(300), notext=True, bgcolor=(0.0,) * 4, badcolor=MASKED,
            margins=(0.0, 0.18, 0.0, 0.0), sub=(1, 1, 1))
cax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=LogNorm(0.3, 300)), cax=cax,
                  orientation="horizontal")
cb.set_ticks([1, 10, 100])
cb.set_ticklabels(["1", "10", "100"])
cb.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
cb.ax.tick_params(labelsize=13, colors=INK)
cb.outline.set_visible(False)
cb.set_label("foreground signal-to-noise per pixel", fontsize=13, color=INK)
im = grab(fig)
im.crop(im.getchannel("A").getbbox()).save(HERE / "snr_map.png")
print("wrote snr_map.png")
