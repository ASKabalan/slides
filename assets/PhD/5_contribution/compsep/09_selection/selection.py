#!/usr/bin/env python3
# ENV: furax-cs
"""
Choosing the sky partition by the 68 % upper limit on r.

selection.mp4 (+ selection_last.png)
    the thesis figure chap5/section_42/variance_vs_residual_r_total, built point by point along
    the monotonic-complexity chain of the low-latitude region GAL060, from the figure's own data
    (decode_thesis_chain.py): upper panel r + sigma(r), lower panel the systematic bias r_sys,
    against the total number of patches and coloured by the variance of the reconstructed CMB.
    Beside it, the patches of the parameter being refined and the likelihood on r of the nearest
    run in the public dataset. The optimum (smallest r + sigma(r)) is ringed at the end, and the
    last frame holds its patches and likelihood. Plays once on the slide.
best_patches_{beta_dust,temp_dust,beta_pl}.png
    the selected partition, all three regions (paper Fig. 9)
gridding_{beta_dust,temp_dust,beta_pl}.svg
    r + sigma(r) against the patch count of one parameter, the other two fixed, per region
    (paper Fig. 8a-c); an isolated spike (above 1.5x both neighbours) is replaced by the mean of
    its neighbours, and reported when the script runs
r_likelihood_regions.svg, r_likelihood_combined.svg
    the likelihood on r of the best configuration of each region, then of the whole sky (Fig. 8d)

Data: selection_data.py (processed rows) and _data.snapshot (raw runs through r_analysis snap).
"""

import csv
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _common import INK, skip_if_built, slide_style
from _compsep import moll_png, plot_r_likelihoods, r_formatters, shuffled
from _data import snapshot
from selection_data import PARAMS, chain, scans

OUTS = (["selection.mp4", "selection_last.png", "r_likelihood_regions.svg", "r_likelihood_combined.svg"]
        + [f"best_patches_{p}.png" for p in PARAMS] + [f"gridding_{p}.svg" for p in PARAMS])
skip_if_built(HERE, *OUTS)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from PIL import Image

BG = "#faf7f0"
REGION = {"GAL020": ("hi-lat", "#3B6FB6"), "GAL040": ("mid-lat", "#E08A1E"), "GAL060": ("low-lat", "#C0392B")}
PNAME = {"beta_dust": r"$\beta_d$", "temp_dust": r"$T_d$", "beta_pl": r"$\beta_s$"}
fmt, resolve = r_formatters()
slide_style(scale=1.25)

# ---------------------------------------------------------------- the chain animation
# Right: the thesis figure's own points (decode_thesis_chain.py). Left: the patches and the
# likelihood on r of the run of the public dataset with the nearest total patch count.
C = chain()
with open(HERE / "thesis_chain.csv") as f:
    pts = list(csv.DictReader(f))
up = sorted((int(r["total_patches"]), float(r["value"]), r["colour"]) for r in pts if r["panel"] == "upper")
lo = sorted((int(r["total_patches"]), float(r["value"]), r["colour"]) for r in pts if r["panel"] == "lower")
steps = [t for t, _, _ in up]
best = min(range(len(up)), key=lambda j: up[j][1])
VMIN, VMAX = 0.9807, 1.0949                          # the thesis colour bar


def nearest(total):
    return int(np.argmin(np.abs(C["total"] - total)))


def patch_image(i):
    ph = int(C["phase"][i])
    buf = io.BytesIO()
    moll_png(shuffled(C["patches"][i][ph], seed=i), buf, vmin=0, vmax=1, figsize=(5.2, 2.8), dpi=110)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGBA"))


PATCH = {}


def draw(T, ring=False, left=None):
    i = nearest(T if left is None else left)
    if i not in PATCH:
        PATCH[i] = patch_image(i)
    fig = plt.figure(figsize=(12.8, 6.6), dpi=100, facecolor=BG)
    axp = fig.add_axes([0.02, 0.56, 0.34, 0.36])
    axl = fig.add_axes([0.07, 0.1, 0.29, 0.38])
    axt = fig.add_axes([0.46, 0.53, 0.43, 0.38])
    axb = fig.add_axes([0.46, 0.1, 0.43, 0.38], sharex=axt)
    cax = fig.add_axes([0.905, 0.1, 0.015, 0.81])
    for ax in (axl, axt, axb):
        ax.set_facecolor(BG)
    ph = int(C["phase"][i])
    axp.imshow(PATCH[i])
    axp.set_axis_off()
    axp.set_title(rf"{PNAME[PARAMS[ph]]} patches:  $K = {C['k'][i][ph]}$", fontsize=16, color=INK)
    r = C["r_grid"][i]
    L = C["L"][i] / C["L"][i].max()
    rb, sp, sn = C["r"][i], C["sp"][i], C["sn"][i]
    axl.plot(r, L, color="#521463", lw=2.2)
    axl.fill_between(r, 0, L, where=(r > rb - sn) & (r < rb + sp), color="#521463", alpha=0.2)
    axl.axvline(rb, color="#521463", ls="--", lw=1.4)
    axl.axvline(0, color=INK, ls=":", lw=1.8)                       # the input sky has r = 0
    axl.text(0, 1.1, r"truth $r=0$", color=INK, fontsize=12, ha="center", va="bottom", clip_on=False)
    axl.set_xlim(-0.001, 0.005)
    axl.set_ylim(0, 1.08)
    axl.set_xlabel(r"$r$")
    axl.set_ylabel(r"$L_{\mathrm{cosmo}}$")
    axl.grid(True, ls=":", alpha=0.5)
    # the two panels of the thesis figure, revealed up to T
    for ax, data, filled in ((axt, up, True), (axb, lo, False)):
        shown = [q for q in data if q[0] <= T]
        ax.plot([q[0] for q in shown], [q[1] for q in shown], color="#999999", lw=1, zorder=1)
        for t, v, c in shown:
            if filled:
                ax.scatter(t, v, color=c, s=58, edgecolors=INK, linewidths=0.5, zorder=2)
            else:
                ax.scatter(t, v, facecolors="none", edgecolors=c, s=58, linewidths=1.4, zorder=2)
    if ring:
        axt.scatter(up[best][0], up[best][1], s=420, facecolors="none", edgecolors="#E41A1C",
                    linewidths=2.6, zorder=3)
    axb.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=8, color=INK, label=r"total $r+\sigma(r)$"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec=INK, mew=1.4, label="systematic bias"),
        Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec="#E41A1C", mew=2.2,
               label=r"optimum (min $r+\sigma$)")],
        loc="upper right", fontsize=12, frameon=True, framealpha=0.95)
    axt.set_xlim(0, 23500)
    axt.set_ylim(1.5, 5.3)
    axt.set_ylabel(r"total $(r+\sigma)\times10^{3}$")
    axt.tick_params(labelbottom=False)
    axb.set_yscale("log")
    axb.set_ylim(1e-7, 1.5e-3)
    axb.set_ylabel(r"systematic bias $r_{\mathrm{sys}}$")
    axb.set_xlabel("total number of patches")
    for ax in (axt, axb):
        ax.grid(True, ls=":", alpha=0.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(VMIN, VMAX), cmap="viridis"), cax=cax)
    cb.set_ticks([1.00, 1.02, 1.04, 1.06, 1.08])
    cb.set_label(r"variance (Q + U)  [$\mu$K$^2$]")
    fig.canvas.draw()                                    # a fixed frame size (no tight bbox)
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


frames = [draw(T) for T in steps]
last = draw(steps[-1], ring=True, left=up[best][0])        # hold on the optimum
fps = 6
h, w = last.shape[:2]
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", str(HERE / "selection.mp4")],
               input=b"".join(f.tobytes() for f in frames + [last] * (3 * fps)), check=True)
Image.fromarray(last).save(HERE / "selection_last.png")
print(f"wrote selection.mp4 ({len(steps)} steps; optimum at {up[best][0]} patches, "
      f"(r + sigma) = {up[best][1]:.3f} x 1e-3; left panels from the run at {C['total'][nearest(up[best][0])]})")

# ---------------------------------------------------------------- per-region material
best_rows = {r["kw"]: r for r in snapshot(
    "per_mask_best", ["GAL020", "GAL040", "GAL060"], ["KMEANS_BEST_BEST"],
    fetch=["KMEANS_BEST_BEST/*"], names=["GAL020", "GAL040", "GAL060"], noise_selection="min-nll")}
combined = snapshot("best_combined", ["GAL020", "GAL040", "GAL060"], ["KMEANS_BEST_BEST"],
                    fetch=["KMEANS_BEST_BEST/*"], combine=True, names=["COMBINED"],
                    noise_selection="min-nll")[0]

for p in PARAMS:
    lab = np.asarray(combined[f"patches_{p}"], dtype=float)
    moll_png(shuffled(lab, seed=len(p) + 3), HERE / f"best_patches_{p}.png", vmin=0, vmax=1,
             figsize=(4.6, 2.5), dpi=180)

def despike(k, y, label, factor=1.5):
    """Replace an isolated spike (a point above `factor` times both of its neighbours) by the
    mean of its neighbours, and say so."""
    y = y.copy()
    for i in range(1, len(y) - 1):
        if y[i] > factor * max(y[i - 1], y[i + 1]):
            fixed = (y[i - 1] + y[i + 1]) / 2
            print(f"  {label}: spike at K = {k[i]} replaced, {y[i] * 1e3:.2f} -> {fixed * 1e3:.2f} x 1e-3")
            y[i] = fixed
    return y


S = scans()
for p in PARAMS:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for region, (name, col) in REGION.items():
        k, rs = S[(p, region)]
        rs = despike(k, rs, f"{p} {region}")
        ax.plot(k, rs * 1e3, "o-", color=col, ms=4, lw=1.8, label=name)
        j = int(np.argmin(rs))
        ax.scatter(k[j], rs[j] * 1e3, s=160, facecolors="none", edgecolors=col, linewidths=2.2)
    ax.set_xlabel(rf"number of patches $K_{{{PNAME[p][1:-1]}}}$")
    ax.set_ylabel(r"$(r+\sigma)\times10^{3}$")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.95)
    fig.savefig(HERE / f"gridding_{p}.svg", transparent=True)
    plt.close(fig)

for out, entries in (
    ("r_likelihood_regions.svg", [("Best hi-lat", best_rows["GAL020"], REGION["GAL020"][1]),
                                  ("Best mid-lat", best_rows["GAL040"], REGION["GAL040"][1]),
                                  ("Best low-lat", best_rows["GAL060"], REGION["GAL060"][1])]),
    ("r_likelihood_combined.svg", [("All combined", combined, "#2E8B3E")])):
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    plot_r_likelihoods(ax, entries, xlim=(-0.001, 0.004), ytop=1.42, exponent=-4)
    ax.legend(loc="upper right", fontsize=11.5, frameon=True, framealpha=0.95)
    fig.savefig(HERE / out, transparent=True)
    plt.close(fig)
print("wrote the per-region figures")
