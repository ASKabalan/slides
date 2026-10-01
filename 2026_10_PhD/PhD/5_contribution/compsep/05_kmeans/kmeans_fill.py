#!/usr/bin/env python3
# ENV: furax-cs
"""
Spherical K-means on the nside 64 sky, from one patch to two thousand, as a looping animation
with a slider under the map showing K. The clustering is jax-healpy's find_kmeans_clusters, the
routine furax-cs uses to build its patches; each K is an independent run with the same seed.

Outputs (this directory): kmeans_fill.gif (+ kmeans_fill.png, the last frame, for print)
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from _common import GREY, INK, KW, skip_if_built, slide_style
from _compsep import CACHE, NSIDE, shuffled

OUTS = ["kmeans_fill.gif", "kmeans_fill.png"]
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

KS = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
KMAX = KS[-1]
BG = (250, 247, 240)


def labels():
    npz = CACHE / f"kmeans_fill_{NSIDE}.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[str(k)] for k in KS}
    import jax
    import jax.numpy as jnp
    from jax_healpy.clustering import find_kmeans_clusters

    npix = hp.nside2npix(NSIDE)
    mask = jnp.ones(npix)
    idx = jnp.arange(npix)
    out = {}
    for k in KS:
        lab = find_kmeans_clusters(mask, idx, k, jax.random.key(0), max_centroids=KMAX,
                                   initial_sample_size=1)
        out[k] = np.asarray(lab).astype(int)
        print(f"  K = {k}: {len(np.unique(out[k]))} patches")
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **{str(k): v for k, v in out.items()})
    return out


LAB = labels()
slide_style(scale=1.4)


def frame(k):
    fig = plt.figure(figsize=(7.0, 4.6), dpi=130)
    vals = shuffled(LAB[k], seed=k) if k > 1 else np.full(len(LAB[k]), 0.45)
    hp.mollview(vals, fig=fig.number, title="", cbar=False, cmap="viridis", min=0, max=1,
                notext=True, bgcolor=(0.0,) * 4, sub=(1, 1, 1), margins=(0.0, 0.2, 0.0, 0.0))
    ax = fig.add_axes([0.1, 0.03, 0.8, 0.14])
    ax.set_xlim(0, KMAX)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()
    ax.plot([0, KMAX], [0, 0], color=GREY, lw=5, solid_capstyle="round", alpha=0.45)
    ax.plot([0, k], [0, 0], color=KW, lw=5, solid_capstyle="round")
    ax.plot([k], [0], "o", color=KW, ms=15, mec="white", mew=2)
    ax.text(k, 0.75, f"K = {k}", ha="center", va="bottom", color=INK, fontsize=16, fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")


frames = [frame(k) for k in KS]
frames[-1].save(HERE / "kmeans_fill.png")
rgb = [Image.alpha_composite(Image.new("RGBA", f.size, BG + (255,)), f).convert("RGB") for f in frames]
pal = rgb[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
gif = []
for f, r in zip(frames, rgb):
    q = r.quantize(palette=pal, dither=Image.Dither.NONE)
    q.paste(255, mask=f.getchannel("A").point(lambda a: 255 if a < 40 else 0))
    gif.append(q)
durations = [1400] + [700] * (len(gif) - 2) + [2600]
gif[0].save(HERE / "kmeans_fill.gif", save_all=True, append_images=gif[1:], duration=durations,
            loop=0, transparency=255, disposal=2, optimize=True)
print(f"wrote kmeans_fill.gif ({(HERE / 'kmeans_fill.gif').stat().st_size / 1e6:.1f} MB)")
