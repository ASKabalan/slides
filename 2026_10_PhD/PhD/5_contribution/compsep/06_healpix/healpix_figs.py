#!/usr/bin/env python3
# ENV: shared
"""
Figures for the HEALPix slide.

  healpix_logo.svg      the HEALPix emblem, redrawn in vector with healpy: a sphere
                        in orthographic view, the nside = 2 pixel grid, one base
                        pixel shaded dark and its neighbour light, pixel centres
                        as dots (after the favicon of healpix.sourceforge.io)
  healpix_nside64.png   the Planck 2018 SMICA temperature (Stokes I) map degraded
                        to nside 64, Mollweide, transparent outside the sky
  nasa_logo.svg         Wikimedia Commons "File:NASA logo.svg"
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import PLANCK_SMICA, PLANCK_SMICA_URL, cached_fetch, cached_file, commons_url, skip_if_built

NSIDES = (64,)
OUTS = ("healpix_logo.svg", "nasa_logo.svg", *(f"healpix_nside{n}.png" for n in NSIDES))
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CACHE = HERE / ".cache"
INK = "#2E2E2E"

# ---------------------------------------------------------------- downloads
(HERE / "nasa_logo.svg").write_bytes(cached_fetch(CACHE, "nasa_logo", commons_url("File:NASA logo.svg")))
print("wrote nasa_logo.svg")


# ---------------------------------------------------------------- the emblem
def ortho(vec, rot):
    """Unit vectors -> (x, y, visible) for a view from direction rot (lon, lat deg)."""
    lon, lat = np.deg2rad(rot)
    v = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    e1 = np.array([-np.sin(lon), np.cos(lon), 0.0])
    e2 = np.cross(v, e1)
    return vec @ e1, vec @ e2, vec @ v > 0


NS, ROT = 2, (30.0, 25.0)
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")
ax.axis("off")
# shade the pixel facing the viewer (dark) and two of its neighbours (light)
_lon, _lat = ROT
_centre = hp.ang2pix(NS, _lon, _lat - 12, lonlat=True)
_nb = [p for p in hp.get_all_neighbours(NS, _centre) if p >= 0]
shade = {_centre: "#5c5c5c", _nb[0]: "#bdbdbd", _nb[2]: "#bdbdbd", _nb[1]: "#bdbdbd"}
for pix in range(hp.nside2npix(NS)):
    b = hp.boundaries(NS, pix, step=16).T
    x, y, vis = ortho(b, ROT)
    if vis.all() and pix in shade:
        ax.fill(x, y, color=shade[pix], lw=0, zorder=1)
    if vis.any():
        segs = np.split(np.arange(len(x)), np.where(~vis)[0])
        for s in segs:
            s = s[vis[s]]
            if len(s) > 1:
                ax.plot(x[s], y[s], color=INK, lw=1.2, zorder=2)
    c = np.array(hp.pix2vec(NS, pix))
    cx, cy, cv = ortho(c[None, :], ROT)
    if cv[0]:
        ax.plot(cx, cy, "o", ms=3.2, color=INK, zorder=3)
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), color=INK, lw=2.2, zorder=4)
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
fig.savefig(HERE / "healpix_logo.svg", transparent=True, bbox_inches="tight")
plt.close(fig)
print("wrote healpix_logo.svg")

# ---------------------------------------------------------------- the Planck sky
smica = cached_file((ROOT / "3_observation/cmb") / ".cache", PLANCK_SMICA, PLANCK_SMICA_URL)
t256 = hp.ud_grade(hp.read_map(smica, field=0), 256)    # Stokes I, K_CMB

XS = 2000
proj = hp.projector.MollweideProj(xsize=XS)
cmap = LinearSegmentedColormap.from_list("deck", ["#3B6FB6", "#ffffff", "#C2560A"])
t = np.linspace(0, 2 * np.pi, 400)
for ns in NSIDES:
    m = hp.ud_grade(t256, ns)
    img = proj.projmap(m, lambda x, y, z: hp.vec2pix(ns, x, y, z))
    inside = np.isfinite(img) & (img > -1e30)
    lim = np.percentile(np.abs(img[inside]), 99)
    rgba = cmap(np.clip(0.5 + 0.5 * np.where(inside, img, 0) / lim, 0, 1))
    rgba[~inside, 3] = 0.0

    fig = plt.figure(figsize=(XS / 200, XS / 400), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgba, origin="lower", extent=(-2, 2, -1, 1), interpolation="nearest")
    ax.plot(2 * np.cos(t), np.sin(t), color="#8A8A96", lw=1.2)
    ax.set_xlim(-2.02, 2.02)
    ax.set_ylim(-1.02, 1.02)
    ax.axis("off")
    fig.savefig(HERE / f"healpix_nside{ns}.png", transparent=True)
    plt.close(fig)
    print(f"wrote healpix_nside{ns}.png")
