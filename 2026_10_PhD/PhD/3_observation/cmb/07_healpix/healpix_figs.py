#!/usr/bin/env python3
# ENV: shared
"""
Figures for the HEALPix slide.

  healpix_logo.svg      the HEALPix emblem, redrawn in vector with healpy: a sphere
                        in orthographic view, the nside = 2 pixel grid, one base
                        pixel shaded dark and its neighbour light, pixel centres
                        as dots (after the favicon of healpix.sourceforge.io)
  healpix_nside{4,16,32,64}.png
                        the simulated sky of the E/B and two-point slides (CAMB,
                        seed 11) degraded to each nside, Mollweide, pixel edges
                        drawn (up to nside 32), transparent outside the sky
  nasa_logo.svg         Wikimedia Commons "File:NASA logo.svg"
  detector_ccd.jpg      Wikimedia Commons, Orthogonal Transfer Array CCD wafer (NOAO)
  detector_bolometer.jpg  Wikimedia Commons PIA17993, BICEP2/Keck bolometer tiles
                        (NASA/JPL-Caltech), cropped to the focal plane
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import cached_fetch, commons_url, skip_if_built

NSIDES = (4, 16, 32, 64)
OUTS = ("healpix_logo.svg", "nasa_logo.svg", "detector_ccd.jpg", "detector_bolometer.jpg",
        *(f"healpix_nside{n}.png" for n in NSIDES))
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io

CACHE = HERE / ".cache"
INK = "#2E2E2E"

# ---------------------------------------------------------------- downloads
(HERE / "nasa_logo.svg").write_bytes(cached_fetch(CACHE, "nasa_logo", commons_url("File:NASA logo.svg")))


def fetch_photo(key, title, out, box=None):
    im = Image.open(io.BytesIO(cached_fetch(CACHE, key, commons_url(title)))).convert("RGB")
    if box:   # fractional crop (left, top, right, bottom)
        w, h = im.size
        im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    im.thumbnail((900, 900), Image.LANCZOS)
    im.save(HERE / out, quality=88)


fetch_photo("ccd_wafer", "File:Orthogonal Transfer Array Charge Coupled Device (noao-04892).jpg",
            "detector_ccd.jpg", (0.08, 0.02, 0.92, 0.98))
fetch_photo("bicep_tiles", "File:PIA17993-DetectorsForInfantUniverseStudies-20140317.jpg",
            "detector_bolometer.jpg", (0.02, 0.55, 0.72, 1.0))
print("wrote nasa_logo.svg, detector_ccd.jpg, detector_bolometer.jpg")


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

# ---------------------------------------------------------------- nside 16 sky
import camb

pars = camb.set_params(H0=67.4, ombh2=0.0224, omch2=0.120, As=2.1e-9, ns=0.965,
                       tau=0.054, lmax=768)
cls = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"]
np.random.seed(11)
t256, _, _ = hp.synfast([cls[:, 0], cls[:, 1], cls[:, 2], cls[:, 3]], 256, new=True)
m16 = hp.ud_grade(t256, 16)

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
    if ns <= 32:   # beyond that the edges would grey the map out
        for pix in range(hp.nside2npix(ns)):
            b = hp.boundaries(ns, pix, step=4)
            th, ph = hp.vec2ang(b.T)
            x, y = proj.ang2xy(th, ph)
            x, y = np.append(x, x[0]), np.append(y, y[0])
            if np.max(np.abs(np.diff(x))) > 1.0:      # pixel cut by the map edge
                continue
            ax.plot(x, y, color="#ffffff", lw={4: 1.2, 16: 0.35, 32: 0.18}[ns], alpha=0.9)
    ax.plot(2 * np.cos(t), np.sin(t), color="#8A8A96", lw=1.2)
    ax.set_xlim(-2.02, 2.02)
    ax.set_ylim(-1.02, 1.02)
    ax.axis("off")
    fig.savefig(HERE / f"healpix_nside{ns}.png", transparent=True)
    plt.close(fig)
    print(f"wrote healpix_nside{ns}.png")
