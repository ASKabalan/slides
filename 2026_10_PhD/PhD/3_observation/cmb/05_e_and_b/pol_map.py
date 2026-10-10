#!/usr/bin/env python3
# ENV: shared
"""
The Planck 2018 SMICA CMB temperature map with its polarisation drawn as
headless sticks, Mollweide, transparent outside the sky, for the "E and B
modes" slide.

T, Q, U are the SMICA I, Q, U maps degraded to nside 256. Temperature is smoothed to 1 degree and shown in the deck's
blue-white-orange map; the sticks use Q, U smoothed to 5 degrees so that the
pattern reads at slide size: angle 1/2 atan2(U, Q), length proportional to P.

Output (this directory): pol_map.png
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import PLANCK_SMICA, PLANCK_SMICA_URL, cached_file, skip_if_built

OUT = "pol_map.png"
skip_if_built(HERE, OUT)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NSIDE = 256
XS, YS = 1600, 800

smica = cached_file(HERE.parent / ".cache", PLANCK_SMICA, PLANCK_SMICA_URL)
t, q, u = 1e6 * np.array(hp.ud_grade(hp.read_map(smica, field=(0, 1, 2)), NSIDE))   # uK_CMB

t1 = hp.smoothing(t, fwhm=np.deg2rad(1.0))
_, q5, u5 = hp.smoothing([t, q, u], fwhm=np.deg2rad(5.0))

proj = hp.projector.MollweideProj(xsize=XS)
tmap = proj.projmap(t1, lambda x, y, z: hp.vec2pix(NSIDE, x, y, z))
inside = np.isfinite(tmap) & (tmap > -1e30)

cmap = LinearSegmentedColormap.from_list("deck", ["#3B6FB6", "#ffffff", "#C2560A"])
lim = np.percentile(np.abs(tmap[inside]), 99)
rgba = cmap(np.clip(0.5 + 0.5 * np.where(inside, tmap, 0) / lim, 0, 1))
rgba[~inside, 3] = 0.0

fig = plt.figure(figsize=(XS / 200, XS / 400), dpi=200)
ax = fig.add_axes((0, 0, 1, 1))
ax.imshow(rgba, origin="lower", extent=(-2, 2, -1, 1), interpolation="bilinear")

# sticks on a regular grid inside the ellipse
step = 0.075
xs, ys = np.meshgrid(np.arange(-2 + step / 2, 2, step), np.arange(-1 + step / 2, 1, step))
xs, ys = xs.ravel(), ys.ravel()
keep = (xs / 2) ** 2 + ys**2 < 0.97
xs, ys = xs[keep], ys[keep]
theta, phi = proj.xy2ang(xs, ys, lonlat=False)
pix = hp.ang2pix(NSIDE, theta, phi)
qq, uu = q5[pix], u5[pix]
p = np.hypot(qq, uu)
ang = 0.5 * np.arctan2(uu, qq)
half = 0.5 * step * 0.9 * p / np.percentile(p, 95)
# healpy's Q/U are in the local (theta, phi) frame; on the map, north is +y and
# the phi direction is -x, so the stick direction in map coordinates is:
dx, dy = -np.sin(ang) * half, np.cos(ang) * half
ax.plot(np.vstack([xs - dx, xs + dx]), np.vstack([ys - dy, ys + dy]), color="#2E2E2E",
        lw=1.1, solid_capstyle="round")
t_edge = np.linspace(0, 2 * np.pi, 400)
ax.plot(2 * np.cos(t_edge), np.sin(t_edge), color="#8A8A96", lw=1.2)
ax.set_xlim(-2.02, 2.02)
ax.set_ylim(-1.02, 1.02)
ax.axis("off")
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
