#!/usr/bin/env python3
# ENV: shared
"""
Data for the two-point animation (two_point_render.py): the same simulated sky as
pol_map.py (same CAMB spectra and seed), temperature only.

  two_point_sky.png      the 1-degree-smoothed temperature, Mollweide, cropped
                         exactly to the ellipse's bounding box (the slide lays it
                         under the transparent GIF), transparent outside the sky
  .cache/two_point.npz   anchors and partners (Mollweide x, y), their angular
                         separations and products dT1 dT2; the per-bin mean of
                         dT1 dT2 over 2e5 random pairs of this sky; and this sky's
                         full C(theta) = sum_l (2l+1)/4pi C_l P_l(cos theta), with
                         C_l measured from the map (so the bars land on it; one sky
                         differs from the ensemble curve by cosmic variance)

The binned statistic is the two-point correlation C(theta) = <dT(n1) dT(n2)>
at separation theta: pair counts alone only measure the geometry.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built

OUT_PNG = "two_point_sky.png"
OUT_NPZ = HERE / ".cache" / "two_point.npz"
if OUT_NPZ.exists():
    skip_if_built(HERE, OUT_PNG)

import camb
import healpy as hp
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.special import eval_legendre

NSIDE, SEED, FWHM_DEG = 256, 11, 1.0          # SEED as in pol_map.py
BINS = np.linspace(0.0, 36.0, 13)             # 12 bins of 3 degrees
N_ANCHORS, N_PARTNERS = 8, 120
N_RANDOM = 200_000
XS = 1600

pars = camb.set_params(H0=67.4, ombh2=0.0224, omch2=0.120, As=2.1e-9, ns=0.965,
                       tau=0.054, lmax=3 * NSIDE)
cls = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"]
np.random.seed(SEED)
t, _, _ = hp.synfast([cls[:, 0], cls[:, 1], cls[:, 2], cls[:, 3]], NSIDE, new=True)
t = hp.smoothing(t, fwhm=np.deg2rad(FWHM_DEG))
t -= t.mean()

# ---------------------------------------------------------------- sky image
proj = hp.projector.MollweideProj(xsize=XS)
img = proj.projmap(t, lambda x, y, z: hp.vec2pix(NSIDE, x, y, z))
inside = np.isfinite(img) & (img > -1e30)
cmap = LinearSegmentedColormap.from_list("deck", ["#3B6FB6", "#ffffff", "#C2560A"])
lim = np.percentile(np.abs(img[inside]), 99)
rgba = cmap(np.clip(0.5 + 0.5 * np.where(inside, img, 0) / lim, 0, 1))
rgba[~inside, 3] = 0.0
ys, xs = np.nonzero(inside)
crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1][::-1]   # north up
Image.fromarray((crop * 255).astype(np.uint8), "RGBA").save(HERE / OUT_PNG)
print(f"wrote {OUT_PNG} {crop.shape[1]}x{crop.shape[0]}")


# ---------------------------------------------------------------- pairs
def unit(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)


def offset(v, sep, az):
    """Directions at angular distance sep (rad) from unit vectors v, azimuth az."""
    ref = np.where(np.abs(v[..., 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    e1 = np.cross(v, ref)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(v, e1)
    return (np.cos(sep)[..., None] * v
            + np.sin(sep)[..., None] * (np.cos(az)[..., None] * e1 + np.sin(az)[..., None] * e2))


def value(v):
    return t[hp.vec2pix(NSIDE, v[..., 0], v[..., 1], v[..., 2])]


def mollxy(v):
    theta, phi = hp.vec2ang(v)
    x, y = proj.ang2xy(theta, phi)
    return np.column_stack([x, y])


rng = np.random.default_rng(3)
# anchors spread across the map, away from the rim
anchor_ll = [(-120, 30), (-50, -25), (20, 35), (80, -10), (140, 25), (-10, -5),
             (-140, -30), (110, -40)]
anchors = unit(np.deg2rad(90 - np.array([b for _, b in anchor_ll])),
               np.deg2rad(np.array([l for l, _ in anchor_ll])))
# separations uniform in theta, so every bin gets the same number of pairs
sep = np.deg2rad(rng.uniform(0.3, BINS[-1] - 0.3, (N_ANCHORS, N_PARTNERS)))
az = rng.uniform(0, 2 * np.pi, (N_ANCHORS, N_PARTNERS))
partners = offset(np.repeat(anchors[:, None, :], N_PARTNERS, 1), sep, az)
prod = value(anchors)[:, None] * value(partners)

# converged per-bin means: random first points, separations uniform within the range
v1 = unit(np.arccos(rng.uniform(-1, 1, N_RANDOM)), rng.uniform(0, 2 * np.pi, N_RANDOM))
s = np.deg2rad(rng.uniform(0.25, BINS[-1], N_RANDOM))
v2 = offset(v1, s, rng.uniform(0, 2 * np.pi, N_RANDOM))
p = value(v1) * value(v2)
idx = np.digitize(np.rad2deg(s), BINS) - 1
converged = np.array([p[idx == i].mean() for i in range(len(BINS) - 1)])

# this sky's full two-point function, from its own C_l
cl_sky = hp.alm2cl(hp.map2alm(t, lmax=3 * NSIDE - 1))
ell = np.arange(len(cl_sky))
th = np.linspace(0.0, BINS[-1], 300)
x = np.cos(np.deg2rad(th))
exact = np.array([np.sum((2 * ell + 1) / (4 * np.pi) * cl_sky * eval_legendre(ell, xi))
                  for xi in x])

# the pairs behind the drawn ones: random points over the sphere, separations
# uniform in theta; the bars count these as the lines are drawn
N_HIDDEN = 15_000
h1 = unit(np.arccos(rng.uniform(-1, 1, N_HIDDEN)), rng.uniform(0, 2 * np.pi, N_HIDDEN))
hs = np.deg2rad(rng.uniform(0.3, BINS[-1] - 0.3, N_HIDDEN))
h2 = offset(h1, hs, rng.uniform(0, 2 * np.pi, N_HIDDEN))
hidden_prod = value(h1) * value(h2)

OUT_NPZ.parent.mkdir(exist_ok=True)
np.savez(OUT_NPZ, bins=BINS, anchors_xy=mollxy(anchors),
         partners_xy=mollxy(partners.reshape(-1, 3)).reshape(N_ANCHORS, N_PARTNERS, 2),
         sep_deg=np.rad2deg(sep), prod=prod,
         hidden_sep_deg=np.rad2deg(hs), hidden_prod=hidden_prod, converged=converged, theta=th, exact=exact)
print("converged per bin:", np.round(converged))
print("exact at bin centres:", np.round(np.interp(0.5 * (BINS[1:] + BINS[:-1]), th, exact)))
print(f"wrote {OUT_NPZ.name}")
