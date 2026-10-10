#!/usr/bin/env python3
# ENV: shared
"""
Data for the two-point animation (two_point_render.py).

Map (printed at run time as well):
  Planck 2018 (PR3) SMICA CMB map, COM_CMB_IQU-smica_2048_R3.00_full.fits,
  Stokes I, K_CMB -> uK; degraded from Nside 2048 to Nside 256 (ud_grade), then
  smoothed with a 1 deg FWHM Gaussian beam; mean removed. No mask: SMICA is a
  full-sky map, inpainted along the Galactic plane.

The statistic is the two-point correlation function, estimated directly:

  C(theta) = < dT(n1) dT(n2) >  over all pairs at separation theta.

Each bar of the animation is the mean product dT1 dT2 of the pairs in its
separation bin (no fitted amplitude). The curve is the same map's C(theta) from
its own power spectrum: C_l = healpy.anafast(map), then
C(theta) = sum_l (2l+1)/4pi C_l P_l(cos theta). Pairs are drawn with separations
uniform in theta, so a bar estimates the bin average of C(theta); the script
checks that every bar agrees with the bin-averaged curve within 4 sigma of the
pair sampling scatter, and stops otherwise.

Binning: theta from 0 to 90 deg in 5 deg bins centred on 0, 5, ..., 90 deg (the two
end bins are half bins). The range shows the small-angle
peak, the zero crossing (~30 deg for this map) and the negative lobe (minimum
near 50 deg, back to zero near 91 deg).

  two_point_sky.png      the smoothed temperature, Mollweide, cropped exactly to
                         the ellipse's bounding box (the slide lays it under the
                         transparent GIF), transparent outside the sky
  .cache/two_point.npz   drawn pairs (anchors, great-circle arcs in Mollweide
                         x, y, separations, products), the pairs behind them,
                         the bars' final values and scatter, the curve, the
                         y-range covering every frame of the animation, and the
                         three rings of the opening (RING_THETA around RING_ANCHOR:
                         Mollweide x, y of the ring and of a spoke from the anchor
                         to it, the sign of dT1 dT2 along the ring)
  ../../../../CLAUDE/two_point_check.png
                         final bars and curve overlaid, for checking
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import PLANCK_SMICA, PLANCK_SMICA_URL, cached_file, skip_if_built

OUT_PNG = "two_point_sky.png"
OUT_NPZ = HERE / ".cache" / "two_point.npz"
CHECK_PNG = ROOT.parent / "CLAUDE" / "two_point_check.png"
if OUT_NPZ.exists():
    skip_if_built(HERE, OUT_PNG)

import healpy as hp
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.special import eval_legendre

NSIDE, FWHM_DEG = 256, 1.0
THETA_MAX, BIN_DEG = 90.0, 5.0
# 5 deg bins centred on multiples of 5 deg (the first, 0 - 2.5, and the last, 87.5 - 90, are half
# bins), so the rings of the opening sit exactly at bin centres: 19 bins
BINS = np.concatenate([[0.0], np.arange(BIN_DEG / 2, THETA_MAX, BIN_DEG), [THETA_MAX]])
NB = len(BINS) - 1
N_ANCHORS, N_PARTNERS, ARC_PTS = 8, 120, 24
N_HIDDEN = 450_000
HIDDEN_POWER = 1.0      # pairs behind the drawn ones: M * (n / N)**HIDDEN_POWER after n drawn lines
MIN_COUNT = 1000        # a bar is shown once its bin holds this many pairs
XS = 1600

# ---------------------------------------------------------------- the map
smica = cached_file(HERE.parent / ".cache", PLANCK_SMICA, PLANCK_SMICA_URL)
print(f"map: Planck 2018 SMICA ({PLANCK_SMICA}), Stokes I, Nside 2048 -> {NSIDE}, "
      f"{FWHM_DEG:g} deg FWHM smoothing, mean removed, no mask")
t = 1e6 * hp.ud_grade(hp.read_map(smica, field=0), NSIDE)   # uK_CMB
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

# ---------------------------------------------------------------- the curve
cl = hp.anafast(t, lmax=3 * NSIDE - 1)
ell = np.arange(len(cl))
theta = np.linspace(0.0, THETA_MAX, 901)
exact = np.array([np.sum((2 * ell + 1) / (4 * np.pi) * cl * eval_legendre(ell, xi))
                  for xi in np.cos(np.deg2rad(theta))])
fine = np.linspace(0.0, THETA_MAX, 9001)
exact_fine = np.interp(fine, theta, exact)
bin_avg = np.array([exact_fine[(fine >= BINS[i]) & (fine <= BINS[i + 1])].mean()
                    for i in range(NB)])


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
    th, ph = hp.vec2ang(v.reshape(-1, 3))
    x, y = proj.ang2xy(th, ph)
    return np.column_stack([x, y]).reshape(v.shape[:-1] + (2,))


rng = np.random.default_rng(3)
anchor_ll = [(-120, 30), (-50, -25), (20, 35), (80, -10), (140, 25), (-10, -5),
             (-140, -30), (110, -40)]
anchors = unit(np.deg2rad(90 - np.array([b for _, b in anchor_ll])),
               np.deg2rad(np.array([l for l, _ in anchor_ll])))
sep = np.deg2rad(rng.uniform(0.0, THETA_MAX, (N_ANCHORS, N_PARTNERS)))
az = rng.uniform(0, 2 * np.pi, (N_ANCHORS, N_PARTNERS))
anc = np.repeat(anchors[:, None, :], N_PARTNERS, 1)
partners = offset(anc, sep, az)
prod = value(anchors)[:, None] * value(partners)

# great-circle arcs on the Mollweide map, broken (NaN) where they cross the map edge
s = np.linspace(0.0, 1.0, ARC_PTS)
arc_v = offset(np.repeat(anc[:, :, None, :], ARC_PTS, 2), sep[..., None] * s,
               np.repeat(az[..., None], ARC_PTS, -1))
arcs = mollxy(arc_v)
jump = np.abs(np.diff(arcs[..., 0], axis=-1)) > 1.0
for a, p, k in zip(*np.nonzero(jump)):
    arcs[a, p, k + 1] = np.nan          # a NaN point splits the polyline at the edge

# the pairs behind the drawn ones: random points over the sphere, separations uniform in theta
h1 = unit(np.arccos(rng.uniform(-1, 1, N_HIDDEN)), rng.uniform(0, 2 * np.pi, N_HIDDEN))
hs = np.deg2rad(rng.uniform(0.0, THETA_MAX, N_HIDDEN))
h2 = offset(h1, hs, rng.uniform(0, 2 * np.pi, N_HIDDEN))
hidden_prod = value(h1) * value(h2)

# ---------------------------------------------------------------- the opening: rings at separation theta
# One anchor n1 and, at three separations (bin centres), the small circle of every n2 at angle theta
# from it, with a spoke (the great-circle arc n1 -> n2 at azimuth RING_SPOKE_AZ) to show the angle.
RING_THETA = np.array([5.0, 10.0, 45.0])        # bin centres, round angles
RING_PTS, SPOKE_PTS, RING_SPOKE_AZ = 361, 40, np.deg2rad(200.0)
ring_az = np.linspace(0, 2 * np.pi, RING_PTS)


def ring_share(v, theta_deg):
    """Fraction of the ring at theta_deg around v with dT1 dT2 > 0."""
    pts = offset(np.repeat(v[None], RING_PTS, 0), np.full(RING_PTS, np.deg2rad(theta_deg)), ring_az)
    return np.mean(value(v) * value(pts) > 0)


# the anchor, near the map centre, whose rings show the sign change of C(theta) best: mostly the
# same sign as the anchor on the smallest ring, mostly the opposite sign on the largest
cands = [(lon, lat) for lon in range(-40, 41, 4) for lat in range(-24, 25, 4)]
score = [ring_share(unit(np.deg2rad(90 - b), np.deg2rad(l)), RING_THETA[0])
         - ring_share(unit(np.deg2rad(90 - b), np.deg2rad(l)), RING_THETA[-1]) for l, b in cands]
RING_ANCHOR = cands[int(np.argmax(score))]
r_anchor = unit(np.deg2rad(90 - RING_ANCHOR[1]), np.deg2rad(RING_ANCHOR[0]))
print(f"ring anchor: lon {RING_ANCHOR[0]}, lat {RING_ANCHOR[1]}")
ring_v = offset(np.repeat(r_anchor[None, None], len(RING_THETA), 0).repeat(RING_PTS, 1),
                np.deg2rad(RING_THETA)[:, None].repeat(RING_PTS, 1), ring_az[None].repeat(3, 0))
ring_sign = np.sign(value(r_anchor) * value(ring_v))
ring_xy = mollxy(ring_v)
spoke_v = offset(np.repeat(r_anchor[None, None], len(RING_THETA), 0).repeat(SPOKE_PTS, 1),
                 np.deg2rad(RING_THETA)[:, None] * np.linspace(0, 1, SPOKE_PTS)[None],
                 np.full((len(RING_THETA), SPOKE_PTS), RING_SPOKE_AZ))
spoke_xy = mollxy(spoke_v)
for xy in (ring_xy, spoke_xy):
    jump = np.abs(np.diff(xy[..., 0], axis=-1)) > 1.0
    for a, k in zip(*np.nonzero(jump)):
        xy[a, k + 1] = np.nan
print("rings: " + ", ".join(f"{th:g} deg ({100 * np.mean(sg > 0):.0f} % positive)"
                            for th, sg in zip(RING_THETA, ring_sign)))

# ---------------------------------------------------------------- bars: mean product per bin
all_sep = np.concatenate([np.rad2deg(sep).ravel(), np.rad2deg(hs)])
all_prod = np.concatenate([prod.ravel(), hidden_prod])
b = np.clip(np.digitize(all_sep, BINS) - 1, 0, NB - 1)
count = np.bincount(b, minlength=NB)
final = np.bincount(b, weights=all_prod, minlength=NB) / count
sq = np.bincount(b, weights=all_prod**2, minlength=NB) / count
sigma = np.sqrt((sq - final**2) / count)

pull = (final - bin_avg) / sigma
print(f"{'bin':>9} {'bars':>8} {'curve':>8} {'sigma':>6} {'pull':>6}")
for i in range(NB):
    print(f"{BINS[i]:4.0f}-{BINS[i + 1]:<4.0f} {final[i]:8.1f} {bin_avg[i]:8.1f} "
          f"{sigma[i]:6.1f} {pull[i]:6.2f}")
if np.any(np.abs(pull) > 4):
    sys.exit(f"bars and curve disagree beyond 4 sigma in bins {np.nonzero(np.abs(pull) > 4)[0]}: "
             "not rescaling; check the map, the binning or the pair sampling")


def zero_crossing(x, y):
    i = np.nonzero(np.diff(np.sign(y)) != 0)[0][0]
    return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])


print(f"zero crossing of C(theta): {zero_crossing(theta, exact):.1f} deg; "
      f"minimum {exact.min():.0f} uK^2 at {theta[exact.argmin()]:.0f} deg")

# ---------------------------------------------------------------- y-range over every frame
# The renderer shows, after n drawn lines, the running mean of the first n drawn
# pairs plus the first M (n / N)**HIDDEN_POWER hidden ones; bars appear at MIN_COUNT.
d_sep, d_prod = np.rad2deg(sep).ravel(), prod.ravel()
d_bin = np.clip(np.digitize(d_sep, BINS) - 1, 0, NB - 1)
h_bin = np.clip(np.digitize(np.rad2deg(hs), BINS) - 1, 0, NB - 1)
lo, hi = min(exact.min(), final.min()), max(exact.max(), final.max())
N = len(d_prod)
for n in range(1, N + 1):
    m = int(N_HIDDEN * (n / N) ** HIDDEN_POWER)
    c = np.bincount(d_bin[:n], minlength=NB) + np.bincount(h_bin[:m], minlength=NB)
    w = (np.bincount(d_bin[:n], weights=d_prod[:n], minlength=NB)
         + np.bincount(h_bin[:m], weights=hidden_prod[:m], minlength=NB))
    shown = c >= MIN_COUNT
    if shown.any():
        v = w[shown] / c[shown]
        lo, hi = min(lo, v.min()), max(hi, v.max())
lo, hi = min(lo, 0.0), max(hi, 0.0)
pad = 0.1 * (hi - lo)
ymin, ymax = lo - pad, hi + pad
print(f"y-range over all frames, 10% margin: {ymin:.0f} to {ymax:.0f} uK^2")

OUT_NPZ.parent.mkdir(exist_ok=True)
np.savez(OUT_NPZ, bins=BINS, anchors_xy=mollxy(anchors), arcs=arcs,
         sep_deg=np.rad2deg(sep), prod=prod, hidden_sep_deg=np.rad2deg(hs),
         hidden_prod=hidden_prod, final=final, sigma=sigma, bin_avg=bin_avg,
         theta=theta, exact=exact, ymin=ymin, ymax=ymax,
         hidden_power=HIDDEN_POWER, min_count=MIN_COUNT, fwhm_deg=FWHM_DEG,
         ring_theta=RING_THETA, ring_xy=ring_xy, ring_sign=ring_sign, spoke_xy=spoke_xy,
         ring_anchor_xy=mollxy(r_anchor[None])[0])
print(f"wrote {OUT_NPZ.name}")

# ---------------------------------------------------------------- check figure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(0.5 * (BINS[1:] + BINS[:-1]), final, width=0.9 * np.diff(BINS), color="#521463", alpha=0.85,
       yerr=sigma, ecolor="#2E2E2E", capsize=0, label="mean product per bin (bars)")
ax.plot(theta, exact, color="#C2560A", lw=2.5, label="from the map's C$_\\ell$ (curve)")
ax.axhline(0, color="#8A8A96", lw=0.8)
ax.set_xlim(0, THETA_MAX)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("separation θ [deg]")
ax.set_ylabel("C(θ) = ⟨ΔT ΔT⟩ [µK²]")
ax.set_title(f"Planck 2018 SMICA, Nside {NSIDE}, {FWHM_DEG:g}° FWHM", fontsize=10)
ax.legend(frameon=False)
fig.tight_layout()
CHECK_PNG.parent.mkdir(exist_ok=True)
fig.savefig(CHECK_PNG, dpi=150)
print(f"wrote {CHECK_PNG}")
