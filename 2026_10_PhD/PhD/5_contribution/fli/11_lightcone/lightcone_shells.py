#!/usr/bin/env python3
# ENV: jax-fli
"""
From 3D density cubes to concentric spherical shells around the observer, for the slide on
creating the lightcone.

Four of the twenty equal-volume lightcone shells of the 3072^3 run of jax-fli experiment 05e (5000
Mpc/h box, observer at the centre, BullFrog, nside 2048), shells 1, 5, 10 and 19 (z = 0.36, 0.61,
0.79, 1.10), each brought to nside 512 with SphericalDensity.ud_sample and drawn as an annular band
around the observer's eye, in the manner of the healpix_shells_eye diagram: the band at radius r shows
the strip of the map within +-8 deg of the equator, over a 48 deg wedge facing the eye. Radii are
schematic (evenly spaced), not to scale. Above each band, the density cube of the tomography slide
(3_observation/weaklensing/05_tomography/box_{0..3}.png, far to near) with an arrow down to it.

Output (this directory): lightcone_shells.png (transparent)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import GREY, INK, skip_if_built

OUT = "lightcone_shells.png"
skip_if_built(HERE, OUT)

CACHE = HERE.parent / ".cache"
RUN = Path("/home/wassim/Projects/NBody/jax-fli-experiments/05-spacing-n-stepping/05e-mesh/"
           "density/exp5e_m3072")
SHELLS = (19, 10, 5, 1)             # far to near, as the cubes box_0..box_3
NSIDE = 512
TOMO = HERE.parents[2] / "3_observation" / "weaklensing" / "05_tomography"


def maps():
    npz = CACHE / "lightcone_shells_512.npz"
    if npz.exists():
        d = np.load(npz)
        return d["maps"], d["z"]
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax_fli as jfli
    from datasets import load_dataset
    from jax_fli.io import Catalog

    out, z = [], []
    for i in SHELLS:
        f = RUN / f"shell_{i:04d}.parquet"
        shell = Catalog.from_dataset(load_dataset("parquet", data_files=str(f), split="train")).field[0]
        shell = shell.ud_sample(NSIDE).to(jfli.DensityUnit.OVERDENSITY)
        out.append(np.asarray(shell.array).ravel())
        z.append(float(np.asarray(shell.z_sources).ravel()[0]))
        print(f"shell {i}: z = {z[-1]:.3f}")
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, maps=np.stack(out), z=np.array(z))
    return np.stack(out), np.array(z)


M, Z = maps()

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RADII = [1.6, 1.25, 0.9, 0.55]      # far to near
THICK = 0.085
HALF = 24                           # half-opening of the wedge [deg]
SPAN = np.radians(np.linspace(180 - HALF, 180 + HALF, 600))
LAT = np.radians(np.linspace(-8, 8, 28))
EYE = "#3B6FB6"

fig = plt.figure(figsize=(10.5, 6.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_aspect("equal")
ax.axis("off")
for r, m, z, k in zip(RADII, M, Z, range(4)):
    v = np.log1p(np.clip(m, -0.99, None))
    lo, hi = np.percentile(v, [2, 99.5])
    th = np.pi / 2 - LAT[:, None] + 0 * SPAN[None, :]
    ph = SPAN[None, :] + 0 * LAT[:, None]
    strip = hp.get_interp_val(v, th, ph)
    R = r + THICK * np.linspace(-1, 1, len(LAT))[:, None]
    ax.pcolormesh(R * np.cos(ph), R * np.sin(ph), strip, cmap="magma", vmin=lo, vmax=hi,
                  shading="gouraud", zorder=2, rasterized=True)
    ro, ri = r + THICK, r - THICK
    ax.plot(np.r_[ro * np.cos(SPAN), ri * np.cos(SPAN[::-1]), ro * np.cos(SPAN[:1])],
            np.r_[ro * np.sin(SPAN), ri * np.sin(SPAN[::-1]), ro * np.sin(SPAN[:1])],
            color="#2b3342", lw=1.2, zorder=3)
    # the cube this shell is cut from, its redshift, and an arrow down to the band
    cube = np.asarray(Image.open(TOMO / f"box_{k}.png").convert("RGBA"))
    w = 0.25
    h = w * cube.shape[0] / cube.shape[1]
    x0, y0 = -r * np.cos(np.radians(HALF - 6)), 1.02     # above the band's upper end
    ax.imshow(cube, extent=(x0 - w / 2, x0 + w / 2, y0, y0 + h), zorder=4, interpolation="lanczos")
    ax.text(x0, y0 - 0.02, f"z = {z:.2f}", ha="center", va="top", fontsize=13, color=INK)
    y_hit = r * np.sin(np.radians(HALF - 6)) + 0.5 * THICK   # onto the band, near its upper end
    ax.annotate("", xy=(x0, y_hit), xytext=(x0, y0 - 0.14),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.4, mutation_scale=18))

# the observer's eye
t = np.linspace(-1, 1, 200)
ew = 0.13
lid = ew * 0.47 * np.cos(t * np.pi / 2) ** 1.5
ax.plot(ew * t, lid, color=EYE, lw=2.4, solid_capstyle="round", zorder=6)
ax.plot(ew * t, -lid, color=EYE, lw=2.4, solid_capstyle="round", zorder=6)
ax.add_patch(plt.Circle((0, 0), ew * 0.24, fc=EYE, ec="none", zorder=7))
ax.text(0, -0.13, "observer", ha="center", va="top", fontsize=13, color=INK)

ax.set_xlim(-1.8, 0.2)
ax.set_ylim(-0.74, 1.36)
fig.savefig(HERE / OUT, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {OUT}")
