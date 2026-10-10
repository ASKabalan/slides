#!/usr/bin/env python3
# ENV: jax-fli
"""
Distant galaxies lensed by the cosmic web, after Mandelbaum (2018, ARA&A, arXiv:1710.03235),
Fig. 1 right (image: CFHT; simulation S. Colombi, IAP), for the closing overlay of the
"Convergence and shear" slide.

The matter: a 12 x 12 deg gnomonic patch of the nearest equal-volume lightcone shell (shell 1,
z = 0.36) of the 3072^3 run of jax-fli experiment 05e, at its native nside 2048, centred on the
densest region of the shell; log(1 + delta) in a hot colour map on black, as the original.
The galaxies: cyan ellipses at random positions, each stretched along the shear of the same
patch, gamma from flat-sky Kaiser-Squires of the smoothed overdensity (kappa taken
proportional to delta). The ellipticities are exaggerated to be seen (|e| up to 0.6, set by
|gamma| / its 95th percentile): the pattern, tangential around the clumps, is the point.

Output (this directory): lensed_galaxies.png (900 x 900)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

OUT = "lensed_galaxies.png"
skip_if_built(HERE, OUT)

CACHE = HERE.parent / ".cache"
RUN = Path("/home/wassim/Projects/NBody/jax-fli-experiments/05-spacing-n-stepping/05e-mesh/"
           "density/exp5e_m3072")
SHELL, XSIZE, RESO = 1, 900, 0.8            # shell index, patch pixels, arcmin per pixel
N_GAL, SEED = 120, 7


def patch():
    npz = CACHE / "lensed_patch.npz"
    if npz.exists():
        d = np.load(npz)
        return d["delta"], float(d["z"])
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax_fli as jfli
    from datasets import load_dataset
    from jax_fli.io import Catalog

    f = RUN / f"shell_{SHELL:04d}.parquet"
    shell = Catalog.from_dataset(load_dataset("parquet", data_files=str(f), split="train")).field[0]
    shell = shell.to(jfli.DensityUnit.OVERDENSITY)
    m = np.asarray(shell.array, dtype=np.float64).ravel()
    z = float(np.asarray(shell.z_sources).ravel()[0])
    nside = hp.npix2nside(m.size)
    # centre: the densest 2-degree region, away from the poles (gnomonic distortion is mild anyway)
    smooth = hp.smoothing(hp.ud_grade(m, 64), fwhm=np.radians(2.0))
    theta, phi = hp.pix2ang(64, np.arange(smooth.size))
    smooth[np.abs(np.pi / 2 - theta) > np.radians(60)] = -np.inf
    th, ph = hp.pix2ang(64, int(np.argmax(smooth)))
    rot = (np.degrees(ph), 90 - np.degrees(th))
    delta = hp.gnomview(m, rot=rot, xsize=XSIZE, reso=RESO, return_projected_map=True,
                        no_plot=True).filled(0.0)
    print(f"shell {SHELL}: nside {nside}, z = {z:.3f}, patch centred on lon {rot[0]:.1f}, lat {rot[1]:.1f}")
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, delta=delta, z=z)
    return delta, z


delta, z = patch()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------- shear (flat-sky Kaiser-Squires)
kappa = gaussian_filter(delta, 10.0)             # rows: y (up, origin="lower"), columns: x
ky, kx = np.meshgrid(np.fft.fftfreq(XSIZE), np.fft.fftfreq(XSIZE), indexing="ij")
k2 = kx ** 2 + ky ** 2
k2[0, 0] = 1.0
kh = np.fft.fft2(kappa)
g1 = np.real(np.fft.ifft2((kx ** 2 - ky ** 2) / k2 * kh))
g2 = np.real(np.fft.ifft2(2 * kx * ky / k2 * kh))

# ---------------------------------------------------------------- galaxies
rng = np.random.default_rng(SEED)
pts = []
while len(pts) < N_GAL:                           # random, but not overlapping
    p = rng.uniform(30, XSIZE - 30, 2)
    if all(np.hypot(*(p - q)) > 55 for q in pts):
        pts.append(p)
pts = np.array(pts)
ix, iy = pts[:, 0].astype(int), pts[:, 1].astype(int)
amp = np.hypot(g1, g2)
e = np.clip(0.6 * amp[iy, ix] / np.percentile(amp, 95), 0.08, 0.6)
ang = 0.5 * np.degrees(np.arctan2(g2[iy, ix], g1[iy, ix]))   # direction of the stretch
SIZE = 11.0

# ---------------------------------------------------------------- figure
img = np.log10(np.clip(1 + gaussian_filter(delta, 0.8), 1e-2, None))
lo, hi = np.percentile(img, [40, 99.8])        # voids black, as the original
fig = plt.figure(figsize=(9, 9), dpi=100)
ax = fig.add_axes((0, 0, 1, 1))
ax.imshow(img, origin="lower", cmap="afmhot", vmin=lo, vmax=hi, interpolation="bilinear")
for (x, y), ei, a in zip(pts, e, ang):
    ax.add_patch(Ellipse((x, y), 2 * SIZE * (1 + ei), 2 * SIZE * (1 - ei), angle=a,
                         facecolor="#2ee6f2", edgecolor="none"))
ax.set_xlim(0, XSIZE)
ax.set_ylim(0, XSIZE)
ax.axis("off")
fig.savefig(HERE / OUT, facecolor="black")
print(f"wrote {OUT} (z = {z:.2f}, {N_GAL} galaxies)")
