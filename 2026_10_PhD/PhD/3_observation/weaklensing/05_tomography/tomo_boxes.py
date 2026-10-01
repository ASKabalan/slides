#!/usr/bin/env python3
# ENV: jax-fli
"""
Pieces of the "Tomographic reconstruction" slide, all from one small run of my
forward model plus the matching theory convergence.

  slab.png           a long 1LPT density box, 256 x 64 x 64 cells, 1600 x 400 x 400
                     Mpc/h, Planck 2018, at a = 1 (magma, transparent)
  box_{0..3}.png     the same box cut into four 64^3 sub-boxes, far to near, on one
                     colour scale
  kappa_{0..3}.png   convergence seen by an observer for source planes at the far
                     edge of each sub-box (comoving distance 1600, 1200, 800, 400
                     Mpc/h from the observer), drawn on the sphere (orthographic,
                     viridis, one colour scale). Gaussian realisations of the
                     jax_cosmo Limber C_l, built from one set of harmonic modes so
                     the four maps are the same sky, growing with source distance.

Runs on CPU in the jax-fli environment; jax_enable_x64 first.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built

OUTS = ["slab.png"] + [f"box_{i}.png" for i in range(4)] + [f"kappa_{i}.png" for i in range(4)]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MESH, BOX = (256, 64, 64), (1600.0, 400.0, 400.0)
cosmo = jc.Planck18()

# ---------------------------------------------------------------- density
ic = jfli.gaussian_initial_conditions(jax.random.PRNGKey(7), MESH, BOX, cosmo=cosmo)
dx, _ = jfli.lpt(cosmo, ic, ts=1.0, order=1)          # snapshot: displacements at a = 1
rho = dx.paint(order="cic")                            # DensityField
vol = np.asarray(rho.array).reshape(MESH)
vol = vol / vol.mean()
vmax = float(np.percentile(vol, 99.3))
vol = np.clip(vol, 0, vmax)


def render(v, out, aspect, figsize, zoom=1.15):
    fig = plt.figure(figsize=figsize, dpi=170)
    ax = fig.add_subplot(projection="3d")
    sub = rho.replace(array=jnp.asarray(v), mesh_size=v.shape,
                      box_size=tuple(b * n / m for b, n, m in zip(BOX, v.shape, MESH)))
    sub.plot(
        ax=ax, project_slices=6, labels=("", "", ""), titles="", vmin=0.0, vmax=vmax,
        colorbar=False, levels=96, elev=18, azim=-80)
    ax.set_box_aspect(aspect, zoom=zoom)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(HERE / out, transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


render(vol, "slab.png", (4, 1, 1), (9.0, 4.2), zoom=0.95)
for i in range(4):
    render(vol[64 * i:64 * (i + 1)], f"box_{i}.png", (1, 1, 1), (3.4, 3.4))

# ---------------------------------------------------------------- convergence
NSIDE, LMAX = 256, 600
h = float(cosmo.h)
chi_src = [1600.0, 1200.0, 800.0, 400.0]            # Mpc/h, far to near
a_grid = jnp.linspace(0.2, 1.0, 400)
chi_grid = np.asarray(jc.background.radial_comoving_distance(cosmo, a_grid))
z_src = [float(1 / np.interp(c, chi_grid[::-1], np.asarray(a_grid)[::-1]) - 1) for c in chi_src]
nzs = [jc.redshift.delta_nz(z) for z in z_src]
probe = jc.probes.WeakLensing(nzs, sigma_e=0.0)
ell = jnp.arange(2, LMAX + 1).astype(float)
cls = np.asarray(jc.angular_cl.angular_cl(cosmo, ell, [probe]))   # all auto/cross pairs
n = len(z_src)
pairs = [(i, j) for i in range(n) for j in range(i, n)]
C = np.zeros((LMAX + 1, n, n))
for k, (i, j) in enumerate(pairs):
    C[2:, i, j] = C[2:, j, i] = cls[k]
# correlated maps: a_lm = L_l @ g_lm, with L_l the Cholesky factor of C_l
rng = np.random.default_rng(3)
nalm = hp.Alm.getsize(LMAX)
g = (rng.standard_normal((n, nalm)) + 1j * rng.standard_normal((n, nalm))) / np.sqrt(2)
ls, ms = hp.Alm.getlm(LMAX)
g[:, ms == 0] = np.sqrt(2) * g[:, ms == 0].real
alm = np.zeros((n, nalm), complex)
for l in range(2, LMAX + 1):
    L = np.linalg.cholesky(C[l] + 1e-30 * np.eye(n))
    sel = ls == l
    alm[:, sel] = L @ g[:, sel]
maps = [hp.alm2map(a, NSIDE, lmax=LMAX) for a in alm]
v = float(np.percentile(np.abs(maps[0]), 99.0))       # the farthest plane sets the scale
for i, m in enumerate(maps):
    fig = plt.figure(figsize=(3.2, 3.2), dpi=170)
    hp.orthview(m, fig=fig.number, half_sky=True, title="", cbar=False, cmap="viridis",
                min=-0.6 * v, max=v, notext=True, rot=(20, 25, 0), bgcolor=(0.0,) * 4)
    fig.savefig(HERE / f"kappa_{i}.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
# trim every image to its visible content (3D axes leave wide empty margins)
from PIL import Image

for name in OUTS:
    im = Image.open(HERE / name)
    im.crop(im.getchannel("A").point(lambda x: 255 if x > 8 else 0).getbbox()).save(HERE / name)

# orthview paints a white square behind the sphere: keep only the disc
for i in range(4):
    im = Image.open(HERE / f"kappa_{i}.png").convert("RGBA")
    rgb = np.asarray(im.convert("RGB")).astype(int)
    inside = rgb.sum(axis=2) < 3 * 245                      # not the white background
    ys, xs = np.nonzero(inside)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = im.size
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / ((min(w, h) - 1) / 2)
    a = np.clip((1.0 - r) * min(w, h) / 2, 0, 1)             # 1-pixel anti-aliased rim
    im.putalpha(Image.fromarray((255 * a).astype(np.uint8)))
    im.save(HERE / f"kappa_{i}.png")

print("z_src:", np.round(z_src, 3))
print("wrote", ", ".join(OUTS))
