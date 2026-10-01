#!/usr/bin/env python3
# ENV: jax-fli
"""
The redshift a lightcone assigns to each particle, with and without the drift on the lightcone, for
the drift backup slide (static: three panels side by side). After jax-fli docs/2-advanced-usage/07-Drift-on-Lightcone.ipynb and the thesis
figure chap6/redshift_assignment.pdf.

A small simulation on CPU: 256^3 particles in a 2000 Mpc/h box, observer at the centre, Planck18,
first-order LPT to a = 0.1, then BullFrog in 10 steps, the particles of one thick radial bin
[60, 950] Mpc/h read out at the scale factor of its centre. A slab |z| < 80 Mpc/h and a wedge
0.1 < phi < 1.3 rad of it (40 000 particles) are drawn in the plane, coloured by the redshift each
one is assigned:
  one redshift per shell  the redshift of the centre of the shell the particle falls in, for 10 or 40
                          shells uniform in the scale factor over [0, 950] Mpc/h;
  drifted                 the redshift of the particle's own comoving distance, z(|x|), the epoch at
                          which it crosses the lightcone. The drift moves particles by a few Mpc/h
                          at most, below what the panel resolves, so the positions are the same in
                          every panel and only the colour changes.

Outputs (this directory):
  drift_a10.png       10 shells uniform in a, one redshift per shell
  drift_smooth.png    the drifted assignment
  drift_a40.png       40 shells uniform in a, one redshift per shell
  drift_cbar.svg      the redshift colour bar
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built, slide_style

OUTS = ["drift_a10.png", "drift_smooth.png", "drift_a40.png", "drift_cbar.svg"]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
MESH, BOX, N_STEPS = 256, 2000.0, 10
R0, R1 = 60.0, 950.0
BG = (250, 247, 240)                        # the slide background, #faf7f0


def cloud():
    npz = CACHE / "drift_cloud.npz"
    if npz.exists():
        d = np.load(npz)
        return d["x"], d["y"], d["r"], d["z_of_r"], d["chi_grid"], d["z_grid"]
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_cosmo as jc
    import jax_fli as jfli

    cosmo = jc.Planck18()
    a_c = float(jc.background.a_of_chi(cosmo, jnp.atleast_1d(0.5 * (R0 + R1)))[0])
    ic = jfli.gaussian_initial_conditions(jax.random.PRNGKey(0), (MESH,) * 3, (BOX,) * 3,
                                          cosmo=cosmo, nside=256)
    dx, p = jfli.lpt(cosmo, ic, ts=0.1, order=1)
    part = jfli.PaintingOptions(target="particles")
    out = jfli.nbody(cosmo, dx, p, ts=jnp.array([a_c]), density_widths=jnp.array([R1 - R0]),
                     shell_spacing="comoving", min_width=1.0,
                     solver=jfli.BullFrog(interp_kernel=jfli.NoInterp(painting=part),
                                          t0=0.1, t1=1.0, n_steps=N_STEPS))
    pos = np.asarray(out.to(jfli.PositionUnit.MPC_H).array).reshape(-1, 3) - 0.5 * BOX
    r = np.linalg.norm(pos, axis=1)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    idx = np.where((abs(pos[:, 2]) < 80) & (phi > 0.1) & (phi < 1.3) & (r > R0) & (r < R1))[0]
    idx = np.random.default_rng(0).choice(idx, min(40000, idx.size), replace=False)
    chi_grid = np.linspace(0, 1000, 2001)
    z_grid = 1 / np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(chi_grid[1:]))) - 1
    z_grid = np.r_[0.0, z_grid]
    a_edges = np.linspace(float(jc.background.a_of_chi(cosmo, jnp.atleast_1d(R1))[0]), 1.0, 41)
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, x=pos[idx, 0], y=pos[idx, 1], r=r[idx], z_of_r=np.interp(r[idx], chi_grid, z_grid),
             chi_grid=chi_grid, z_grid=z_grid)
    return cloud()


X, Y, R, Z_R, CHI_GRID, Z_GRID = cloud()


def z_of(chi):
    return np.interp(chi, CHI_GRID, Z_GRID)


def chi_of(z):
    return np.interp(z, Z_GRID, CHI_GRID)


def uniform_a_edges(n):
    """Edges uniform in a over [0, R1]: a_i = a_min + (i/n)(1 - a_min), r_i = chi(a_i)."""
    a_min = 1 / (1 + z_of(R1))
    a = a_min + np.arange(n + 1) / n * (1 - a_min)
    return np.sort(chi_of(1 / a - 1))


def banded(edges):
    b = np.clip(np.digitize(R, edges) - 1, 0, len(edges) - 2)
    return z_of(0.5 * (edges[:-1] + edges[1:]))[b]


slide_style()
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["savefig.bbox"] = None
VMIN, VMAX = z_of(R0), z_of(R1)
FIGSIZE, RECT, DPI = (3.9, 3.8), [0.17, 0.13, 0.8, 0.83], 200


def render(z):
    """One panel as an RGB image on the slide background (fixed canvas and axes box)."""
    fig = plt.figure(figsize=FIGSIZE, facecolor=np.array(BG) / 255)
    ax = fig.add_axes(RECT, facecolor=np.array(BG) / 255)
    ax.scatter(X, Y, c=z, s=1.6, cmap="turbo", vmin=VMIN, vmax=VMAX, lw=0, rasterized=True)
    ax.set_xlim(-20, 960)
    ax.set_ylim(-20, 940)
    ax.set_aspect("equal")
    ax.set_xticks([0, 400, 800])
    ax.set_yticks([0, 400, 800])
    ax.set_xlabel(r"$x$  [Mpc/$h$]", labelpad=1)
    ax.set_ylabel(r"$y$  [Mpc/$h$]", labelpad=1)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return im


plt.rcParams["figure.dpi"] = DPI
for out, z in (("drift_a10.png", banded(uniform_a_edges(10))), ("drift_smooth.png", Z_R),
               ("drift_a40.png", banded(uniform_a_edges(40)))):
    render(z).save(HERE / out)
    print(f"wrote {out}")

# the colour bar, on the same canvas height and axes span as the panels
fig = plt.figure(figsize=(0.95, FIGSIZE[1]))
cax = fig.add_axes([0.08, RECT[1], 0.2, RECT[3]])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(VMIN, VMAX), cmap="turbo"), cax=cax)
cb.set_label("assigned redshift $z$", labelpad=4)
cb.outline.set_linewidth(0.8)
fig.savefig(HERE / "drift_cbar.svg")
print("wrote drift_cbar.svg; edges uniform a (10):", uniform_a_edges(10).round(0))
