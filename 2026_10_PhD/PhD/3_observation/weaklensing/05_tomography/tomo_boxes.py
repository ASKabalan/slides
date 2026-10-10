#!/usr/bin/env python3
# ENV: jax-fli
"""
The four density cubes that 5_contribution/fli/08_lightcone/lightcone_shells.py draws above its
shells: a long 1LPT density box from one small run of my forward model, 256 x 64 x 64 cells,
1600 x 400 x 400 Mpc/h, Planck 2018, at a = 1, cut into four 64^3 sub-boxes, far to near, on one
colour scale (magma, transparent).

  box_{0..3}.png

Runs on CPU in the jax-fli environment; jax_enable_x64 first.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

OUTS = [f"box_{i}.png" for i in range(4)]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
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


for i in range(4):
    render(vol[64 * i:64 * (i + 1)], f"box_{i}.png", (1, 1, 1), (3.4, 3.4))

# trim every image to its visible content (3D axes leave wide empty margins)
from PIL import Image

for name in OUTS:
    im = Image.open(HERE / name)
    im.crop(im.getchannel("A").point(lambda x: 255 if x > 8 else 0).getbbox()).save(HERE / name)

print("wrote", ", ".join(OUTS))
