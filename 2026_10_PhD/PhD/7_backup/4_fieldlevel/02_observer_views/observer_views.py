#!/usr/bin/env python3
# ENV: jax-fli
"""
Where the observer sits in the box and the sky it sees, for the second slide on the volume a
DES Y3 analysis needs.

Two observers inside the unit box [0, 1]^3: at the centre, and in the shifted-quadrant position
(0.1, 0.5, 0.9) of the masked-shear experiment. Top row: the box as a wireframe with the observer
marked (the drawing of These_wassim/figures/chap6/observer_wireframe.py). Bottom row: the
Mollweide view of the visible sky in the two-colour seen / unseen map of the thesis, with the
DES Y3 footprint (jax_fli.data.get_desy3_mask, smoothed 1.5 deg so the holes inside it do not
draw) outlined. The visible sky is jaxpm.spherical.spherical_visibility_mask with
threshold = 1.0 (pixels whose cone stays inside the box out to half a box), the call of
observer_wireframe.py and masked_shear.py.

Output (this directory): observer_views.png
"""

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, KW, KW2, skip_if_built, slide_style

OUTS = ["observer_views.png"]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import jax.numpy as jnp
import jax_fli as jfli
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.spherical import spherical_visibility_mask
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

NSIDE = 128
OBS = "#C0392B"
CASES = (("centre", (0.5, 0.5, 0.5)), ("shifted quadrant", (0.1, 0.5, 0.9)))
FOOTPRINT = ListedColormap(
    [plt.get_cmap("viridis")(0.0), plt.get_cmap("viridis")(1.0)], name="seen_unseen"
).with_extremes(bad=(0, 0, 0, 0))

slide_style(scale=1.3)


def draw_box(ax, observer):
    """Unit-box wireframe with the observer marked and a drop line to the base."""
    corners = list(itertools.product([0, 1], repeat=3))
    for a, b in itertools.combinations(corners, 2):
        if sum(ai != bi for ai, bi in zip(a, b)) == 1:
            ax.plot3D(*zip(a, b), color="0.45", lw=1.4)
    ox, oy, oz = observer
    ax.plot([ox, ox], [oy, oy], [0, oz], color=OBS, ls=":", lw=1.8, clip_on=False)
    ax.scatter([ox], [oy], [oz], color=OBS, s=170, depthshade=False, zorder=6, clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.93, 0.93, 0.93, 0.5))
        axis.pane.set_edgecolor("0.7")
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks, labels=[""] * 5)
    ax.set_yticks(ticks, labels=[""] * 5)
    ax.set_zticks(ticks, labels=[""] * 5)
    ax.tick_params(axis="both", length=0, pad=0)
    for x, y, z, name in ((0.5, -0.38, 0.0, "x"), (1.38, 0.5, 0.0, "y")):
        ax.text(x, y, z, name, ha="center", va="center", color=INK, fontsize=16)
    ax.text2D(1.0, 0.55, "z", transform=ax.transAxes, ha="center", va="center", color=INK,
              fontsize=16)
    ax.set_box_aspect((1, 1, 1), zoom=1.0)
    ax.view_init(elev=18, azim=-52)


proj = hp.projector.MollweideProj(xsize=1000)
vec2pix = lambda x, y, z: hp.vec2pix(NSIDE, x, y, z)
finite = lambda img, fill: np.where(np.isfinite(img) & (img > -1e30), img, fill)
des = np.asarray(jfli.data.get_desy3_mask(NSIDE)).astype(float)
# smoothed before contouring, so the outline follows the footprint and not the holes in it
des_img = finite(proj.projmap(hp.smoothing(des, fwhm=np.radians(1.5)), vec2pix), 0.0)

fig = plt.figure(figsize=(10.0, 6.6))
for k, (name, obs) in enumerate(CASES):
    ax = fig.add_axes([0.03 + 0.5 * k, 0.47, 0.42, 0.52], projection="3d")
    draw_box(ax, obs)

    mask = np.asarray(spherical_visibility_mask(NSIDE, jnp.asarray(obs), threshold=1.0))
    fsky = float(mask.mean())
    print(f"{name:16s} {obs}  f_sky = {fsky:.3f}  DES covered = {mask[des > 0.5].mean():.3f}")
    axm = fig.add_axes([0.02 + 0.5 * k, 0.13, 0.46, 0.32])
    axm.imshow(finite(proj.projmap(mask.astype(float), vec2pix), np.nan), cmap=FOOTPRINT,
               vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axm.contour(des_img, levels=[0.5], colors=[KW], linewidths=2.4, origin="lower")
    axm.set_axis_off()
    axm.set_title(rf"{name}:  $f_\mathrm{{sky}} = {fsky:.2f}$", color=KW2, fontsize=17, pad=6)

bar = fig.colorbar(ScalarMappable(norm=BoundaryNorm([0.0, 0.5, 1.0], 2), cmap=FOOTPRINT),
                   cax=fig.add_axes([0.385, 0.05, 0.23, 0.035]), orientation="horizontal")
bar.set_ticks([0.25, 0.75], labels=["unseen", "seen"])
bar.ax.tick_params(length=0, labelsize=15, colors=INK)
bar.outline.set_edgecolor(INK)

fig.savefig(HERE / "observer_views.png", dpi=200)
print("wrote", ", ".join(OUTS))
