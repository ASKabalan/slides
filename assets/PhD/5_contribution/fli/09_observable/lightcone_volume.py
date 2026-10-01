#!/usr/bin/env python3
# ENV: jax-fli
"""
Where the observer sits in the box, and the sky it sees, for the slide on the volume a DES Y3
analysis needs.

Two observers inside the unit box [0, 1]^3: at the centre, and in the shifted-quadrant position
(0.1, 0.5, 0.9) of the masked-shear experiment. The visible sky is
jaxpm.spherical.spherical_visibility_mask with threshold = 1.0 (pixels whose cone stays inside
the box out to half a box), the call the thesis figures observer_wireframe.pdf and
observer_masks.pdf make; f_sky is printed, and quoted on the slide. The DES Y3 footprint
(jax_fli.data.get_desy3_mask) is drawn as an outline on the same projections for the last
fragment, smoothed by 1.5 deg so the holes inside it do not draw. The wireframe drawing follows These_wassim/figures/chap6/observer_wireframe.py.

Outputs (this directory):
  wire_centre.svg, wire_shifted.svg              the box and the observer
  sky_centre.png, sky_shifted.png                the visible sky (seen / unseen)
  sky_centre_des.png, sky_shifted_des.png        the same, with the DES Y3 footprint outlined
"""

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import INK, KW, skip_if_built, slide_style

OUTS = ["wire_centre.svg", "wire_shifted.svg", "sky_centre.png", "sky_shifted.png",
        "sky_centre_des.png", "sky_shifted_des.png"]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import jax.numpy as jnp
import jax_fli as jfli
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.spherical import spherical_visibility_mask
from matplotlib.colors import ListedColormap

NSIDE = 128
CASES = {"centre": (0.5, 0.5, 0.5), "shifted": (0.1, 0.5, 0.9)}
OBS = "#C0392B"
FOOTPRINT = ListedColormap([plt.get_cmap("viridis")(0.0), plt.get_cmap("viridis")(1.0)])

slide_style(scale=1.3)


def draw_box(ax, observer):
    """Unit-box wireframe with the observer marked and a drop line to the base."""
    corners = list(itertools.product([0, 1], repeat=3))
    for a, b in itertools.combinations(corners, 2):
        if sum(ai != bi for ai, bi in zip(a, b)) == 1:
            ax.plot3D(*zip(a, b), color="0.45", lw=1.4)
    ox, oy, oz = observer
    ax.plot([ox, ox], [oy, oy], [0, oz], color=OBS, ls=":", lw=1.6, clip_on=False)
    ax.scatter([ox], [oy], [oz], color=OBS, s=160, depthshade=False, zorder=6, clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.pane.set_facecolor((0.93, 0.93, 0.93, 0.5))
        axis.pane.set_edgecolor("0.7")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis="both", length=0, pad=0)
    for x, y, z, name in ((0.5, -0.38, 0.0, "x"), (1.38, 0.5, 0.0, "y")):
        ax.text(x, y, z, name, ha="center", va="center", color=INK, fontsize=16)
    ax.text2D(1.02, 0.55, "z", transform=ax.transAxes, ha="center", va="center", color=INK,
              fontsize=16)
    ax.set_box_aspect((1, 1, 1), zoom=1.1)
    ax.view_init(elev=18, azim=-52)


des = np.asarray(jfli.data.get_desy3_mask(NSIDE)).astype(float)
proj = hp.projector.MollweideProj(xsize=1400)
vec2pix = lambda x, y, z: hp.vec2pix(NSIDE, x, y, z)
# smoothed before contouring, so the outline follows the footprint and not the holes in it
des_img = proj.projmap(hp.smoothing(des, fwhm=np.radians(1.5)), vec2pix)

for name, obs in CASES.items():
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_axes([0, 0, 0.86, 1], projection="3d")
    draw_box(ax, obs)
    fig.savefig(HERE / f"wire_{name}.svg", transparent=True)
    plt.close(fig)

    mask = np.asarray(spherical_visibility_mask(NSIDE, jnp.asarray(obs), threshold=1.0))
    print(f"{name:8s} {obs}  f_sky = {mask.mean():.3f}  DES covered = {mask[des > 0.5].mean():.3f}")
    img = proj.projmap(mask.astype(float), vec2pix)
    for with_des in (False, True):
        fig = plt.figure(figsize=(7.0, 3.6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.where(np.isfinite(img) & (img > -1e30), img, np.nan), cmap=FOOTPRINT,
                  vmin=0, vmax=1, origin="lower", interpolation="nearest")
        if with_des:
            ax.contour(np.where(np.isfinite(des_img) & (des_img > -1e30), des_img, 0.0),
                       levels=[0.5], colors=[KW], linewidths=2.6, origin="lower")
        ax.set_axis_off()
        fig.savefig(HERE / f"sky_{name}{'_des' if with_des else ''}.png", transparent=True,
                    dpi=200)
        plt.close(fig)
print("wrote", ", ".join(OUTS))
