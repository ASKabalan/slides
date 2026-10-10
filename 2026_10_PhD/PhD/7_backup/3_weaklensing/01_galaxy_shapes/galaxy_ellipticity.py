#!/usr/bin/env python3
# ENV: shared
"""
A real galaxy seen as an ellipse, for the slide "From galaxy shapes to shear": the M101 cut-out of
galaxy_distortion.py (Hubble; Wikimedia Commons "File:M101 hires STScI-PRC2006-10a.jpg"), on the
same dark rounded tile, stretched by A = R(phi) diag(1 + s, 1 - s) R(-phi) so that it appears
elliptical, with its ellipse drawn over it: the semi-major axis a (orange), the semi-minor axis b
(lavender) and the position angle phi from the x axis.

Output (this directory): galaxy_ellipticity.png (transparent outside the tile)
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import KW, cached_fetch, skip_if_built

OUT = "galaxy_ellipticity.png"
skip_if_built(HERE, OUT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyBboxPatch
from PIL import Image

SKY = (10, 12, 22)
LAVENDER = "#C9A8E0"
PHI, S = np.radians(30.0), 0.30            # position angle, stretch: a/b = 1.3/0.7
N = 900                                    # tile pixels

# the galaxy, cut and faded exactly as in galaxy_distortion.py
M101 = ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/"
        "M101_hires_STScI-PRC2006-10a.jpg/1920px-M101_hires_STScI-PRC2006-10a.jpg")
src = Image.open(io.BytesIO(cached_fetch(HERE / ".cache", "m101_1920", M101))).convert("RGB")
w, h = src.size
side = int(0.62 * min(w, h))
cx, cy = int(0.5 * w), int(0.52 * h)
src = src.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))
G = int(0.64 * N)
gal = src.resize((G, G), Image.LANCZOS)
yy, xx = np.mgrid[0:G, 0:G]
r = np.hypot(xx - G / 2, yy - G / 2) / (G / 2)
gal.putalpha(Image.fromarray((255 * np.clip((1.0 - r) / 0.28, 0, 1) ** 1.5).astype(np.uint8)))

# stretch along phi (image y points down, so the angle is mirrored)
c, s_ = np.cos(-PHI), np.sin(-PHI)
R = np.array([[c, -s_], [s_, c]])
A = R @ np.diag([1 + S, 1 - S]) @ R.T
Ainv = np.linalg.inv(A)
off = np.array([G / 2, G / 2]) - Ainv @ np.array([N / 2, N / 2])
tile = Image.new("RGBA", (N, N), SKY + (255,))
tile.alpha_composite(gal.transform((N, N), Image.AFFINE,
                                   (Ainv[0, 0], Ainv[0, 1], off[0], Ainv[1, 0], Ainv[1, 1], off[1]),
                                   resample=Image.BICUBIC))

# the overlay, in tile units: the tile spans [-1, 1], the faded disc has radius R0 before the stretch
R0 = 0.78 * (G / 2) / (N / 2)
a, b = (1 + S) * R0, (1 - S) * R0
fig = plt.figure(figsize=(4.4, 4.4), dpi=N / 4.4)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.axis("off")
ax.imshow(np.asarray(tile.convert("RGB")), extent=(-1, 1, -1, 1), zorder=0)
clip = FancyBboxPatch((-1, -1), 2, 2, boxstyle="round,pad=0,rounding_size=0.15", transform=ax.transData,
                      fc="none", ec="none")
ax.add_patch(clip)
ax.images[0].set_clip_path(clip)

ax.plot([-0.9, 0.9], [0, 0], color="white", lw=1.2, alpha=0.45, ls=(0, (4, 4)), zorder=1)
ax.text(0.92, 0.03, "$x$", color="white", alpha=0.7, fontsize=15, ha="left", va="bottom")
ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, angle=np.degrees(PHI), fill=False, ec="white", lw=2.0,
                     ls=(0, (6, 4)), zorder=2))
ua = np.array([np.cos(PHI), np.sin(PHI)])
ub = np.array([-np.sin(PHI), np.cos(PHI)])
ax.plot([0, a * ua[0]], [0, a * ua[1]], color=KW, lw=3.4, solid_capstyle="round", zorder=3)
ax.plot([0, b * ub[0]], [0, b * ub[1]], color=LAVENDER, lw=3.4, solid_capstyle="round", zorder=3)
ax.text(*(0.62 * a * ua + 0.07 * ub), "$a$", color=KW, fontsize=24, ha="center", va="bottom",
        fontweight="bold", zorder=4)
ax.text(*(0.55 * b * ub - 0.08 * ua), "$b$", color=LAVENDER, fontsize=24, ha="right", va="center",
        zorder=4)
ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=np.degrees(PHI), color="white", lw=1.8,
                 zorder=3))
ax.text(0.40 * np.cos(PHI / 2), 0.40 * np.sin(PHI / 2), r"$\varphi$", color="white", fontsize=20,
        ha="center", va="center", zorder=4)
ax.plot(0, 0, "o", color="white", ms=4, zorder=4)

buf = io.BytesIO()
fig.savefig(buf, format="png", transparent=True, dpi=N / 4.4)
plt.close(fig)
im = Image.open(buf).convert("RGBA")
# round the tile corners, as on the previous slide
mask = Image.new("L", im.size, 0)
from PIL import ImageDraw

ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius=int(0.075 * im.size[0]),
                                       fill=255)
im.putalpha(mask)
im.save(HERE / OUT)
print(f"wrote {OUT} {im.size}, e = (a - b)/(a + b) = {(a - b) / (a + b):.2f}")
