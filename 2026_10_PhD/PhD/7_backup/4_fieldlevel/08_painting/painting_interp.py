#!/usr/bin/env python3
# ENV: jax-fli
"""
How one particle is painted onto HEALPix pixels, for the slide on spherical painting.

A patch of the N_side = 4 sphere around one particle, drawn in longitude and latitude with the
pixel outlines of healpy.boundaries. The pixels and weights are those of jax-healpy, as the
lightcone painter of jaxpm uses them:

  bilinear.mp4      jax_healpy.get_interp_weights: four pixels, weights from the particle's
                    position between the pixel centres; the particle drifts and the weights
                    follow it continuously
  rbf.mp4           jax_healpy.get_all_neighbours(get_center=True): the pixel under the particle
                    and its eight neighbours, weighted by a Gaussian of the angular distance to
                    each centre and normalised to one (jaxpm.spherical.
                    paint_particles_spherical_rbf_neighbor), 1.5 pixel FWHM; same drift
  rbf_kernel.mp4    the particle at rest while the FWHM breathes from 0.5 to 2.5 pixels and back,
                    the dashed circle at half the FWHM

Line width and fill show each pixel's weight. A PNG of each last frame is kept for print.
No text on the frames; the slide holds it.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, KW, KW2, skip_if_built

OUTS = ["bilinear.mp4", "rbf.mp4", "rbf_kernel.mp4"]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import jax_healpy as jhp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

NSIDE = 4
BG = "#faf7f0"
FPS = 20
RES = hp.nside2resol(NSIDE)                       # pixel scale [rad]
T0, P0 = 1.12, 0.92                               # the particle's start (theta, phi) [rad]
WIN = (np.degrees(P0) - 34, np.degrees(P0) + 34, 90 - np.degrees(T0) - 26, 90 - np.degrees(T0) + 26)

# every pixel whose centre falls near the window, with its outline in (lon, lat) degrees
cth, cph = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
lon = (np.degrees(cph) - np.degrees(P0) + 180) % 360 - 180 + np.degrees(P0)
near = np.where((lon > WIN[0] - 20) & (lon < WIN[1] + 20)
                & (90 - np.degrees(cth) > WIN[2] - 20) & (90 - np.degrees(cth) < WIN[3] + 20))[0]
unwrap = lambda lon: (lon - np.degrees(P0) + 180) % 360 - 180 + np.degrees(P0)  # no 0/360 jump
OUTLINE = {}
for p in near:
    th, ph = hp.vec2ang(hp.boundaries(NSIDE, p, step=10).T)
    OUTLINE[p] = np.c_[unwrap(np.degrees(ph)), 90 - np.degrees(th)]
CENTRE = {p: (unwrap(np.degrees(cph[p])), 90 - np.degrees(cth[p])) for p in near}


def bilinear(theta, phi):
    pix, w = jhp.get_interp_weights(NSIDE, np.array([theta]), np.array([phi]))
    return dict(zip(np.asarray(pix).ravel().tolist(), np.asarray(w).ravel().tolist()))


def rbf(theta, phi, fwhm_pix):
    sigma = fwhm_pix * RES / (2 * np.sqrt(2 * np.log(2)))
    pix9 = np.asarray(jhp.get_all_neighbours(NSIDE, np.array([theta]), np.array([phi]),
                                             get_center=True)).ravel()
    u = hp.ang2vec(theta, phi)
    vecs = np.array(hp.pix2vec(NSIDE, np.where(pix9 >= 0, pix9, 0))).T      # (9, 3)
    gamma = np.arccos(np.clip(vecs @ u, -1.0, 1.0))                      # angular distance
    w = np.exp(-gamma ** 2 / (2 * sigma ** 2))
    w = np.where(pix9 >= 0, w, 0.0)
    return dict(zip(pix9.tolist(), (w / w.sum()).tolist())), sigma


def draw(theta, phi, weights, colour, radius=None):
    fig = plt.figure(figsize=(6.4, 4.9), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    wmax = max(weights.values()) if weights else 1.0
    for p, xy in OUTLINE.items():
        w = weights.get(p, 0.0)
        ax.add_patch(Polygon(xy, closed=True, facecolor=colour if w > 0 else "none",
                             alpha=None if w == 0 else 0.12 + 0.7 * w / wmax,
                             edgecolor="none"))
        ax.add_patch(Polygon(xy, closed=True, facecolor="none", edgecolor="#8A8A8A", lw=1.2))
    x, y = np.degrees(phi), 90 - np.degrees(theta)
    for p, w in weights.items():
        if w > 0:
            cx, cy = CENTRE[p]
            ax.plot([x, cx], [y, cy], color=colour, lw=0.6 + 5.5 * w, alpha=0.9, zorder=3)
            ax.plot(cx, cy, "o", color=colour, ms=5, zorder=4)
    if radius is not None:
        ax.add_patch(Circle((x, y), np.degrees(radius), fill=False, ls="--", lw=2.2,
                            edgecolor=KW, zorder=5))
    ax.plot(x, y, "o", color=INK, ms=11, mec="white", mew=2, zorder=6)
    ax.set_xlim(WIN[0], WIN[1])
    ax.set_ylim(WIN[2], WIN[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def write(frames, name):
    h, w = frames[0].shape[:2]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(HERE / name)], input=b"".join(f.tobytes() for f in frames),
                   check=True)
    from PIL import Image

    Image.fromarray(frames[-1]).save(HERE / name.replace(".mp4", "_last.png"))
    print(f"wrote {name} ({len(frames)} frames)")


# the drift: a slow loop of about one pixel around the start
s = np.linspace(0, 2 * np.pi, 5 * FPS)
path = [(T0 + 0.09 * np.sin(t), P0 + 0.16 * (1 - np.cos(t)) / 2 + 0.05 * np.sin(2 * t)) for t in s]
FWHM = 1.5

for name, colour in (("bilinear.mp4", KW2), ("rbf.mp4", KW)):
    weight = (lambda th, ph: bilinear(th, ph)) if name == "bilinear.mp4" else \
             (lambda th, ph: rbf(th, ph, FWHM)[0])
    frames = [draw(T0, P0, {}, colour)] * (FPS // 2)
    frames += [draw(T0, P0, weight(T0, P0), colour)] * FPS
    frames += [draw(th, ph, weight(th, ph), colour) for th, ph in path]
    frames += [frames[-1]] * FPS
    write(frames, name)

widths = 1.5 + np.sin(np.linspace(0, 2 * np.pi, 6 * FPS)) * 1.0      # 0.5 .. 2.5 pixels
frames = []
for f in widths:
    wts, sigma = rbf(T0, P0, f)
    frames.append(draw(T0, P0, wts, KW, radius=f * RES / 2))
frames += [frames[-1]] * (FPS // 2)
write(frames, "rbf_kernel.mp4")
