#!/usr/bin/env python3
# ENV: shared
"""
Energy content of the Universe, Planck 2018, as a textured donut.

The thesis version (figures/chap2/content_pie.py) uses flat colours. On a slide
each wedge instead displays a texture that says what the component is: a star
field for the baryons, a slice of one of my own particle-mesh density fields for
the dark matter, and a smooth outward-brightening gradient for the dark energy.

Outputs (this directory):
  content_pie.svg        the finished donut (also the PDF stand-in for the GIF)
  content_pie_fill.gif   the donut filling up clockwise from twelve o'clock,
                         each label appearing as its wedge completes; plays
                         once (no loop). Transparent GIF: alpha is 1-bit, so
                         the anti-aliased rim is pre-blended onto the slide
                         colour (#faf7f0) before the alpha is thresholded.
Both come from the same draw() and the same fixed frame, so they match.

Density-field texture: assets/Fields/LPT_density_field_z0_1024.png (this work).
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# Text as paths, so the figure reads the same in every browser.
plt.rcParams["svg.fonttype"] = "path"

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
OUT = HERE / "content_pie.svg"
GIF = HERE / "content_pie_fill.gif"
FIELD = ROOT.parent / "assets" / "Fields" / "LPT_density_field_z0_1024.png"   # the shared slides assets, through the deck's assets link
FORCE_REGEN = os.getenv("FORCE_REGEN", "0").lower() in ("1", "true", "t")

if OUT.exists() and GIF.exists() and not FORCE_REGEN:
    print(f"{OUT.name}, {GIF.name} already exist. Set FORCE_REGEN=1 to regenerate them.")
    sys.exit(0)

# Planck 2018 TT,TE,EE+lowE+lensing+BAO
FRACTIONS = {"dark energy": 0.6847, "dark matter": 0.2645, "baryons": 0.0493}
KW = "#C2560A"   # the deck keyword orange
INK = "#2E2E2E"
N = 900


def star_field(n=N, seed=3) -> np.ndarray:
    """Warm points of light on near-black: the baryons."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n))
    ys, xs = rng.integers(0, n, 2600), rng.integers(0, n, 2600)
    img[ys, xs] = rng.uniform(0.45, 1.0, 2600)
    # A few brighter ones, spread with a small Gaussian so they read as stars.
    from scipy.ndimage import gaussian_filter

    img = gaussian_filter(img, 1.1)
    bright = np.zeros((n, n))
    ys, xs = rng.integers(0, n, 240), rng.integers(0, n, 240)
    bright[ys, xs] = 1.0
    img = img / img.max() + 2.2 * gaussian_filter(bright, 2.6)
    img += 0.10 * gaussian_filter(rng.random((n, n)), 26)  # faint diffuse gas
    rgb = np.zeros((n, n, 3))
    v = np.clip(img / np.percentile(img, 99.7), 0, 1)
    rgb[..., 0] = 0.06 + 0.94 * v
    rgb[..., 1] = 0.07 + 0.86 * v**1.15
    rgb[..., 2] = 0.13 + 0.72 * v**1.45
    return rgb


def density_field(n=N) -> np.ndarray:
    """A crop of one of my particle-mesh fields: the dark matter."""
    from PIL import Image

    im = Image.open(FIELD).convert("RGB")
    side = min(im.size)
    left = (im.size[0] - side) // 2
    top = (im.size[1] - side) // 2
    im = im.crop((left, top, left + side, top + side)).resize((n, n), Image.LANCZOS)
    return np.asarray(im) / 255.0


def expansion(n=N) -> np.ndarray:
    """A smooth field brightening outward: the accelerating expansion."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.clip(np.hypot(x, y), 0, 1)
    v = r**1.25
    rgb = np.zeros((n, n, 3))
    rgb[..., 0] = 0.07 + 0.16 * v        # deep indigo, brightening outward
    rgb[..., 1] = 0.05 + 0.33 * v
    rgb[..., 2] = 0.28 + 0.62 * v
    return rgb


TEXTURES = {
    "dark energy": expansion(),
    "dark matter": density_field(),
    "baryons": star_field(),
}

R_OUT, R_IN = 1.0, 0.44
XLIM, YLIM = (-2.2, 2.3), (-1.15, 1.45)   # fixed frame: SVG and GIF match
SLIDE_BG = (250, 247, 240)


def draw(sweep: float = 360.0):
    """The donut with the first `sweep` degrees filled (clockwise from 12)."""
    fig = plt.figure(figsize=(XLIM[1] - XLIM[0], YLIM[1] - YLIM[0]) , dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_aspect("equal")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    fig.patch.set_alpha(0.0)

    start = 0.0
    theta = 90.0
    for name, frac in FRACTIONS.items():
        span = 360.0 * frac
        shown = float(np.clip(sweep - start, 0.0, span))
        if shown > 0:
            wedge = Wedge((0, 0), R_OUT, theta - shown, theta, width=R_OUT - R_IN,
                          facecolor="none", edgecolor="white", linewidth=2.2, zorder=3)
            ax.add_patch(wedge)
            im = ax.imshow(TEXTURES[name], extent=(-R_OUT, R_OUT, -R_OUT, R_OUT),
                           origin="lower", zorder=2, interpolation="bilinear")
            im.set_clip_path(wedge)
        if shown >= span - 1e-6:
            # Label just outside the mid-angle of the finished wedge.
            mid = np.deg2rad(theta - span / 2)
            lx, ly = 1.17 * np.cos(mid), 1.17 * np.sin(mid)
            ha = "left" if lx > 0.08 else ("right" if lx < -0.08 else "center")
            ax.text(lx, ly, f"{name}\n{100 * frac:.1f}%", ha=ha, va="center",
                    fontsize=11.5, color=INK, fontweight="bold", linespacing=1.35,
                    family="DejaVu Sans", zorder=4)
        start += span
        theta -= span

    if sweep >= 360.0:
        ax.text(0, 0, r"$\Lambda$CDM", ha="center", va="center", fontsize=17,
                color=KW, fontweight="bold", family="DejaVu Sans", zorder=4)
    return fig


def gif_frame(fig, dpi: int = 240):
    """Render to a 1-bit-alpha palette frame, rim pre-blended onto the slide."""
    from PIL import Image

    fig.set_dpi(dpi)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).astype(float)
    plt.close(fig)
    a = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * a + np.array(SLIDE_BG) * (1 - a)
    q = Image.fromarray(rgb.astype(np.uint8), "RGB").quantize(255)
    idx = np.asarray(q).copy()
    idx[rgba[..., 3] < 90] = 255
    out = Image.fromarray(idx, "P")
    out.putpalette(q.getpalette()[:765] + [0, 0, 0])
    return out


fig = draw()
fig.savefig(OUT, transparent=True)
plt.close(fig)
print(f"wrote {OUT.name}")

# Fill over 1.8 s at 25 fps with an ease-out, then hold the finished donut.
n = 45
u = np.linspace(0, 1, n)
sweeps = 360.0 * (1 - (1 - u) ** 2.2)
frames = [gif_frame(draw(sw)) for sw in sweeps] + [gif_frame(draw(360.0))]
durations = [40] * n + [1000]
frames[0].save(GIF, save_all=True, append_images=frames[1:], duration=durations,
               transparency=255, disposal=2, optimize=False)   # no loop: plays once
print(f"wrote {GIF.name} ({GIF.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
