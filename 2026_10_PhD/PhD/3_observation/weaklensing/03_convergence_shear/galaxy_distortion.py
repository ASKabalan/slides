#!/usr/bin/env python3
# ENV: shared
"""
What weak lensing does to a galaxy, for the "Convergence and shear" slide.

Three dark cut-outs of the same real galaxy (M101, Hubble; Wikimedia Commons
"File:M101 hires STScI-PRC2006-10a.jpg"), each warped frame by frame by an affine
map around its centre:
  1. the galaxy as it is (reference)
  2. convergence kappa: an isotropic magnification (1 + kappa), with radial
     arrows pointing outward (spin 0: no preferred direction)
  3. shear gamma = (gamma1, gamma2): a stretch along one axis and a squeeze along
     the other, the axis turning from the "plus" (gamma1) to the "cross" (gamma2)
     pattern and back (spin 2); a dashed circle keeps the original outline
Rendered with PIL (supersampled, then reduced), on the slide colour, so the GIF
needs no transparency. Loops; each cycle starts and ends on the undeformed galaxy.

Outputs (this directory): galaxy_distortion.gif, galaxy_distortion_last.png,
galaxy_shapes_vertical.png (the same still, tiles stacked, for the pipeline slide)
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import cached_fetch, skip_if_built

from PIL import Image, ImageDraw, ImageFilter

OUT_GIF, OUT_LAST = "galaxy_distortion.gif", "galaxy_distortion_last.png"
OUT_VERT = "galaxy_shapes_vertical.png"
skip_if_built(HERE, OUT_GIF, OUT_LAST, OUT_VERT)

SS = 2                          # supersampling
TILE, GAP, PAD = 340, 50, 8     # final pixels
BG = (250, 247, 240)            # slide colour
SKY = (10, 12, 22)
ARROW = (194, 86, 10)           # deck orange
GHOST = (235, 235, 245)
KAPPA, GAMMA = 0.28, 0.30
FPS, T = 18, 6.0

# the original is a 196-megapixel scan: fetch a 1920 px Commons rendition instead
M101 = ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/"
        "M101_hires_STScI-PRC2006-10a.jpg/1920px-M101_hires_STScI-PRC2006-10a.jpg")
raw = cached_fetch(HERE / ".cache", "m101_1920", M101)
src = Image.open(io.BytesIO(raw)).convert("RGB")
w, h = src.size
side = int(0.62 * min(w, h))
cx, cy = int(0.5 * w), int(0.52 * h)
src = src.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))
G = int(0.62 * TILE * SS)       # galaxy image size inside a tile
gal = src.resize((G, G), Image.LANCZOS)
# soft circular fade into the sky, so the square crop never shows
yy, xx = np.mgrid[0:G, 0:G]
r = np.hypot(xx - G / 2, yy - G / 2) / (G / 2)
alpha = np.clip((1.0 - r) / 0.28, 0, 1) ** 1.5
gal.putalpha(Image.fromarray((255 * alpha).astype(np.uint8)))


def warp(M):
    """Tile with the galaxy mapped by the 2x2 matrix M about the tile centre."""
    S = TILE * SS
    tile = Image.new("RGBA", (S, S), SKY + (255,))
    Minv = np.linalg.inv(M)
    c_out = np.array([S / 2, S / 2])
    c_in = np.array([G / 2, G / 2])
    # PIL AFFINE maps output (x, y) -> input: in = Minv @ (out - c_out) + c_in
    off = c_in - Minv @ c_out
    data = (Minv[0, 0], Minv[0, 1], off[0], Minv[1, 0], Minv[1, 1], off[1])
    g = gal.transform((S, S), Image.AFFINE, data, resample=Image.BICUBIC)
    tile.alpha_composite(g)
    return tile


def arrow(d, p, q, width):
    L = np.linalg.norm(q - p)
    v = (q - p) / (L + 1e-9)
    n = np.array([-v[1], v[0]])
    head = min(5.5 * width, 0.45 * L)        # the head never outgrows the shaft
    d.line([tuple(p), tuple(q - 0.5 * v * head)], fill=ARROW, width=width)
    d.polygon([tuple(q + 0.4 * v * head), tuple(q - v * head + 0.6 * n * head),
               tuple(q - v * head - 0.6 * n * head)], fill=ARROW)


def dashed_circle(d, c, rad, width):
    for k in range(40):
        if k % 2:
            continue
        a0, a1 = 2 * np.pi * k / 40, 2 * np.pi * (k + 1) / 40
        t = np.linspace(a0, a1, 6)
        pts = [(c[0] + rad * np.cos(x), c[1] + rad * np.sin(x)) for x in t]
        d.line(pts, fill=GHOST, width=width)


def rounded(tile):
    S = tile.size[0]
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, S - 1, S - 1], radius=28 * SS, fill=255)
    tile.putalpha(m)
    return tile


def frame(t):
    S = TILE * SS
    u = t / T
    # convergence: breathe in and out once
    kap = KAPPA * np.sin(np.pi * np.clip(u / 0.5, 0, 1)) if u < 0.5 else 0.0
    # shear: grow as plus, turn to cross, shrink
    if u < 0.2:
        g, phi = GAMMA * np.sin(0.5 * np.pi * u / 0.2), 0.0
    elif u < 0.75:
        g, phi = GAMMA, (np.pi / 2) * (u - 0.2) / 0.55     # 2*theta from 0 to 90 deg
    else:
        g, phi = GAMMA * np.cos(0.5 * np.pi * (u - 0.75) / 0.25), np.pi / 2
    g1, g2 = g * np.cos(phi), g * np.sin(phi)

    tiles = []
    # 1. reference
    tiles.append(rounded(warp(np.eye(2))))
    # 2. convergence
    t2 = warp((1 + kap) * np.eye(2))
    d2 = ImageDraw.Draw(t2)
    c = np.array([S / 2, S / 2])
    for k in range(8):
        a = 2 * np.pi * k / 8 + np.pi / 8
        v = np.array([np.cos(a), np.sin(a)])
        r0 = 0.30 * G
        if kap > 0.06:
            arrow(d2, c + v * r0 * (1 + kap), c + v * (r0 * (1 + kap) + 0.9 * kap * G), 4 * SS)
    tiles.append(rounded(t2))
    # 3. shear
    M = np.array([[1 + g1, g2], [g2, 1 - g1]])
    t3 = warp(M)
    d3 = ImageDraw.Draw(t3)
    dashed_circle(d3, c, 0.38 * G, 2 * SS)
    if g > 0.06:
        th = phi / 2                                           # stretch axis angle
        e = np.array([np.cos(th), -np.sin(th)])                # screen y points down
        f = np.array([np.sin(th), np.cos(th)])
        L = 0.40 * G
        for s in (1, -1):
            arrow(d3, c + s * e * L * (1 + 0.3 * g), c + s * e * L * (1 + 1.6 * g), 4 * SS)
            arrow(d3, c + s * f * L * (1 + 1.3 * g), c + s * f * L * (1 - 0.1 * g), 4 * SS)
    tiles.append(rounded(t3))

    W = 3 * S + 2 * GAP * SS + 2 * PAD * SS
    canvas = Image.new("RGBA", (W, S + 2 * PAD * SS), BG + (255,))
    for i, tl in enumerate(tiles):
        canvas.alpha_composite(tl, (PAD * SS + i * (S + GAP * SS), PAD * SS))
    return canvas.convert("RGB").resize((W // SS, (S + 2 * PAD * SS) // SS), Image.LANCZOS)


n = int(FPS * T)
frames = [frame(T * i / n) for i in range(n)]
# one shared palette (built from a frame showing both effects), so unchanged pixels
# stay identical from frame to frame and the GIF compresses them away
ref = frame(0.45 * T).quantize(colors=255, method=Image.Quantize.MEDIANCUT)
pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
pal[0].save(HERE / OUT_GIF, save_all=True, append_images=pal[1:], duration=int(1000 / FPS),
            optimize=True, disposal=1, loop=0)               # loops forever
last = frame(0.45 * T)                                     # both effects visible
last.save(HERE / OUT_LAST)
tiles = [last.crop((PAD + i * (TILE + GAP), PAD, PAD + i * (TILE + GAP) + TILE, PAD + TILE))
         for i in range(3)]
vert = Image.new("RGBA", (TILE, 3 * TILE + 2 * GAP // 2), (0, 0, 0, 0))
for i, tl in enumerate(tiles):
    vert.paste(tl, (0, i * (TILE + GAP // 2)))
vert.save(HERE / OUT_VERT)
print(f"wrote {OUT_GIF} ({(HERE / OUT_GIF).stat().st_size / 1e6:.1f} MB, {n} frames), {OUT_LAST}, {OUT_VERT}")
