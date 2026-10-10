#!/usr/bin/env python3
# ENV: shared
"""
The shear thumbnail the inverse-problem diagram embeds: the second shear component of my forward
model, seen on the sphere, with the DES Y3 footprint opaque over a faint full sky.

The map is the full-sky gamma_2 of the masked-shear experiment in jax-fli (CosmoGrid convergence,
one tomographic bin, Kaiser-Squires to shear, nside 128), stored with the DES Y3 mask in its
data/masked_shear.npz. Regenerate that file with

    cd /home/wassim/Projects/NBody/jax-fli/docs/5-experiments/08-masked-shear
    JAX_PLATFORMS=cpu uv run --no-sync python build.py

Orthographic view centred on the footprint, magma as in the experiment's figures: the whole sky at
30 % opacity, the footprint at full opacity with a thin dark outline.

Output: .cache/shear_thumb.png, read by inverse_problem.tex
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import ortho_frame, skip_if_built

SRC = Path("/home/wassim/Projects/NBody/jax-fli/docs/5-experiments/08-masked-shear/data/masked_shear.npz")
THUMB = HERE / ".cache" / "shear_thumb.png"
PX = 900
SKY_ALPHA = 0.30

skip_if_built(HERE, ".cache/shear_thumb.png")
if not SRC.exists():
    sys.exit(f"missing {SRC}; run the masked-shear experiment first")

import healpy as hp
from PIL import Image, ImageChops, ImageFilter

d = np.load(SRC)
g2, des = d["g2_full"], d["des_binary"] > 0
nside = hp.npix2nside(len(g2))

# centre of the footprint
vx, vy, vz = hp.pix2vec(nside, np.where(des)[0])
lon, lat = hp.vec2ang(np.array([vx.mean(), vy.mean(), vz.mean()]), lonlat=True)
lon, lat = float(np.atleast_1d(lon)[0]), float(np.atleast_1d(lat)[0])

lo, hi = np.percentile(g2[des], [2, 98])
kw = dict(cmap="magma", min=lo, max=hi)
sky = ortho_frame(g2, lon, lat, PX, **kw)
foot = ortho_frame(np.where(des, g2, hp.UNSEEN), lon, lat, PX, badcolor=(0.0,) * 4, **kw)

# the sky faint, the footprint opaque on top, a dark outline round the footprint and the disc
sky.putalpha(sky.getchannel("A").point(lambda a: round(a * SKY_ALPHA)))
out = Image.new("RGBA", sky.size, (0, 0, 0, 0))
out.alpha_composite(sky)
out.alpha_composite(foot)
ink = Image.new("RGBA", sky.size, (46, 46, 46, 255))
for a, width in ((foot.getchannel("A"), 5), (sky.getchannel("A").point(lambda v: 255 if v else 0), 3)):
    a = a.point(lambda v: 255 if v > 128 else 0)
    edge = ImageChops.subtract(a.filter(ImageFilter.MaxFilter(width)), a.filter(ImageFilter.MinFilter(width)))
    out.paste(ink, (0, 0), edge)
out = out.crop(out.getchannel("A").getbbox())
THUMB.parent.mkdir(parents=True, exist_ok=True)
out.save(THUMB)
print(f"wrote {THUMB.relative_to(HERE)} {out.size}, centred on (lon, lat) = ({lon:.1f}, {lat:.1f})")
