#!/usr/bin/env python3
# ENV: furax-cs
"""
From the sky to multi-frequency data, for the opening slide of the component-separation part.

sky_stack_cmb.png    the CMB alone, as a tilted disc in the style of the SciPol pipeline graphic
sky_stack_full.png   the same disc with the lensing convergence, thermal dust and synchrotron
                     stacked beneath it, each labelled (same canvas, so the two swap in place)
freq_cards.png       the summed c1d1s1 sky in temperature at LiteBIRD's lowest, middle and
                     highest bands (40, 119 and 402 GHz), as offset cards

Sky maps: PySM c1, d1 and s1 through furax's loader, at nside 64 (the resolution of the fit).
The lensing disc is the Born convergence of the weak-lensing section
(3_observation/weaklensing/06_kaiser_squires/.cache). Band centres: furax-cs instruments.yaml
(LiteBIRD Collaboration 2023, Table 13).
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, skip_if_built, slide_style

OUTS = ["sky_stack_cmb.png", "sky_stack_full.png", "freq_cards.png"]
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw, ImageFont

CACHE = HERE.parent / ".cache"
NSIDE = 64
KAPPA = ROOT / "3_observation/weaklensing/06_kaiser_squires/.cache/kappa_born.parquet"
SHOWN = (40, 119, 402)


# ---------------------------------------------------------------- maps
def emission(tag, freq):
    import pysm3.units as u
    from furax._instruments.sky import get_sky

    em = get_sky(NSIDE, tag).get_emission(freq * u.GHz)
    return np.asarray(em.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq * u.GHz)))


def maps():
    npz = CACHE / f"sky_to_maps_{NSIDE}.npz"
    if npz.exists():
        return dict(np.load(npz))
    import pyarrow.parquet as pq

    out = {
        "cmb": emission("c1", 140.0)[0],
        "dust": emission("d1", 353.0)[0],
        "sync": emission("s1", 30.0)[0],
    }
    kappa = np.asarray(pq.read_table(KAPPA, columns=["array"]).column("array").to_pylist()[0][0])
    out["kappa"] = hp.ud_grade(kappa, NSIDE)
    for f in SHOWN:
        out[f"sky{f}"] = emission("c1d1s1", float(f))[0]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


M = maps()


# ---------------------------------------------------------------- tilted discs
def ortho(m, lat=22.0, n=420):
    """The visible hemisphere of a map, seen from Galactic latitude `lat` above the centre."""
    b = np.deg2rad(lat)
    fwd = np.array([np.cos(b), 0.0, np.sin(b)])
    right = np.array([0.0, 1.0, 0.0])
    up = np.cross(fwd, right)
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(1, -1, n))
    r2 = u ** 2 + v ** 2
    inside = r2 < 1
    w = np.sqrt(np.clip(1 - r2, 0, 1))
    vec = w[..., None] * fwd + u[..., None] * right + v[..., None] * up
    pix = hp.vec2pix(NSIDE, vec[..., 0], vec[..., 1], vec[..., 2])
    img = np.where(inside, m[pix], np.nan)
    return img


LAYERS = [   # top to bottom; values mapped through (transform, colormap, percentiles)
    ("CMB", M["cmb"], lambda x: x, "RdYlBu_r", (1, 99)),
    # the N-body lattice shows at small scales: keep the large ones
    ("Gravitational lensing", hp.smoothing(M["kappa"], fwhm=np.deg2rad(2.5)), lambda x: x,
     "Purples", (2, 99.5)),
    ("Thermal dust", M["dust"], np.log10, "YlOrRd", (5, 99.5)),
    ("Synchrotron", M["sync"], np.log10, "GnBu", (5, 99.5)),
]
SQUASH, GAP, THICK = 0.36, 0.52, 0.07


def stack(n_shown, out):
    slide_style(scale=1.2)
    fig = plt.figure(figsize=(5.9, 4.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-2.75, 1.08)
    ax.set_ylim(-(len(LAYERS) - 1) * GAP - SQUASH - 0.12, SQUASH + 0.08)
    ax.set_axis_off()
    for i in reversed(range(n_shown)):
        name, m, tf, cmap, pct = LAYERS[i]
        y0 = -i * GAP
        img = tf(ortho(m))
        lo, hi = np.nanpercentile(img, pct)
        edge = plt.get_cmap(cmap)(0.85)
        # the plate's thickness, then its face
        ax.add_patch(Ellipse((0, y0 - THICK), 2, 2 * SQUASH, color=edge, alpha=0.9, zorder=2 * (10 - i)))
        face = Ellipse((0, y0), 2, 2 * SQUASH, fc="none", ec=INK, lw=0.9, zorder=2 * (10 - i) + 1)
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, extent=(-1, 1, y0 - SQUASH, y0 + SQUASH),
                       interpolation="bilinear", zorder=2 * (10 - i) + 1)
        ax.add_patch(face)
        im.set_clip_path(face)
        ax.text(-1.1, y0, name, ha="right", va="center", color=INK, fontsize=17)
    ax.set_aspect("auto")
    fig.savefig(HERE / out, transparent=True)
    plt.close(fig)
    print(f"wrote {out}")


stack(1, "sky_stack_cmb.png")
stack(len(LAYERS), "sky_stack_full.png")


# ---------------------------------------------------------------- frequency cards
CARD_W, PAD, STEP = 620, 14, (70, 62)


def mollweide(m):
    lo, hi = np.percentile(m, [2, 98])
    fig = plt.figure(figsize=(6.2, 3.3), dpi=100)
    hp.mollview(np.clip(m, lo, hi), fig=fig.number, title="", cbar=False, cmap="RdYlBu_r",
                min=lo, max=hi, notext=True, margins=(0, 0, 0, 0), bgcolor=(0.0,) * 4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=100)
    plt.close(fig)
    im = Image.open(buf).convert("RGBA")
    return im.crop(im.getchannel("A").getbbox())


def card(im, label):
    im = im.resize((CARD_W - 2 * PAD, round(im.height * (CARD_W - 2 * PAD) / im.width)))
    c = Image.new("RGBA", (CARD_W, im.height + 2 * PAD + 34), (255, 255, 255, 255))
    d = ImageDraw.Draw(c)
    d.rounded_rectangle((0, 0, c.width - 1, c.height - 1), radius=14, outline=(200, 194, 184), width=3)
    d.text((PAD + 4, PAD - 4), label, fill=INK, font=ImageFont.truetype("DejaVuSans-Bold.ttf", 26))
    c.alpha_composite(im, (PAD, PAD + 30))
    mask = Image.new("L", c.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, c.width - 1, c.height - 1), radius=14, fill=255)
    c.putalpha(mask)
    return c


order = (40, 402, 119)                                   # back to front, CMB-dominated in front
cards = [card(mollweide(M[f"sky{f}"]), f"{f} GHz") for f in order]
W = cards[0].width + STEP[0] * (len(cards) - 1)
H = cards[0].height + STEP[1] * (len(cards) - 1)
sheet = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for i, c in enumerate(cards):
    sheet.alpha_composite(c, (STEP[0] * (len(cards) - 1 - i), STEP[1] * i))
sheet.save(HERE / "freq_cards.png")
print("wrote freq_cards.png")
