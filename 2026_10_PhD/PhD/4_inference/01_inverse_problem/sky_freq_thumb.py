#!/usr/bin/env python3
# ENV: furax-cs
"""
The microwave sky at three frequencies, as a stack of offset cards, for the inverse-problem
diagram: what a CMB experiment measures is the sky signal band by band, not the CMB itself.

Temperature maps of the c1d1s1 sky (PySM through furax's loader, as in
../../5_contribution/compsep/01_sky_to_maps/sky_to_maps.py) at 40, 140 and 402 GHz, two ends and a
middle band of the LiteBIRD range: synchrotron dominates the first, the CMB the second, dust the
third.

Output: .cache/sky_freq_thumb.png (embedded by inverse_problem.tex)
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import INK, force_regen

OUT = HERE / ".cache" / "sky_freq_thumb.png"
if OUT.exists() and not force_regen():
    print(f"{OUT.name} already exists. Set FORCE_REGEN=1 to regenerate it.")
    sys.exit(0)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

NSIDE = 64
FREQS = (40.0, 402.0, 140.0)                  # back to front: the CMB-dominated band on top
CARD_W, PAD, STEP = 620, 14, (70, 62)          # card width, inner padding, offset between cards


def sky(freq):
    import pysm3.units as u
    from furax._instruments.sky import get_sky

    em = get_sky(NSIDE, "c1d1s1").get_emission(freq * u.GHz)
    em = em.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq * u.GHz))
    return np.asarray(em[0])


def mollweide(m):
    lo, hi = np.percentile(m, [2, 98])
    fig = plt.figure(figsize=(6.2, 3.3), dpi=100)
    hp.mollview(np.clip(m, lo, hi), fig=fig.number, title="", cbar=False, cmap="RdYlBu_r",
                min=lo, max=hi, notext=True, margins=(0, 0, 0, 0))
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
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    d.text((PAD + 4, PAD - 4), label, fill=INK, font=font)
    c.alpha_composite(im, (PAD, PAD + 30))
    # rounded corners
    mask = Image.new("L", c.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, c.width - 1, c.height - 1), radius=14, fill=255)
    c.putalpha(mask)
    return c


cards = [card(mollweide(sky(f)), f"{f:g} GHz") for f in FREQS]
W = cards[0].width + STEP[0] * (len(cards) - 1)
H = cards[0].height + STEP[1] * (len(cards) - 1)
out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for i, c in enumerate(cards):
    out.alpha_composite(c, (STEP[0] * (len(cards) - 1 - i), STEP[1] * i))
OUT.parent.mkdir(exist_ok=True)
out.save(OUT)
print(f"wrote {OUT.name} {out.size}")
