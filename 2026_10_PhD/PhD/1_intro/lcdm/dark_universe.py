#!/usr/bin/env python3
# ENV: shared
"""
The two classic pieces of evidence for the dark Universe, from the papers that made them.

sn_hubble.svg        Perlmutter et al. (1999, ApJ 517, 565), Fig. 1 redrawn from the paper's own
                     Tables 1 and 2: effective B magnitude (width-luminosity corrected) against
                     redshift for the 18 Calan/Tololo SNe Ia (open) and the 42 SCP SNe Ia
                     (filled), measurement errors only. The flat model with dark energy
                     (Omega_M, Omega_Lambda) = (0.28, 0.72), the paper's flat best fit, against
                     matter only (1, 0); the magnitude zero point (script M_B) is fitted on the
                     18 low-redshift SNe, where every model agrees, so only the shape at high z
                     carries information. Nobel Prize in Physics 2011.
rubin1970_fig9.png   Rubin & Ford (1970, ApJ 159, 379), Fig. 9: the rotation curve of M31 (OB
                     associations, NE and SW), nearly flat out to 24 kpc instead of falling off.
                     Cropped from the ADS scan of page 390 (the line art is a 600 ppi bitmap), ink
                     in the deck's text colour on a transparent background.

Data:
  arXiv source of astro-ph/9812133 (deluxetables "SCP SNe Ia Data" and "Calan Tololo SNe Ia Data":
    z in column 2, m_B^effective / m_B^corr in column 9, its uncertainty in column 10)
  articles.adsabs.harvard.edu/pdf/1970ApJ...159..379R, page 15 of the PDF

Output (this directory): sn_hubble.svg, rubin1970_fig9.png
"""

import io
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, INK, KW, cached_fetch, skip_if_built, slide_style

OUT_SN, OUT_ROT = "sn_hubble.svg", "rubin1970_fig9.png"
skip_if_built(HERE, OUT_SN, OUT_ROT)
slide_style(scale=1.5)
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.integrate import cumulative_trapezoid

CACHE = HERE.parent / ".cache"
PERLMUTTER_URL = "https://arxiv.org/e-print/astro-ph/9812133"
RUBIN_URL = "https://articles.adsabs.harvard.edu/pdf/1970ApJ...159..379R"
FIGSIZE = (5.0, 3.7)
C_KMS = 299792.458

# --------------------------------------------------------------------------- #
# Perlmutter et al. (1999), Fig. 1
# --------------------------------------------------------------------------- #

with tarfile.open(fileobj=io.BytesIO(cached_fetch(CACHE, "perlmutter1999_src", PERLMUTTER_URL))) as tf:
    tex = tf.extractfile("40sneemulate.tex").read().decode("latin-1")


def table(num):
    """(z, m_B effective, sigma) of the rows of deluxetable number num."""
    start = tex.index(f"\\tablenum{{{num}}}")
    body = tex[tex.index("\\startdata", start):tex.index("\\end{deluxetable}", start)]
    rows = []
    for line in body.splitlines():
        cols = [c.strip() for c in line.split("&")]
        if len(cols) >= 10 and re.match(r"^\d", cols[1]):
            rows.append([float(cols[k].replace("$-$", "-")) for k in (1, 8, 9)])
    return np.array(rows)


scp, ct = table(1), table(2)
if (len(scp), len(ct)) != (42, 18):
    sys.exit(f"parsed {len(scp)} SCP and {len(ct)} Calan/Tololo SNe, expected 42 and 18")

zg = np.geomspace(1e-3, 1.05, 3000)


def lum_dist(om, ol):
    """H0 d_L / c on zg, any curvature."""
    ok = 1 - om - ol
    z = np.concatenate([[0.0], zg])
    chi = cumulative_trapezoid(1 / np.sqrt(om * (1 + z) ** 3 + ok * (1 + z) ** 2 + ol), z)
    if abs(ok) < 1e-9:
        dm = chi
    elif ok > 0:
        dm = np.sinh(np.sqrt(ok) * chi) / np.sqrt(ok)
    else:
        dm = np.sin(np.sqrt(-ok) * chi) / np.sqrt(-ok)
    return (1 + zg) * dm


def mag(om, ol):
    """m_B effective minus the zero point: 5 log10 of the Hubble-constant-free distance."""
    return 5 * np.log10(C_KMS * lum_dist(om, ol))


m_de, m_matter = mag(0.28, 0.72), mag(1.0, 0.0)
w = 1 / ct[:, 2] ** 2
script_m = np.sum(w * (ct[:, 1] - np.interp(ct[:, 0], zg, m_de))) / np.sum(w)
print(f"script M_B fitted on the low-z SNe: {script_m:.2f} (the paper: -3.32 +- 0.05)")

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(zg, script_m + m_matter, color=BLUE, ls="--", lw=2.4, zorder=2)
ax.plot(zg, script_m + m_de, color=KW, lw=3.0, zorder=3)
for data, face in ((ct, "white"), (scp, INK)):
    ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt="o", ms=5.5, color=INK, mfc=face,
                mew=1.4, elinewidth=1.3, capsize=0, zorder=4)

ax.text(0.0115, 24.6, "accelerating", color=KW, fontweight="bold", fontsize=17)
ax.text(0.0115, 23.45, "with dark energy", color=KW, fontsize=14)
ax.text(0.95, 16.0, "matter only", color=BLUE, fontweight="bold", fontsize=17, ha="right")
ax.text(0.95, 14.85, "no dark energy", color=BLUE, fontsize=14, ha="right")

ax.set_xscale("log")
ax.set_xlim(0.01, 1.0)
ax.set_ylim(14, 25.6)
ax.set_xticks([0.01, 0.1, 1.0], ["0.01", "0.1", "1"])
ax.set_yticks([15, 20, 25])
ax.set_xlabel("redshift $z$")
ax.set_ylabel("magnitude $m_B$")
fig.savefig(HERE / OUT_SN, transparent=True)
plt.close(fig)
print(f"wrote {OUT_SN} ({len(ct)} + {len(scp)} SNe)")

# --------------------------------------------------------------------------- #
# Rubin & Ford (1970), Fig. 9
# --------------------------------------------------------------------------- #

PAGE, DPI = 15, 600
BOX = (1060, 1750, 4085, 3955)         # Fig. 9 with its axis titles, in 600 dpi page pixels
WIDTH = 1400

pdf = cached_fetch(CACHE, "rubin1970_scan", RUBIN_URL)
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "r.pdf").write_bytes(pdf)
    subprocess.run(["pdftoppm", "-f", str(PAGE), "-l", str(PAGE), "-r", str(DPI), "-gray", "-png",
                    "r.pdf", "p"], cwd=tmp, check=True)
    page = Image.open(next(Path(tmp).glob("p*.png"))).convert("L")
# the scan's hairlines vanish at slide size: thicken them (a 5-pixel max filter at 600 dpi) before
# the downsampling, and darken the anti-aliased edges
page = page.crop(BOX).filter(ImageFilter.MinFilter(5))
ink = (1.0 - np.asarray(page, dtype=float) / 255.0) ** 0.7     # 1 where the line art is
img = Image.fromarray((ink * 255).astype(np.uint8))
img = img.resize((WIDTH, round(WIDTH * img.height / img.width)), Image.LANCZOS)
rgb = tuple(int(INK[i:i + 2], 16) for i in (1, 3, 5))
out = Image.new("RGBA", img.size, rgb + (0,))
out.putalpha(img)
out.save(HERE / OUT_ROT, optimize=True)
print(f"wrote {OUT_ROT} {out.size[0]}x{out.size[1]}")
