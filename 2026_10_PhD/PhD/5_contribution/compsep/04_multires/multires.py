#!/usr/bin/env python3
# ENV: furax-cs
"""
Multi-resolution patches, the partition used by the LiteBIRD Collaboration (2023, PTEP).

nside_{4,16,32}.png   the sky cut into HEALPix pixels at N_side 4, 16 and 32 (12 N_side^2 patches)
regions.png           LiteBIRD's three analysed 20 % regions, low, mid and high Galactic latitude;
                      the Galactic plane is masked
multires_td.png       the dust-temperature partition of the paper: N_side 8, 4 and 0 (a single
                      patch) in the low, mid and high regions

All at nside 64, drawn with _compsep.moll_png (transparent Mollweide, masked sky in grey).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from _common import KW, KW2
from _common import skip_if_built
from _compsep import NSIDE, moll_png, regions_by_latitude, shuffled

OUTS = ["nside_4.png", "nside_16.png", "nside_32.png", "regions.png", "multires_td.png"]
skip_if_built(HERE, *OUTS)

import healpy as hp
from matplotlib.colors import ListedColormap

npix = hp.nside2npix(NSIDE)
ipix = np.arange(npix)


def superpixels(nside_out):
    """Label of the N_side `nside_out` pixel containing each nside-64 pixel (0: one patch)."""
    if nside_out == 0:
        return np.zeros(npix, dtype=int)
    return hp.ang2pix(nside_out, *hp.pix2ang(NSIDE, ipix))


for n in (4, 16, 32):
    moll_png(shuffled(superpixels(n), seed=n), HERE / f"nside_{n}.png", vmin=0, vmax=1)
    print(f"wrote nside_{n}.png ({12 * n * n} patches)")

regions, order = regions_by_latitude()
print("low, mid, high =", order)
UNSEEN = hp.UNSEEN
reg = np.full(npix, UNSEEN)
for k, name in enumerate(("low", "mid", "high")):
    reg[regions[name]] = k
moll_png(reg, HERE / "regions.png", cmap=ListedColormap([KW, KW2, "#3B6FB6"]), vmin=-0.5, vmax=2.5)
print("wrote regions.png")

# N_side 8 / 4 / 0 for the dust temperature, as in Table 11 of the PTEP paper
td = np.full(npix, UNSEEN)
offset = 0
for name, n in (("low", 8), ("mid", 4), ("high", 0)):
    lab = superpixels(n)[regions[name]]
    uniq, inv = np.unique(lab, return_inverse=True)
    td[regions[name]] = inv + offset
    offset += len(uniq)
moll_png(shuffled(td, seed=8), HERE / "multires_td.png", vmin=0, vmax=1)
print(f"wrote multires_td.png ({offset} patches)")
