#!/usr/bin/env python3
# ENV: shared
"""
Stokes Q and U maps of one flat-sky patch, before and after rotating the
reference frame by 45 degrees, for the "Q and U Stokes parameters" slide.

A Gaussian E-mode-only polarisation field is drawn from the CAMB EE spectrum
(Planck 2018 cosmology), smoothed by a 40' beam, on a 1024^2 flat-sky grid, Q = E cos 2phi_l,
U = E sin 2phi_l in Fourier space. The slide shows the central square.

When the frame turns by psi, Q' = Q cos 2psi + U sin 2psi and
U' = -Q sin 2psi + U cos 2psi: at 45 degrees Q' = U and U' = -Q. On the slide the
tiles turn by 45 degrees (CSS, clockwise); the rotated-frame maps are pre-rotated
by 45 degrees the other way, so the sky itself stays put while the frame turns.

Outputs (this directory), one shared diverging colour scale:
  qu_Q.png, qu_U.png     the original frame
  qu_Qp.png, qu_Up.png   the rotated frame: Q' = U, U' = -Q (pre-rotated)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

OUTS = ("qu_Q.png", "qu_U.png", "qu_Qp.png", "qu_Up.png")
skip_if_built(HERE, *OUTS)

import camb
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

N, SIDE_DEG, CROP, OUT_PX = 1024, 20.0, 512, 600
SEED = 7
BEAM_ARCMIN = 40.0

pars = camb.set_params(H0=67.4, ombh2=0.0224, omch2=0.120, As=2.1e-9, ns=0.965,
                       tau=0.054, lmax=4000)
cl_ee = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK",
                                                     raw_cl=True)["unlensed_scalar"][:, 1]

dx = np.deg2rad(SIDE_DEG) / N
lx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
LX, LY = np.meshgrid(lx, lx)
ell = np.hypot(LX, LY)
phi = np.arctan2(LY, LX)
cl = np.interp(ell, np.arange(len(cl_ee)), cl_ee, right=0.0)
cl *= np.exp(-ell**2 * (np.deg2rad(BEAM_ARCMIN / 60) / 2.355) ** 2)  # smooth: readable blobs
rng = np.random.default_rng(SEED)
e_k = np.fft.fft2(rng.standard_normal((N, N))) * np.sqrt(cl / dx**2)
q = np.fft.ifft2(e_k * np.cos(2 * phi)).real
u = np.fft.ifft2(e_k * np.sin(2 * phi)).real

cmap = LinearSegmentedColormap.from_list("deck", ["#3B6FB6", "#ffffff", "#C2560A"])
lim = np.percentile(np.abs(np.concatenate([q.ravel(), u.ravel()])), 99)


def to_image(field: np.ndarray) -> Image.Image:
    rgb = (cmap(np.clip(0.5 + 0.5 * field / lim, 0, 1))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def crop(im: Image.Image) -> Image.Image:
    a = (N - CROP) // 2
    return im.crop((a, a, a + CROP, a + CROP)).resize((OUT_PX, OUT_PX), Image.LANCZOS)


for name, field, turn in (("qu_Q.png", q, 0), ("qu_U.png", u, 0),
                          ("qu_Qp.png", u, 45), ("qu_Up.png", -q, 45)):
    im = to_image(field)
    if turn:
        im = im.rotate(turn, resample=Image.BICUBIC)   # counter-clockwise: undoes the CSS turn
    crop(im).save(HERE / name)
    print(f"wrote {name}")
