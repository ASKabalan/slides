#!/usr/bin/env python3
# ENV: furax-cs
"""
Figures for the "Spherical Harmonic Transforms" slide: I, Q, U maps of the PySM
CMB component c1 (a lensed CMB realisation), their T, E, B harmonic coefficients
laid out as s2fft does, and the three auto-spectra.

  sht_map_{I,Q,U}.svg    Mollweide maps (nside 128, 100 GHz, uK_CMB, smoothed to
                         1.5 deg for display only), deck
                         blue-white-orange, transparent outside the sky
  sht_alm_{T,E,B}.svg    log10 |a_lm| in the s2fft layout: an L x (2L - 1) array,
                         row l, column m + L - 1 (m from -l to l, so the filled
                         part is a triangle); L = 64. Negative m from
                         a_{l,-m} = (-1)^m a*_{lm} (real fields)
  sht_cl_{TT,EE,BB}.svg  D_l = l(l+1) C_l / 2pi from the same a_lm, log y

Runs in the furax-cs environment (pysm3).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built

OUTS = [f"sht_map_{s}.svg" for s in "IQU"] + [f"sht_alm_{s}.svg" for s in "TEB"] \
    + [f"sht_cl_{s}.svg" for s in ("TT", "EE", "BB")]
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
from matplotlib.colors import LinearSegmentedColormap

NSIDE, L, FREQ = 128, 64, 100 * u.GHz
INK, GREY = "#2E2E2E", "#6E6E6E"
COLS = {"T": "#C2560A", "E": "#3B6FB6", "B": "#C0392B"}
DECK = LinearSegmentedColormap.from_list("deck", ["#3B6FB6", "#ffffff", "#C2560A"])
plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans"})

sky = pysm3.Sky(nside=NSIDE, preset_strings=["c1"])
iqu = sky.get_emission(FREQ).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(FREQ)).value

# ---------------------------------------------------------------- maps
proj = hp.projector.MollweideProj(xsize=900)
shown = hp.smoothing(iqu, fwhm=np.deg2rad(1.5), pol=False)   # display only
for name, m in zip("IQU", shown):
    img = proj.projmap(m, lambda x, y, z: hp.vec2pix(NSIDE, x, y, z))
    inside = np.isfinite(img) & (img > -1e30)
    lim = np.percentile(np.abs(img[inside]), 99)
    rgba = DECK(np.clip(0.5 + 0.5 * np.where(inside, img, 0) / lim, 0, 1))
    rgba[~inside, 3] = 0.0
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgba, origin="lower", extent=(-2, 2, -1, 1), interpolation="bilinear")
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(2 * np.cos(t), np.sin(t), color="#8A8A96", lw=1.0)
    ax.set_xlim(-2.02, 2.02)
    ax.set_ylim(-1.02, 1.02)
    ax.axis("off")
    fig.savefig(HERE / f"sht_map_{name}.svg", transparent=True)
    plt.close(fig)

# ---------------------------------------------------------------- a_lm, s2fft layout
alms = hp.map2alm(iqu, lmax=L - 1, pol=True)          # T, E, B
ell_idx, m_idx = hp.Alm.getlm(L - 1)


def to_s2fft(alm):
    flm = np.full((L, 2 * L - 1), np.nan, dtype=complex)
    flm[ell_idx, L - 1 + m_idx] = alm
    neg = m_idx > 0
    flm[ell_idx[neg], L - 1 - m_idx[neg]] = (-1.0) ** m_idx[neg] * np.conj(alm[neg])
    return flm


for name, alm in zip("TEB", alms):
    a = np.abs(to_s2fft(alm))
    a[:2] = np.nan                                     # monopole and dipole removed
    la = np.log10(a)
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.imshow(la, cmap="magma", origin="upper", aspect="auto", interpolation="nearest",
              vmin=np.nanpercentile(la, 5), vmax=np.nanpercentile(la, 99.5),
              extent=(-(L - 0.5), L - 0.5, L - 0.5, -0.5))
    ax.set_xlabel("m", color=INK, fontsize=12, labelpad=1)
    ax.set_ylabel("ℓ", color=INK, fontsize=12, labelpad=1)
    ax.set_xticks([-(L - 1), 0, L - 1])
    ax.set_yticks([0, L - 1])
    ax.tick_params(labelsize=9, colors=GREY, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(HERE / f"sht_alm_{name}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- spectra
cls = hp.alm2cl(alms)                                  # TT, EE, BB, TE, EB, TB
ell = np.arange(L)
for name, cl in zip(("TT", "EE", "BB"), cls[:3]):
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.semilogy(ell[2:], dl[2:], color=COLS[name[0]], lw=2.2)
    ax.set_xlabel("ℓ", color=INK, fontsize=12, labelpad=1)
    ax.set_ylabel(rf"$D_\ell^{{{name}}}$", color=INK, fontsize=12, labelpad=1)
    ax.set_xticks([2, 20, 40, 60])
    ax.tick_params(labelsize=9, colors=GREY, length=2, which="both")
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GREY)
    ax.set_facecolor("none")
    fig.savefig(HERE / f"sht_cl_{name}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

print("wrote", ", ".join(OUTS))
