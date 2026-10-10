#!/usr/bin/env python3
# ENV: shared
"""
Figures for the "Spherical Harmonic Transforms" slide: the Planck 2018 SMICA
CMB maps (I, Q, U; the inpainted columns, so the Galactic plane is filled),
their T, E, B harmonic coefficients laid out as s2fft does, and the three
LCDM auto-spectra as C_l.

  sht_map_{I,Q,U}.svg    Mollweide maps (I/Q/U_STOKES_INP, degraded to nside 128,
                         smoothed to 1.5 deg for display only), deck
                         blue-white-orange, transparent outside the sky, with a
                         colour bar in uK (symmetric, the 99th percentile rounded)
  sht_alm_{T,E,B}.svg    |a_lm|^2 in uK^2 (logarithmic colour scale, with its colour bar) in the
                         s2fft layout: an L x (2L - 1) array,
                         row l, column m + L - 1 (m from -l to l, so the filled
                         part is a triangle); L = 64. Negative m from
                         a_{l,-m} = (-1)^m a*_{lm} (real fields)
  sht_cl_{TT,EE,BB}.svg  the Planck 2018 LCDM C_l (CAMB, the spectra of the next
                         slide: 3_observation/cmb/07_power_spectrum/.cache/camb_cl_all.npz), log-log;
                         BB = lensing + primordial at r = 0.01, so both bumps show; uK^2

"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import PLANCK_SMICA, PLANCK_SMICA_URL, cached_file, skip_if_built

OUTS = [f"sht_map_{s}.svg" for s in "IQU"] + [f"sht_alm_{s}.svg" for s in "TEB"] \
    + [f"sht_cl_{s}.svg" for s in ("TT", "EE", "BB")]
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

NSIDE, L = 128, 64
INK, GREY = "#2E2E2E", "#6E6E6E"
COLS = {"T": "#C2560A", "E": "#3B6FB6", "B": "#C0392B"}
DECK = LinearSegmentedColormap.from_list("deck", ["#3B6FB6", "#ffffff", "#C2560A"])
plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans"})

smica = cached_file((ROOT / "3_observation/cmb") / ".cache", PLANCK_SMICA, PLANCK_SMICA_URL)
iqu = 1e6 * np.array(hp.ud_grade(hp.read_map(smica, field=(5, 6, 7)), NSIDE))   # inpainted, uK_CMB

# ---------------------------------------------------------------- maps
proj = hp.projector.MollweideProj(xsize=900)
shown = hp.smoothing(iqu, fwhm=np.deg2rad(1.5), pol=False)   # display only
for name, m in zip("IQU", shown):
    img = proj.projmap(m, lambda x, y, z: hp.vec2pix(NSIDE, x, y, z))
    inside = np.isfinite(img) & (img > -1e30)
    lim = float(f"{np.percentile(np.abs(img[inside]), 99):.1g}")      # one significant digit
    rgba = DECK(np.clip(0.5 + 0.5 * np.where(inside, img, 0) / lim, 0, 1))
    rgba[~inside, 3] = 0.0
    # the map on top, a thin colour bar in uK under it
    fig = plt.figure(figsize=(4, 2.8))
    ax = fig.add_axes((0, 0.286, 1, 0.714))
    cax = fig.add_axes((0.17, 0.165, 0.62, 0.06))
    cb = fig.colorbar(ScalarMappable(Normalize(-lim, lim), DECK), cax=cax, orientation="horizontal")
    cb.set_ticks([-lim, 0, lim], labels=[f"−{lim:g}", "0", f"{lim:g}"])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=21, colors=GREY, length=2, pad=1)
    cax.text(1.04, 0.5, "µK", transform=cax.transAxes, ha="left", va="center", fontsize=23, color=INK)
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
    p2 = np.abs(to_s2fft(alm)) ** 2
    p2[:2] = np.nan                                    # monopole and dipole removed
    fig, ax = plt.subplots(figsize=(4, 2.2))
    # |a_lm|^2 itself, on a logarithmic colour scale (for display only)
    im = ax.imshow(p2, cmap="magma", origin="upper", aspect="auto", interpolation="nearest",
                   norm=LogNorm(vmin=np.nanpercentile(p2, 5), vmax=np.nanpercentile(p2, 99.5)),
                   extent=(-(L - 0.5), L - 0.5, L - 0.5, -0.5))
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=18, colors=GREY, length=2, which="both")
    cb.ax.set_title("µK²", fontsize=22, color=INK, pad=6)
    ax.set_xlabel("m", color=INK, fontsize=24, labelpad=0)
    ax.set_ylabel("ℓ", color=INK, fontsize=24, labelpad=2)
    ax.set_xticks([-(L - 1), 0, L - 1])
    ax.set_yticks([0, L - 1])
    ax.tick_params(labelsize=17, colors=GREY, length=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(HERE / f"sht_alm_{name}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- spectra
# the LCDM prediction of the next slide (CAMB, Planck 2018 cosmology, D_l in uK^2 with the tensors
# at r = 0.01), back to C_l = 2 pi D_l / (l (l + 1))
th = np.load(ROOT / "3_observation/cmb/07_power_spectrum/.cache/camb_cl_all.npz")
ell = th["ell"][2:2001]
to_cl = 2 * np.pi / (ell * (ell + 1.0))
spectra = {"TT": th["TT"], "EE": th["EE"], "BB": th["BB_lens"] + th["BB_tensor"]}
for name, dl in spectra.items():
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.loglog(ell, dl[2:2001] * to_cl, color=COLS[name[0]], lw=2.6)
    ax.set_xlabel("ℓ", color=INK, fontsize=24, labelpad=0)
    ax.set_ylabel(rf"$C_\ell^{{{name}}}$", color=INK, fontsize=24, labelpad=2)
    ax.set_xlim(2, 2000)
    ax.set_xticks([2, 10, 100, 1000], ["2", "10", "100", "1000"])
    ax.text(0.0, 1.03, "µK²", transform=ax.transAxes, ha="center", va="bottom", fontsize=22, color=INK)
    ax.tick_params(labelsize=17, colors=GREY, length=3, which="both")
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GREY)
    ax.set_facecolor("none")
    fig.savefig(HERE / f"sht_cl_{name}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

print("wrote", ", ".join(OUTS))
