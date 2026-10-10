#!/usr/bin/env python3
# ENV: shared
"""
Polarised Galactic foregrounds against the primordial B modes, for the SED slide of
Contribution 1. A slide version of figures/chap3/fig_foregrounds.py in the thesis.

Spectral energy distributions in thermodynamic (CMB) units at a fixed angular scale, ell = 80:
the CMB is flat by construction, synchrotron falls and thermal dust rises, with a foreground
minimum near 70 GHz. Foreground amplitudes: BICEP/Keck fiducial values at the pivot frequencies,
A_d(1 %) = 4.7 uK^2 at 353 GHz and A_s(1 %) = 1.0 uK^2 at 23 GHz, with beta_d = 1.55,
T_d = 19.6 K, beta_s = -3.1 (BICEP/Keck XIII 2021; Planck 2018 IV). The primordial B-mode
amplitude at ell = 80 comes from CAMB (Planck 2018 cosmology) for r = 1e-2 and 1e-3.

Output (this directory): foreground_sed.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, KW2, skip_if_built, slide_style

OUT = "foreground_sed.svg"
skip_if_built(HERE, OUT)

h_pl, k_B, T_CMB = 6.62607015e-34, 1.380649e-23, 2.7255
SYNC, DUST = "#3b6fb6", "#d68910"      # as on the spectral-model slide


def _x(nu, T):
    return h_pl * nu * 1e9 / (k_B * T)


def dBdT(nu):
    x = _x(nu, T_CMB)
    return nu**4 * np.exp(x) / np.expm1(x) ** 2


def sed_sync(nu, beta_s=-3.1):
    return nu ** (beta_s + 2.0) / dBdT(nu)


def sed_dust(nu, beta_d=1.55, T_d=19.6):
    return nu**beta_d * nu**3 / np.expm1(_x(nu, T_d)) / dBdT(nu)


A_D, A_S = 4.7, 1.0

import camb

pars = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200, ns=0.9649, As=2.1e-9,
                       tau=0.054, r=1.0)
pars.WantTensors = True
pars.set_for_lmax(500, lens_potential_accuracy=1)
tensor = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=False)["tensor"]
b80_r1 = tensor[80, 2]                 # D_ell^BB at ell = 80 for r = 1

slide_style(scale=1.4)
import matplotlib.pyplot as plt

nu = np.logspace(np.log10(10.0), np.log10(500.0), 500)
sync = np.sqrt(A_S) * sed_sync(nu) / sed_sync(23.0)
dust = np.sqrt(A_D) * sed_dust(nu) / sed_dust(353.0)
nu_min = nu[np.argmin(np.abs(sync - dust))]

fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.axvspan(90, 150, color=KW2, alpha=0.07, lw=0)
ax.loglog(nu, sync, color=SYNC, lw=3.4)
ax.loglog(nu, dust, color=DUST, lw=3.4)
for r, lw in ((1e-2, 3.0), (1e-3, 1.8)):
    amp = np.sqrt(r * b80_r1)
    ax.axhline(amp, color=KW2, lw=lw, ls="--")
    ax.text(470, amp * 1.25, rf"primordial $B$, $r = 10^{{{int(np.log10(r))}}}$", color=KW2,
            fontsize=15, ha="right", va="bottom")
ax.text(14, 6.0, "synchrotron", color=SYNC, fontsize=19, fontweight="bold")
ax.text(330, 6.0, "thermal dust", color=DUST, fontsize=19, fontweight="bold", ha="right")
ax.axvline(nu_min, color=GREY, lw=1.0, ls=":")
ax.text(nu_min, 0.55, f"foreground\nminimum\n~{nu_min:.0f} GHz", color=GREY, fontsize=14,
        ha="center", va="center", linespacing=1.15,
        bbox=dict(facecolor="#faf7f0", edgecolor="none", pad=1.5))
ax.text(118, 18, "CMB channels", color=KW2, fontsize=14, ha="center")

ax.set_xlabel(r"frequency $\nu$ [GHz]")
ax.set_ylabel(r"$B$-mode amplitude at $\ell = 80$ [$\mu\mathrm{K}_\mathrm{CMB}$]")
ax.set_xlim(10, 500)
ax.set_ylim(3e-3, 40)
ax.set_xticks([10, 30, 100, 300], ["10", "30", "100", "300"])
ax.minorticks_off()
ax.tick_params(top=False, right=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT} (foreground minimum {nu_min:.0f} GHz; r=1e-2 amplitude {np.sqrt(1e-2 * b80_r1):.3f} uK)")
