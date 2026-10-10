#!/usr/bin/env python3
# ENV: shared
"""
What the r likelihood sees, kept simple: the lensing B modes, the band of primordial B modes the
analysis is after (r = 1e-3 to 1e-2, r times the CAMB tensor template), and the observed spectrum
C_l^obs of 1000 simulated skies, drawn as the band holding 68 % of them, with r = 0: lensing plus white noise, each drawn with its cosmic
and noise variance (chi^2 with f_sky (2l + 1) degrees of freedom).

CAMB, Planck 2018 cosmology. Noise: LiteBIRD-like, 2.2 uK-arcmin in polarisation with a 30 arcmin
Gaussian beam; f_sky = 0.6. Multipoles 2 <= l <= 150, the range of the likelihood.

Output (this directory): r_budget.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, KW2, skip_if_built, slide_style

OUT = "r_budget.svg"
skip_if_built(HERE, OUT)

import camb

pars = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200, ns=0.9649, As=2.1e-9,
                       tau=0.0544, r=1.0)
pars.WantTensors = True
pars.set_for_lmax(400, lens_potential_accuracy=1)
cls = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
LMIN, LMAX = 2, 150
ell = np.arange(LMIN, LMAX + 1)
tens = cls["tensor"][LMIN:LMAX + 1, 2]                              # r = 1
lens = cls["lensed_scalar"][LMIN:LMAX + 1, 2]

W_NOISE, FWHM, FSKY, NREAL = 2.2, 30.0, 0.6, 1000
sig_b = np.deg2rad(FWHM / 60) / np.sqrt(8 * np.log(2))
noise = np.deg2rad(W_NOISE / 60) ** 2 * np.exp(ell * (ell + 1) * sig_b**2)

rng = np.random.default_rng(20261012)
nu = np.round(FSKY * (2 * ell + 1)).astype(int)
obs = (lens + noise) * rng.chisquare(nu, size=(NREAL, len(ell))) / nu

dl = ell * (ell + 1) / (2 * np.pi)
slide_style(scale=1.8)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.2, 6.0))
lo, hi = np.percentile(dl * obs, [16, 84], axis=0)
ax.fill_between(ell, lo, hi, color=KW2, alpha=0.35, lw=0,
                label=r"$C_\ell^{\mathrm{obs}}$, $r = 0$")
ax.fill_between(ell, dl * 1e-3 * tens, dl * 1e-2 * tens, color=KW, alpha=0.30, lw=0)
ax.loglog(ell, dl * 1e-2 * tens, color=KW, lw=2.0,
          label=r"$r\,C_\ell^{\mathrm{tensor}}$, $10^{-3} < r < 10^{-2}$")
ax.loglog(ell, dl * 1e-3 * tens, color=KW, lw=2.0)
ax.loglog(ell, dl * lens, color=INK, lw=2.6, ls="--", label=r"$C_\ell^{\mathrm{lens}}$")
ax.legend(loc="upper left", fontsize=18, handlelength=1.6, labelspacing=0.4)

ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{BB}/2\pi$ [$\mu$K$^2$]")
ax.set_xlim(LMIN, LMAX)
ax.set_ylim(5e-7, 3e-2)
ax.set_xticks([2, 10, 50, 150], ["2", "10", "50", "150"])
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
