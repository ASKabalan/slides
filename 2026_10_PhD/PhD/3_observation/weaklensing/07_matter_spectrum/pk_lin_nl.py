#!/usr/bin/env python3
# ENV: shared
"""
The matter power spectrum, and what the two parameters do to it.

The non-linear spectrum at z = 0 alone, with the two quantities weak lensing
measures marked on the curve: sigma_8, which sets the amplitude, and Omega_m,
which sets where the spectrum turns over.

Computed with CAMB (halofit for the non-linear spectrum) and cached in .cache/.

Outputs (this directory):
  pk_lin_nl.svg         with the sigma_8 / Omega_m annotations (the two-point slide)
  pk_lin_nl_plain.svg   the bare curve, small and with large type (the pipeline slide,
                        before sigma_8 is introduced)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, skip_if_built, slide_style

OUT, OUT_PLAIN = "pk_lin_nl.svg", "pk_lin_nl_plain.svg"
CACHE = HERE / ".cache"
skip_if_built(HERE, OUT, OUT_PLAIN)


def spectra():
    npz = CACHE / "camb_pk.npz"
    if npz.exists():
        d = np.load(npz)
        return d["k"], d["lin"], d["nl"]
    import camb

    out = {}
    for tag, nonlinear in (("lin", False), ("nl", True)):
        pars = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200,
                               ns=0.9649, As=2.1e-9, tau=0.0544)
        pars.set_matter_power(redshifts=[0.0], kmax=30.0)
        pars.NonLinear = (camb.model.NonLinear_both if nonlinear
                          else camb.model.NonLinear_none)
        res = camb.get_results(pars)
        k, _, pk = res.get_matter_power_spectrum(minkh=1e-3, maxkh=20.0,
                                                 npoints=500)
        out["k"], out[tag] = k, pk[0]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out["k"], out["lin"], out["nl"]


k, lin, nl = spectra()

import matplotlib.pyplot as plt

slide_style(scale=1.2)

# the pipeline slide shows it at 200 px: one curve, short labels, big type
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.loglog(k, nl, color=KW, lw=3.0)
ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$", fontsize=17)
ax.set_ylabel(r"$P(k)$", fontsize=17)
ax.set_xlim(1e-3, 20)
ax.set_ylim(3e1, 1e5)
ax.set_xticks([1e-2, 1])
ax.set_yticks([1e2, 1e4])
ax.tick_params(labelsize=14)
ax.minorticks_off()
fig.savefig(HERE / OUT_PLAIN, transparent=True)
plt.close(fig)
print(f"wrote {OUT_PLAIN}")

# the two-point slide: the same curve, with what sigma_8 and Omega_m do to it
fig, ax = plt.subplots(figsize=(7.0, 5.0))
ax.loglog(k, nl, color=KW, lw=2.8)
ax.set_xlabel(r"wavenumber $k\ \ [h\,\mathrm{Mpc}^{-1}]$")
ax.set_ylabel(r"$P(k)\ \ [h^{-3}\mathrm{Mpc}^{3}]$")
ax.set_xlim(1e-3, 20)
ax.set_ylim(3e1, 1e5)

# sigma_8 moves the whole curve up and down.
i = np.argmin(np.abs(k - 0.0045))
ax.annotate("", xy=(k[i], nl[i] * 2.4), xytext=(k[i], nl[i] / 2.4),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.6))
ax.text(k[i] * 1.35, nl[i] / 3.4, r"$\sigma_8$ sets" "\n" "the amplitude",
        color=GREY, fontsize=12.5, va="center", linespacing=1.3)

# Omega_m moves the turnover left and right.
j = np.argmax(nl)
ax.annotate("", xy=(k[j] * 3.0, nl[j] * 1.55), xytext=(k[j] / 3.0, nl[j] * 1.55),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.6))
ax.text(k[j], nl[j] * 1.9, r"$\Omega_\mathrm{m}$ sets the turnover",
        color=GREY, fontsize=12.5, ha="center")

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
