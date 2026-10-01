#!/usr/bin/env python3
# ENV: shared
"""
The matter power spectrum, and what the two parameters do to it.

Linear theory against the non-linear spectrum at z = 0, with the two quantities
weak lensing measures marked on the curve: sigma_8, which sets the amplitude,
and Omega_m, which sets where the spectrum turns over. The gap between the two
curves at small scales is the part a two-point analysis cannot describe from
linear theory alone, and the reason the forward model of Chapter 6 runs a
particle-mesh solver.

Computed with CAMB (halofit for the non-linear spectrum) and cached in .cache/.

Output (this directory): pk_lin_nl.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import GREY, INK, KW, skip_if_built, slide_style

OUT = "pk_lin_nl.svg"
CACHE = HERE / ".cache"
skip_if_built(HERE, OUT)


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

slide_style(scale=1.2)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.0, 5.0))

ax.fill_between(k, lin, nl, where=nl > lin, color=KW, alpha=0.10, lw=0)
ax.loglog(k, lin, color="#25406B", lw=2.4, ls="--", label="linear theory")
ax.loglog(k, nl, color=KW, lw=2.8, label="non-linear")

# sigma_8 moves the whole curve up and down.
i = np.argmin(np.abs(k - 0.0045))
ax.annotate("", xy=(k[i], lin[i] * 2.4), xytext=(k[i], lin[i] / 2.4),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.6))
ax.text(k[i] * 1.35, lin[i] / 3.4, r"$\sigma_8$ sets" "\n" "the amplitude",
        color=GREY, fontsize=12.5, va="center", linespacing=1.3)

# Omega_m moves the turnover left and right.
j = np.argmax(lin)
ax.annotate("", xy=(k[j] * 3.0, lin[j] * 1.55), xytext=(k[j] / 3.0, lin[j] * 1.55),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.6))
ax.text(k[j], lin[j] * 1.9, r"$\Omega_\mathrm{m}$ sets the turnover",
        color=GREY, fontsize=12.5, ha="center")

ax.text(0.5, 50, "non-linear growth,\nabsent from linear theory",
        color=KW, fontsize=12.5, ha="right", linespacing=1.3)

ax.set_xlabel(r"wavenumber $k\ \ [h\,\mathrm{Mpc}^{-1}]$")
ax.set_ylabel(r"$P(k)\ \ [h^{-3}\mathrm{Mpc}^{3}]$")
ax.set_xlim(1e-3, 20)
ax.set_ylim(3e1, 1e5)
ax.legend(loc="upper right", labelcolor=INK)

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
