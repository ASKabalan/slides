#!/usr/bin/env python3
# ENV: shared
"""
What linear theory misses in the matter power spectrum.

The z = 0 matter power spectrum, once from linear theory and once with halofit,
with their ratio below. Above k ~ 0.2 h/Mpc the two separate by factors that grow
with wavenumber: the non-linear collapse of structure that perturbation theory
does not follow. Same layout as limber_lin_vs_halofit.py (now a backup), which
shows the same gap in the lensing C_ell.

Spectra from CAMB (halofit), shared with the two-point slide through the cache of
3_observation/weaklensing/07_matter_spectrum/pk_lin_nl.py.

Output (this directory): pk_lin_vs_halofit.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, skip_if_built, slide_style

OUT = "pk_lin_vs_halofit.svg"
CACHE = ROOT / "3_observation/weaklensing/07_matter_spectrum/.cache"
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

fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.8, 4.9), sharex=True,
                              gridspec_kw={"height_ratios": [2.3, 1.0],
                                           "hspace": 0.06})
ax.loglog(k, nl, color=KW, lw=2.6, label="halofit")
ax.loglog(k, lin, color=KW, lw=2.0, ls="--", label="linear")
ax.legend(loc="lower left", labelcolor=INK)
ax.set_ylabel(r"$P(k)\ \ [h^{-3}\mathrm{Mpc}^{3}]$")
ax.set_ylim(3e0, 1e5)

r = nl / lin
axr.axhline(1.0, color=GREY, lw=1.0)
axr.loglog(k, r, color=KW, lw=2.3)
axr.set_ylabel("ratio")
axr.set_ylim(0.7, 120)
axr.set_yticks([1, 10, 100], ["1", "10", "100"])
axr.minorticks_off()
axr.set_xlabel(r"wavenumber $k\ \ [h\,\mathrm{Mpc}^{-1}]$")
axr.set_xlim(1e-3, 20)

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}  (ratio {r[np.argmin(np.abs(k - 0.1))]:.2f} at k = 0.1, "
      f"{r[np.argmin(np.abs(k - 1))]:.1f} at k = 1, max {r.max():.1f})")
