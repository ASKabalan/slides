# ENV: jax-fli
"""
What Omega_m and sigma_8 do to the lensing power spectrum.

The convergence spectrum of the last DES Y3 source bin (jax-fli ``compute_theory_cl``, halofit),
for the fiducial Planck 2018 model and with sigma_8, then Omega_m, moved by +-10 %. Both
parameters mostly raise or lower the same curve: lensing measures one amplitude, which is why
it constrains the combination S_8 = sigma_8 sqrt(Omega_m / 0.3) best.

Output (this directory): cl_kappa.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import GREY, KW, KW2, skip_if_built, slide_style

OUT = "cl_kappa.svg"
skip_if_built(HERE, OUT)

import jax

jax.config.update("jax_enable_x64", True)  # the Limber integral needs float64
import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
from jax_fli.io import get_des_y3_nz_shear

ELL = np.geomspace(10, 3000, 60)
NZ = get_des_y3_nz_shear()[-1:]
FID = jc.Planck18()
OM = FID.Omega_c + FID.Omega_b


def cl(**kw):
    cosmo = jc.Planck18(**kw)
    c = jfli.compute_theory_cl(cosmo, jnp.asarray(ELL), z_source=NZ, probe_type="weak_lensing")
    return ELL * (ELL + 1) * np.asarray(c.array).reshape(-1) / (2 * np.pi)


fid = cl()
s8 = {s: cl(sigma8=FID.sigma8 * (1 + s)) for s in (-0.1, 0.1)}
om = {s: cl(Omega_c=OM * (1 + s) - FID.Omega_b) for s in (-0.1, 0.1)}

slide_style(scale=1.2)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
for ax, var, col, name in ((axes[0], s8, KW, r"$\sigma_8$"), (axes[1], om, KW2, r"$\Omega_\mathrm{m}$")):
    ax.fill_between(ELL, var[-0.1], var[0.1], color=col, alpha=0.18, lw=0)
    ax.loglog(ELL, var[0.1], color=col, lw=1.6, ls="--")
    ax.loglog(ELL, var[-0.1], color=col, lw=1.6, ls="--")
    ax.loglog(ELL, fid, color="#2E2E2E", lw=2.4)
    ax.set_title(f"{name} $\\pm$ 10 %", color=col, fontsize=15)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_xlim(ELL[0], ELL[-1])
axes[0].set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
axes[1].text(2500, fid[0] * 1.3, "Planck 2018", color="#2E2E2E", fontsize=12, ha="right")

fig.tight_layout()
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
