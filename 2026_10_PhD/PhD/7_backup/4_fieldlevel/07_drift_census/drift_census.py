#!/usr/bin/env python3
# ENV: jax-fli
"""
What the drift on the lightcone does to each shell, for the slide after the drift slide: the slide
version of the thesis figure chap6/density_census.pdf (its script is
These_wassim/figures/chap6/density_census.py), equal-volume row only, its first shell.

jax-fli experiment 05c (ASKabalan/jax-fli-experiments, 05-spacing-n-stepping/05c-equal-volume,
density_spectra/spectra_exp5c_{nodrift,drift}_10): 2560^3 mesh in a 5000 Mpc/h box, BullFrog with
50 steps, ten equal-volume shells, nside 2048, the same particles painted once at each shell's
single redshift and once drifted to the redshift at which each particle crosses the lightcone.
Shell 1, the widest (a ball reaching 1160 Mpc/h, its centre at 580 Mpc/h), with and without
drift, against the Limber prediction for its number counts times the squared pixel window, in bands
of 16 multipoles; the wedge beside it shows where that shell sits in the lightcone. The dotted line is the shell's
particle-mesh Nyquist multipole pi chi / dx. Both runs share their particles, so their shot noise is
the same realisation and the gap between them is the drift. The medians printed for shells 1, 5
and 10 show that the thinner shells do not move.

Output (this directory): drift_census_shell1.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, GREY, INK, KW, KW2, skip_if_built, slide_style

OUTS = ["drift_census_shell1.svg"]
skip_if_built(HERE, *OUTS)

CACHE = (ROOT / "5_contribution/fli") / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
RUNS = {"nodrift": "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_10.parquet",
        "drift": "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_10.parquet"}
SHELLS = (1, 5, 10)                     # counted from the observer outwards, one-based; shell 1 is drawn
BOX, MESH, LMAX, NSIDE, NLB = 5000.0, 2560, 1500, 2048, 16


def load():
    npz = CACHE / f"drift_census_05c_nlb{NLB}.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli import compute_theory_cl_for_density
    from jax_fli.io import Catalog

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=list(RUNS.values()))
    out = {}
    for key, path in RUNS.items():
        cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}", split="train"))
        spec = cat.field[0]
        o = np.argsort(np.asarray(spec.comoving_centers))
        binned = spec.bin(nlb=NLB, lmin=2)
        out["ell"] = np.asarray(binned.wavenumber)
        out[f"cl_{key}"] = np.asarray(binned.array)[o]
        if key == "nodrift":
            theory = (compute_theory_cl_for_density(cat.cosmology[0], spec, jnp.arange(LMAX + 1))
                      * hp.pixwin(NSIDE, lmax=LMAX) ** 2).bin(nlb=NLB, lmin=2)
            out["theory"] = np.asarray(theory.array)[o]
            out["chi"] = np.asarray(spec.comoving_centers)[o]
            out["edges"] = np.r_[0.0, np.cumsum(np.asarray(spec.density_width)[o])]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()
ELL = D["ell"]
DL = ELL * (ELL + 1) / (2 * np.pi)
BAND = (ELL >= 30) & (ELL <= 300)
for s in SHELLS:
    i = s - 1
    med = {k: np.median((D[f"cl_{k}"][i] / D["theory"][i])[BAND]) - 1 for k in RUNS}
    change = np.median((D["cl_drift"][i] / D["cl_nodrift"][i])[BAND]) - 1
    print(f"shell {s:2d} chi = {D['chi'][i]:6.0f} Mpc/h: meas/Limber - 1 over 30 <= l <= 300: "
          f"no drift {med['nodrift']:+.3f}, drift {med['drift']:+.3f}; drift / no drift - 1 = {change:+.3f}")

slide_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

plt.rcParams["savefig.bbox"] = None
HALF, R_MAX = 11, 2500.0                # the wedge of the dropped number-of-shells slide
FAINT = "#b9bfca"


def wedge(ax, edges, fill):
    """The lightcone as a wedge of its shells, as on the dropped number-of-shells slide."""
    for k in range(len(edges) - 1):
        fc = fill.get(k, "#d8dde6" if k % 2 else "#eef1f5")
        ax.add_patch(Wedge((0, 0), edges[k + 1], -HALF, HALF, width=edges[k + 1] - edges[k],
                           fc=fc, ec="#9aa3b2", lw=0.5))
    ax.plot(0, 0, "o", color=INK, ms=3)
    ax.set_xlim(-60, R_MAX + 40)
    h = R_MAX * np.sin(np.radians(HALF))
    ax.set_ylim(-h - 30, h + 30)
    ax.set_aspect("equal")
    ax.axis("off")


i = 0                                   # shell 1
th = D["theory"][i]
fig = plt.figure(figsize=(9.0, 4.2))
ax_w = fig.add_axes([0.0, 0.3, 0.27, 0.4])
wedge(ax_w, D["edges"], {i: KW2})
ax_w.set_title(f"shell 1 of 10\n$0 \\leq \\chi \\leq {D['edges'][1]:.0f}$ Mpc/$h$", fontsize=12, color=KW2, pad=2,
               linespacing=1.3)
ax = fig.add_axes([0.38, 0.42, 0.59, 0.53])
ar = fig.add_axes([0.38, 0.12, 0.59, 0.26], sharex=ax)
ax.loglog(ELL, DL * th, color=INK, ls="--", lw=1.6, label="Limber")
ax.loglog(ELL, DL * D["cl_nodrift"][i], color=KW, lw=2.2, label="one redshift per shell")
ax.loglog(ELL, DL * D["cl_drift"][i], color=BLUE, lw=2.2, label="drifted")
ar.axhspan(-0.05, 0.05, color="#d8dde6", lw=0)
ar.axhline(0, color=INK, lw=0.8, ls=":")
ar.semilogx(ELL, D["cl_nodrift"][i] / th - 1, color=KW, lw=2.0)
ar.semilogx(ELL, D["cl_drift"][i] / th - 1, color=BLUE, lw=2.0)
nyq = np.pi * D["chi"][i] / (BOX / MESH)
for a in (ax, ar):
    a.axvline(nyq, color=GREY, ls=":", lw=1.0)
ax.set_xlim(ELL[0], LMAX)
ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(LogLocator(numticks=4))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.tick_params(labelbottom=False)
ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
ax.legend(loc="upper left", fontsize=11.5, handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)
ar.set_ylim(-0.32, 0.32)
ar.set_yticks([-0.2, 0, 0.2])
ar.set_yticklabels(["−20 %", "0", "+20 %"])
ar.set_ylabel("/ Limber − 1", fontsize=12)
ar.set_xlabel(r"$\ell$", labelpad=-1)
fig.savefig(HERE / OUTS[0])
plt.close(fig)
print(f"wrote {OUTS[0]}")
