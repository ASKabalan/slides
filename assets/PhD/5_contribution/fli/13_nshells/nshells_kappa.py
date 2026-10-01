#!/usr/bin/env python3
# ENV: jax-fli
"""
Convergence accuracy against the number of equal-volume shells, for the slide on choosing it.

jax-fli experiment 05c (ASKabalan/jax-fli-experiments, 05-spacing-n-stepping/05c-equal-volume):
2560^3 mesh in a 5000 Mpc/h box, observer at the centre, BullFrog, equal-volume shells drifted on
the lightcone, Born convergence integrated with Gauss-Legendre quadrature across each shell, for
three tomographic bins of the stage-3 source distribution (effective z_s = 0.31, 0.48, 0.75). The
thesis figure chap6/lensing_spacing.pdf reads the same files. One image per shell count:
  top     the lightcone as a wedge from the observer, each shell a band at its true comoving edges,
          with the comoving distance of each bin's effective source redshift as a coloured arc;
  bottom  C_l / C_l^Limber - 1 for the three bins in 15 log-spaced multipole bands from 30 to 1000,
          the theory multiplied by the squared nside-2048 pixel window, over a grey band at +-10 %.

Outputs (this directory), same canvas and axes boxes: nshells_kappa_{10,20,30}.svg. An alternative to
nshells.py (uniform scale factor, density shells), kept out of the deck for now.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, INK, KW, KW2, skip_if_built, slide_style

COUNTS = (10, 20, 30)
OUTS = [f"nshells_kappa_{n}.svg" for n in COUNTS]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
KAPPA = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_{n}.parquet"
DENSITY = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_{n}.parquet"
Z_SOURCE = (0.31, 0.48, 0.75)
LMAX, NSIDE, R_MAX = 1500, 2048, 2500.0


def load():
    npz = CACHE / "nshells_05c_gl_raw.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    import jax_cosmo as jc
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli import compute_theory_cl
    from jax_fli.data import get_stage3_nz_shear
    from jax_fli.io import Catalog

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[
        p.format(n=n) for n in COUNTS for p in (KAPPA, DENSITY)])

    def catalog(path):
        return Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}",
                                                 split="train"))

    out = {}
    for n in COUNTS:
        dens = catalog(DENSITY.format(n=n))
        cosmo = dens.cosmology[0]
        chi, w = np.asarray(dens.field[0].comoving_centers), np.asarray(dens.field[0].density_width)
        o = np.argsort(chi)
        out[f"edges_{n}"] = np.r_[0.0, np.cumsum(w[o])]
        out[f"kappa_{n}"] = np.asarray(catalog(KAPPA.format(n=n)).field[0].array)
    theory = (compute_theory_cl(cosmo, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3])
              * hp.pixwin(NSIDE, lmax=LMAX) ** 2)
    out["theory"] = np.asarray(theory.array)
    out["chi_s"] = np.asarray(jc.background.radial_comoving_distance(
        cosmo, jc.utils.z2a(jnp.asarray(Z_SOURCE))))
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()
EDGES = np.unique(np.geomspace(30, 1001, 16).astype(int))


def binned(cl):
    """Sums over log-spaced multipole bands [EDGES[i], EDGES[i+1])."""
    return np.stack([cl[..., a:b].sum(-1) for a, b in zip(EDGES[:-1], EDGES[1:])], -1)


ELL = np.sqrt(EDGES[:-1] * (EDGES[1:] - 1))
LOW = (ELL >= 30) & (ELL <= 100)
for n in COUNTS:
    r = binned(D[f"kappa_{n}"]) / binned(D["theory"]) - 1
    print(f"{n} shells: median C_l/theory - 1 over l in [30, 100]:",
          np.median(r[:, LOW], axis=1).round(3), " edges", D[f"edges_{n}"][:4].round(0), "...")

slide_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

plt.rcParams["savefig.bbox"] = None
BINS = [(KW, "bin 1"), (KW2, "bin 2"), (BLUE, "bin 3")]
HALF = 11                                   # half-opening of the wedge [deg]

for n, out in zip(COUNTS, OUTS):
    fig = plt.figure(figsize=(3.5, 4.0))
    ax_w = fig.add_axes([0.03, 0.64, 0.94, 0.28])
    ax_r = fig.add_axes([0.235, 0.14, 0.7, 0.45])

    # the lightcone, shell by shell, from the observer at the left
    edges = D[f"edges_{n}"]
    for j in range(n):
        ax_w.add_patch(Wedge((0, 0), edges[j + 1], -HALF, HALF, width=edges[j + 1] - edges[j],
                             fc="#d8dde6" if j % 2 else "#eef1f5", ec="#9aa3b2", lw=0.5))
    th = np.radians(np.linspace(-HALF, HALF, 50))
    for (c, _), chi in zip(BINS, D["chi_s"]):
        ax_w.plot(chi * np.cos(th), chi * np.sin(th), color=c, lw=2.4)
    for i, ((c, _), chi) in enumerate(zip(BINS, D["chi_s"])):
        ax_w.text(chi * np.cos(np.radians(HALF)), chi * np.sin(np.radians(HALF)) + 45,
                  f"bin {i + 1}", color=c, fontsize=10.5, ha="center", va="bottom")
    ax_w.plot(0, 0, "o", color=INK, ms=4)
    ax_w.set_xlim(-60, R_MAX + 40)
    ax_w.set_ylim(-R_MAX * np.sin(np.radians(HALF)) - 30, R_MAX * np.sin(np.radians(HALF)) + 60)
    ax_w.set_aspect("equal")
    ax_w.axis("off")
    ax_w.set_title(f"{n} shells", fontsize=15, color=INK, pad=4)

    # convergence against Limber
    r = binned(D[f"kappa_{n}"]) / binned(D["theory"]) - 1
    ax_r.axhspan(-0.1, 0.1, color="#9aa3b2", alpha=0.25, lw=0)
    ax_r.axhline(0, color=INK, lw=0.8, ls=":")
    for i, (c, lab) in enumerate(BINS):
        ax_r.semilogx(ELL, r[i], color=c, lw=2.0, label=lab)
    ax_r.set_xlim(30, 1000)
    ax_r.set_ylim(-0.62, 0.22)
    ax_r.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000]))
    ax_r.xaxis.set_major_formatter(FixedFormatter(["30", "100", "300", "1000"]))
    ax_r.xaxis.set_minor_formatter(NullFormatter())
    ax_r.set_yticks([-0.4, -0.2, 0, 0.2])
    ax_r.set_yticklabels(["−40 %", "−20 %", "0", "+20 %"])
    ax_r.set_xlabel(r"$\ell$", labelpad=0)
    ax_r.set_title(r"$C_\ell^{\kappa} / C_\ell^{\mathrm{Limber}} - 1$", fontsize=13, color=INK, pad=5)
    fig.savefig(HERE / out)
    plt.close(fig)
    print(f"wrote {out}")
