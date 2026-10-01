#!/usr/bin/env python3
# ENV: jax-fli
"""
What the number of shells does to the lightcone, for the slide on its impact: one figure per row,
the same three columns (10, 20 and 30 shells) in both.

nshells_density.svg  many shells: shot noise. jax-fli experiment 05b (ASKabalan/jax-fli-experiments,
    05-spacing-n-stepping/05b-3bins, spectra_exp5b_nodrift_{n}): 2560^3 mesh in a 5000 Mpc/h box,
    BullFrog with 50 steps, shells uniform in the scale factor, nside 2048. Per column, the lightcone
    as a wedge with its innermost shell filled, and the spectrum of that shell (bands of 32
    multipoles) against the Limber prediction times the squared pixel window (dashed) and its
    shot-noise level 4 pi / N, N = nbar 4/3 pi r^3 particles with nbar = 2560^3 / 5000^3 (dotted).
nshells_kappa.svg  few shells: bias. jax-fli experiment 05c (05-spacing-n-stepping/05c-equal-volume,
    spectra_gauss_legendre/spectra_gl_drift_{n}): the same box and mesh, equal-volume shells drifted
    on the lightcone, Born convergence with Gauss-Legendre quadrature, three stage-3 source bins
    (z_s = 0.31, 0.48, 0.75). Per column, the wedge with each bin's source distance as an arc in the
    bin's colour, and
    C_l / C_l^Limber - 1 per bin in 15 log-spaced bands from l = 30 to 1000, the theory times the
    squared nside-2048 pixel window, over a grey band at +-10 %.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, GREY, INK, KW, KW2, skip_if_built, slide_style

OUTS = ["nshells_density.svg", "nshells_kappa.svg"]
skip_if_built(HERE, *OUTS)

COUNTS = (10, 20, 30)
CACHE = HERE.parent / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
DENSITY = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_{n}.parquet"
GEOM_EV = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_{n}.parquet"
KAPPA = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_{n}.parquet"
Z_SOURCE = (0.31, 0.48, 0.75)
BOX, MESH, LMAX, NSIDE, NLB, R_MAX = 5000.0, 2560, 1500, 2048, 32, 2500.0


def _setup():
    import jax

    jax.config.update("jax_enable_x64", True)
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli.io import Catalog

    def catalog(root, path):
        return Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}", split="train"))

    return snapshot_download, catalog


def load_density():
    npz = CACHE / "nshells_05b_nodrift.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    snapshot_download, catalog = _setup()
    import healpy as hp
    import jax.numpy as jnp
    from jax_fli import compute_theory_cl_for_density

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[DENSITY.format(n=n) for n in COUNTS])
    out = {}
    for n in COUNTS:
        cat = catalog(root, DENSITY.format(n=n))
        spec = cat.field[0]
        theory = (compute_theory_cl_for_density(cat.cosmology[0], spec, jnp.arange(LMAX + 1))
                  * hp.pixwin(NSIDE, lmax=LMAX) ** 2).bin(nlb=NLB, lmin=2)
        binned = spec.bin(nlb=NLB, lmin=2)
        o = np.argsort(np.asarray(spec.comoving_centers))
        out["ell"] = np.asarray(binned.wavenumber)
        out[f"cl_{n}"] = np.asarray(binned.array)[o]
        out[f"th_{n}"] = np.asarray(theory.array)[o]
        out[f"edges_{n}"] = np.r_[0.0, np.cumsum(np.asarray(spec.density_width)[o])]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


def load_kappa():
    npz = CACHE / "nshells_05c_gl_raw.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    snapshot_download, catalog = _setup()
    import healpy as hp
    import jax.numpy as jnp
    import jax_cosmo as jc
    from jax_fli import compute_theory_cl
    from jax_fli.data import get_stage3_nz_shear

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[
        p.format(n=n) for n in COUNTS for p in (KAPPA, GEOM_EV)])
    out = {}
    for n in COUNTS:
        dens = catalog(root, GEOM_EV.format(n=n))
        cosmo = dens.cosmology[0]
        chi, w = np.asarray(dens.field[0].comoving_centers), np.asarray(dens.field[0].density_width)
        out[f"edges_{n}"] = np.r_[0.0, np.cumsum(w[np.argsort(chi)])]
        out[f"kappa_{n}"] = np.asarray(catalog(root, KAPPA.format(n=n)).field[0].array)
    out["theory"] = np.asarray((compute_theory_cl(cosmo, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3])
                                * hp.pixwin(NSIDE, lmax=LMAX) ** 2).array)
    out["chi_s"] = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(Z_SOURCE))))
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


DD, DK = load_density(), load_kappa()
NBAR = MESH**3 / BOX**3
ELL = DD["ell"]
DL = ELL * (ELL + 1) / (2 * np.pi)
EDGES_K = np.unique(np.geomspace(30, 1001, 16).astype(int))
ELL_K = np.sqrt(EDGES_K[:-1] * (EDGES_K[1:] - 1))


def banded(cl):
    return np.stack([cl[..., a:b].sum(-1) for a, b in zip(EDGES_K[:-1], EDGES_K[1:])], -1)


for n in COUNTS:
    N = NBAR * 4 / 3 * np.pi * DD[f"edges_{n}"][1] ** 3
    r = banded(DK[f"kappa_{n}"]) / banded(DK["theory"]) - 1
    low = (ELL_K >= 30) & (ELL_K <= 100)
    print(f"{n} shells: innermost uniform-a shell {N:.2e} particles; kappa/Limber - 1 below l = 100:",
          np.median(r[:, low], axis=1).round(3))

slide_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter

plt.rcParams["savefig.bbox"] = None
HALF = 11                                   # half-opening of the wedge [deg]
BIN_COLOURS = (KW, KW2, BLUE)
FIGSIZE = (9.0, 2.6)


def column(fig, j):
    x0 = 0.075 + j * 0.31
    ax_w = fig.add_axes([x0 - 0.03, 0.66, 0.29, 0.25])
    ax_p = fig.add_axes([x0 + 0.03, 0.15, 0.22, 0.47])
    return ax_w, ax_p


def wedge(ax, edges, fill=None, title=None):
    for k in range(len(edges) - 1):
        fc = (fill or {}).get(k, "#d8dde6" if k % 2 else "#eef1f5")
        ax.add_patch(Wedge((0, 0), edges[k + 1], -HALF, HALF, width=edges[k + 1] - edges[k],
                           fc=fc, ec="#9aa3b2", lw=0.5))
    ax.plot(0, 0, "o", color=INK, ms=3)
    ax.set_xlim(-60, R_MAX + 40)
    h = R_MAX * np.sin(np.radians(HALF))
    ax.set_ylim(-h - 30, h + 150)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, color=INK, pad=1)


# --- row 1: many shells, shot noise
lims = np.concatenate([np.r_[DL * DD[f"cl_{n}"][0], DL * DD[f"th_{n}"][0],
                             DL * 4 * np.pi / (NBAR * 4 / 3 * np.pi * DD[f"edges_{n}"][1] ** 3)]
                       for n in COUNTS])
lims = lims[np.isfinite(lims) & (lims > 0)]
fig = plt.figure(figsize=FIGSIZE)
for j, n in enumerate(COUNTS):
    ax_w, ax = column(fig, j)
    edges = DD[f"edges_{n}"]
    wedge(ax_w, edges, {0: KW}, f"{n} shells")
    N = NBAR * 4 / 3 * np.pi * edges[1] ** 3
    ax.loglog(ELL, DL * DD[f"th_{n}"][0], color=INK, ls="--", lw=1.4, label="Limber")
    ax.loglog(ELL, DL * 4 * np.pi / N + 0 * ELL, color=GREY, ls=":", lw=1.8, label="shot noise")
    ax.loglog(ELL, DL * DD[f"cl_{n}"][0], color=KW, lw=2.2, label="innermost shell")
    ax.set_xlim(ELL[0], LMAX)
    ax.set_ylim(lims.min() / 1.5, lims.max() * 1.5)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(numticks=4))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=10.5)
    ax.set_xlabel(r"$\ell$", labelpad=-1, fontsize=12)
    mant, expo = f"{N:.1e}".split("e")
    ax.text(0.04, 0.94, rf"$N = {mant}\times10^{{{int(expo)}}}$", transform=ax.transAxes,
            fontsize=10.5, color=KW, va="top")
    if j == 2:
        ax.legend(loc="lower right", fontsize=9.5, handlelength=1.5, borderaxespad=0.15, labelspacing=0.1)
fig.savefig(HERE / OUTS[0])
plt.close(fig)
print(f"wrote {OUTS[0]}")

# --- row 2: few shells, bias
fig = plt.figure(figsize=FIGSIZE)
for j, n in enumerate(COUNTS):
    ax_w, ax = column(fig, j)
    wedge(ax_w, DK[f"edges_{n}"], None, f"{n} shells")
    th = np.radians(np.linspace(-HALF, HALF, 50))
    for b, (c, chi) in enumerate(zip(BIN_COLOURS, DK["chi_s"])):
        ax_w.plot(chi * np.cos(th), chi * np.sin(th), color=c, lw=2.2)
    r = banded(DK[f"kappa_{n}"]) / banded(DK["theory"]) - 1
    ax.axhspan(-0.1, 0.1, color="#d8dde6", lw=0)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    for b, c in enumerate(BIN_COLOURS):
        ax.semilogx(ELL_K, r[b], color=c, lw=2.0, label=f"bin {b + 1}")
    if j == 2:          # the 30-shell curves hug zero at low l, which leaves that corner free
        ax.legend(loc="lower left", fontsize=9.5, handlelength=1.3, borderaxespad=0.15, labelspacing=0.1)
    ax.set_xlim(30, 1000)
    ax.set_ylim(-0.62, 0.2)
    ax.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["30", "100", "300", "1000"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([-0.4, -0.2, 0])
    ax.set_yticklabels(["−40 %", "−20 %", "0"])
    ax.tick_params(labelsize=10.5)
    ax.set_xlabel(r"$\ell$", labelpad=-1, fontsize=12)
fig.savefig(HERE / OUTS[1])
plt.close(fig)
print(f"wrote {OUTS[1]}")
