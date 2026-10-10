"""Data and panels for the shell-spacing backup figure (spacing_compare.py, beside this file).

The two phase-matched 30-shell lightcones of jax-fli experiments 05b and 05c (2560^3 mesh in a
5000 Mpc/h box, observer at the centre, BullFrog with 50 steps, drift on the lightcone, nside 2048,
seed 0): exp5b_drift_30, uniform in the scale factor, and exp5c_drift_30, equal volume with its outer
shells floored at 60 Mpc/h. Shell geometry and spectra are read from ASKabalan/jax-fli-experiments
(HuggingFace) and cached per binning in 5_contribution/fli/.cache.
"""

from pathlib import Path

import numpy as np
from _common import INK, KW

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_common.py").exists())
CACHE = ROOT / "5_contribution/fli/.cache"
REPO = "ASKabalan/jax-fli-experiments"
RUNS = {"a": "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_30.parquet",
        "equal_vol": "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_30.parquet"}
BOX, MESH, NSIDE, LMAX = 5000.0, 2560, 2048, 1500
NBAR = MESH**3 / BOX**3
NPIX = 12 * NSIDE**2
BINS = ["#f2b58a", "#d9772e", "#a3450f", "#5c1f04"]   # DES Y3 bins 1-4, light to dark
COUNT_LIM = (5e4, 2e9)


def load(nlb):
    """Spectra of both runs in bandpowers of nlb multipoles, their Limber predictions, the shell
    geometry and the DES Y3 lensing kernels."""
    npz = CACHE / f"shell_spacing_30_nlb{nlb}.npz"
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
    from jax_fli import compute_theory_cl_for_density
    from jax_fli.data import get_des_y3_nz_shear
    from jax_fli.io import Catalog

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=list(RUNS.values()))
    out = {}
    for tag, run in RUNS.items():
        cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{run}", split="train"))
        spec, cosmo = cat.field[0], cat.cosmology[0]
        theory = (compute_theory_cl_for_density(cosmo, spec, jnp.arange(LMAX + 1))
                  * hp.pixwin(NSIDE, lmax=LMAX) ** 2).bin(nlb=nlb, lmin=2)
        binned = spec.bin(nlb=nlb, lmin=2)
        out["ell"] = np.asarray(binned.wavenumber)
        out[f"{tag}_cl"] = np.asarray(binned.array)
        out[f"{tag}_th"] = np.asarray(theory.array)
        out[f"{tag}_chi"] = np.asarray(spec.comoving_centers)
        out[f"{tag}_w"] = np.asarray(spec.density_width)
        print(tag, "chi", out[f"{tag}_chi"].round(0), "w", out[f"{tag}_w"].round(0))
    z = np.linspace(0.005, 2.995, 600)
    out["q_chi"] = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z)))
    out["q"] = np.asarray(jc.probes.WeakLensing(get_des_y3_nz_shear()).kernel(
        cosmo, jnp.asarray(z), 1000.0))
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


def shells(D, tag):
    """Shells of one run, observer outwards: near and far edges, particle counts, spectra."""
    chi, w = D[f"{tag}_chi"], D[f"{tag}_w"]
    order = np.argsort(chi)
    chi, w = chi[order], w[order]
    near, far = chi - w / 2, chi + w / 2
    count = NBAR * 4 / 3 * np.pi * (far**3 - np.maximum(near, 0.0) ** 3)
    return near, far, count, D[f"{tag}_cl"][order], D[f"{tag}_th"][order]


def draw_kernels(ax_k, D, near, far):
    """Panel 1: the DES Y3 lensing kernels over the shells, drawn as alternating bands."""
    for j in range(len(near)):
        ax_k.axvspan(near[j], far[j], color="#d8dde6" if j % 2 else "#eef1f5", lw=0, zorder=0)
    q = D["q"] / D["q"].max()
    for i, c in enumerate(BINS):
        ax_k.plot(D["q_chi"], q[i], color=c, lw=2.0, zorder=2)
    ax_k.set_xlim(0, 2500)
    ax_k.set_ylim(0, 1.08)
    ax_k.set_yticks([])
    ax_k.set_xticks([0, 1000, 2000])
    ax_k.set_xlabel(r"$\chi$  [Mpc/$h$]", labelpad=2)
    ax_k.set_title("lensing kernels and shells", fontsize=13, color=INK, pad=6)


def draw_counts(ax_n, count):
    """Panel 2: the particles each shell receives, against one particle per pixel."""
    from matplotlib.ticker import LogLocator, NullFormatter

    x = np.arange(1, len(count) + 1)
    ax_n.bar(x, count, width=0.78, color=[KW if c < NPIX else "#9aa3b2" for c in count], lw=0)
    ax_n.axhline(NPIX, color=INK, ls="--", lw=1.4, label="1 particle per pixel")
    ax_n.legend(loc="upper left", fontsize=11, handlelength=1.8, borderaxespad=0.3)
    ax_n.set_yscale("log")
    ax_n.set_ylim(*COUNT_LIM)
    ax_n.set_xlim(0.3, len(count) + 0.7)
    ax_n.set_xticks([1, 10, 20, len(count)])
    ax_n.set_xlabel("shell (observer outwards)", labelpad=2)
    ax_n.set_title("particles per shell", fontsize=13, color=INK, pad=6)
    ax_n.yaxis.set_major_locator(LogLocator(numticks=10))
    ax_n.yaxis.set_minor_formatter(NullFormatter())
