"""Shared data and drawing for the mesh ladder against CosmoGrid: the slide frames (mesh_ladder.py,
beside this file) and the finer-bandpower backup (7_backup/4_fieldlevel/09_ladder_nlb4).

jax-fli experiment 05e (jax-fli-experiments, 05-spacing-n-stepping/05e-mesh/kappa_spectra): the
production lightcone (5000 Mpc/h box, BullFrog with 50 steps, 20 equal-volume shells drifted on the
lightcone, nside 2048) run at a 512^3 to 4096^3 mesh; the 2560^3 point is the thesis run
(05c exp5c_drift_20). Each Born convergence spectrum is divided by the CosmoGrid pkdgrav3 lightcone
at grid point cosmo_172798, recomputed through the same Born integral, source distribution and
pixelisation (00-cosmogrid/cosmo_172798/kappa_spectra/spectra_kappa_born_s3.parquet, the reference
of the thesis figure chap6/lensing_vs_cosmogrid.pdf). Tomographic bins 2 and 3, the two kept for
inference. The grey band is sqrt(2) times the fractional scatter of the 200 fiducial CosmoGrid
permutations (00-cosmogrid/fiducial_kappa_spectra): the run and the reference are two independent
realisations. The top axis of each panel converts the multipole to the transverse comoving scale it
probes, 2 pi chi* / l, with chi* the comoving distance where the DES Y3 lensing kernel of that bin
peaks (jax_cosmo WeakLensing, the reference run's cosmology). A teal band marks the linear scales
l < 100; the legend sits above the panels.
"""

from pathlib import Path

import numpy as np
from _common import GREY, INK, KW, TEAL

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_common.py").exists())
CACHE = ROOT / "5_contribution/fli/.cache"
EXP = Path("/home/wassim/Projects/NBody/jax-fli-experiments")
RUN = EXP / "05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m{m}.parquet"
REF = EXP / "00-cosmogrid/cosmo_172798/kappa_spectra/spectra_kappa_born_s3.parquet"
FID = EXP / "00-cosmogrid/fiducial_kappa_spectra/cosmo_fiducial_part{p}.parquet"
BINS = (1, 2)                        # tomographic bins 2 and 3, zero-based
MESHES = (512, 1024, 2048, 2560, 3072, 4096)
PAST = "#b9bfca"
LINEAR, LINEAR_ELL = TEAL, 100   # the linear-scale band, l < 100


def load(nlb):
    npz = CACHE / f"mesh_ladder_05e_nlb{nlb}.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    from datasets import load_dataset
    from jax_fli.io import Catalog

    def catalog(path):
        return Catalog.from_dataset(load_dataset("parquet", data_files=str(path), split="train"))

    ref = catalog(REF).field[0].bin(nlb=nlb, lmin=2)
    out = {"ell": np.asarray(ref.wavenumber), "ref": np.asarray(ref.array)[:3]}
    for m in MESHES:
        out[f"m{m}"] = np.asarray(catalog(str(RUN).format(m=m)).field[0].bin(nlb=nlb, lmin=2).array)[:3]
    fid = np.stack([np.asarray(f.bin(nlb=nlb, lmin=2).array)
                    for p in range(4) for f in catalog(str(FID).format(p=p)).field])
    out["cv"] = (np.sqrt(2) * fid.std(0, ddof=1) / fid.mean(0))[:3]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


def kernel_peaks():
    """chi* of each DES Y3 bin: where its lensing kernel peaks, in Mpc/h."""
    npz = CACHE / "ladder_chi_star.npz"
    if npz.exists():
        return np.load(npz)["chi_star"]
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_cosmo as jc
    from datasets import load_dataset
    from jax_fli.data import get_des_y3_nz_shear
    from jax_fli.io import Catalog

    cosmo = Catalog.from_dataset(load_dataset("parquet", data_files=str(REF), split="train")).cosmology[0]
    z = np.linspace(0.005, 2.995, 1200)
    q = np.asarray(jc.probes.WeakLensing(get_des_y3_nz_shear()).kernel(cosmo, jnp.asarray(z), 1000.0))
    chi = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(z))))
    chi_star = chi[np.argmax(q, axis=1)]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, chi_star=chi_star)
    return chi_star


# the two bins share the y axis, so only the left panel carries tick labels, and the gap between
# the panels holds the mesh label at the end of the left curve
def frame(D, k, path):
    """Frame k of the ladder, written to path: mesh MESHES[k] in orange, the meshes before it grey."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    CHI_STAR = kernel_peaks()
    plt.rcParams["savefig.bbox"] = None
    mesh = MESHES[k]
    ELL = D["ell"]
    KEEP = (ELL >= 20) & (ELL <= 1000)
    fig = plt.figure(figsize=(10.4, 6.3))   # tall enough to fill the slide under its title
    for j, b in enumerate(BINS):
        ax = fig.add_axes([0.095 + 0.455 * j, 0.084, 0.385, 0.70])
        ax.fill_between(ELL[KEEP], -D["cv"][b][KEEP], D["cv"][b][KEEP], color="#d8dde6", lw=0, zorder=0)
        ax.axvspan(1, LINEAR_ELL, color=LINEAR, alpha=0.12, lw=0, zorder=0)   # linear scales, l < 100
        for y in (-0.1, 0.1):
            ax.axhline(y, color=GREY, ls=":", lw=1.2, zorder=1)
        ax.axhline(0, color=INK, lw=0.9, zorder=1)
        for m in MESHES[:k]:
            ax.semilogx(ELL[KEEP], (D[f"m{m}"][b] / D["ref"][b] - 1)[KEEP], color=PAST, lw=1.6, zorder=2)
        r = (D[f"m{mesh}"][b] / D["ref"][b] - 1)[KEEP]
        ax.semilogx(ELL[KEEP], r, color=KW, lw=2.6, zorder=3)
        # the mesh named at the right-hand end of its own curve
        ax.text(ELL[KEEP][-1] * 1.04, np.clip(r[-1], -0.72, 0.18), rf"${mesh}^3$", color=KW,
                fontsize=13, va="center", ha="left", clip_on=False)
        ax.set_xlim(20, 1000)
        ax.set_ylim(-0.75, 0.22)
        ax.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000]))
        ax.xaxis.set_major_formatter(FixedFormatter(["30", "100", "300", "1000"]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([-0.6, -0.4, -0.2, -0.1, 0, 0.1])
        ax.set_yticklabels(["−60 %", "−40 %", "−20 %", "−10 %", "0", "+10 %"] if j == 0 else [])
        ax.set_xlabel(r"$\ell$", labelpad=0)
        # the multipole as a transverse comoving scale 2 pi chi*/l at the kernel peak, labelled by its unit only
        cs = 2 * np.pi * CHI_STAR[b]
        top = ax.secondary_xaxis("top", functions=(lambda l, c=cs: c / np.asarray(l, float),
                                                   lambda r, c=cs: c / np.asarray(r, float)))
        top.xaxis.set_major_locator(FixedLocator([200, 100, 50, 20, 10, 5]))
        top.xaxis.set_major_formatter(FixedFormatter(["200", "100", "50", "20", "10", "5"]))
        top.xaxis.set_minor_formatter(NullFormatter())
        top.tick_params(labelsize=11, colors=INK)
        top.set_xlabel(r"Mpc/$h$", fontsize=11, color=INK, labelpad=4)
        ax.tick_params(axis="x", which="both", top=False)
        ax.set_title(f"tomographic bin {b + 1}", fontsize=14, color=INK, pad=44)
        if j == 0:
            ax.set_ylabel(r"$C_\ell^{\kappa} / C_\ell^{\kappa,\,\mathrm{CosmoGrid}} - 1$", fontsize=13, labelpad=2)
    fig.legend(handles=[Patch(color="#d8dde6", label="CosmoGrid cosmic variance"),
                        Patch(color=LINEAR, alpha=0.12, label=rf"linear scales, $\ell < {LINEAR_ELL}$")],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=12, frameon=False,
               handlelength=1.8, columnspacing=1.8, labelcolor=INK)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.name}")

