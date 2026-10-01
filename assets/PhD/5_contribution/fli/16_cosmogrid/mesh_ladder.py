#!/usr/bin/env python3
# ENV: jax-fli
"""
The forward model against CosmoGrid as the particle-mesh resolution grows, for the CosmoGrid slide.

jax-fli experiment 05e (jax-fli-experiments, 05-spacing-n-stepping/05e-mesh/kappa_spectra): the
production lightcone (5000 Mpc/h box, BullFrog with 50 steps, 20 equal-volume shells drifted on the
lightcone, nside 2048) run at a 512^3 to 4096^3 mesh; the 2560^3 point is the thesis run
(05c exp5c_drift_20). Each Born convergence spectrum is divided by the CosmoGrid pkdgrav3 lightcone
at grid point cosmo_172798, recomputed through the same Born integral, source distribution and
pixelisation (00-cosmogrid/cosmo_172798/kappa_spectra/spectra_kappa_born_s3.parquet, the reference
of the thesis figure chap6/lensing_vs_cosmogrid.pdf). Tomographic bins 2 and 3, the two kept for
inference, in bandpowers of 32 multipoles. The grey band is sqrt(2) times the fractional scatter of
the 200 fiducial CosmoGrid permutations (00-cosmogrid/fiducial_kappa_spectra): the run and the
reference are two independent realisations.

Outputs (this directory), one per mesh on the same canvas, the meshes before it greyed:
  ladder_512.svg ... ladder_4096.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, GREY, INK, KW, skip_if_built, slide_style

MESHES = (512, 1024, 2048, 2560, 3072, 4096)
OUTS = [f"ladder_{m}.svg" for m in MESHES]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
EXP = Path("/home/wassim/Projects/NBody/jax-fli-experiments")
RUN = EXP / "05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m{m}.parquet"
REF = EXP / "00-cosmogrid/cosmo_172798/kappa_spectra/spectra_kappa_born_s3.parquet"
FID = EXP / "00-cosmogrid/fiducial_kappa_spectra/cosmo_fiducial_part{p}.parquet"
BINS = (1, 2)                        # tomographic bins 2 and 3, zero-based
NLB = 32


def load():
    npz = CACHE / "mesh_ladder_05e.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    from datasets import load_dataset
    from jax_fli.io import Catalog

    def catalog(path):
        return Catalog.from_dataset(load_dataset("parquet", data_files=str(path), split="train"))

    ref = catalog(REF).field[0].bin(nlb=NLB, lmin=2)
    out = {"ell": np.asarray(ref.wavenumber), "ref": np.asarray(ref.array)[:3]}
    for m in MESHES:
        out[f"m{m}"] = np.asarray(catalog(str(RUN).format(m=m)).field[0].bin(nlb=NLB, lmin=2).array)[:3]
    fid = np.stack([np.asarray(f.bin(nlb=NLB, lmin=2).array)
                    for p in range(4) for f in catalog(str(FID).format(p=p)).field])
    out["cv"] = (np.sqrt(2) * fid.std(0, ddof=1) / fid.mean(0))[:3]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()
ELL = D["ell"]
KEEP = (ELL >= 20) & (ELL <= 1000)
for m in MESHES:
    r = D[f"m{m}"] / D["ref"] - 1
    at = [np.argmin(abs(ELL - l)) for l in (100, 200, 300)]
    print(f"{m:5d}^3: C_l/CosmoGrid - 1 at l = {ELL[at].round(0)}: bin 2 {r[1, at].round(2)}, "
          f"bin 3 {r[2, at].round(2)}")

slide_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

plt.rcParams["savefig.bbox"] = None
PAST = "#b9bfca"

for k, (mesh, out) in enumerate(zip(MESHES, OUTS)):
    fig = plt.figure(figsize=(10.4, 4.4))
    for j, b in enumerate(BINS):
        ax = fig.add_axes([0.1 + 0.485 * j, 0.12, 0.34, 0.79])
        ax.fill_between(ELL[KEEP], -D["cv"][b][KEEP], D["cv"][b][KEEP], color="#d8dde6", lw=0, zorder=0)
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
        ax.set_yticklabels(["−60 %", "−40 %", "−20 %", "−10 %", "0", "+10 %"])
        ax.set_xlabel(r"$\ell$", labelpad=0)
        ax.set_title(f"tomographic bin {b + 1}", fontsize=14, color=INK, pad=6)
        if j == 0:
            ax.set_ylabel(r"$C_\ell^{\kappa} / C_\ell^{\kappa,\,\mathrm{CosmoGrid}} - 1$", fontsize=13)
            ax.text(22, 0.13, "grey: CosmoGrid cosmic variance", fontsize=10.5, color=GREY,
                    va="bottom")
    fig.savefig(HERE / out)
    plt.close(fig)
    print(f"wrote {out}")
