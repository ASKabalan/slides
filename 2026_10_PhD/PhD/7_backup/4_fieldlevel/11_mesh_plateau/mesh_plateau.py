#!/usr/bin/env python3
# ENV: jax-fli
"""
How the lightcone approaches theory as the particle mesh grows, for two backup slides: the slide
versions of figures 1 and 3 of jax-fli experiment 05f (docs/5-experiments/05f-mesh-plateau-diagnosis,
build.py), replotted from the same spectra.

jax-fli-experiments 05-spacing-n-stepping/05f-mesh: 5000 Mpc/h box, 100 steps, 20 shells drifted on the
lightcone, nside 2048, meshes 512^3 to 4096^3, two solvers (BullFrog stepped in D, kick-drift-kick
stepped in a) and five shell spacings (uniform in a; equal volume; equal volume with the width capped at
150 Mpc/h, the lightcone starting at 0, 150 or 300 Mpc/h). Every spectrum is binned in bands of 32
multipoles and divided by its Limber prediction times the squared nside-2048 pixel window, and each
ratio is reduced to its median over band centres in [150, 250), "l ~ 200".

  mesh_plateau_density.svg   density shells (density_spectra/spectra_exp5f_*), one line per shell
                             coloured by its comoving centre
  mesh_plateau_kappa.svg     Born convergence of the three Stage-3 source bins
                             (kappa_gl_spectra/spectra_born_gl_*)

Rows are the solvers, columns the spacings. Values above 1.3 (shot noise at the coarsest meshes) sit on
the top edge as triangles.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, GREY, INK, KW, KW2, skip_if_built, slide_style

OUTS = ["mesh_plateau_density.svg", "mesh_plateau_kappa.svg"]
skip_if_built(HERE, *OUTS)

CACHE = (ROOT / "5_contribution/fli") / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
BASE = "05-spacing-n-stepping/05f-mesh"
DENSITY = BASE + "/density_spectra/spectra_exp5f_m{m}_{s}_{sp}.parquet"
KAPPA = BASE + "/kappa_gl_spectra/spectra_born_gl_m{m}_{s}_{sp}.parquet"
MESHES = (512, 1024, 2048, 2560, 3072, 4096)
SOLVERS = ("bf", "kdk")
SPACINGS = ("a", "equal_vol", "eqvolc_w150_r0", "eqvolc_w150_r150", "eqvolc_w150_r300")
LMAX, NSIDE, NLB, BAND = 1500, 2048, 32, (150, 250)


def load():
    npz = CACHE / "mesh_plateau_05f.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli import compute_theory_cl, compute_theory_cl_for_density
    from jax_fli.io import Catalog, get_stage3_nz_shear

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[
        p.format(m=m, s=s, sp=sp) for p in (DENSITY, KAPPA) for m in MESHES for s in SOLVERS for sp in SPACINGS])
    catalog = lambda path: Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}", split="train"))
    ell = jnp.arange(LMAX + 1)
    pw = hp.pixwin(NSIDE, lmax=LMAX) ** 2
    out = {}
    ref = catalog(KAPPA.format(m=2048, s="bf", sp="a"))
    cosmo = ref.cosmology[0]
    kappa_th = compute_theory_cl(cosmo, ell, get_stage3_nz_shear()[:3]) * pw
    kappa_th_b = kappa_th.bin(nlb=NLB, lmin=2)
    bc = np.asarray(kappa_th_b.wavenumber)
    keep = (bc >= BAND[0]) & (bc < BAND[1])
    med = lambda field, th: np.median((np.asarray(field.bin(nlb=NLB, lmin=2).array) / th)[:, keep], axis=1)
    out["z_sources"] = np.asarray(ref.field[0].z_sources)[:3]
    for sp in SPACINGS:
        dens_ref = catalog(DENSITY.format(m=512, s="bf", sp=sp)).field[0]
        dens_th = np.asarray((compute_theory_cl_for_density(cosmo, dens_ref, ell) * pw).bin(nlb=NLB, lmin=2).array)
        out[f"chi_{sp}"] = np.asarray(dens_ref.comoving_centers)
        for s in SOLVERS:
            out[f"dens_{s}_{sp}"] = np.stack([med(catalog(DENSITY.format(m=m, s=s, sp=sp)).field[0], dens_th)
                                              for m in MESHES])
            out[f"kappa_{s}_{sp}"] = np.stack([med(catalog(KAPPA.format(m=m, s=s, sp=sp)).field[0],
                                                   np.asarray(kappa_th_b.array))[:3] for m in MESHES])
            print(f"{s:3s} {sp:18s} kappa / Limber at l ~ 200, 4096^3: {out[f'kappa_{s}_{sp}'][-1].round(2)}")
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()

slide_style()
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["savefig.bbox"] = None
SOLVER_LABEL = {"bf": "BullFrog\nsteps in $D$", "kdk": "kick-drift-kick\nsteps in $a$"}
SPACING_LABEL = {
    "a": "uniform in $a$",
    "equal_vol": "equal volume",
    "eqvolc_w150_r0": "equal volume,\nwidth ≤ 150 Mpc/$h$",
    "eqvolc_w150_r150": "width ≤ 150 Mpc/$h$,\nfrom 150 Mpc/$h$",
    "eqvolc_w150_r300": "width ≤ 150 Mpc/$h$,\nfrom 300 Mpc/$h$",
}
X = np.arange(len(MESHES))
Y_MAX = 1.3
BAND_GREY = "#d8dde6"
BIN_COLOURS = (KW, KW2, BLUE)                 # source bins 1-3, as on the number-of-shells slide
SHELLS = plt.get_cmap("viridis")                # as in the 05f documentation (build.py, fig01)
CHI_NORM = Normalize(0.0, 2500.0)
LEFT, W, GAP = 0.105, 0.147, 0.012


def grid():
    fig = plt.figure(figsize=(10.4, 5.2))
    axes = np.array([[fig.add_axes([LEFT + j * (W + GAP), 0.53 - i * 0.355, W, 0.3]) for j in range(5)]
                     for i in range(2)])
    for i, s in enumerate(SOLVERS):
        for j, sp in enumerate(SPACINGS):
            ax = axes[i, j]
            ax.axhspan(0.95, 1.05, color=BAND_GREY, lw=0, zorder=0)
            ax.axhline(1.0, color=INK, lw=0.9, zorder=1)
            ax.set_xlim(-0.35, len(MESHES) - 0.65)
            ax.set_ylim(0.0, Y_MAX)
            ax.set_xticks(X)
            ax.set_xticklabels([rf"${m}^3$" for m in MESHES] if i == 1 else [], rotation=50, fontsize=9.5)
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=10, labelleft=(j == 0))
            ax.grid(True, axis="y", ls=":", alpha=0.4)
            if i == 0:
                ax.set_title(SPACING_LABEL[sp], fontsize=10.5, color=INK, pad=4, linespacing=1.15)
        axes[i, 0].set_ylabel(SOLVER_LABEL[s], fontsize=11.5, color=INK, linespacing=1.2, labelpad=4)
    fig.text(LEFT + 2.5 * W + 2 * GAP, 0.0, "particle-mesh resolution", ha="center", va="bottom", fontsize=12,
             color=INK)
    fig.text(0.008, 0.5, r"$C_\ell \,/\, C_\ell^{\mathrm{Limber}}$ at $\ell \approx 200$", rotation=90,
             ha="left", va="center", fontsize=12.5, color=INK)
    return fig, axes


def line(ax, y, colour, lw, ms):
    ax.plot(X, np.minimum(y, Y_MAX - 0.025), "-o", color=colour, lw=lw, ms=ms, zorder=3)
    off = y > Y_MAX
    ax.plot(X[off], np.full(off.sum(), Y_MAX - 0.025), "^", color=colour, ms=ms + 3, zorder=4)


OFF = Line2D([], [], color=GREY, marker="^", ls="none", ms=7, label=f"off scale (> {Y_MAX})")
FIVE = Patch(color=BAND_GREY, label="±5 % of Limber")

# --- density shells
fig, axes = grid()
for i, s in enumerate(SOLVERS):
    for j, sp in enumerate(SPACINGS):
        for k, chi in enumerate(D[f"chi_{sp}"]):
            line(axes[i, j], D[f"dens_{s}_{sp}"][:, k], SHELLS(CHI_NORM(chi)), 1.1, 2.6)
cax = fig.add_axes([LEFT + 5 * W + 4 * GAP + 0.014, 0.175, 0.012, 0.655])
cb = fig.colorbar(ScalarMappable(norm=CHI_NORM, cmap=SHELLS), cax=cax)
cb.set_label(r"shell centre $\chi$  [Mpc/$h$]", fontsize=11)
cb.ax.tick_params(labelsize=9.5)
fig.legend(handles=[OFF, FIVE], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=11,
           frameon=False, handlelength=1.6)
fig.savefig(HERE / OUTS[0])
plt.close(fig)
print(f"wrote {OUTS[0]}")

# --- Born convergence
fig, axes = grid()
for i, s in enumerate(SOLVERS):
    for j, sp in enumerate(SPACINGS):
        for b, c in enumerate(BIN_COLOURS):
            line(axes[i, j], D[f"kappa_{s}_{sp}"][:, b], c, 1.9, 3.6)
bins = [Line2D([], [], color=c, marker="o", ms=4, lw=1.9, label=f"bin {b + 1}, $z_s = {z:.2f}$")
        for b, (c, z) in enumerate(zip(BIN_COLOURS, D["z_sources"]))]
fig.legend(handles=bins + [OFF, FIVE], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=11,
           frameon=False, handlelength=1.6, columnspacing=1.4)
fig.savefig(HERE / OUTS[1])
plt.close(fig)
print(f"wrote {OUTS[1]}")
