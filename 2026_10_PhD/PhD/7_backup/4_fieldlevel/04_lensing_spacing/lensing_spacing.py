#!/usr/bin/env python3
# ENV: jax-fli
"""
Equal-volume against uniform scale-factor shell spacing, judged on the convergence: the slide version
of figure 18 of jax-fli experiment 05c (docs/5-experiments/05c-spacing-n-stepping-equal-vol, build.py,
`lensing_spacing(25, ...)`), replotted from the same spectra with its top two panels only (the
band-median bars are left out).

jax-fli-experiments, 25 shells drifted on the lightcone, 2560^3 in a 5000 Mpc/h box, nside 2048:
  equal volume     05c spectra_gauss_legendre/spectra_gl_drift_25 (Born, Gauss-Legendre across shells)
  uniform in a     05b kappa_spectra/spectra_born_drift_25 (midpoint; for its thin shells midpoint and
                   Gauss-Legendre agree to < 0.2 % on the lensing weight, as in the experiment)
  reference        CosmoGrid grid point 172798 (00-cosmogrid/cosmo_172798), the grid cosmology closest
                   to the run's
Stage-3 source bins 1-3, bandpowers of 16 multipoles. Top: l(l+1)C_l/2pi. Bottom: C_l/CosmoGrid - 1,
with the grey band sqrt(2) x the empirical cosmic variance of the 200 fiducial CosmoGrid permutations
(worst bin) around the expected cosmology + pixel-window offset (dotted).

Output (this directory): lensing_spacing_25.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, skip_if_built, slide_style

OUT = "lensing_spacing_25.svg"
skip_if_built(HERE, OUT)

CACHE = (ROOT / "5_contribution/fli") / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
EV = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_25.parquet"
SF = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_25.parquet"
CG = "00-cosmogrid/cosmo_172798/kappa_spectra/spectra_cosmogrid_sample_kappa.parquet"
FID = "00-cosmogrid/fiducial_kappa_spectra/cosmo_fiducial_part{p}.parquet"
LMAX, NLB, ELL_MAX_PLOT = 1500, 16, 1300
BIN_COLOURS = ("#4477aa", "#ee7733", "#117733")  # the thesis tomographic-bin colours, as in 05c


def load():
    npz = CACHE / f"lensing_spacing_05c_25_nlb{NLB}.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli import compute_theory_cl
    from jax_fli.io import Catalog, get_stage3_nz_shear

    fids = [FID.format(p=p) for p in range(4)]
    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[EV, SF, CG, *fids])
    catalog = lambda path: Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}", split="train"))
    binned = lambda field: np.asarray(field.bin(nlb=NLB, lmin=2).array)

    fid = np.stack([binned(f) for p in fids for f in catalog(p).field])
    cv = np.sqrt(2.0) * (fid.std(axis=0, ddof=1) / fid.mean(axis=0))[:3].max(axis=0)

    cg_cat, ev_cat = catalog(CG), catalog(EV)
    ell, nz = jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]
    th_cg = binned(compute_theory_cl(cg_cat.cosmology[0], ell, nz) * hp.pixwin(512, lmax=LMAX) ** 2)[:3]
    th_run = (compute_theory_cl(ev_cat.cosmology[0], ell, nz) * hp.pixwin(2048, lmax=LMAX) ** 2).bin(nlb=NLB, lmin=2)
    out = {
        "ell": np.asarray(th_run.wavenumber),
        "cg": binned(cg_cat.field[0])[:3],
        "ev": binned(ev_cat.field[0])[:3],
        "sf": binned(catalog(SF).field[0])[:3],
        "offset": (np.asarray(th_run.array)[:3] / th_cg - 1.0).mean(axis=0),
        "cv": cv,
        "z": np.asarray(ev_cat.field[0].z_sources)[:3],
    }
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()

slide_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

bc = D["ell"]
keep = bc <= ELL_MAX_PLOT
dl = bc * (bc + 1) / (2 * np.pi)
r_ev, r_sf = D["ev"] / D["cg"] - 1.0, D["sf"] / D["cg"] - 1.0
off, cv = D["offset"], D["cv"]

fig = plt.figure(figsize=(10.4, 5.9))
ax = fig.add_axes((0.10, 0.45, 0.60, 0.50))
axr = fig.add_axes((0.10, 0.10, 0.60, 0.32), sharex=ax)

for b, c in enumerate(BIN_COLOURS):
    ax.plot(bc[keep], (dl * D["cg"][b])[keep], "-.", color=c, lw=1.8)
    ax.plot(bc[keep], (dl * D["ev"][b])[keep], "-", color=c, lw=2.2)
    ax.plot(bc[keep], (dl * D["sf"][b])[keep], "--", color=c, lw=1.8)
ax.set(xscale="log", yscale="log", xlim=(20, ELL_MAX_PLOT))
ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
ax.tick_params(labelbottom=False)

axr.fill_between(bc[keep], (off - cv)[keep], (off + cv)[keep], color="0.86", lw=0, zorder=0)
for b, c in enumerate(BIN_COLOURS):
    axr.plot(bc[keep], r_ev[b][keep], "-", color=c, lw=2.2)
    axr.plot(bc[keep], r_sf[b][keep], "--", color=c, lw=1.8)
axr.plot(bc[keep], off[keep], ":", color="0.3", lw=1.2)
axr.axhline(0.0, color=INK, lw=0.9)
axr.set_xscale("log")
axr.set_xlabel(r"multipole $\ell$")
axr.set_ylabel(r"$C_\ell / C_\ell^{\mathrm{CosmoGrid}} - 1$")
# the same y cap as the experiment: the offset line keeps rising with the pixel window above l ~ 400
m_le = (bc <= 400) & keep
lo = min(r_ev[:, keep].min(), r_sf[:, keep].min())
hi = max(r_ev[:, m_le].max(), r_sf[:, m_le].max(), (off + cv)[m_le].max())
pad = 0.06 * (hi - lo)
axr.set_ylim(lo - pad, hi + pad)

handles = [Patch(color=c, label=rf"bin {b + 1}, $z_s = {z:.2f}$") for b, (c, z) in enumerate(zip(BIN_COLOURS, D["z"]))]
handles += [
    Line2D([], [], color="0.3", ls="-", lw=2.2, label="equal volume"),
    Line2D([], [], color="0.3", ls="--", lw=1.8, label="uniform in $a$"),
    Line2D([], [], color="0.3", ls="-.", lw=1.8, label=r"CosmoGrid $N$-body"),
    Patch(color="0.86", label="$\\sqrt{2}\\,\\times$ cosmic variance\n(200 CosmoGrid permutations)"),
    Line2D([], [], color="0.3", ls=":", lw=1.2, label="expected cosmology\n+ pixel-window offset"),
]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.72, 0.52), frameon=False,
           fontsize=12.5, handlelength=2.0, labelspacing=0.7)
fig.text(0.40, 0.965, "25 shells", ha="center", va="bottom", fontsize=14, color=INK)
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
for name, r in (("equal volume", r_ev), ("uniform a", r_sf)):
    sel = (bc >= 150) & (bc < 300)
    print(f"{name}: median C_l/CG - 1 over 150 <= l < 300 per bin", np.round(np.median(r[:, sel], axis=1), 3))
