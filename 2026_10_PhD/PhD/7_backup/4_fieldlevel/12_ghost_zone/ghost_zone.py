#!/usr/bin/env python3
# ENV: jax-fli
"""
The ghost zone, for its backup slide: the slide version of the thesis figures chap6/ghost_zone.pdf and
chap6/ghost_zone_pencil.pdf (These_wassim/figures/chap6/ghost_zone{,_pencil}.py), as one figure.

jax-fli experiment 01 (ASKabalan/jax-fli-experiments, 01-resolution/spectra): a lightcone in a
2000 Mpc/h box, nside 512, at a 512^3 to 3072^3 mesh. A slab decomposition puts each mesh on 4, 8, 64,
128 and 256 GPUs, so the ghost zone (half the local domain along the split axis) shrinks to 250, 125,
15.6, 7.8 and 3.9 Mpc/h; the two finest meshes are re-run as pencils (ghost zones 31.2 and 15.6 Mpc/h).
Left, per shell beyond 400 Mpc/h (colour: comoving distance), the measured over the Limber number-count
prediction (times the squared pixel window), (2l+1)-weighted over l in [270, 330], minus one, over a
+-5 % band. Right, the ghost-zone width against the rms linear displacement (1D at z = 0.35 to 3D at
z = 0); a ghost zone below it loses the particles that leave their subdomain.
Top row slab, bottom row with the two finest meshes as pencils.

Output (this directory): ghost_zone.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, RED, TEAL, skip_if_built, slide_style

OUT = "ghost_zone.svg"
skip_if_built(HERE, OUT)

CACHE = (ROOT / "5_contribution/fli") / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
NSIDE, BOX = 512, 2000.0
MESHES = [512, 1024, 2048, 2560, 3072]
PX = {512: 4, 1024: 8, 2048: 64, 2560: 128, 3072: 256}
HALO = {m: 0.5 * BOX / PX[m] for m in MESHES}
PENCILS = {2560: 31.25, 3072: 15.625}
BAND, CHI_CUT = (270, 330), 400.0


def load():
    npz = CACHE / "ghost_zone_01.npz"
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
    from jax_fli.io import Catalog

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=["01-resolution/spectra/*"])
    cat = lambda e: Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{e}", split="train"))
    slab = {m: cat(f"01-resolution/spectra/spectra_m{m}.parquet") for m in MESHES}
    pencil = {m: cat(f"01-resolution/spectra/spectra_m{m}_pencils.parquet") for m in PENCILS}
    f512 = slab[512].field[0]
    cosmo = slab[512].cosmology[0]
    lmax = int(f512.wavenumber.max())
    ell = np.asarray(f512.wavenumber)
    th = np.asarray(compute_theory_cl_for_density(cosmo, f512, jnp.arange(lmax + 1)).array) \
        * hp.pixwin(NSIDE, lmax=lmax) ** 2
    m_ = (ell >= BAND[0]) & (ell <= BAND[1])
    w = 2 * ell[m_] + 1

    def ratio(arr):
        # the pencil re-runs stop at l = 1500, the slab runs at 1535; the band is far below both
        arr = np.asarray(arr)[:, : len(ell)]
        mm = m_[: arr.shape[1]]
        return (arr[:, mm] * w).sum(1) / (th[:, m_] * w).sum(1)

    out = {"chi": np.asarray(f512.comoving_centers)}
    for m in MESHES:
        out[f"slab_{m}"] = ratio(slab[m].field[0].array)
    for m in PENCILS:
        out[f"pencil_{m}"] = ratio(pencil[m].field[0].array)

    def sigma(z):
        k = jnp.logspace(-4, 1.3, 4000)
        return float(jnp.sqrt(jnp.trapezoid(jc.power.linear_matter_power(cosmo, k, a=1 / (1 + z)), k)
                              / (6 * np.pi**2)))

    out["band"] = np.array([sigma(0.35), np.sqrt(3) * sigma(0.0)])
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()
chi = D["chi"]
kept = [i for i in range(len(chi)) if chi[i] >= CHI_CUT]
s_lo, s_hi = D["band"]

slide_style()
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

plt.rcParams["savefig.bbox"] = None
xs = np.arange(len(MESHES))
norm = Normalize(vmin=float(chi[kept].min()), vmax=float(chi[kept].max()))
fig = plt.figure(figsize=(10.4, 5.8))
for row, pencil in enumerate((False, True)):
    y0 = 0.58 - row * 0.47
    axL = fig.add_axes([0.08, y0, 0.38, 0.36])
    axR = fig.add_axes([0.655, y0, 0.33, 0.36])
    labels = [f"{m}\n{'pencil' if (pencil and m in PENCILS) else 'slab'}" for m in MESHES]
    rat = {m: (D[f"pencil_{m}"] if (pencil and m in PENCILS) else D[f"slab_{m}"]) for m in MESHES}
    halo = {m: (PENCILS[m] if (pencil and m in PENCILS) else HALO[m]) for m in MESHES}
    axL.axhspan(-0.05, 0.05, color="#d8dde6", lw=0)
    axL.axhline(0, color=INK, lw=0.8, ls="--")
    for i in kept:
        axL.plot(xs, [rat[m][i] - 1 for m in MESHES], "o-", color=cm.plasma(norm(chi[i])), lw=1.6, ms=4)
    axL.set_ylim(-1.02, 0.22)
    axL.set_ylabel(r"$C_\ell / C_\ell^{\mathrm{Limber}} - 1$", fontsize=12)
    axR.axhspan(s_lo, s_hi, color=RED, alpha=0.15, lw=0)
    axR.text(-0.3, s_lo * 0.8, f"rms displacement {s_lo:.1f}–{s_hi:.1f} Mpc/$h$", fontsize=11, color=RED, va="top")
    for k, m in enumerate(MESHES):
        ok = halo[m] > s_hi
        axR.plot(k, halo[m], "o", ms=9, color=TEAL if ok else RED, mec=INK, mew=0.8)
        axR.annotate(f"{round(halo[m], 1):g}", (k, halo[m]), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=11)
    axR.set_yscale("log")
    axR.yaxis.set_major_formatter(ScalarFormatter())
    axR.set_ylim(0.8, 2500)
    axR.set_ylabel("ghost zone  [Mpc/$h$]", fontsize=12)
    for a in (axL, axR):
        a.set_xticks(xs)
        a.set_xticklabels(labels, linespacing=1.05, fontsize=11)
        a.set_xlim(-0.42, 4.42)
        a.grid(alpha=0.25)
    axL.set_title("slab decomposition" if not pencil else "the two finest meshes as pencils",
                  fontsize=13, color=INK, loc="left", pad=4)
sm = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
cax = fig.add_axes([0.475, 0.11, 0.012, 0.83])
cb = fig.colorbar(sm, cax=cax)
cb.set_label(r"$\chi$  [Mpc/$h$]", fontsize=11)
cb.ax.tick_params(labelsize=10)
fig.savefig(HERE / OUT)
for m in MESHES:
    print(m, "slab median ratio-1 (kept shells):", round(float(np.median(D[f'slab_{m}'][kept] - 1)), 3),
          "" if m not in PENCILS else f"pencil {round(float(np.median(D[f'pencil_{m}'][kept] - 1)), 3)}")
print(f"wrote {OUT}")
