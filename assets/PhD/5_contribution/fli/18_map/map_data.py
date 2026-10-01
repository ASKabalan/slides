#!/usr/bin/env python3
# ENV: jax-fli
"""
Everything the MAP slides draw, computed once from the DES Y3 MAP run of jax-fli experiment 13 and
cached in fli/.cache/map_des.npz (map_anim.py renders from it).

Run: ASKabalan/jax-fli-sampling, 13-LPT-MassMapping/mesh_416_DES (HuggingFace). The initial
conditions (IC) of a 416^3 mesh are reconstructed by maximum a posteriori (Adam, 300 steps, cosine
decay from 0.05) from two DES Y3 convergence maps (source bins 2 and 3, nside 256, sigma_e = 0.26,
1.48 galaxies per arcmin^2 and bin), with 2LPT on 22 equal-volume shells from 300 Mpc/h to z = 1 and
a pixel likelihood on kappa; eleven log-spaced steps are saved (0, 1, 2, 4, ..., 159, 300). The
constants are those of jax-fli docs/5-experiments/13-map-lpt2-mass-mapping/build.py.

Per saved step and for the truth:
  proj    the IC projected on the sky with the lensing efficiency of the two bins, averaged
          (DensityField.sky_projection), nside 256 -- how lensing sees the IC
  kappa   the convergence of both bins, nside 256
Per saved step, against the truth:
  r_kappa, r_proj           Pearson correlation of the maps
  coh, trans                coherence r(l) and transfer T(l) = sqrt(C_l^MAP / C_l^truth) of kappa, 2 <= l < 188
  l1                        starlet l1 norm of kappa (5 scales at nside 64, nu = w_j / sigma_j^truth,
                            40 bins over [-5, 5]); l1_truth for the truth
plus the joint Wiener-filter coherence of both bins (the best a linear reconstruction reaches), and
the steps and the loss.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache"
NPZ = CACHE / "map_des.npz"
if NPZ.exists() and "--force" not in sys.argv:
    print(f"{NPZ.name} exists, pass --force to recompute")
    sys.exit(0)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import jax_cosmo as jc
import jax_fli as jfli
from huggingface_hub import snapshot_download
from jax_fli.data.nz import get_des_y3_nz_shear
from jax_fli.summary_statistics import starlet_coefficients_spherical

REPO, RUN = "ASKabalan/jax-fli-sampling", "13-LPT-MassMapping/mesh_416_DES"
NSIDE, ELL_MIN, ELL_MAX = 256, 2, 188
DES_BINS, MAX_Z, SIGMA_E = (1, 2), 1.0, 0.26
ST_NSIDE, N_SCALES = 64, 5
NU_EDGES = np.linspace(-5.0, 5.0, 41)

out = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{RUN}/**"])) / RUN
load = lambda p: jfli.io.Catalog.from_parquet(str(p)).field[0]
frames = lambda d, pre: sorted((out / d).glob(f"{pre}_*.parquet"), key=lambda p: int(p.stem.split("_")[-1]))
rows = sorted(json.loads((out / "metrics.json").read_text()), key=lambda r: r["frame"])
cosmo = jc.Planck18()
nz = [get_des_y3_nz_shear(zmax=MAX_Z)[i] for i in DES_BINS]
lc = load(out / "truth_density_lightcone.parquet")
r_min = float(np.min(np.asarray(lc.comoving_centers) - 0.5 * np.asarray(lc.density_width)))
del lc


def project(ic):
    return np.asarray(ic.sky_projection(cosmo, nz, nside=NSIDE, r_min=r_min).array, dtype=np.float64).mean(axis=0)


def starlet(m):
    m = hp.alm2map(hp.map2alm(np.asarray(m, dtype=np.float64), lmax=ELL_MAX), ST_NSIDE, lmax=ELL_MAX)
    return np.asarray(starlet_coefficients_spherical(m, nside=ST_NSIDE, nscales=N_SCALES)[0])


def l1_norm(coef, sigma):
    res = np.zeros((N_SCALES, len(NU_EDGES) - 1))
    for j in range(N_SCALES):
        nu = coef[j] / sigma[j]
        i = np.digitize(nu, NU_EDGES) - 1
        ok = (i >= 0) & (i < res.shape[1])
        res[j] = np.bincount(i[ok], weights=np.abs(nu[ok]), minlength=res.shape[1])
    return res


kt = np.asarray(load(out / "truth_kappa.parquet").array, dtype=np.float64)
proj_t = project(load(out / "true_ic.parquet"))
st_t = [starlet(kt[b]) for b in range(len(DES_BINS))]
sig_t = [c.std(axis=1) for c in st_t]
ell = np.arange(ELL_MAX + 1)
sel = (ell >= ELL_MIN) & (ell < ELL_MAX)
ctt = [hp.anafast(kt[b], lmax=ELL_MAX) for b in range(len(DES_BINS))]

# the joint Wiener filter of both bins: r_a^2 = [C (C + N)^-1 C]_aa / C_aa
C = np.array([[hp.anafast(kt[a], kt[b], lmax=ELL_MAX) for b in range(2)] for a in range(2)]).transpose(2, 0, 1)
N_ell = SIGMA_E**2 / (np.array([float(n.gals_per_arcmin2) for n in nz]) * (180 * 60 / np.pi) ** 2)
M = np.einsum("lab,lbc,lcd->lad", C[sel], np.linalg.inv(C[sel] + np.diag(N_ell)[None]), C[sel])
wiener = np.array([np.sqrt(M[:, b, b] / C[sel, b, b]) for b in range(2)])

proj, kappa, r_kappa, r_proj, coh, trans, l1 = [], [], [], [], [], [], []
for f_ic, f_k in zip(frames("ic_evolution", "ic"), frames("kappa_evolution", "kappa")):
    pp = project(load(f_ic))
    kp = np.asarray(load(f_k).array, dtype=np.float64)
    proj.append(pp.astype(np.float32))
    kappa.append(kp.astype(np.float32))
    r_proj.append(np.corrcoef(proj_t, pp)[0, 1])
    r_kappa.append([np.corrcoef(kt[b], kp[b])[0, 1] for b in range(2)])
    c_, t_ = [], []
    for b in range(2):
        cpp, ctp = hp.anafast(kp[b], lmax=ELL_MAX), hp.anafast(kt[b], kp[b], lmax=ELL_MAX)
        c_.append((ctp / np.sqrt(ctt[b] * cpp))[sel])
        t_.append(np.sqrt(cpp / ctt[b])[sel])
    coh.append(c_)
    trans.append(t_)
    l1.append([l1_norm(starlet(kp[b]), sig_t[b]) for b in range(2)])
    print(f"{f_ic.stem}: r(kappa) {np.round(r_kappa[-1], 3)}, r(projected IC) {r_proj[-1]:.3f}", flush=True)

CACHE.mkdir(exist_ok=True)
np.savez(NPZ, steps=np.array([r["step"] for r in rows]), loss=np.array([r["loss"] for r in rows]),
         proj_truth=proj_t.astype(np.float32), proj=np.array(proj), kappa_truth=kt.astype(np.float32),
         kappa=np.array(kappa), r_kappa=np.array(r_kappa), r_proj=np.array(r_proj), ell=ell[sel],
         coh=np.array(coh), trans=np.array(trans), wiener=wiener, l1=np.array(l1),
         l1_truth=np.array([l1_norm(st_t[b], sig_t[b]) for b in range(2)]),
         nu=0.5 * (NU_EDGES[1:] + NU_EDGES[:-1]))
print(f"wrote {NPZ}")
