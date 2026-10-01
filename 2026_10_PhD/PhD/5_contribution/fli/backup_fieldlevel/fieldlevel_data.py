#!/usr/bin/env python3
# ENV: jax-fli
"""
Everything the field-level sampling backups draw, gathered once into fli/.cache/fieldlevel.npz
(kappa_anim.py, starlet_anim.py and posterior.py render from it).

Joint chain, jax-fli notebook docs/3-sampling-and-inference/16-LPT-FieldLevel-IC.ipynb, published in
ASKabalan/jax-fli-sampling, 14-LPT-Sampling/ (HuggingFace). MCLMC over (Omega_c, sigma8) and the 128^3
white-noise initial conditions, from two convergence maps (DES Y3 source bins 2 and 3 at the Euclid
IST:F density, 7.5 galaxies per arcmin^2 and bin, sigma_e = 0.30 / sqrt 2), 2LPT on 22 capped
equal-volume shells from 300 Mpc/h to z = 1, Born, a Gaussian pixel likelihood on kappa band-limited to
2 <= l <= 57 at nside 64. 300 tuning steps, then 100 draws, one every 5 MCLMC steps; the chain starts
at the truth. Read from chain/samples/observable_fields/: the noiseless band-limited kappa each draw
predicts for the data, and the cosmology of the draw.
  kappa_truth (bin, pix), kappa (draw, bin, pix)     nside 64
  om16, s816 (draw)                                  Omega_c and sigma8 of the draws
  l1_truth (bin, scale, nu), l1 (draw, bin, scale, nu)
      starlet l1 norm, as notebook 16 section 14: each map resampled to nside 32 through its a_lm
      (l <= 57), five starlet scales, nu = w_j / sigma_j with sigma_j the truth's, 40 nu bins over
      [-5, 5], the sum of |nu| over the pixels of each bin

Conditional chain, jax-fli notebook 17-LPT-FieldLevel-Cosmology.ipynb, run on the cluster on
29 September 2026 and read from the local copy of its output (mcmc_results/ is gitignored and this run
is not on HuggingFace: 14-LPT-Sampling/chain_fixed_ic/ there is an older version of the notebook).
The same model, truth and data, conditioned on the true white field: NUTS on (Omega_c, sigma8) alone,
100 draws after 100 warmup steps, and the log-posterior on a 31 x 31 grid around them.
  om17, s817 (draw)                                  the NUTS draws
  grid_om, grid_s8, grid_density (om, s8), grid_levels (95 %, 68 %)
      the grid density in (Omega_c, sigma8), e^-U times the Jacobian of the prior's base map, as
      notebook 17 section 7, normalised to unit sum, and its highest-density levels
plus the truth, Planck18 (truth_om, truth_s8).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache"
NPZ = CACHE / "fieldlevel.npz"
if NPZ.exists() and "--force" not in sys.argv:
    print(f"{NPZ.name} exists, pass --force to recompute")
    sys.exit(0)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import jax_cosmo as jc
import jax_fli as jfli
from huggingface_hub import snapshot_download
from jax_fli.summary_statistics import starlet_coefficients_spherical
from scipy.special import ndtri

REPO, RUN = "ASKabalan/jax-fli-sampling", "14-LPT-Sampling"
RUN17 = Path("/home/wassim/Projects/NBody/jax-fli/docs/3-sampling-and-inference/mcmc_results/"
             "FIELDLEVEL_COSMOLOGY_MESH128")
ELL_MAX, ST_NSIDE, N_SCALES = 57, 32, 5   # notebook 16: ELL_MAX 57, NSIDE 64; ST_NSIDE = 2^ceil(log2(ELL_MAX / 3))
NU_EDGES = np.linspace(-5.0, 5.0, 41)
PRIORS = {"Omega_c": (0.1, 0.5), "sigma8": (0.6, 1.0)}   # PreconditionnedUniform(low, high)
index = lambda p: int(p.stem.split("_")[-1])

# ------------------------------------------------------------------ notebook 16, the joint chain
out = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[
    f"{RUN}/truth/kappa_truth.parquet", f"{RUN}/chain/samples/observable_fields/*.parquet"])) / RUN
kt = np.asarray(jfli.io.Catalog.from_parquet(str(out / "truth/kappa_truth.parquet")).field[0].array, np.float64)
kappa, om16, s816 = [], [], []
for f in sorted((out / "chain/samples/observable_fields").glob("fields_*.parquet"), key=index):
    cat = jfli.io.Catalog.from_parquet(str(f))
    kappa += [np.asarray(k.array, np.float64) for k in cat.field]
    om16 += [float(c.Omega_c) for c in cat.cosmology]
    s816 += [float(c.sigma8) for c in cat.cosmology]
kappa = np.array(kappa)
print(f"notebook 16: {len(kappa)} draws, kappa {kappa.shape[1:]}")


def starlet(m):
    m = hp.alm2map(hp.map2alm(np.asarray(m, dtype=np.float64), lmax=ELL_MAX), ST_NSIDE, lmax=ELL_MAX)
    return np.asarray(starlet_coefficients_spherical(m, nside=ST_NSIDE, nscales=N_SCALES)[0])


def l1_norm(coef, sigma):
    return np.array([np.histogram(coef[j] / sigma[j], bins=NU_EDGES, weights=np.abs(coef[j] / sigma[j]))[0]
                     for j in range(N_SCALES)])


st_t = [starlet(kt[b]) for b in range(2)]
sig_t = [c.std(axis=1) for c in st_t]
l1_truth = np.array([l1_norm(st_t[b], sig_t[b]) for b in range(2)])
l1 = np.array([[l1_norm(starlet(k[b]), sig_t[b]) for b in range(2)] for k in kappa])

for b in range(2):
    r = [np.corrcoef(kt[b], k[b])[0, 1] for k in kappa]
    print(f"  bin {b + 2}: r(draw, truth) {np.mean(r):.3f} +/- {np.std(r):.3f}, "
          f"std over draws / truth rms {kappa[:, b].std(0).mean() / kt[b].std():.3f}")
print(f"  Omega_c {np.mean(om16):.4f} +/- {np.std(om16):.4f}, sigma8 {np.mean(s816):.4f} +/- {np.std(s816):.4f}")

# ------------------------------------------------------------------ notebook 17, the conditional chain
draws = [np.load(f) for f in sorted((RUN17 / "chain/samples/samples").glob("cosmo_*.npz"), key=index)]
om17 = np.concatenate([d["Omega_c"] for d in draws])
s817 = np.concatenate([d["sigma8"] for d in draws])
grid = np.load(RUN17 / "grid.npz")


def base_jacobian(name, value):
    low, high = PRIORS[name]
    base = ndtri((value - low) / (high - low))
    return 1.0 / ((high - low) * np.exp(-0.5 * base**2) / np.sqrt(2 * np.pi))


density = np.exp(-(grid["potential"] - grid["potential"].min())) * np.outer(
    base_jacobian("Omega_c", grid["Omega_c"]), base_jacobian("sigma8", grid["sigma8"]))
density /= density.sum()
flat = np.sort(density.ravel())[::-1]
levels = np.array([flat[np.searchsorted(np.cumsum(flat), mass)] for mass in (0.95, 0.68)])
print(f"notebook 17: {om17.size} NUTS draws")
for name, d, axis, marginal in (("Omega_c", om17, grid["Omega_c"], density.sum(1)),
                                ("sigma8", s817, grid["sigma8"], density.sum(0))):
    mean = (marginal * axis).sum()
    print(f"  {name:8s} NUTS {d.mean():.4f} +/- {d.std():.4f}   "
          f"grid {mean:.4f} +/- {np.sqrt((marginal * (axis - mean) ** 2).sum()):.4f}")

cosmo = jc.Planck18()
CACHE.mkdir(exist_ok=True)
np.savez(NPZ, kappa_truth=kt.astype(np.float32), kappa=kappa.astype(np.float32), om16=om16, s816=s816,
         l1_truth=l1_truth, l1=l1, nu=0.5 * (NU_EDGES[1:] + NU_EDGES[:-1]), om17=om17, s817=s817,
         grid_om=grid["Omega_c"], grid_s8=grid["sigma8"], grid_density=density, grid_levels=levels,
         truth_om=float(cosmo.Omega_c), truth_s8=float(cosmo.sigma8))
print(f"wrote {NPZ}")
