# ENV: skip
# (a module imported by the figure scripts, not a generator of its own)
"""Computations behind the minimisation slides, cached in ../.cache/newton_data.npz.

Everything is the real FURAX spectral likelihood (furax.obs.negative_log_likelihood, LiteBIRD
bands, nside 64, the ALL-GALACTIC mask), set up as in the thesis:

- one faint high-latitude K-means patch (out of 100) holding (l, b) = (90, 60) deg: its
  negative log-likelihood on a (beta_d, T_d) grid at beta_s fixed, for noise levels from zero to
  the nominal LiteBIRD depth (always weighted by the nominal covariance, one realisation scaled);
- an exact damped Newton path on the noise-free grid's likelihood (jax.grad, jax.hessian);
- the Hessian of the likelihood of the high-latitude region (GAL020) with 10 patches x 3
  parameters = 30 parameters, at the input values, for the same noise levels, in the normalised
  coordinates AdaTopK works in (each parameter mapped to [0, 1] across its bounds), with its
  eigenvalues;
- the signal-to-noise of the c1d1s1 foreground polarisation per pixel at 140 GHz, a CMB channel,
  at nominal depth.
"""

import os
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache"
NPZ = CACHE / "newton_data.npz"

NSIDE = 64
MASK = "ALL-GALACTIC"
DUST_NU0, SYNC_NU0 = 150.0, 20.0
REF_DIR = (90.0, 60.0)
BD = np.linspace(1.05, 2.15, 71)
TD = np.linspace(8.0, 40.0, 71)
TRUTH = {"beta_dust": 1.54, "temp_dust": 20.0, "beta_pl": -3.0}
NOISE = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
K_HESS = 10
HESS_MASK = "GAL020"
LOWER = {"beta_dust": 0.5, "temp_dust": 10.0, "beta_pl": -7.0}   # AdaTopK bounds (furax-cs)
UPPER = {"beta_dust": 3.0, "temp_dust": 40.0, "beta_pl": -0.5}
START = np.array([1.14, 37.0])


def load():
    if NPZ.exists():
        return dict(np.load(NPZ))
    os.chdir(CACHE)                   # furax-cs keeps its PySM cache under ./freq_maps_cache
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    from furax._instruments.sky import get_noise_sigma_from_instrument
    from furax.obs import negative_log_likelihood
    from furax.obs.landscapes import FrequencyLandscape
    from furax.obs.operators import NoiseDiagonalOperator
    from furax.obs.stokes import Stokes
    from furax_cs import kmeans_clusters
    from furax_cs.data import get_instrument, get_mask, load_from_cache, save_to_cache
    from jax_healpy.clustering import get_cutout_from_mask

    def sky(tag):
        try:
            nu, maps = load_from_cache(NSIDE, sky=tag)
        except FileNotFoundError:
            save_to_cache(NSIDE, sky=tag)
            nu, maps = load_from_cache(NSIDE, sky=tag)
        return jnp.asarray(nu), np.asarray(maps)

    instrument = get_instrument("LiteBIRD")
    mask = get_mask(MASK, nside=NSIDE)
    (indices,) = jnp.where(mask)
    nu, maps = sky("c1d0s0")
    d_full = Stokes.from_stokes(jnp.asarray(maps[:, 1]), jnp.asarray(maps[:, 2]))
    landscape = FrequencyLandscape(NSIDE, instrument.frequency, "QU")
    white_full = landscape.normal(jax.random.key(7))
    sigma = get_noise_sigma_from_instrument(instrument, NSIDE, stokes_type="QU")
    small_n = jax.tree.map(lambda s: s ** 2, sigma)

    def data(r):
        noised = jax.tree.map(lambda x, w, s: x + r * w * s, d_full, white_full, sigma)
        return get_cutout_from_mask(noised, indices, axis=-1)

    def nll_on(sub, patches=None):
        N = NoiseDiagonalOperator(small_n, in_structure=sub.structure)
        kw = dict(nu=nu, N=N, d=sub, dust_nu0=DUST_NU0, synchrotron_nu0=SYNC_NU0)
        if patches is not None:
            kw["patch_indices"] = patches
        return jax.jit(partial(negative_log_likelihood, **kw))

    # ---- the reference patch and its (beta_d, T_d) grids
    counts = {k: 100 for k in ("beta_dust_patches", "temp_dust_patches", "beta_pl_patches")}
    lab100 = np.asarray(kmeans_clusters(jax.random.key(1), mask, indices, counts, max_patches=counts)["beta_dust_patches"])
    ref_pos = int(np.searchsorted(np.asarray(indices), hp.ang2pix(NSIDE, *REF_DIR, lonlat=True)))
    members = np.nonzero(lab100 == lab100[ref_pos])[0]
    out = {"noise": NOISE, "bd": BD, "td": TD, "npix_patch": np.array(members.size)}
    grids = []
    for r in NOISE:
        sub = jax.tree.map(lambda x: x[..., members], data(r))
        nll = nll_on(sub)
        at = lambda bd, td: nll({"beta_dust": jnp.atleast_1d(bd), "temp_dust": jnp.atleast_1d(td),
                                 "beta_pl": jnp.atleast_1d(TRUTH["beta_pl"])})
        row = jax.jit(jax.vmap(at, in_axes=(None, 0)))
        grids.append(np.stack([np.asarray(row(b, jnp.asarray(TD))) for b in BD]))
        if r == 0:
            f = lambda x: at(x[0], x[1])
            grad, hess = jax.jit(jax.grad(f)), jax.jit(jax.hessian(f))
            x = jnp.asarray(START)
            path, models = [np.asarray(x)], []
            for _ in range(12):
                g, H = grad(x), hess(x)
                w = np.linalg.eigvalsh(np.asarray(H))
                lam = 0.0 if w.min() > 0 else 1.1 * abs(w.min()) + 1e-3
                Hd = H + lam * jnp.eye(2)
                step = -jnp.linalg.solve(Hd, g)
                t = 1.0
                while f(x + t * step) > f(x) and t > 1e-3:
                    t *= 0.5
                models.append(np.concatenate([np.asarray(x + step), np.asarray(Hd).ravel(), [float(f(x))]]))
                x = x + t * step
                path.append(np.asarray(x))
                if float(jnp.linalg.norm(t * step)) < 1e-4:
                    break
            out["path"] = np.array(path)
            out["models"] = np.array(models)
    out["grids"] = np.stack(grids)

    # ---- the 30-parameter Hessian (10 patches x 3 parameters) at the input values
    hmask = get_mask(HESS_MASK, nside=NSIDE)
    (hidx,) = jnp.where(hmask)
    counts = {k: K_HESS for k in ("beta_dust_patches", "temp_dust_patches", "beta_pl_patches")}
    lab = kmeans_clusters(jax.random.key(3), hmask, hidx, counts, max_patches=counts)
    keys = ("beta_dust", "temp_dust", "beta_pl")
    lo = jnp.asarray(np.tile([LOWER[k] for k in keys], K_HESS))
    span = jnp.asarray(np.tile([UPPER[k] - LOWER[k] for k in keys], K_HESS))

    def vec_to_params(u):                             # u in [0, 1], grouped by patch
        v = (lo + span * u).reshape(K_HESS, 3)
        return {k: v[:, i] for i, k in enumerate(keys)}

    u0 = (jnp.asarray(np.tile([TRUTH[k] for k in keys], K_HESS)) - lo) / span
    hs, eig, gs = [], [], []
    for r in NOISE:
        noised = jax.tree.map(lambda x, w, s: x + r * w * s, d_full, white_full, sigma)
        nll = nll_on(get_cutout_from_mask(noised, hidx, axis=-1), lab)
        f = lambda u: nll(vec_to_params(u))
        H = np.asarray(jax.jit(jax.hessian(f))(u0))
        hs.append(H)
        eig.append(np.linalg.eigvalsh(H))
        gs.append(np.asarray(jax.jit(jax.grad(f))(u0)))
    out["hess30"], out["eig30"], out["grad30"] = np.stack(hs), np.stack(eig), np.stack(gs)

    # ---- the foreground SNR per pixel
    sig_q = np.asarray(sigma.q).reshape(len(nu), -1)[:, 0]
    _, fg = sky("d1s1")
    i140 = int(np.argmin(np.abs(np.asarray(nu) - 140.0)))
    out["snr_fg"] = np.sqrt(fg[i140, 1] ** 2 + fg[i140, 2] ** 2) / sig_q[i140]
    out["sigma_q"] = sig_q
    out["mask"] = np.asarray(mask)
    CACHE.mkdir(exist_ok=True)
    np.savez(NPZ, **out)
    return out


if __name__ == "__main__":
    d = load()
    print("patch pixels", d["npix_patch"], "path steps", len(d["path"]) - 1)
    for r, e in zip(d["noise"], d["eig30"]):
        print(f"noise {r:4.2f}: eig min {e.min():.3e} max {e.max():.3e} cond {e.max() / e.min():.3e}")
    s = d["snr_fg"][d["mask"].astype(bool)]
    print("fg SNR percentiles 5/50/95:", np.percentile(s, [5, 50, 95]))
