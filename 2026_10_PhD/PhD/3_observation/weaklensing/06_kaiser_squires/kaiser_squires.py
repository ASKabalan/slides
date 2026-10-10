# ENV: jax-fli
"""Kaiser-Squires on the sphere: kappa -> (gamma1, gamma2).

Paints a full-sky jax-fli LPT lightcone onto spherical shells (no N-body step), makes the
Born convergence of one source bin, then takes the shear with jax-fli's
``SphericalKappaField.get_shear`` (forward KS).

This script only draws: each map spins as a looping transparent orthographic GIF, all with the
same rotation so the three read as one sky.

Writes kappa.gif, gamma1.gif, gamma2.gif with a still PNG of each (first frame) for print.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import ortho_frame, skip_if_built, write_alpha_gif

NAMES = ["kappa", "gamma1", "gamma2"]
OUTS = [f"{n}.{e}" for n in NAMES for e in ("gif", "png")]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import jax_cosmo as jc
import jax_fli as jfli

NSIDE, MESH = 256, (256, 256, 256)
N_FRAMES, LAT, LON0 = 40, -30.0, 30.0
PX = 260                                     # GIF width in pixels
L_CUT = 120                                  # above ~150 the 256^3 lattice shows as great circles

# ---------------------------------------------------------------- simulation
# The Born convergence of the LPT lightcone is kept in .cache/ (git-ignored).
cosmo = jc.Planck18()
CACHE = HERE / ".cache" / "kappa_born.parquet"
if not CACHE.exists():
    # observer half a cell off the particle lattice: on a lattice plane, that whole plane of
    # particles projects onto one great circle in every shell
    obs = tuple(0.5 + 0.5 / n for n in MESH)
    box = tuple(float(b) for b in jfli.compute_box_size_from_redshift(cosmo, 1.0, obs))
    ic = jfli.gaussian_initial_conditions(jax.random.PRNGKey(3), MESH, box, cosmo=cosmo, nside=NSIDE,
                                          observer_position=obs)
    # LPT lightcone straight onto spherical shells (no N-body step)
    lightcone, _ = jfli.lpt(cosmo, ic, nb_shells=16, order=2,
                            painting=jfli.PaintingOptions(target="spherical", scheme="ngp"))
    kappa = jfli.born(cosmo, lightcone, nz_shear=jfli.io.get_stage3_nz_shear()[-1:])
    CACHE.parent.mkdir(exist_ok=True)
    jfli.io.Catalog(field=kappa, cosmology=cosmo).to_parquet(str(CACHE))
kappa = jfli.io.Catalog.from_parquet(str(CACHE)).field[0][0]
# shot noise dominates the pixel scale at this resolution: keep the large scales only
kappa = kappa.scale_cut(L_CUT, L_CUT // 3, l_min=2)

# ---------------------------------------------------------------- Kaiser-Squires
shear = kappa.get_shear()

k = np.asarray(kappa.array)
g = np.asarray(shear.array)
maps = {
    "kappa": (k, "viridis"),
    "gamma1": (g[0], "magma"),
    "gamma2": (g[1], "magma"),
}
scale = {"kappa": np.percentile(k, [1, 99.5])}
gmax = np.percentile(np.abs(g), 99.5)
scale["gamma1"] = scale["gamma2"] = (-gmax, gmax)


# ---------------------------------------------------------------- frames
lons = LON0 + np.arange(N_FRAMES) * 360.0 / N_FRAMES
for name in NAMES:
    m, cmap = maps[name]
    vmin, vmax = scale[name]
    frames = [ortho_frame(m, lon, LAT, PX, cmap=cmap, min=vmin, max=vmax) for lon in lons]
    write_alpha_gif(frames, HERE / f"{name}.gif")
    print("wrote", name)
