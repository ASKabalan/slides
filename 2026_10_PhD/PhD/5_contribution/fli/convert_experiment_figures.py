#!/usr/bin/env python3
# ENV: shared
"""
Figures from the jax-fli experiment suite, brought across for the slides.

Every experiment under jax-fli/docs/5-experiments/ ships a build.py that
recomputes its data and rewrites its assets; several of them replot on CPU
without a GPU, for example

    cd /home/wassim/Projects/NBody/jax-fli/docs/5-experiments/08-masked-shear
    JAX_PLATFORMS=cpu uv run --no-sync python build.py

This script only copies or converts the results, so the experiment stays the
single source of truth for every number shown. The field-level sampling backups
come from the payload of jax-fli docs/3-sampling-and-inference (notebooks 16 and
17), rendered there by fieldlevel_data.py, kappa_anim.py, starlet_anim.py and
posterior.py (JAX_PLATFORMS=cpu uv run --no-sync python <script> in that folder). The forward-model diagram comes
from jax-fli too (assets/PIPELINE.svg, built from assets/pipeline/pipeline.tex).

Outputs (this directory, or the subfolder named in the key): one file per entry
in FIGURES below.
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import force_regen

EXP = Path("/home/wassim/Projects/NBody/jax-fli/docs/5-experiments")
MASK = EXP / "08-masked-shear/assets"
MAP = EXP / "13-map-lpt2-mass-mapping/assets"
PAYLOAD = Path("/home/wassim/Projects/NBody/jax-fli/docs/3-sampling-and-inference/payload")

FIGURES = {
    # the forward model end to end (jax-fli README diagram)
    "5_contribution/fli/01_pipeline/pipeline.svg": Path("/home/wassim/Projects/NBody/jax-fli/assets/PIPELINE.svg"),
    # one lightcone shell, the end of the painting step (the particles: 07_painting/particles_box.py)
    "5_contribution/fli/07_painting/density_shell.png": Path("/home/wassim/Projects/NBody/jax-fli/assets/pipeline/pipeline_images/03_spherical_projection.png"),
    # the survey mask and what it costs
    "7_backup/3_weaklensing/03_masked_shear/masked_shear_masks.svg": MASK / "fig01-masks.svg",
    "7_backup/3_weaklensing/03_masked_shear/masked_shear_residual_maps.svg": MASK / "fig08-gamma1-residual-maps.svg",
    # the MAP reconstruction under DES Y3 and Euclid noise (experiment 13)
    "7_backup/4_fieldlevel/20_map_euclid/map_des_euclid_coherence.svg": MAP / "fig13-des-vs-euclid-coherence.svg",
    "7_backup/4_fieldlevel/20_map_euclid/map_des_euclid_starlet.svg": MAP / "fig14-des-vs-euclid-starlet.svg",
}
# the field-level sampling backups: posterior kappa and its starlet l1 norm (notebook 16), and the
# cosmology at fixed initial conditions (notebook 17)
SAMPLING = [f"kappa_{q}_bin{b}.png" for q in ("truth", "std") for b in (2, 3)]
SAMPLING += [f"{v}_bin{b}{ext}" for v in ("kappa_samples", "kappa_diff", "starlet_l1", "starlet_resid")
             for b in (2, 3) for ext in (".mp4", "_last.png")]
SAMPLING += [f"cbar_{q}.svg" for q in ("kappa", "std", "diff")] + ["posterior_nb17.svg"]
FIGURES.update({f"7_backup/4_fieldlevel/21_sampling/{name}": PAYLOAD / name for name in SAMPLING})
made, missing = 0, []
for out, src in FIGURES.items():
    if (ROOT / out).exists() and not force_regen():
        continue
    if not src.exists():
        missing.append(f"{out}  <-  {src}")
        continue
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, ROOT / out)
    made += 1
    print(f"copied {out}")

print(f"\n{made} experiment figures brought across")
if missing:
    print("MISSING:")
    for m in missing:
        print("   ", m)
