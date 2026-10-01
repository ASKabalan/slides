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
single source of truth for every number shown. The forward-model diagram comes
from jax-fli too (assets/PIPELINE.svg, built from assets/pipeline/pipeline.tex).

Outputs (this directory, or the subfolder named in the key): one file per entry
in FIGURES below.
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import force_regen

EXP = Path("/home/wassim/Projects/NBody/jax-fli/docs/5-experiments")
MASK = EXP / "08-masked-shear/assets"

FIGURES = {
    # the forward model end to end (jax-fli README diagram)
    "01_pipeline/pipeline.svg": Path("/home/wassim/Projects/NBody/jax-fli/assets/PIPELINE.svg"),
    # the evolved density and one lightcone shell, the two ends of the painting step
    "10_painting/density_box.png": Path("/home/wassim/Projects/NBody/jax-fli/assets/pipeline/pipeline_images/02_final_density.png"),
    "10_painting/density_shell.png": Path("/home/wassim/Projects/NBody/jax-fli/assets/pipeline/pipeline_images/03_spherical_projection.png"),
    # the survey mask and what it costs
    "masked_shear_masks.svg": MASK / "fig01-masks.svg",
    "masked_shear_observer_quad.svg": MASK / "fig02-observer-quad.svg",
    "masked_shear_observer_large.svg": MASK / "fig03-observer-large.svg",
    "masked_shear_residual_pdf.svg": MASK / "fig07-gamma1-residual-pdf.svg",
    "masked_shear_residual_maps.svg": MASK / "fig08-gamma1-residual-maps.svg",
    "masked_shear_ee.svg": MASK / "fig09-ee-spectra.svg",
}
# These are PDFs and need converting rather than copying.
CONVERT = {
    "shear_gamma1_des.svg": MASK / "fig10-gamma1-des.pdf",
    "shear_gamma2_des.svg": MASK / "fig11-gamma2-des.pdf",
}

if shutil.which("pdftocairo") is None:
    sys.exit("pdftocairo not found")

made, missing = 0, []
for out, src in FIGURES.items():
    if (HERE / out).exists() and not force_regen():
        continue
    if not src.exists():
        missing.append(f"{out}  <-  {src}")
        continue
    (HERE / out).parent.mkdir(exist_ok=True)
    shutil.copy(src, HERE / out)
    made += 1
    print(f"copied {out}")

for out, src in CONVERT.items():
    if (HERE / out).exists() and not force_regen():
        continue
    if not src.exists():
        missing.append(f"{out}  <-  {src}")
        continue
    subprocess.run(["pdftocairo", "-svg", str(src), str(HERE / out)], check=True)
    made += 1
    print(f"wrote {out}")

print(f"\n{made} experiment figures brought across")
if missing:
    print("MISSING:")
    for m in missing:
        print("   ", m)
