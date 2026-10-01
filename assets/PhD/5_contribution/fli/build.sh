#!/usr/bin/env bash
# Build the field-level section. The slides rewritten on 2026-09-30 have one folder each
# (01_pipeline ... 19_born; numbered in the order they were made, not the slide order); the backup slides still use the flat files of this directory.
# Staged TikZ figures have several renders from one source.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FORCE_REGEN="${FORCE_REGEN:-0}"
JAXFLI=/home/wassim/Projects/NBody/jax-fli

# 02 forward model: one source, three layers on one bounding box (shared arrow, implicit, explicit)
( cd 02_forward_model
  for st in 0 1 2; do
      out=$(echo fm_forward.svg fm_implicit.svg fm_explicit.svg | cut -d' ' -f$((st + 1)))
      if [ -f "$out" ] && [ "$FORCE_REGEN" != "1" ]; then echo "  02_forward_model/$out exists, skipping"; continue; fi
      echo "  building 02_forward_model/$out"
      pdflatex -interaction=nonstopmode -halt-on-error -jobname=fm_tmp \
          "\def\stage{$st}\input{forward_model.tex}" >/dev/null
      pdftocairo -svg fm_tmp.pdf "$out"
  done
  rm -f fm_tmp.* )

# 05 particle mesh: the six-step loop lit seven ways (all, then one step at a time), for pm_run.py
( cd 05_pm
  for lit in 0 1 2 3 4 5 6; do
      out="pm_cycle_$lit.png"
      if [ -f "$out" ] && [ "$FORCE_REGEN" != "1" ]; then continue; fi
      echo "  building 05_pm/$out"
      pdflatex -interaction=nonstopmode -halt-on-error -jobname=cyc_tmp \
          "\def\lit{$lit}\input{pm_cycle.tex}" >/dev/null
      pdftocairo -png -r 220 -singlefile -transp cyc_tmp.pdf "pm_cycle_$lit"
  done
  rm -f cyc_tmp.* )

if [ ! -f adjoints.svg ] || [ "$FORCE_REGEN" = "1" ]; then
    echo "  building adjoints.svg"
    pdflatex -interaction=nonstopmode -halt-on-error adjoints.tex >/dev/null
    pdftocairo -svg adjoints.pdf adjoints.svg
fi

# every generator, in its own folder, in the environment its `# ENV:` header names
run() {   # run <dir> <script>
    local env_line
    env_line=$(sed -n 's/^# *ENV: *//p' "$1/$2" | head -1)
    echo "  [${env_line:-shared}] $1/$2"
    case "${env_line:-shared}" in
      jax-fli) ( cd "$1" && JAX_PLATFORMS=cpu uv run --project "$JAXFLI" --no-sync python "$2" ) ;;
      skip)    ;;
      *)       ( cd "$1" && uv run --project "$(cd ../.. && pwd)" python "$2" ) ;;
    esac
}
run . convert_thesis_figures.py
run . convert_experiment_figures.py
run 07_distributed fetch_images.py
run 04_lpt limber_lin_vs_halofit.py
run 05_pm pm_run.py
run 07_distributed halo_exchange.py
run 09_observable lightcone_volume.py
run 10_painting painting_interp.py
run 10_painting painting_cl.py
run 11_lightcone lightcone_shells.py
run 12_spacing shell_spacing.py
run 13_nshells nshells.py
run 13_nshells nshells_kappa.py
run 14_drift drift.py
run 19_born born.py
run 15_scaling fetch_logo.py
run 15_scaling scaling.py
run 16_cosmogrid mesh_ladder.py
run 17_starlet fetch_tersenov.py
run 17_starlet starlet_l1.py
run 18_map map_data.py        # the MAP run, from HF jax-fli-sampling, into ../.cache
run 18_map map_anim.py
for p in pm3d_fragments.py; do
    [ -f "$p" ] && run . "$p"
done
rm -f ./*.aux ./*.log
