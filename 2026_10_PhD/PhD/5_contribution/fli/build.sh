#!/usr/bin/env bash
# Build the field-level section: one folder per main slide, numbered in slide order (01_pipeline ...
# 12_conclusion), the shared cache in .cache/. The field-level backups live in 7_backup/4_fieldlevel and
# are built by ../../bake.sh; the figures copied from jax-fli and the thesis come through the two
# convert_*.py scripts. Staged TikZ figures have several renders from one source.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FORCE_REGEN="${FORCE_REGEN:-0}"
JAXFLI=/home/wassim/Projects/NBody/jax-fli

# 03 particle mesh: the six-step loop lit seven ways (all, then one step at a time), for pm_run.py
( cd 03_pm
  for lit in 0 1 2 3 4 5 6; do
      out="pm_cycle_$lit.png"
      if [ -f "$out" ] && [ "$FORCE_REGEN" != "1" ]; then continue; fi
      echo "  building 03_pm/$out"
      pdflatex -interaction=nonstopmode -halt-on-error -jobname=cyc_tmp \
          "\def\lit{$lit}\input{pm_cycle.tex}" >/dev/null
      pdftocairo -png -r 220 -singlefile -transp cyc_tmp.pdf "pm_cycle_$lit"
  done
  rm -f cyc_tmp.* )

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
run 02_lpt pk_lin_vs_halofit.py
run 03_pm pm_run.py
run 04_accuracy pm3d_fragments.py
run 05_distributed pencils.py
run 06_observable depth.py
run 07_painting particles_box.py
run 07_painting painting_interp.py
run 08_lightcone lightcone_shells.py
run 09_born born.py
run 10_cosmogrid mesh_ladder.py
run 11_map map_data.py        # the MAP run, from HF jax-fli-sampling, into ../.cache
run 11_map map_anim.py
run 12_conclusion moving_mesh.py
rm -f ./*.aux ./*.log
