#!/usr/bin/env bash
# Rebuild every figure and animation for the PhD defense deck.
#
#   bash bake.sh                 # build what is missing
#   FORCE_REGEN=1 bash bake.sh   # rebuild everything
#   bash bake.sh cmb compsep     # build only these sections
#
# Each generator sits beside its output and declares its environment on a
# header line "# ENV: shared | jax-fli | furax-cs | manim". TikZ sources carry
# their own compile line and are turned into SVG with pdflatex + pdftocairo.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FORCE_REGEN="${FORCE_REGEN:-0}"



SECTIONS=(0_cover 1_intro 2_outline 3_observation/cmb 3_observation/weaklensing 4_inference 5_contribution/compsep 5_contribution/fli 6_ending/software 6_ending/backup)
[ $# -gt 0 ] && SECTIONS=("$@")

fail=0
run_py() {
    local f="$1" env
    env=$(sed -n 's/^# *ENV: *//p' "$f" | head -1)
    env="${env:-shared}"
    echo "  [py/$env] $(basename "$f")"
    case "$env" in
        shared)   ( cd "$(dirname "$f")" && uv run --project "$HERE" python "$(basename "$f")" ) ;;
        jax-fli)  ( cd "$(dirname "$f")" && uv run --project "$HERE" --group jax-fli python "$(basename "$f")" ) ;;
        furax-cs) ( cd "$(dirname "$f")" && uv run --project "$HERE" --group furax-cs python "$(basename "$f")" ) ;;
        manim)    ( cd "$(dirname "$f")" && uv run --project "$HERE" --group manim python "$(basename "$f")" ) ;;
        skip)     echo "        (skipped by header)" ;;
        *)        echo "        unknown ENV '$env'"; return 1 ;;
    esac
}

compile_tex() {
    local f="$1" stem dir
    dir="$(dirname "$f")"; stem="$(basename "$f" .tex)"
    if [ -f "$dir/$stem.svg" ] && [ "$FORCE_REGEN" != "1" ]; then
        echo "  [tex] $stem.svg exists, skipping"
        return 0
    fi
    echo "  [tex] $stem.tex"
    ( cd "$dir" \
      && pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >/dev/null \
      && pdftocairo -svg "$stem.pdf" "$stem.svg" )
}

for s in "${SECTIONS[@]}"; do
    d="$HERE/$s"
    [ -d "$d" ] || { echo "== $s: no such section"; continue; }
    echo "== $s"
    # A section may ship a build.sh when its figures need a driver of their own
    # (several variants from one source, an external replot, a manim render).
    if [ -f "$d/build.sh" ]; then
        echo "  [sh]  build.sh"
        ( cd "$d" && bash build.sh ) || { echo "  !! build.sh failed"; fail=1; }
        continue
    fi
    shopt -s nullglob
    # generators sit in the section or one level down (one subfolder per slide)
    for f in "$d"/*.tex "$d"/*/*.tex; do compile_tex "$f" || { echo "  !! $f failed"; fail=1; }; done
    for f in "$d"/*.py  "$d"/*/*.py;  do run_py      "$f" || { echo "  !! $f failed"; fail=1; }; done
    shopt -u nullglob
done

# TeX by-products; the .svg is the artefact that ships.
find "$HERE" -maxdepth 4 \( -name '*.aux' -o -name '*.log' -o -name '*.out' \) -delete

[ "$fail" = 0 ] && echo "bake: all sections built" || echo "bake: finished with failures"
exit "$fail"
