#!/usr/bin/env bash
# Render the outline map: the full map plus one focused variant per section.
# Every variant shares rail.tex's \useasboundingbox, so the SVGs overlay 1:1.
#   bash build.sh            # build what is missing
#   FORCE_REGEN=1 bash build.sh
#
# The badges are vector figures from other sections, converted to PDF in
# .cache/ for pdflatex (the ΛCDM badges are viewBox crops of the boxed region).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FORCE_REGEN="${FORCE_REGEN:-0}"
PHD=..

mkdir -p .cache
# the ΛCDM badges show only the boxed region (CMB / present-day ellipse)
crop() {  # crop <src.svg> <out stem> <viewBox>
    sed -e "0,/width=\"[^\"]*\" height=\"[^\"]*\" viewBox=\"[^\"]*\"/s//viewBox=\"$3\"/" "$1" > ".cache/$2.svg"
    rsvg-convert -f pdf -o ".cache/$2.pdf" ".cache/$2.svg"
}
crop "$PHD/1_intro/lcdm/lcdm_model_cmb.svg" lcdm_cmb "1020 780 500 860"
crop "$PHD/1_intro/lcdm/lcdm_model_lss.svg" lcdm_lss "3270 560 630 1280"
rsvg-convert -f pdf -o .cache/mle_vs_bayes_icon.pdf "$PHD/4_inference/03_bayesian_inference/mle_vs_bayes_icon.svg"
# the field-level forward model (prior -> forward -> observable)
rsvg-convert -f pdf -o .cache/fli_pipeline.pdf "$PHD/5_contribution/fli/01_pipeline/pipeline.svg"

names=(full cmb lensing stats c1 c2 conclusion)
focus=(0    2   3       4     5  6  7)

for i in "${!names[@]}"; do
    [ "${names[$i]}" = full ] && out="rail_full.svg" || out="rail_focus_${names[$i]}.svg"
    if [ -f "$out" ] && [ "$FORCE_REGEN" != "1" ]; then
        echo "  $out exists, skipping"; continue
    fi
    echo "  building $out (focus=${focus[$i]})"
    pdflatex -interaction=nonstopmode -halt-on-error \
        -jobname="rail_tmp" "\def\focus{${focus[$i]}}\input{rail.tex}" >/dev/null
    pdftocairo -svg rail_tmp.pdf "$out"
done
rm -f rail_tmp.pdf rail_tmp.aux rail_tmp.log
# by-products of compiling rail.tex by hand (editor / latexmk)
rm -f rail.pdf rail.aux rail.log rail.fls rail.fdb_latexmk rail.synctex.gz
ls -1 rail_*.svg
