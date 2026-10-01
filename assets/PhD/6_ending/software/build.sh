#!/usr/bin/env bash
# Software section: the ecosystem graph has two renders from one source (all
# packages, and only the four the uptake slide follows).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FORCE_REGEN="${FORCE_REGEN:-0}"
names=(ecosystem ecosystem_keep)
keep=(0 1)
for i in "${!names[@]}"; do
    out="${names[$i]}.svg"
    if [ -f "$out" ] && [ "$FORCE_REGEN" != "1" ]; then
        echo "  $out exists, skipping"; continue
    fi
    echo "  building $out (keep=${keep[$i]})"
    pdflatex -interaction=nonstopmode -halt-on-error -jobname=eco_tmp \
        "\def\keep{${keep[$i]}}\input{ecosystem.tex}" >/dev/null
    pdftocairo -svg eco_tmp.pdf "$out"
done
rm -f eco_tmp.pdf eco_tmp.aux eco_tmp.log
echo "  [py] fetch_logos.py"
uv run --project ../.. python fetch_logos.py
