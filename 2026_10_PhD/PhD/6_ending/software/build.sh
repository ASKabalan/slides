#!/usr/bin/env bash
# Software section: the ecosystem graph, then the logos of the projects that use it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FORCE_REGEN="${FORCE_REGEN:-0}"
if [ -f ecosystem.svg ] && [ "$FORCE_REGEN" != "1" ]; then
    echo "  ecosystem.svg exists, skipping"
else
    echo "  building ecosystem.svg"
    pdflatex -interaction=nonstopmode -halt-on-error -jobname=eco_tmp ecosystem.tex >/dev/null
    pdftocairo -svg eco_tmp.pdf ecosystem.svg
    rm -f eco_tmp.pdf eco_tmp.aux eco_tmp.log
fi
echo "  [py] fetch_logos.py"
uv run --project ../.. python fetch_logos.py
