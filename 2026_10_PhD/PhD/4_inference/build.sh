#!/usr/bin/env bash
# Build every figure in the inference section, one folder per slide. The implicit/explicit
# diagrams share one source with two renders, so the section needs a driver of its own.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE_REGEN="${FORCE_REGEN:-0}"

tex() {   # tex <dir> <stem>: compile <stem>.tex to <stem>.svg in <dir>
    ( cd "$HERE/$1"
      if [ -f "$2.svg" ] && [ "$FORCE_REGEN" != "1" ]; then echo "  $1/$2.svg exists, skipping"; exit 0; fi
      echo "  building $1/$2.svg"
      pdflatex -interaction=nonstopmode -halt-on-error "$2.tex" >/dev/null
      pdftocairo -svg "$2.pdf" "$2.svg"
      rm -f "$2.pdf" "$2.aux" "$2.log" )
}

py() {    # py <dir> <script>: run a shared-env generator in <dir>
    echo "  [py] $1/$2"
    ( cd "$HERE/$1" && uv run --project "$HERE/.." python "$2" )
}

# 01 inverse problem: the two thumbnails the TikZ figure embeds come first.
echo "  [py/furax-cs] 01_inverse_problem/sky_freq_thumb.py"
( cd "$HERE/01_inverse_problem" && mkdir -p .cache && cd .cache
  uv run --project "$HERE/.." --group furax-cs python ../sky_freq_thumb.py )
py 01_inverse_problem gamma2_des.py
tex 01_inverse_problem inverse_problem

# 02 forward modelling: one source, two renders (implicit.svg, explicit.svg).
( cd "$HERE/02_forward_modelling"
  for st in 1 2; do
      out=$([ "$st" = 1 ] && echo implicit.svg || echo explicit.svg)
      if [ -f "$out" ] && [ "$FORCE_REGEN" != "1" ]; then echo "  02_forward_modelling/$out exists, skipping"; continue; fi
      echo "  building 02_forward_modelling/$out"
      pdflatex -interaction=nonstopmode -halt-on-error -jobname=ie_tmp \
          "\def\stage{$st}\input{implicit_explicit.tex}" >/dev/null
      pdftocairo -svg ie_tmp.pdf "$out"
  done
  rm -f ie_tmp.pdf ie_tmp.aux ie_tmp.log )

# 03 Bayesian inference
py 03_bayesian_inference bayes_posterior.py
tex 03_bayesian_inference hierarchical_model

# 04 optimising the posterior
py 04_optimising_posterior posterior_moves.py
