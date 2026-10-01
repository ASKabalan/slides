# Assets for the PhD defense deck (`2026_10_PhD`)

Every figure and animation in the deck is generated here, and its generator sits
beside it, the way `figures/chapN/fig_x.py` sits beside `fig_x.pdf` in the thesis.
Nothing in the deck points at a figure that cannot be rebuilt from this directory.

```
assets/PhD/
├── pyproject.toml, uv.lock   shared light environment (matplotlib, healpy, camb, …)
├── bake.sh                   rebuild everything, or one section
├── 1_intro/  2_outline/  4_inference/
├── 3_observation/            the two observables
│   ├── cmb/                  one subfolder per slide, in deck order:
│   │                         01_inflation … 10_r_constraints, scipol/ (shared)
│   └── weaklensing/
├── 5_contribution/           the thesis's two contributions
│   ├── compsep/
│   └── fli/
├── 6_ending/                 closing sections
│   └── software/ conclusion/ backup/
└── FOR_USAGE/                source material handed over, not generated here
```

## Rebuilding

```bash
bash bake.sh                 # build whatever is missing
FORCE_REGEN=1 bash bake.sh   # rebuild everything from scratch
bash bake.sh 3_observation/cmb 5_contribution/compsep     # one or more sections only
```

A generator skips itself when its output already exists, unless `FORCE_REGEN=1`.
Downloads and expensive intermediates are cached in `<section>/.cache/`, so a
rebuild costs no network and no second minimisation. A couple of handed-over
originals live in `source/` folders (the horn-antenna photo in
`1_intro/CMB/source/`, the weak-lensing illustration) — bake never touches a `source/` directory.

## The three environments

Each `.py` declares its environment on a header line, and `bake.sh` reads it:

| Header | Runs as | For |
|---|---|---|
| `# ENV: shared` | `uv run --project ../.. python x.py` | matplotlib, healpy, CAMB, HuggingFace replots, image fetching |
| `# ENV: jax-fli` | `JAX_PLATFORMS=cpu uv run --project ../.. --group jax-fli python x.py` (jax-fli installed from GitHub) | lightcone shells, spherical painting, Born maps, experiment replots |
| `# ENV: furax-cs` | `uv run --project ../.. --group furax-cs python x.py` (furax-cs installed from GitHub `main`) | PySM skies, the spectral likelihood, NLL scans |
| `# ENV: manim` | `uv run --project ../.. --group manim python x.py` | the animated scenes (Manim CE is its own dependency group); each file renders its own scenes to MP4 and keeps the last frame as a PNG beside itself |

`jax_enable_x64` must be set **before** importing `jax_fli` or `furax`: a masked
spin-2 `angular_cl` returns all-NaN in float32, and the initial-condition
gradients diverge.

A section whose figures need a driver of their own — several variants from one
TikZ source, an external replot, a manim render — ships a `build.sh`, and
`bake.sh` calls that instead of walking the files.

## TikZ

Each `.tex` carries its compile line as a comment on line 3, and `bake.sh` runs
the same command:

```
% Compile: pdflatex x.tex && pdftocairo -svg x.pdf x.svg
```

`pdf2svg` is not installed on this machine; `pdftocairo` is what the repo uses.
Figures meant to overlay inside an `.r-stack` share an explicit
`\useasboundingbox`, so successive fragments register exactly.

## Conventions

- SVG where the figure is vector, high-DPI PNG where it is raster, GIF for animation.
- Animations are built for the slide: white background, no title, no caption, no
  changing text. They are not standalone figures.
- Every image taken from outside carries a `[Credit: …]{.credit}` line on its slide.
- The SciPol illustrations are by **Ève Barlier**; **Josquin Errard** holds the
  SciPol ERC grant (No. 101044073). Both are credited wherever those images appear.
