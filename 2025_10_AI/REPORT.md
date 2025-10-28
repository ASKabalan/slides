# Slides Status Report — 2025_10_AI

Author: Wassim Kabalan
Date: October 2025

## Summary

- Deck is substantially complete and aligned with the brief. Approximately 47 slides detected (within 40±5 when considering backup slides), covering Generative Models, Evaluation, JAX ecosystem, and cosmology demos.
- Two major deliverables are missing: the three runnable notebooks and `env_setup.md`.
- A couple of image references appear broken or non-existent; some build artifacts are checked in; and styling is duplicated locally instead of using shared styles.

## Deliverables Compliance

- Slides: Present at `2025_10_AI/index.qmd` (rendered `index.html` exists).
- Notebooks: Missing
  - `2025_10_AI/Generative_AI_JAX_GZ10_VAE.ipynb` – not found
  - `2025_10_AI/Generative_AI_JAX_GZ10_VAE_NF.ipynb` – not found
  - `2025_10_AI/Generative_AI_JAX_PSF_Bayesian.ipynb` – not found
- Environment: Missing
  - `2025_10_AI/env_setup.md` – not found
- Assets: Present under `assets/` (both repo root `assets/` and local `2025_10_AI/assets/` include required figures). Slide references resolve to repo‑root `assets/` via `../assets/...`.

## Slide Coverage vs Blueprint

- Title and Outline: Present and styled; background image and logos render variables defined.
- Generative Models: GANs → WGAN; VAE and β‑VAE; Normalizing Flows; Diffusion/Score models; model comparison table; evaluation metrics (FID, Precision/Recall).
- JAX Ecosystem: What is JAX; JAX vs PyTorch; RNG pattern; Transforms (grad/jit/vmap/pmap); Flax example; ecosystem overview; performance/RNG best practices.
- Cosmology Demos: GZ10 dataset overview; VAE + redshift example; Flow prior upgrade; PSF Bayesian inference with BlackJAX.
- Backup Slides: Mathematical derivations (GAN optimal D, FID, ELBO) and troubleshooting.

Observed slide count (by `##` headings): ~47 (includes backup and checklist slides).

## Asset and Link Status

- Most images referenced via `../assets/...` exist in repo‑root `assets/` (also duplicated under `2025_10_AI/assets/`). Examples: `gan_diagram.png`, `vae_arch.png`, `flow_diagram.png`, `ddpm_box.png`, `jax_openxla.png`.
- Broken/missing references in slides:
  - `2025_10_AI/index.qmd`: VAE in Cosmology slide points to:
    - `TASK/Final_Report_Binh_Nguyen/images/deblender_diagram.png` — images/ directory does not exist.
    - `TASK/Final_Report_Binh_Nguyen/images/recons_t2.png` — images/ directory does not exist.
  - Suggestion: Extract figures from `2025_10_AI/TASK/Final_Report_Binh_Nguyen.pdf` (or replace with locally available assets) and save under `../assets/` (or `2025_10_AI/assets/`), then update paths in slides.
- Placeholder note: PSF concept slide uses `../assets/placeholder.png` which exists; consider replacing with notebook‑generated figure when Notebook C is ready.

## Quarto Config & Styling

- Front matter in `2025_10_AI/index.qmd` uses `format: revealjs` with theme `[default, css/custom.scss]` and `template-partials: [css/title-slide.html]` — OK.
- Title‑slide background image and `logo1` variable are set and used by the custom `title-slide.html` template.
- Local duplicated styles present under `2025_10_AI/css/`:
  - `2025_10_AI/css/custom.scss`
  - `2025_10_AI/css/title-slide.html`
  - Repo guideline prefers shared styling in repo‑root `css/`. If no deck‑specific overrides are required, reference `../../css/...` or keep only minimal overrides locally.
- Deck `_quarto.yml` is set to a website project:
  - `2025_10_AI/_quarto.yml` contains `project: type: website` and outputs into `.`. For a single reveal deck, this is optional; consider simplifying to deck‑local rendering without generating site scaffolding.
- Minor config nits to verify during render:
  - `pdfSeparateFragments: true` may need `pdf-separate-fragments: true` in Quarto’s revealjs options.
  - `presentation-size: max-scale` is uncommon; Quarto typically supports `width`, `height`, `margin`, `min-scale`, `max-scale`. Confirm desired behavior.

## Build Artifacts in Repo

- The deck directory contains build output:
  - `2025_10_AI/index.html`
  - `2025_10_AI/search.json`
  - `2025_10_AI/site_libs/`
  - `.quarto/` is ignored locally, but other artifacts are not. Repo guideline advises not to commit generated files. Consider adding to `2025_10_AI/.gitignore`:
    - `/index.html`
    - `/search.json`
    - `/site_libs/`

## Content Quality Checks

- Equations and code samples: Present and readable; math is set in LaTeX blocks; code matches JAX/Flax idioms and uses explicit PRNG keys.
- Notes vs on‑slide text: Slides stay concise; speaker notes include details — aligns with “minimal‑text, diagram‑first” guidance.
- Slide numbering enabled and transitions configured.

## Blocking Gaps to Ship

1. Missing notebooks (A/B/C) per TASK:
   - `Generative_AI_JAX_GZ10_VAE.ipynb`
   - `Generative_AI_JAX_GZ10_VAE_NF.ipynb`
   - `Generative_AI_JAX_PSF_Bayesian.ipynb`
2. Missing environment instructions:
   - `2025_10_AI/env_setup.md` with CPU and CUDA install commands (provided in TASK).
3. Broken images:
   - Extract or replace the two `TASK/Final_Report_Binh_Nguyen/images/*.png` references.
4. Build artifacts tracked:
   - Clean and ignore `index.html`, `search.json`, `site_libs/`.
5. Styling duplication:
   - Decide between local overrides vs shared repo‑root `css/` (recommended: reuse shared styles, keep only deck‑specific changes locally if needed).

## Suggested Next Actions

- Create `env_setup.md` and copy the install blocks from `2025_10_AI/TASK/TASK.md` (CPU + CUDA), plus a troubleshooting section.
- Implement the three notebooks following the specs already summarized in `2025_10_AI/README.md` and referenced in slides. Save diagnostics/plots to `../assets/generated/`.
- Fix slide image paths for the VAE cosmology slide by adding the figures to `assets/` and updating `index.qmd`.
- Normalize styling: point `index.qmd` to repo‑root `css/custom.scss` and `css/title-slide.html` unless deck‑specific variants are required.
- Update `.gitignore` in `2025_10_AI/` to exclude build outputs; run `quarto clean 2025_10_AI` before committing.
- Run: `quarto render 2025_10_AI` and manually verify layout, link targets, speaker notes, and title‑slide background/logo rendering.

## Quick Checklist

- [ ] `env_setup.md` added and referenced
- [ ] A/B/C notebooks implemented and runnable (CPU+GPU)
- [ ] Broken image references replaced/fixed
- [ ] Slide asset paths verified
- [ ] Styling deduplicated or justified
- [ ] Build artifacts cleaned and ignored
- [ ] Full deck renders without warnings

