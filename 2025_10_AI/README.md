# Generative AI with JAX Workshop

**Workshop Title**: Generative AI with JAX (after CNNs)
**Presenter**: Wassim Kabalan
**Affiliation**: APC/CNRS/Université Paris Cité
**Date**: October 2025

## Overview

This workshop provides a comprehensive introduction to generative AI models (GANs, VAEs, Flows, Diffusion) and their implementation in JAX, with hands-on applications to cosmological data analysis.

## Contents

### 1. Slide Deck (`index.qmd`)

A 40+ slide presentation covering:

- **Introduction** (Slides 1-2): What is generative AI?
- **Generative Models** (Slides 3-13):
  - GANs and WGANs
  - VAEs and β-VAEs
  - Normalizing Flows
  - Diffusion Models
  - Model comparison and evaluation metrics
- **JAX Ecosystem** (Slides 14-22):
  - JAX fundamentals vs PyTorch
  - Core transforms (grad, jit, vmap, pmap)
  - Flax for neural networks
  - Ecosystem overview (Optax, BlackJAX, Distrax, etc.)
- **Hands-On Applications** (Slides 23-30):
  - GZ10 galaxy dataset
  - VAE for morphology analysis
  - PSF Bayesian inference
- **Best Practices** (Slides 31-33): RNG management and performance tips
- **Backup Slides** (Slides 34-40): Mathematical derivations and troubleshooting

**To render slides**:
```bash
quarto render index.qmd
```

View the rendered slides by opening `index.html` in a browser.

### 2. Environment Setup (`env_setup.md`)

Complete installation instructions for:
- CPU-only setup (universal)
- NVIDIA CUDA 12 setup (Linux GPU)
- All required libraries (JAX, Flax, Optax, BlackJAX, Distrax, etc.)
- Troubleshooting guide

### 3. Jupyter Notebooks

Three complete, runnable notebooks demonstrating practical applications:

#### Notebook A: `Generative_AI_JAX_GZ10_VAE.ipynb`
**Goal**: Train a VAE on Galaxy Zoo 10 dataset and predict redshift from latent space

**Contents**:
- Load and preprocess GZ10 dataset (~17k galaxy images)
- Implement Flax Encoder/Decoder with 3-layer conv architecture
- ELBO loss with β warm-up schedule (0→1 over first 10% training)
- Training loop with Adam optimizer (5 epochs, batch size 64)
- Visualization: reconstructions, generated samples, training curves
- Scientific application: predict galaxy redshift from latent μ using Ridge regression
- Save outputs to `../assets/generated/`

**Runtime**: ~10-15 min on CPU, ~2-3 min on GPU

#### Notebook B: `Generative_AI_JAX_GZ10_VAE_NF.ipynb`
**Goal**: Improve VAE by replacing Gaussian prior with Normalizing Flow

**Contents**:
- Load trained VAE from Notebook A (or retrain quickly)
- Implement RealNVP flow with 4 coupling layers using Distrax
- Modified ELBO using flexible flow prior
- Train for 3 epochs
- Comparison: Gaussian prior vs Flow prior samples
- Visualize latent space distributions
- Re-evaluate redshift prediction with flow-enhanced latents
- Demonstrate improved latent space utilization

**Runtime**: ~8-12 min on CPU, ~2-3 min on GPU

#### Notebook C: `Generative_AI_JAX_PSF_Bayesian.ipynb`
**Goal**: Bayesian inference of PSF width using BlackJAX NUTS sampler

**Contents**:
- Generate synthetic galaxy data with known PSF (σ=2.0)
- Implement forward model: convolution using `lax.conv_general_dilated`
- Define Bayesian model: likelihood + prior on log(σ)
- MCMC sampling with BlackJAX NUTS (2000 steps, 1000 burn-in)
- Diagnostics: trace plots, posterior histograms, acceptance rates
- Posterior predictive checks
- Discussion of surrogate likelihoods (conditional flows) for speedup
- All diagnostics saved to `../assets/generated/`

**Runtime**: ~5-10 min on CPU, ~1-2 min on GPU

### 4. Assets

Downloaded assets in `../assets/`:
- `gan_diagram.png`, `gan_title.png`, `gan_paper.pdf`
- `vae_arch.png`, `vae_title.png`, `vae_paper.pdf`, `vae_no_kl.png`, `vae_with_kl.png`
- `flow_diagram.png`, `flow_title.png`, `flow_paper.pdf`
- `ddpm_box.png`, `ddpm_title.png`, `ddpm_paper.pdf`, `diffusion_spiral.png`, `sde_paper.pdf`
- Placeholder images for blocked URLs (`generated_faces.jpg`, `discrete_cdf.png`, etc.)

Generated outputs will be saved to `../assets/generated/` by the notebooks.

## Workshop Structure

**Total Duration**: ~3-4 hours

1. **Presentation** (45-60 min): Overview of generative models and JAX
2. **Hands-On Session 1** (60 min): Notebooks A & B (VAE applications)
3. **Break** (15 min)
4. **Hands-On Session 2** (45 min): Notebook C (Bayesian inference)
5. **Discussion & Q&A** (15-30 min)

## Prerequisites

**Required knowledge**:
- Python programming
- Basic machine learning (neural networks, backpropagation)
- NumPy array operations
- Basic probability and statistics

**Helpful but not required**:
- PyTorch or TensorFlow experience
- Bayesian inference concepts
- Astronomical imaging knowledge

## Running the Notebooks

1. **Install environment** (see `env_setup.md`):
   ```bash
   python -m venv genjax && source genjax/bin/activate
   pip install -U pip wheel
   pip install -U "jax[cpu]"  # or "jax[cuda12]" for GPU
   pip install flax==0.8.* optax==0.2.* orbax-checkpoint==0.5.* clu==0.0.12
   pip install blackjax==1.* distrax==0.1.* flowjax==0.5.* diffrax==0.5.*
   pip install datasets==3.* huggingface_hub==0.24.* einops matplotlib seaborn scikit-learn pillow tqdm
   ```

2. **Launch Jupyter**:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

3. **Run notebooks in order**: A → B → C

4. **Check outputs** in `../assets/generated/`

## Key Learning Outcomes

By the end of this workshop, participants will:

1. Understand the mathematical foundations of generative models (VAE, GAN, Flow, Diffusion)
2. Know when to use each generative model type based on application requirements
3. Be proficient with JAX's core transformations (grad, jit, vmap, pmap)
4. Implement neural networks in Flax/Linen
5. Apply generative models to scientific data (galaxy morphology)
6. Perform Bayesian inference with gradient-based MCMC (NUTS)
7. Understand how to integrate JAX ecosystem libraries (Optax, BlackJAX, Distrax)
8. Follow best practices for reproducible scientific computing in JAX

## References

### Papers
- **GAN**: Goodfellow et al. (2014) - `../assets/gan_paper.pdf`
- **WGAN**: Arjovsky et al. (2017)
- **VAE**: Kingma & Welling (2013) - `../assets/vae_paper.pdf`
- **β-VAE**: Higgins et al. (2017)
- **Normalizing Flows**: Rezende & Mohamed (2015) - `../assets/flow_paper.pdf`
- **DDPM**: Ho et al. (2020) - `../assets/ddpm_paper.pdf`
- **Score-based Models**: Song et al. (2020) - `../assets/sde_paper.pdf`

### Blog Posts
- Lilian Weng's [blog](https://lilianweng.github.io/posts/) (excellent visual explanations)
  - [GANs](https://lilianweng.github.io/posts/2017-08-20-gan/)
  - [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
  - [Flows](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
  - [Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

### Documentation
- [JAX](https://jax.readthedocs.io/)
- [Flax](https://flax.readthedocs.io/)
- [Optax](https://optax.readthedocs.io/)
- [BlackJAX](https://blackjax.readthedocs.io/)
- [Distrax](https://github.com/deepmind/distrax)
- [FlowJAX](https://flowjax.readthedocs.io/)

### Dataset
- [MultimodalUniverse GZ10](https://huggingface.co/datasets/MultimodalUniverse/gz10)

## Attribution

**Slide Content**: Diagrams and explanations inspired by Lilian Weng's blog posts
**Papers**: arXiv papers as cited
**Dataset**: MultimodalUniverse Galaxy Zoo 10 (Hugging Face)
**Libraries**: JAX, Flax, Optax, BlackJAX, Distrax ecosystems

## Troubleshooting

### Common Issues

**1. CUDA not found**:
- Verify CUDA installation: `nvcc --version`
- Reinstall JAX with CUDA support: `pip install --upgrade "jax[cuda12]"`

**2. Out of memory**:
- Reduce batch size in notebooks
- Use smaller image resolution (32×32 instead of 64×64)
- Reduce number of epochs

**3. Dataset download slow/fails**:
- Notebooks include fallback to synthetic data generation
- Manually cache dataset: `from datasets import load_dataset; load_dataset("MultimodalUniverse/gz10")`

**4. Quarto rendering fails**:
- Check Quarto installation: `quarto --version`
- Verify asset paths are correct
- Check `_quarto.yml` configuration

### Getting Help

- Check `env_setup.md` troubleshooting section
- Consult library documentation
- Review backup slides for common issues
- File issues for code problems

## License

Workshop materials created for educational purposes. Please cite appropriately if reusing.

Papers and figures: © respective authors (fair use for educational purposes)
Code: MIT License (or as specified by library licenses)
