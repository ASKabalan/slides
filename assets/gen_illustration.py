#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow, Circle


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Figures created separately in each function with identical setup (4:5 aspect)


def _gaussian_samples(seed: int = 42, n: int = 30):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=n)


def _gaussian_quantiles(seed: int, qs):
    """Approximate standard normal quantiles via Monte Carlo and np.quantile."""
    rng = np.random.default_rng(seed)
    # Large sample for stable quantiles without external deps
    samples = rng.normal(0.0, 1.0, size=200000)
    return np.quantile(samples, qs)


def discrete_select(save_path: str, seed: int = 42) -> None:
    # Five points chosen from spread-out quantiles for better coverage
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    x = _gaussian_quantiles(seed=seed, qs=qs)
    # Evaluate on Gaussian PDF so points lie on the curve
    y = 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)

    # Create figure (3x3 inches) and styling
    fig, ax = plt.subplots(figsize=(5, 3), dpi=600)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0.0, 0.55)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True, which="both", alpha=0.3)

    # Plot all points in red
    ax.scatter(x, y, s=60, color="#d32f2f", edgecolor="black", linewidth=0.5)

    # Select one point and highlight it
    # Highlight a central sample for readability
    sel_idx = 2
    ax.scatter([x[sel_idx]], [y[sel_idx]], s=90, color="#2e7d32", edgecolor="black", linewidth=0.5, zorder=3)
    ax.annotate(
        "selected",
        xy=(x[sel_idx], y[sel_idx]),
        xytext=(x[sel_idx] + 0.6, min(0.5, y[sel_idx] + 0.25)),
        color="#2e7d32",
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2),
        fontsize=12,
    )

    fig.savefig(save_path, dpi=100, transparent=True)
    plt.close(fig)


def continuos_generate(save_path: str, seed: int = 123, discrete_seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    # Create figure (3x3 inches) and styling
    fig, ax = plt.subplots(figsize=(5, 3), dpi=600)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0.0, 0.55)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True, which="both", alpha=0.3)

    # Overlay the same discrete samples (red) from the first figure (5 quantiles)
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    xd = _gaussian_quantiles(seed=discrete_seed, qs=qs)
    yd = 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * xd ** 2)
    ax.scatter(xd, yd, s=60, color="#d32f2f", edgecolor="black", linewidth=0.5, zorder=2)

    # Gaussian PDF (same distribution), drawn in green
    xs = np.linspace(-3.5, 3.5, 600)
    pdf = 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * xs ** 2)
    ax.plot(xs, pdf, color="#2e7d32", lw=3, zorder=1)

    # A generated sample from the same distribution
    xg = float(rng.normal(loc=0.0, scale=1.0))
    yg = 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * xg ** 2)

    ax.scatter([xg], [yg], s=90, color="#2e7d32", edgecolor="black", linewidth=0.5, zorder=3)
    ax.annotate(
        "generated",
        xy=(xg, yg),
        xytext=(xg + 0.6, min(0.5, yg + 0.25)),
        color="#2e7d32",
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2),
        fontsize=12,
    )

    fig.savefig(save_path, dpi=100, transparent=True)
    plt.close(fig)


def autoencoder_architecture(save_path: str) -> None:
    """Simple autoencoder block diagram as a transparent PNG (600 dpi)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=600)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, fc="#e6e6ff", ec="#3b0a68"):
        bb = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.12",
                            linewidth=2, edgecolor=ec, facecolor=fc)
        ax.add_patch(bb)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=12, color="#1a1a1a")

    # Input and output placeholders
    box(0.5, 2.1, 1.4, 1.8, "Input")
    box(8.1, 2.1, 1.4, 1.8, "Output")

    # Encoder and decoder blocks
    box(2.4, 2.1, 2.0, 1.8, "Encoder")
    box(5.6, 2.1, 2.0, 1.8, "Decoder")

    # Latent circle (z)
    c = Circle((5.0, 3.0), 0.35, facecolor="#ffe6e6", edgecolor="#3b0a68", linewidth=2)
    ax.add_patch(c)
    ax.text(5.0, 3.0, "z", ha="center", va="center", fontsize=12, color="#1a1a1a")

    # Arrows
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrow(x1, y1, x2-x1, y2-y1, width=0.05, length_includes_head=True,
                                head_width=0.35, head_length=0.35, color="#3b0a68"))

    arrow(1.9, 3.0, 2.3, 3.0)   # Input -> Encoder
    arrow(4.5, 3.0, 4.7, 3.0)   # Encoder -> z
    arrow(5.35, 3.0, 5.55, 3.0) # z -> Decoder
    arrow(7.65, 3.0, 8.05, 3.0) # Decoder -> Output

    # Titles
    ax.text(0.5, 5.5, "Autoencoder", fontsize=14, fontweight="bold", ha="left", color="#3b0a68")

    fig.savefig(save_path, dpi=600, transparent=True)
    plt.close(fig)


def main():
    # Resolve output directory relative to the deck to keep assets local
    here = os.path.abspath(os.path.dirname(__file__))
    out_dir = os.path.join(here, "generated")
    _ensure_dir(out_dir)

    targets = {"discrete_select", "continuos_generate", "autoencoder_architecture"}

    # Simple CLI: supports either `autoencoder_architecture` or `--figure name` or `all`
    to_make = set()
    if len(sys.argv) <= 1:
        to_make = targets
    else:
        arglist = sys.argv[1:]
        if arglist[0] in ("--figure", "-f") and len(arglist) >= 2:
            to_make = {arglist[1].strip()}
        else:
            a = arglist[0].strip()
            if a == "all":
                to_make = targets
            else:
                to_make = {a}

    for t in to_make:
        if t not in targets:
            print(f"Unknown target '{t}'. Expected one of: {sorted(targets)} or 'all'", file=sys.stderr)
            sys.exit(2)

    if "discrete_select" in to_make:
        discrete_select(os.path.join(out_dir, "discrete_select.png"))
    if "continuos_generate" in to_make:
        continuos_generate(os.path.join(out_dir, "continuos_generate.png"))
    if "autoencoder_architecture" in to_make:
        autoencoder_architecture(os.path.join(out_dir, "autoencoder_architecture.png"))

    print("Generated:")
    for t in sorted(to_make):
        print(f" - {os.path.join(out_dir, t + '.png')}")


if __name__ == "__main__":
    main()
