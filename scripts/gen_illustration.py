#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _setup_axes(ax, xlim=(-3, 3), ylim=(-0.05, 0.45)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _normal_pdf(x, mu=0.0, sigma=1.0):
    inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return inv * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_discrete_select(out_path: str):
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=28)
    y = np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=100)
    _setup_axes(ax)

    # All points in red
    ax.scatter(x, y, s=50, color="#d32f2f", edgecolor="none")

    # Highlight one selected point in green
    sel_idx = int(np.argmax(x))
    ax.scatter([x[sel_idx]], [y[sel_idx]], s=80, color="#2e7d32", zorder=3)
    ax.annotate(
        "selected",
        xy=(x[sel_idx], y[sel_idx]),
        xytext=(x[sel_idx] + 0.6, 0.12),
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2),
        color="#2e7d32",
    )

    fig.savefig(out_path, dpi=100, transparent=True, bbox_inches="tight")
    plt.close(fig)


def generate_continuos_generate(out_path: str):
    x = np.linspace(-3, 3, 600)
    y = _normal_pdf(x)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=100)
    _setup_axes(ax)

    # Red PDF curve
    ax.plot(x, y, color="#d32f2f", lw=2)

    # A generated sample outside the previously shown discrete points
    xg = 2.2
    yg = _normal_pdf(xg)
    ax.scatter([xg], [yg], s=90, color="#2e7d32", zorder=3)
    ax.annotate(
        "generated",
        xy=(xg, yg),
        xytext=(xg - 1.4, 0.28),
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2),
        color="#2e7d32",
    )

    fig.savefig(out_path, dpi=100, transparent=True, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate illustration figures")
    parser.add_argument(
        "which",
        choices=["discrete_select", "continuos_generate", "all"],
        help="Which illustration to generate",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("assets", "ai"),
        help="Output directory for PNGs (default: assets/ai)",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    if args.which in ("discrete_select", "all"):
        generate_discrete_select(os.path.join(args.out_dir, "discrete_select.png"))
    if args.which in ("continuos_generate", "all"):
        generate_continuos_generate(
            os.path.join(args.out_dir, "continuos_generate.png")
        )


if __name__ == "__main__":
    main()

