from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgb
from matplotlib.ticker import MaxNLocator

# Bright but restrained, print-safe colors for the main latent figures.
DATASET_COLORS = {
    "Lettuce": "#5F9D69",
    "Tomato": "#D98262",
    "Komatsuna": "#5F88B2",
}


def dataset_rgb(dataset_name: str) -> Tuple[float, float, float]:
    return to_rgb(DATASET_COLORS.get(dataset_name, "#8B8F94"))


def muted_shade(
    base_rgb: Tuple[float, float, float],
    value: float,
) -> Tuple[float, float, float]:
    """Return a readable light-to-dark shade while retaining the base hue."""
    t = float(np.clip(value, 0.0, 1.0)) ** 0.9
    base = np.asarray(base_rgb, dtype=float)
    light = 0.62 * np.ones(3, dtype=float) + 0.38 * base
    dark = 0.66 * base + 0.34 * np.asarray((0.16, 0.18, 0.17), dtype=float)
    color = (1.0 - t) * light + t * dark
    return tuple(float(channel) for channel in color)


def soften_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#666A6D")


ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    }
)


def save(fig, name, output=None):
    output = Path(output) if output is not None else ROOT / "results"
    output.mkdir(exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output / f"{name}.png")


METHODS = ["DINO", "MLP", "SSA"]
DISPLAY_NAMES = {"DINO": "DINOv2", "MLP": "Multi", "SSA": "Multi+SSA"}


def draw_population(data, coverage, metric, name, output):
    rows = len(coverage)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.25 * rows), squeeze=False)
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.06, top=0.96, hspace=0.38, wspace=0.34)
    cmap = ListedColormap(
        plt.get_cmap("YlGn" if metric == "leaf_count" else "Blues")(np.linspace(0.35, 0.95, 256))
    )
    for row, info in enumerate(coverage):
        crop = info["species"]
        values = data[f"{crop}__{metric}"]
        norm = Normalize(values.min(), values.max())
        order = np.random.default_rng(42).permutation(len(values))
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            key = f"{crop}__{method}"
            xy, ratio = data[key], data[key + "__ratio"]
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            pad = (hi - lo) * 0.06
            ax.scatter(
                *xy[order].T,
                c=values[order],
                cmap=cmap,
                norm=norm,
                s=3,
                alpha=0.45,
                linewidths=0,
                rasterized=True,
            )
            ax.set(
                xlim=(lo[0] - pad[0], hi[0] + pad[0]),
                ylim=(lo[1] - pad[1], hi[1] + pad[1]),
                xlabel=f"PC1 ({ratio[0]:.1%})",
                ylabel=f"PC2 ({ratio[1]:.1%})",
            )
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(labelsize=9)
            soften_axes(ax)
            if row == 0:
                ax.set_title(DISPLAY_NAMES[method], fontsize=14, pad=12)
        pos = axes[row, 0].get_position()
        fig.text(
            0.055,
            pos.y0 + pos.height / 2,
            f"{crop.title()}\n{info['plants']} plant{'s' if info['plants'] > 1 else ''}\nn = {info['images']:,}",
            ha="center",
            va="center",
            fontsize=11,
        )
        cax = fig.add_axes([0.915, pos.y0 + 0.02, 0.013, pos.height - 0.04])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
        cb.locator = MaxNLocator(nbins=4, integer=True)
        cb.update_ticks()
        cb.set_label("Leaf count" if metric == "leaf_count" else "Age (days)")
    save(fig, name, output)
