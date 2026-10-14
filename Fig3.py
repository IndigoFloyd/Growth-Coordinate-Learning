from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
import argparse
import json

from src.analysis import fit_pca, normalize_weights
from src.dataloader import CONFIG, load_features, load_metadata
from src.models import set_global_seed
from src.plotting import dataset_rgb, muted_shade, soften_axes

BACKBONES = [
    ("DINOv2", "dino_register"),
    ("CLIP", "clip_vitb32"),
    ("VAE", "vae"),
    ("MAE", "mae"),
    ("ResNet50", "resnet50"),
]
POINT_SIZE = 6.0
SHADE_LEVELS = 16
DPI = 300
FIGURE_WIDTH_IN = 7.2
PANEL_HEIGHT_IN = 1.55
BASE_FONTSIZE = 8.0
TITLE_FONTSIZE = 9.0
AXIS_LABEL_FONTSIZE = 7.5
ROW_LABEL_FONTSIZE = 8.5
LEGEND_FONTSIZE = 8.0


@dataclass
class LatentEntry:
    vector: np.ndarray
    dataset: str
    weight: Optional[float]


def _soft_palette(dataset_names: List[str]) -> Dict[str, Tuple[float, float, float]]:
    return {name: dataset_rgb(name) for name in dataset_names}


def _quantize_levels(values: List[float], levels: int) -> List[float]:
    if levels <= 1:
        return [0.5 for _ in values]
    step = 1.0 / (levels - 1)
    quantized = []
    for value in values:
        snapped = round(value / step) * step
        quantized.append(min(1.0, max(0.0, snapped)))
    return quantized


def load_latent_entries(encoder, datasets, config):
    entries = []
    for name in sorted(datasets):
        metadata = load_metadata(name, encoder, config)
        if encoder == "clip_vitb32":
            keys = set(load_metadata(name, "dino_register", config).key)
            metadata = metadata[metadata.key.isin(keys)].reset_index(drop=True)
        vectors = load_features(metadata, "raw")
        trait = (
            "timestamp" if name == "Komatsuna" else "FreshWeightShoot" if name == "Lettuce" else "fw_plant"
        )
        weights = metadata[trait].to_numpy() if trait in metadata else np.full(len(metadata), np.nan)
        entries.extend(
            LatentEntry(vector, name, (None if np.isnan(weight) else float(weight)))
            for vector, weight in zip(vectors, weights)
        )
    return entries


def plot_panel(
    ax: plt.Axes,
    entries: List[LatentEntry],
    coords: np.ndarray,
    ratio: np.ndarray,
    dataset_order: List[str],
    dataset_colors: Dict[str, Tuple[float, float, float]],
) -> None:
    datasets = [entry.dataset for entry in entries]
    weights = [entry.weight for entry in entries]
    ordered = [name for name in dataset_order if name in set(datasets)]
    for name in sorted(set(datasets)):
        if name not in ordered:
            ordered.append(name)

    for dataset_name in ordered:
        indices = [idx for idx, name in enumerate(datasets) if name == dataset_name]
        ds_coords = coords[indices]
        ds_weights = [weights[idx] for idx in indices]
        norm_weights = _quantize_levels(normalize_weights(ds_weights), SHADE_LEVELS)
        base_rgb = dataset_colors[dataset_name]
        colors = [muted_shade(base_rgb, norm) for norm in norm_weights]
        ax.scatter(ds_coords[:, 0], ds_coords[:, 1], s=POINT_SIZE, c=colors, linewidths=0, alpha=0.82)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        f"PC2 ({ratio[1] * 100:.1f}%)" if ratio.size > 1 else "PC2",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    soften_axes(ax)


def _dataset_legend_handles(
    dataset_order: List[str],
    dataset_colors: Dict[str, Tuple[float, float, float]],
) -> List[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=muted_shade(dataset_colors[name], 0.72),
            markeredgecolor="none",
            markersize=4.0,
            linestyle="None",
            label=name,
        )
        for name in dataset_order
    ]


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": BASE_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
        }
    )
    parser = argparse.ArgumentParser(description="Compute backbone PCA panels from frozen features.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    set_global_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_order = ["Tomato", "Komatsuna", "Lettuce"]
    model_entries = {
        name: load_latent_entries(encoder, dataset_order, args.config) for name, encoder in BACKBONES
    }
    all_datasets = list(dataset_order)
    for entries in model_entries.values():
        for entry in entries:
            if entry.dataset not in all_datasets:
                all_datasets.append(entry.dataset)
    dataset_colors = _soft_palette(all_datasets)

    row_datasets = ["ALL"] + dataset_order
    nrows = len(row_datasets)
    ncols = len(BACKBONES)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGURE_WIDTH_IN, PANEL_HEIGHT_IN * nrows + 0.45),
        dpi=DPI,
        squeeze=False,
    )
    summary_rows: List[Dict[str, object]] = []

    for row_idx, dataset_name in enumerate(row_datasets):
        row_label = "All datasets" if dataset_name == "ALL" else dataset_name
        for col_idx, (model_name, _) in enumerate(BACKBONES):
            entries = model_entries[model_name]
            subset = (
                entries
                if dataset_name == "ALL"
                else [entry for entry in entries if entry.dataset == dataset_name]
            )
            ax = axes[row_idx][col_idx]
            if not subset:
                ax.axis("off")
                continue
            coords, _, ratio = fit_pca(np.stack([entry.vector for entry in subset], axis=0))
            plot_panel(
                ax,
                subset,
                coords,
                ratio,
                dataset_order,
                dataset_colors,
            )
            if row_idx == 0:
                ax.set_title(model_name, fontsize=TITLE_FONTSIZE, fontweight="normal", pad=3)
            if col_idx == 0:
                ax.annotate(
                    row_label,
                    xy=(-0.48, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=ROW_LABEL_FONTSIZE,
                    fontweight="normal",
                    annotation_clip=False,
                )
            summary_rows.append(
                {
                    "panel": row_label,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "num_points": len(subset),
                    "pc1_ratio": float(ratio[0]),
                    "pc2_ratio": float(ratio[1]) if ratio.size > 1 else None,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legend_handles = _dataset_legend_handles(dataset_order, dataset_colors)
    fig.legend(
        handles=legend_handles,
        labels=dataset_order,
        loc="upper center",
        ncol=len(dataset_order),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.35,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.055, top=0.925, wspace=0.24, hspace=0.34)
    fig.savefig(args.output_dir / "Fig3.png", bbox_inches="tight", pad_inches=0.03, dpi=DPI)
    plt.close(fig)

    (args.output_dir / "Fig3.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    print(args.output_dir / "Fig3.png")


if __name__ == "__main__":
    main()
