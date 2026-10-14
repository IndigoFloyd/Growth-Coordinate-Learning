"""Compute source/target projections from trained MLP and SSA weights."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.analysis import (
    compute_hidden_features,
    oriented_source_pca,
    project_with_source_pca,
)
from src.dataloader import CONFIG, load_dataset, load_features
from src.models import ROOT, load_model, model_paths, set_global_seed
from src.plotting import DATASET_COLORS, save, soften_axes

SOURCE_DATASET = "Lettuce"
TARGET_DATASET = "Tomato"
AXIS_LABEL_FONTSIZE = 8.0
TITLE_FONTSIZE = 9.5
LEGEND_FONTSIZE = 9.0


def _plot_panel(
    ax: plt.Axes,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    ratio: np.ndarray,
    title: str,
) -> None:
    ax.scatter(
        source_coords[:, 0],
        source_coords[:, 1],
        s=6.0,
        c="#D1D3D4",
        alpha=0.62,
        linewidths=0,
        label=SOURCE_DATASET,
    )
    ax.scatter(
        target_coords[:, 0],
        target_coords[:, 1],
        s=6.5,
        c=DATASET_COLORS.get(TARGET_DATASET, "#8B8F94"),
        alpha=0.78,
        linewidths=0,
        label=TARGET_DATASET,
    )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        f"PC2 ({ratio[1] * 100:.1f}%)" if ratio.size > 1 else "PC2",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    soften_axes(ax)


def main():
    global SOURCE_DATASET, TARGET_DATASET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Lettuce")
    parser.add_argument("--target", default="Tomato")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_global_seed(42)
    SOURCE_DATASET, TARGET_DATASET = args.source, args.target
    source = load_dataset(args.source, config=args.config)
    target = load_dataset(args.target, config=args.config)
    paths = model_paths(args.models)
    methods = [("DINO latent", load_features(source.metadata, "raw"), load_features(target.metadata, "raw"))]
    for branch, label in [("single", "Single"), ("multi", "Multi")]:
        base, _ = load_model(paths["source"][branch], args.device)
        adapted, _ = load_model(paths[args.target][branch + "_ssa"], args.device)
        src = compute_hidden_features(base, source.features, batch_size=256, device=args.device)
        plain = compute_hidden_features(base, target.features, batch_size=256, device=args.device)
        tuned = compute_hidden_features(adapted, target.features, batch_size=256, device=args.device)
        methods.extend([(label, src, plain), (label + " + SSA", src, tuned)])
    labels = source.metadata[source.target_names].to_numpy(dtype=float)
    std = np.nanstd(labels, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    score = np.nanmean((labels - np.nanmean(labels, axis=0, keepdims=True)) / std, axis=1)
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.85), squeeze=False)
    rows = []
    coordinates = {}
    for ax, (name, src, tgt) in zip(axes[0], methods):
        source_xy, transform, association = oriented_source_pca(src, score)
        target_xy = project_with_source_pca(tgt, transform)
        _plot_panel(ax, source_xy, target_xy, transform.ratio, name)
        coordinates[name + "-source"] = source_xy
        coordinates[name + "-target"] = target_xy
        rows.append(
            dict(
                method=name,
                pc1_ratio=float(transform.ratio[0]),
                pc2_ratio=float(transform.ratio[1]),
                source_pc1_score_corr=association,
            )
        )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.85, wspace=0.3)
    save(fig, "Fig5", args.output_dir)
    np.savez_compressed(args.output_dir / "Fig5-coordinates.npz", **coordinates)
    (args.output_dir / "Fig5.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
