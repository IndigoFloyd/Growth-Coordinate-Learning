"""Compute source-domain PCA and the image-feature baseline for Fig4."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.analysis import compute_hidden_features, fit_pca, normalize_weights
from src.dataloader import CONFIG, image_paths, load_dataset
from src.features import baseline_features
from src.models import ROOT, load_model, model_paths, set_global_seed
from src.plotting import dataset_rgb, muted_shade, save, soften_axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Lettuce")
    parser.add_argument("--trait", default="FreshWeightShoot")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_global_seed(42)
    data = load_dataset(args.dataset, config=args.config)
    paths = model_paths(args.models)["source"]
    features = {"DINOv2": data.features}
    for key, label in [("single", "Single"), ("multi", "Multi")]:
        model, _ = load_model(paths[key], args.device)
        features[label] = compute_hidden_features(model, data.features, batch_size=256, device=args.device)
    features["Baseline"] = np.stack(
        [baseline_features(p) for p in image_paths(args.dataset, data.metadata, args.config)]
    )
    values = data.targets[:, data.target_names.index(args.trait)]
    colors = [muted_shade(dataset_rgb(args.dataset), v) for v in normalize_weights(values.tolist())]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.7), squeeze=False)
    rows = []
    coordinates = {}
    for ax, (name, features) in zip(axes[0], features.items()):
        xy, _, ratio = fit_pca(features)
        coordinates[name] = xy
        ax.scatter(xy[:, 0], xy[:, 1], s=6.5, c=colors, linewidths=0, alpha=0.82)
        ax.set(
            xticks=[], yticks=[], xlabel=f"PC1 ({ratio[0]:.1%})", ylabel=f"PC2 ({ratio[1]:.1%})", title=name
        )
        ax.tick_params(labelsize=6.5)
        soften_axes(ax)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.title.set_size(8.5)
        rows.append(
            dict(
                metric_name=args.trait,
                model_name=name,
                num_points=len(xy),
                pc1_ratio=float(ratio[0]),
                pc2_ratio=float(ratio[1]),
            )
        )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.84, wspace=0.3)
    save(fig, "Fig4", args.output_dir)
    np.savez_compressed(args.output_dir / "Fig4-coordinates.npz", **coordinates)
    (args.output_dir / "Fig4.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
