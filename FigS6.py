"""Recompute SSA update associations from frozen inputs and per-step model snapshots."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.analysis import (
    ViewAssociations,
    compute_hidden_features,
    fit_pca_transform,
    weighted_delta_corr_from_arrays,
)
from src.dataloader import CONFIG, load_dataset
from src.models import ROOT, load_model, model_paths, set_global_seed
from src.plotting import save, soften_axes

CROPS = ["mustard", "okra", "radish", "wheat"]
METRICS = ["Age", "leaf_count"]
DELTA_LABEL = r"$|\Delta\mathrm{Corr}|$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_global_seed(42)
    paths = model_paths(args.models)
    rows = []
    for crop in CROPS:
        dataset = load_dataset(crop, config=args.config)
        associations = ViewAssociations(dataset.metadata)
        snapshots = sorted(paths[crop]["snapshots"].rglob("step*.pth"))
        if not snapshots:
            raise FileNotFoundError(
                f"No SSA step snapshots for {crop}: train with TrainSSA.py --snapshot-steps."
            )
        for path in snapshots:
            model, payload = load_model(path, args.device)
            step = int(path.stem.removeprefix("step"))
            seed = payload.get("seed")
            if seed is None:
                seed = int(next(p.name[4:] for p in path.parents if re.fullmatch(r"seed\d+", p.name)))
            hidden = compute_hidden_features(model, dataset.features, batch_size=1024, device=args.device)
            xy, _ = fit_pca_transform(hidden)
            if np.corrcoef(xy[:, 0], dataset.metadata.age.to_numpy())[0, 1] < 0:
                xy[:, 0] *= -1
            dcor, _ = associations.score(xy[:, 0])
            for i, (metric, field) in enumerate([("Age", "age"), ("leaf_count", "leaf_count")]):
                delta = weighted_delta_corr_from_arrays(
                    xy[:, 0], dataset.metadata[field].to_numpy(), dataset.metadata.group.to_numpy()
                )
                rows.append(
                    dict(
                        species=crop,
                        branch="multi",
                        seed=seed,
                        step=step,
                        epoch=step / np.ceil(len(dataset) / payload.get("batch_size", 256)),
                        top_k=payload.get("top_k", 70),
                        metric=metric,
                        abs_weighted_mean=abs(delta),
                        distance_corr=float(dcor[i]),
                    )
                )
    data = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "FigS6-seeds.csv", index=False)
    keys = ["species", "branch", "metric", "step", "epoch", "top_k"]
    summaries = []
    for value, statistic in [("abs_weighted_mean", "delta_corr_abs"), ("distance_corr", "distance_corr")]:
        summaries.append(
            data.groupby(keys)[value]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
            .assign(statistic=statistic)
        )
    summary = pd.concat(summaries, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "FigS6.csv", index=False)
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharey=True, layout="constrained")
    for i, crop in enumerate(CROPS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for statistic, label, color in [
                ("delta_corr_abs", DELTA_LABEL, "#1F77B4"),
                ("distance_corr", "dCor", "#FF7F0E"),
            ]:
                part = summary.query(
                    "species == @crop and metric == @metric and statistic == @statistic"
                ).sort_values("step")
                ax.plot(part.step, part["mean"], "o-", ms=3.5, color=color, label=label)
                ax.fill_between(
                    part.step, part["mean"] - part["std"], part["mean"] + part["std"], color=color, alpha=0.13
                )
            ax.set(
                title=f"{crop.title()} · {'Plant age' if metric == 'Age' else 'Leaf count'}", ylim=(0, 1.03)
            )
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(left=0)
            ax.grid(alpha=0.15)
            soften_axes(ax)
            if i == 3:
                ax.set_xlabel("Optimizer updates")
    axes[0, 0].legend(frameon=False, loc="lower right")
    save(fig, "FigS6", args.output_dir)


if __name__ == "__main__":
    main()
