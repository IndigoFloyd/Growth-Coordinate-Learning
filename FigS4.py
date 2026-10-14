"""Compute full-population PCA coordinates and age associations from features/models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import (
    ViewAssociations,
    population_coordinates,
    weighted_delta_corr_from_arrays,
)
from src.dataloader import CONFIG, load_dataset
from src.models import ROOT, load_model, model_paths, set_global_seed
from src.plotting import draw_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mustard", "okra", "radish", "wheat"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_global_seed(42)
    paths = model_paths(args.models)
    source, _ = load_model(paths["source"]["multi"], args.device)
    population = {}
    coverage = []
    rows = []
    for name in args.datasets:
        data = load_dataset(name, config=args.config)
        adapted, _ = load_model(paths[name]["multi_ssa"], args.device)
        projections = population_coordinates(data, source, adapted, args.device)
        population.update({name + "__" + k: v for k, v in projections.items()})
        population[name + "__age"] = data.metadata["age"].to_numpy()
        coverage.append(dict(species=name, plants=int(data.metadata.plant.nunique()), images=len(data)))
        associations = ViewAssociations(data.metadata, targets=["age"])
        for method in ["DINO", "MLP", "SSA"]:
            pc1 = projections[method][:, 0]
            delta = weighted_delta_corr_from_arrays(
                pc1, data.metadata["age"].to_numpy(), data.metadata.group.to_numpy()
            )
            dcor, _ = associations.score(pc1)
            rows.append(
                dict(dataset=name, method=method, delta_corr_abs=abs(delta), distance_corr=float(dcor[0]))
            )
    draw_population(population, coverage, "age", "FigS4", args.output_dir)
    np.savez_compressed(args.output_dir / "FigS4-coordinates.npz", **population)
    pd.DataFrame(rows).to_csv(args.output_dir / "FigS4-metrics.csv", index=False)


if __name__ == "__main__":
    main()
