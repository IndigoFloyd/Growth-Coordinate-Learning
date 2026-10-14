"""Fixed-view association calculations shared by Table10, TableS7 and TableS8."""

import numpy as np

from .analysis import ViewAssociations, fit_pca, weighted_delta_corr_from_arrays
from .dataloader import load_dataset
from .evaluation import checkpoint, hidden_at

REFERENCES = {"lettuce": "S/S (SSA)", "target_mean": "T/S", "target_variance": "S/T", "target_both": "T/T"}


def reference_associations(crop, paths, config, seeds, device, traits=("age", "leaf_count")):
    data = load_dataset(crop, config=config)
    associations = ViewAssociations(data.metadata, targets=traits)
    groups = data.metadata.group.to_numpy()
    age = data.metadata.age.to_numpy()
    rows = []

    def evaluate(method, seed, features):
        xy, _, _ = fit_pca(features)
        if np.corrcoef(xy[:, 0], age)[0, 1] < 0:
            xy[:, 0] *= -1
        dcor, _ = associations.score(xy[:, 0])
        for i, trait in enumerate(traits):
            delta = weighted_delta_corr_from_arrays(xy[:, 0], data.metadata[trait].to_numpy(), groups)
            rows.append(
                dict(
                    species=crop,
                    method=method,
                    seed=seed,
                    metric="Age" if trait == "age" else trait,
                    delta_corr_abs=abs(delta),
                    distance_corr=float(dcor[i]),
                )
            )

    evaluate("DINO", None, data.features)
    hidden, _ = hidden_at(paths["source"]["multi"], data.features, device, batch_size=1024)
    evaluate("MLP", None, hidden)
    for reference, label in REFERENCES.items():
        for seed in seeds:
            path = checkpoint(paths, crop, "reference_runs", reference=reference, seed=seed)
            hidden, _ = hidden_at(path, data.features, device, batch_size=1024)
            evaluate(label, seed, hidden)
    return rows
