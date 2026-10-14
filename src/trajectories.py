"""Shared time-series calculations for Fig6, FigS2 and FigS3."""

from datetime import datetime
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from .analysis import apply_pca_transform, compute_hidden_features, fit_pca_transform
from .dataloader import CONFIG, load_dataset
from .models import ROOT, load_model, model_paths

SPLIT_DATE = datetime(2025, 12, 8).date()
METHOD_ORDER = ["DINO", "Single", "Single+SSA", "Multi", "Multi+SSA"]


def fit_coordinates(features, samples):
    before = np.array([s.timestamp.date() < SPLIT_DATE for s in samples])
    _, transform = fit_pca_transform(features[before], n_components=2)
    xy = apply_pca_transform(features, transform, n_components=2)
    correlations = []
    for identity in sorted({s.unit_id for s in samples}):
        indices = [i for i, s in enumerate(samples) if s.unit_id == identity and before[i]]
        times = np.array([samples[i].timestamp.timestamp() for i in indices])
        values = xy[indices, 0]
        if len(indices) >= 3 and np.std(times) > 1e-9 and np.std(values) > 1e-9:
            correlations.append(np.corrcoef(times, values)[0, 1])
    if correlations and np.mean(correlations) < 0:
        xy *= -1
    return xy


def before_normalize(values, before):
    std = float(np.std(values[before]))
    return (values - np.mean(values[before])) / (std if std >= 1e-8 else 1.0)


def time_colors(timestamps: List[datetime]) -> np.ndarray:
    colors = np.zeros((len(timestamps), 4), dtype=float)
    before_indices = [idx for idx, ts in enumerate(timestamps) if ts.date() < SPLIT_DATE]
    after_indices = [idx for idx, ts in enumerate(timestamps) if ts.date() >= SPLIT_DATE]
    blues = plt.get_cmap("Blues")
    reds = plt.get_cmap("Reds")

    def _fill(indices: List[int], cmap) -> None:
        if not indices:
            return
        ordered = sorted(indices, key=lambda idx: timestamps[idx])
        if len(ordered) == 1:
            colors[ordered[0]] = cmap(0.8)
            return
        for rank, idx in enumerate(ordered):
            value = 0.35 + 0.55 * (rank / (len(ordered) - 1))
            colors[idx] = cmap(value)

    _fill(before_indices, blues)
    _fill(after_indices, reds)
    return colors


def compute_trajectories(dataset_name, config=CONFIG, models=ROOT / "data/models.yml", device="cuda"):
    data = load_dataset(dataset_name, config=config, targets=[])
    samples = list(data.metadata.itertuples(index=False))
    weights = model_paths(models)
    coordinates = {"DINO": fit_coordinates(data.features, samples)}
    for branch in ["single", "multi"]:
        for adapted in [False, True]:
            path = weights[dataset_name][branch + "_ssa"] if adapted else weights["source"][branch]
            model, _ = load_model(path, device)
            hidden = compute_hidden_features(model, data.features, batch_size=256, device=device)
            coordinates[branch.title() + ("+SSA" if adapted else "")] = fit_coordinates(hidden, samples)
    return samples, coordinates
