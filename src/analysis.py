"""Shared PCA, representation inference and association statistics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCATransform:
    components: np.ndarray
    ratio: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    standardized_mean: np.ndarray


def fit_pca_transform(features, n_components=2):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    coordinates = pca.fit_transform(standardized)
    transform = PCATransform(
        pca.components_,
        pca.explained_variance_ratio_,
        np.asarray(scaler.mean_, dtype=float),
        np.asarray(scaler.scale_, dtype=float),
        np.asarray(pca.mean_, dtype=float),
    )
    return coordinates, transform


def fit_pca(features, n_components=2):
    coordinates, transform = fit_pca_transform(features, n_components)
    return coordinates, transform.components, transform.ratio


def apply_pca_transform(
    features: np.ndarray, transform: PCATransform, n_components: Optional[int] = None
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array for PCA.")
    mean = transform.mean
    std = transform.std
    std = np.where(std < 1e-8, 1.0, std)
    standardized = (features - mean) / std
    centered = standardized - transform.standardized_mean
    comps = transform.components
    if n_components is None:
        n_components = comps.shape[0]
    coords = centered @ comps[:n_components].T
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return coords


def normalize_weights(weights: List[Optional[float]]) -> List[float]:
    valid = [w for w in weights if w is not None]
    if not valid:
        return [0.5 for _ in weights]
    w_min = min(valid)
    w_max = max(valid)
    if np.isclose(w_min, w_max):
        return [0.5 if w is None else 0.8 for w in weights]
    return [0.5 if w is None else (w - w_min) / (w_max - w_min) for w in weights]


def corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    if np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pair_count(n_samples: int) -> int:
    if n_samples < 2:
        return 0
    return int(n_samples * (n_samples - 1) // 2)


def pair_weight(n_samples: int) -> float:
    return float(pair_count(n_samples))


def pair_weight_vector(group_sizes: Sequence[int], normalize: bool = True) -> np.ndarray:
    raw = np.asarray([pair_weight(int(size)) for size in group_sizes], dtype=float)
    if not normalize:
        return raw
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.zeros_like(raw)
    return raw / total


def _groupwise_delta_stats_from_arrays(
    pc1: np.ndarray,
    y_vals: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, object]]:
    if pc1.shape[0] != y_vals.shape[0] or pc1.shape[0] != groups.shape[0]:
        raise ValueError("pc1, y_vals, and groups must have the same length.")
    group_stats: List[Dict[str, object]] = []
    for group_name in sorted({str(name) for name in groups.tolist()}):
        indices = np.where(groups == group_name)[0]
        if indices.size < 2:
            continue
        pc1_group = pc1[indices].astype(float)
        y_group = y_vals[indices].astype(float)
        delta_pc1 = pairwise_deltas(pc1_group)
        delta_y = pairwise_deltas(y_group)
        corr_val = corr(delta_pc1, delta_y)
        if corr_val is None:
            continue
        group_stats.append(
            {
                "group": group_name,
                "n_samples": int(indices.size),
                "n_pairs": int(delta_pc1.size),
                "corr": float(corr_val),
                "delta_pc1": delta_pc1,
                "delta_y": delta_y,
            }
        )
    return group_stats


def weighted_delta_corr_from_arrays(
    pc1: np.ndarray,
    y_vals: np.ndarray,
    groups: np.ndarray,
    weight_mode: str = "sample",
) -> Optional[float]:
    group_stats = _groupwise_delta_stats_from_arrays(pc1, y_vals, groups)
    if not group_stats:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for stat in group_stats:
        if weight_mode == "pair":
            weight = float(stat["n_pairs"])
        else:
            weight = float(stat["n_samples"])
        weighted_sum += float(stat["corr"]) * weight
        weight_total += weight
    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


def compute_hidden_features(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            if not hasattr(model, "forward_feature"):
                raise ValueError("Model missing forward_feature.")
            hidden = model.forward_feature(batch)
            outputs.append(hidden.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def oriented_source_pca(
    source_features: np.ndarray,
    source_score: np.ndarray,
) -> Tuple[np.ndarray, PCATransform, float]:
    src_coords, transform = fit_pca_transform(source_features, n_components=2)
    sign_corr = corr(src_coords[:, 0], source_score)
    if sign_corr is not None and sign_corr < 0:
        src_coords[:, 0] *= -1.0
        transform.components[0] *= -1.0
        sign_corr *= -1.0
    return src_coords, transform, float(sign_corr) if sign_corr is not None else float("nan")


def project_with_source_pca(features: np.ndarray, transform: PCATransform) -> np.ndarray:
    return apply_pca_transform(features, transform, n_components=2)


def centered_distances(values):
    values = np.asarray(values, dtype=np.float64)
    distances = squareform(pdist(values.reshape(len(values), -1)))
    mean = distances.mean(axis=0)
    return distances - mean[:, None] - mean[None, :] + mean.mean()


def centered_correlation(a, b):
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator == 0:
        return 0.0
    return float(np.sqrt(np.clip(np.sum(a * b) / denominator, 0, 1)))


def distance_correlation(x, y):
    return centered_correlation(centered_distances(x), centered_distances(y))


def regression_metrics(truth, prediction):
    difference = prediction - truth
    denominator = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    denominator[denominator == 0] = np.nan
    return dict(
        r2=1 - np.sum(difference**2, axis=0) / denominator,
        mae=np.mean(np.abs(difference), axis=0),
        rmse=np.sqrt(np.mean(difference**2, axis=0)),
    )


class ViewAssociations:
    def __init__(self, metadata, targets=("age", "leaf_count")):
        groups = metadata.group.to_numpy()
        self.indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        self.weights = np.array([len(i) for i in self.indices])
        values = metadata[list(targets)].to_numpy()
        self.targets = [
            [centered_distances(values[i, j]) for j in range(values.shape[1])] for i in self.indices
        ]

    def score(self, pc1):
        values = [
            [centered_correlation(centered_distances(pc1[i]), target) for target in targets]
            for i, targets in zip(self.indices, self.targets)
        ]
        return np.average(values, axis=0, weights=self.weights), np.array(values)


def pairwise_deltas(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    if n < 2:
        return np.array([])
    diffs = values[:, None] - values[None, :]
    tri = np.triu_indices(n, k=1)
    return diffs[tri]


def population_coordinates(data, source_model, adapted_model, device="cuda"):
    sets = {
        "DINO": data.features,
        "MLP": compute_hidden_features(source_model, data.features, batch_size=1024, device=device),
        "SSA": compute_hidden_features(adapted_model, data.features, batch_size=1024, device=device),
    }
    output = {}
    for method, features in sets.items():
        xy, transform = fit_pca_transform(features)
        if np.corrcoef(xy[:, 0], data.metadata.age.to_numpy())[0, 1] < 0:
            xy[:, 0] *= -1
        output[method] = xy
        output[method + "__ratio"] = transform.ratio
    return output
