"""Reusable held-out projections, data splits and linear-probe calculations."""

from pathlib import Path

import numpy as np
import torch

from .analysis import apply_pca_transform, compute_hidden_features, fit_pca_transform
from .models import load_model

BACKBONES = [
    ("DINOv2", "dino_register"),
    ("CLIP", "clip_vitb32"),
    ("VAE", "vae"),
    ("MAE", "mae"),
    ("ResNet50", "resnet50"),
]


def checkpoint(paths, dataset, key, **values):
    return Path(str(paths[dataset][key]).format(**values))


def source_split(payload, n, seed, val_split=0.2):
    if "train_indices" in payload:
        return np.asarray(payload["train_indices"]), np.asarray(payload["validation_indices"])
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    n_val = max(1, int(n * val_split))
    return indices[:-n_val], indices[-n_val:]


def stratified_split(groups, seed, val_split=0.2, shuffle_result=False):
    rng = np.random.default_rng(seed)
    train = []
    val = []
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        rng.shuffle(indices)
        if len(indices) == 1:
            train.extend(indices)
            continue
        count = max(1, min(len(indices) - 1, round(len(indices) * val_split)))
        val.extend(indices[:count])
        train.extend(indices[count:])
    if shuffle_result:
        rng.shuffle(train)
        rng.shuffle(val)
    return np.asarray(train, dtype=int), np.asarray(val, dtype=int)


def projected_split(features, train, val, apply_to_train=True):
    coordinates, transform = fit_pca_transform(features[train])
    train_pc1 = apply_pca_transform(features[train], transform)[:, 0] if apply_to_train else coordinates[:, 0]
    return train_pc1, apply_pca_transform(features[val], transform)[:, 0]


def hidden_at(path, features, device, batch_size=256):
    model, payload = load_model(path, device)
    return compute_hidden_features(model, features, batch_size=batch_size, device=device), payload


def proxy_rows(pc1_train, pc1_val, proxy_train, proxy_val, train_y, val_y, traits, **identity):
    designs = {
        "proxy_only": (proxy_train, proxy_val),
        "pc1_only": (pc1_train[:, None], pc1_val[:, None]),
        "proxy_plus_pc1": (np.column_stack([proxy_train, pc1_train]), np.column_stack([proxy_val, pc1_val])),
    }
    rows = []
    for j, trait in enumerate(traits):
        scores = {}
        for name, (train, val) in designs.items():
            x = np.column_stack([np.ones(len(train)), train.astype(float)])
            beta = np.linalg.lstsq(x, train_y[:, j].astype(float), rcond=None)[0]
            prediction = np.column_stack([np.ones(len(val)), val.astype(float)]) @ beta
            y = val_y[:, j]
            denominator = float(np.sum((y - np.mean(y)) ** 2))
            scores[name] = float(1.0 - np.sum((prediction - y) ** 2) / denominator)
            rows.append(dict(identity, target=trait, model_type=name, r2=scores[name]))
        rows.append(
            dict(
                identity,
                target=trait,
                model_type="delta_r2_vs_proxy_only",
                r2=scores["proxy_plus_pc1"] - scores["proxy_only"],
            )
        )
    return rows
