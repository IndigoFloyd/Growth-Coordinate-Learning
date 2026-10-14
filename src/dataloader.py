"""One feature/label loader for configured datasets, including user-supplied CSVs."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PFAL_REPRESENTATION_DATA_ROOT", "/mnt/data/PFAL_Latent_Representation"))
CONFIG = ROOT / "data/datasets.yml"


def dataset_config(name, config=CONFIG):
    entries = yaml.safe_load(Path(config).read_text())["datasets"]
    return next(row for row in entries if row["name"].lower() == name.lower())


def _path(value, config, encoder):
    path = Path(str(value).format(data_root=DATA_ROOT, encoder=encoder))
    return path if path.is_absolute() else Path(config).parent / path


def _records(path):
    data = json.loads(path.read_text())
    data = data.get("Measurements", data)
    records = {}
    for key, values in data.items():
        keys = [str(key), Path(str(key)).stem]
        keys += [str(values[k]) for k in ["RGB_Image", "rgb_image", "image", "image_name"] if k in values]
        for item in keys:
            records[item] = values
            records[Path(item).stem] = values
    return records


def _time_index(key):
    match = re.search(r"(\d{2})_(\d{3})_(\d{2})", Path(key).stem)
    if match:
        return float(int(match[2]) * 6 + int(match[3]))
    tokens = re.findall(r"\d+", Path(key).stem)
    value = (
        tokens[0] + tokens[1]
        if len(tokens) > 1 and len(tokens[0]) >= 8 and len(tokens[1]) >= 4
        else max(tokens, key=len)
    )
    return float(int(value))


def load_metadata(name, encoder="dino_register", config=CONFIG):
    """Return rows in the experiment's original order; paths are explicit in YAML."""
    cfg = dataset_config(name, config)
    directory = _path(cfg["latents"], config, encoder)
    kind = cfg.get("format", "json")
    if kind in ["csv", "gromo"]:
        frame = pd.read_csv(_path(cfg["groundtruth"], config, encoder))
        frame["key"] = frame["filename"].map(lambda x: Path(x).as_posix())
        frame["latent_path"] = frame["key"].map(lambda x: directory / Path(x).with_suffix(".pt"))
        if kind == "gromo":
            image_root = _path(cfg["path"], config, encoder)
            frame["image_path"] = frame.filename.map(lambda p: image_root / p)
            frame = frame[frame.image_path.map(Path.is_file)].copy()
            parts = frame.key.map(lambda x: Path(x).stem).str.extract(
                r"(?P<species>\w+)_p(?P<plant>\d+)_d?(?P<day>\d+)_L(?P<level>\d+)_(?P<angle>\d+)$"
            )
            for column in ["plant", "day", "level", "angle"]:
                frame[column] = parts[column].astype(int)
            frame["age"] = frame["Age"].astype(float)
            frame["group"] = [
                f"{name}/p{p}/L{l}/{a}" for p, l, a in zip(frame.plant, frame.level, frame.angle)
            ]
            frame = frame.sort_values(["plant", "level", "angle", "day"])
        else:
            if "group" not in frame:
                frame["group"] = "all"
            if "timestamp" in frame and frame.timestamp.dtype == object:
                frame["timestamp"] = pd.to_datetime(frame.timestamp)
        return frame.reset_index(drop=True)
    files = sorted(directory.rglob("*.pt"))
    if kind == "lab_fullimage":
        files.sort(key=lambda p: f"{p.parts[-3]}_{p.parts[-2]}_{p.name}")
    rows = []
    records = _records(_path(cfg["groundtruth"], config, encoder)) if kind == "json" else {}
    for path in files:
        key = path.relative_to(directory).with_suffix("").as_posix()
        row = dict(key=key, latent_path=path, group="all")
        if kind == "json":
            values = records.get(key, records.get(Path(key).stem))
            if values is None:
                values = {}
            row.update({k: float(v) for k, v in values.items() if isinstance(v, (int, float))})
            row["group"] = str(
                values.get(
                    "Variety",
                    values.get(
                        "variety", values.get("Class", values.get("class", Path(key).stem.split("_")[0]))
                    ),
                )
            )
        elif kind == "time_index":
            row["timestamp"] = _time_index(key)
            row["plant"] = Path(key).stem.split("_")[1]
        elif kind == "lab_fullimage":
            cam, day, _ = path.parts[-3:]
            row.update(
                cam=cam, unit_id=cam, timestamp=datetime.strptime(f"{day}_{path.stem}", "%Y-%m-%d_%H%M%S_%f")
            )
        elif kind == "lab_crop":
            layer, side, grid, cam, _ = path.parts[-5:]
            row.update(
                cam=cam,
                unit_id="|".join([layer, side, grid, cam]),
                timestamp=datetime.strptime(path.stem, "%Y-%m-%d_%H%M%S_%f"),
            )
        else:
            raise ValueError(f"Unknown dataset format: {kind}")
        rows.append(row)
    return pd.DataFrame(rows)


def raw_vector(tensor):
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 5:
        tensor = tensor.squeeze(1)
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=1)
    if tensor.ndim > 2:
        tensor = tensor.flatten(start_dim=1)
    return tensor.squeeze(0).detach().cpu().numpy()


def pooled_vector(tensor):
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 1:
        return tensor.detach().cpu().numpy()
    if tensor.ndim == 5:
        tensor = tensor.squeeze(1)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.flatten(start_dim=1).transpose(0, 1)
    return F.layer_norm(tensor, (tensor.shape[-1],)).mean(dim=0).detach().cpu().numpy()


def load_features(metadata, representation="model"):
    values = []
    for path in metadata.latent_path:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(tensor, dict):
            vector = tensor["ssa" if representation == "model" else "raw"].numpy()
        else:
            vector = pooled_vector(tensor) if representation == "model" else raw_vector(tensor)
        values.append(vector)
    return np.stack(values)


class FeatureDataset(Dataset):
    def __init__(self, features, metadata=None, targets=()):
        self.features = np.asarray(features)
        self.metadata = (
            metadata.reset_index(drop=True)
            if metadata is not None
            else pd.DataFrame(index=range(len(features)))
        )
        self.target_names = list(targets)
        self.targets = self.metadata[self.target_names].to_numpy(dtype=np.float32)
        self.x = torch.as_tensor(self.features, dtype=torch.float32)
        self.y = torch.from_numpy(self.targets)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"latent": self.x[index], "target": self.y[index]}


def load_dataset(
    name, encoder="dino_register", config=CONFIG, representation="model", targets=None, before=None
):
    cfg = dataset_config(name, config)
    metadata = load_metadata(name, encoder, config)
    if before is not None:
        metadata = metadata[metadata.timestamp < pd.Timestamp(before)].reset_index(drop=True)
    targets = cfg.get("targets", []) if targets is None else list(targets)
    if targets:
        metadata = metadata.dropna(subset=targets).reset_index(drop=True)
    return FeatureDataset(load_features(metadata, representation), metadata, targets)


def make_loader(features, batch_size=256, shuffle=False, seed=42):
    data = features if isinstance(features, Dataset) else FeatureDataset(features)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def image_paths(name, metadata, config=CONFIG):
    directory = _path(dataset_config(name, config)["path"], config, "dino_register")
    index = {}
    for path in directory.rglob("*"):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            index.setdefault(path.relative_to(directory).with_suffix("").as_posix(), path)
            index.setdefault(path.stem, path)
    return [index[key] if key in index else index[Path(key).stem] for key in metadata.key]
