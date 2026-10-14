"""Table 8: temporal DeltaCorr, ordering and smoothness on komatsuna."""

from pathlib import Path

import numpy as np

from src.analysis import fit_pca, weighted_delta_corr_from_arrays
from src.dataloader import load_dataset, load_features
from src.evaluation import checkpoint, hidden_at
from src.models import model_paths, set_global_seed
from src.tables import save_table, table_parser


def _infer_plant_id(key: str) -> str:
    tokens = Path(key).stem.split("_")
    if len(tokens) >= 4:
        return tokens[1]
    return "all"


def _pairwise_order_accuracy(pc1: np.ndarray, timestamps: np.ndarray, plant_ids: np.ndarray) -> float:
    total = 0.0
    correct = 0.0
    for plant_id in sorted(set(plant_ids.tolist())):
        mask = plant_ids == plant_id
        if np.sum(mask) < 2:
            continue
        ts = timestamps[mask]
        vals = pc1[mask]
        i_idx, j_idx = np.triu_indices(ts.shape[0], k=1)
        dt = ts[j_idx] - ts[i_idx]
        valid = np.isfinite(dt) & (dt != 0)
        if not np.any(valid):
            continue
        dt = dt[valid]
        dp = vals[j_idx][valid] - vals[i_idx][valid]
        correct += float(np.sum((dp * dt) > 0))
        total += float(dt.size)
    return float(correct / total) if total > 0 else float("nan")


def _orient_pc1(pc1: np.ndarray, timestamps: np.ndarray, plant_ids: np.ndarray) -> np.ndarray:
    order_acc = _pairwise_order_accuracy(pc1, timestamps, plant_ids)
    return -pc1 if np.isfinite(order_acc) and order_acc < 0.5 else pc1


def _smoothness_efficiency(pc1: np.ndarray, timestamps: np.ndarray, plant_ids: np.ndarray) -> float:
    scores = []
    for plant_id in sorted(set(plant_ids.tolist())):
        mask = plant_ids == plant_id
        if np.sum(mask) < 3:
            continue
        ts = timestamps[mask]
        vals = pc1[mask]
        valid = np.isfinite(ts) & np.isfinite(vals)
        if np.sum(valid) < 3:
            continue
        ts = ts[valid]
        vals = vals[valid]
        order = np.argsort(ts)
        vals = vals[order]
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))
        if v_max <= v_min:
            continue
        vals = (vals - v_min) / (v_max - v_min)
        total_path = float(np.sum(np.abs(np.diff(vals))))
        if total_path <= 1e-8:
            continue
        net_disp = float(abs(vals[-1] - vals[0]))
        scores.append(net_disp / total_path)
    return float(np.mean(scores)) if scores else float("nan")


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    data = load_dataset("Komatsuna", config=args.config)
    paths = model_paths(args.models)
    plants = np.asarray([_infer_plant_id(key) for key in data.metadata.key])
    timestamps = data.metadata.timestamp.to_numpy(dtype=float)
    baseline = {"DINO latent": load_features(data.metadata, "raw")}
    for branch in ["single", "multi"]:
        baseline[branch.title()], _ = hidden_at(paths["source"][branch], data.features, args.device)
    rows = []
    for seed in args.seeds:
        features = {"DINO latent": baseline["DINO latent"]}
        for branch in ["single", "multi"]:
            features[branch.title()] = baseline[branch.title()]
            features[branch.title() + " + SSA"], _ = hidden_at(
                checkpoint(paths, "Komatsuna", branch + "_runs", seed=seed), data.features, args.device
            )
        for method, values in features.items():
            xy, _, _ = fit_pca(values)
            pc1 = _orient_pc1(xy[:, 0], timestamps, plants)
            value = weighted_delta_corr_from_arrays(pc1, timestamps, data.metadata.group.to_numpy())
            rows.append(
                dict(
                    method=method,
                    seed=seed,
                    delta_corr_abs=abs(value),
                    ordering_accuracy=_pairwise_order_accuracy(pc1, timestamps, plants),
                    smoothness=_smoothness_efficiency(pc1, timestamps, plants),
                )
            )
    save_table(
        rows, "Table8", args.output_dir, ["method"], ["delta_corr_abs", "ordering_accuracy", "smoothness"]
    )


if __name__ == "__main__":
    main()
