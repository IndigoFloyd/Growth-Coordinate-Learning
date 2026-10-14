"""Table 7: statistical alignment baselines and SSA parameterizations on tomato."""

from typing import Tuple

import numpy as np

from src.analysis import fit_pca, weighted_delta_corr_from_arrays
from src.dataloader import load_dataset
from src.evaluation import checkpoint, hidden_at
from src.models import model_paths, set_global_seed
from src.tables import save_table, table_parser


def _safe_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = np.std(x, axis=0)
    return np.where(std < eps, 1.0, std)


def mean_var_align(source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    src_mean = np.mean(source_x, axis=0, keepdims=True)
    tgt_mean = np.mean(target_x, axis=0, keepdims=True)
    src_std = _safe_std(source_x)[None, :]
    tgt_std = _safe_std(target_x)[None, :]
    return ((target_x - tgt_mean) / tgt_std) * src_std + src_mean


def _covariance(x: np.ndarray, eps: float) -> np.ndarray:
    centered = x - np.mean(x, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov += np.eye(cov.shape[0], dtype=float) * float(eps)
    return cov


def _eigh_sqrt(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    sqrt_cov = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    invsqrt_cov = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return sqrt_cov, invsqrt_cov


def coral_align(source_x: np.ndarray, target_x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    src_mean = np.mean(source_x, axis=0, keepdims=True)
    tgt_mean = np.mean(target_x, axis=0, keepdims=True)
    src_cov = _covariance(source_x, eps)
    tgt_cov = _covariance(target_x, eps)
    src_sqrt, _ = _eigh_sqrt(src_cov)
    _, tgt_invsqrt = _eigh_sqrt(tgt_cov)
    return (target_x - tgt_mean) @ tgt_invsqrt @ src_sqrt + src_mean


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    source = load_dataset("Lettuce", config=args.config)
    target = load_dataset("Tomato", config=args.config)
    paths = model_paths(args.models)
    rows = []
    for branch in ["single", "multi"]:
        source_hidden, _ = hidden_at(paths["source"][branch], source.features, args.device)
        target_hidden, _ = hidden_at(paths["source"][branch], target.features, args.device)
        deterministic = {
            "no_adaptation": target_hidden,
            "mean_var_align": mean_var_align(source_hidden, target_hidden),
            "coral": coral_align(source_hidden, target_hidden),
        }
        deterministic["ssa_ln"], _ = hidden_at(
            checkpoint(paths, "Tomato", branch + "_ablation", mode="ln", seed=args.seeds[0]),
            target.features,
            args.device,
        )
        cases = [(method, None, values) for method, values in deterministic.items()]
        for seed in args.seeds:
            adapter, _ = hidden_at(
                checkpoint(paths, "Tomato", branch + "_ablation", mode="lora", seed=seed),
                target.features,
                args.device,
            )
            full, _ = hidden_at(
                checkpoint(paths, "Tomato", branch + "_ablation_full", seed=seed),
                target.features,
                args.device,
            )
            cases.extend([("ssa_lora", seed, adapter), ("ssa_ln_lora", seed, full)])
        for method, seed, values in cases:
            for trait in target.target_names:
                xy, _, _ = fit_pca(values)
                score = weighted_delta_corr_from_arrays(
                    xy[:, 0], target.metadata[trait].to_numpy(), target.metadata.group.to_numpy()
                )
                rows.append(
                    dict(
                        branch=branch,
                        method=method,
                        seed=seed,
                        target_metric=trait,
                        delta_corr_abs=abs(score),
                    )
                )
    import pandas as pd

    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "Table7-traits.csv", index=False)
    averaged = (
        frame.groupby(["branch", "method", "seed"], dropna=False, sort=False)
        .delta_corr_abs.mean()
        .reset_index()
    )
    save_table(averaged, "Table7", args.output_dir, ["method", "branch"], ["delta_corr_abs"])


if __name__ == "__main__":
    main()
