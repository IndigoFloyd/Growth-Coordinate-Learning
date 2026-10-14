"""Compute five-fold scaling results from data/checkpoints; optionally train this experiment."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.analysis import (
    apply_pca_transform,
    compute_hidden_features,
    corr,
    fit_pca_transform,
    regression_metrics,
)
from src.dataloader import CONFIG, load_dataset
from src.mlp import forward_outputs, train_mlp
from src.models import ROOT, load_model, model_paths, save_checkpoint, set_global_seed
from src.plotting import soften_axes

TRAITS = ["FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def make_splits(groups, fractions, seed):
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    splits = []
    for fold, (pool, test) in enumerate(splitter.split(np.zeros(len(groups)), groups)):
        train, val = train_test_split(pool, test_size=0.2, stratify=groups[pool], random_state=seed + fold)
        order = np.random.default_rng(seed + fold).permutation(train)
        subsets = {str(f): order[: max(2, round(len(order) * f))].tolist() for f in fractions}
        assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        for small, large in zip(fractions[:-1], fractions[1:]):
            assert set(subsets[str(small)]).issubset(subsets[str(large)])
        splits.append(
            dict(
                fold=fold,
                seed=40 + fold,
                train_pool=order.tolist(),
                validation=val.tolist(),
                test=test.tolist(),
                subsets=subsets,
            )
        )
    assert sorted(i for split in splits for i in split["test"]) == list(range(len(groups)))
    return splits


def evaluate(model, saved, x, y, identity, device):
    train = saved["train_indices"]
    features = compute_hidden_features(model, x, batch_size=256, device=device)
    np.random.seed(42)
    train_z, transform = fit_pca_transform(features[train])
    train_mean, train_std = y[train].mean(axis=0), y[train].std(axis=0)
    score = (
        y[train, 0] if len(saved["target_names"]) == 1 else ((y[train] - train_mean) / train_std).mean(axis=1)
    )
    sign = 1 if corr(train_z[:, 0], score) >= 0 else -1
    with torch.no_grad():
        pred = forward_outputs(model, torch.tensor(x, device=device), saved["target_names"]).cpu().numpy()
    pred = pred * saved["target_std"] + saved["target_mean"]
    rows, predictions = [], []
    for scope, ids in [("train", train), ("test", saved["test_indices"])]:
        pc1 = sign * apply_pca_transform(features[ids], transform)[:, 0]
        metrics = regression_metrics(y[ids, : pred.shape[1]], pred[ids])
        r2, mae, rmse = metrics["r2"], metrics["mae"], metrics["rmse"]
        for j, trait in enumerate(TRAITS):
            r = corr(pc1, y[ids, j])
            row = dict(
                identity,
                scope=scope,
                trait=trait,
                n_evaluation=len(ids),
                pc1_pearson=r,
                pc1_pearson_abs=abs(r),
            )
            if trait in saved["target_names"]:
                row.update(head_r2=r2[j], head_mae=mae[j], head_rmse=rmse[j])
            rows.append(row)
        predictions.append(
            pd.DataFrame(
                dict(
                    identity,
                    scope=scope,
                    sample_index=ids,
                    pc1=pc1,
                    true_fresh_weight=y[ids, 0],
                    predicted_fresh_weight=pred[ids, 0],
                )
            )
        )
    return rows, pd.concat(predictions, ignore_index=True)


def plot(rows, output):
    keys = ["mode", "fold", "fraction", "n_train", "scope"]
    pc1 = rows.groupby(keys).pc1_pearson_abs.mean().rename("pc1_mean_abs_r").reset_index()
    fw = rows[rows.trait.eq("FreshWeightShoot")][keys + ["head_mae"]]
    per_fold = pc1.merge(fw, on=keys, validate="one_to_one")
    per_fold.to_csv(output / "FigS1-folds.csv", index=False)
    summary = (
        per_fold.groupby(["mode", "fraction", "scope"])
        .agg(
            n_train_min=("n_train", "min"),
            n_train_max=("n_train", "max"),
            pc1_mean=("pc1_mean_abs_r", "mean"),
            pc1_sd=("pc1_mean_abs_r", "std"),
            mae_mean=("head_mae", "mean"),
            mae_sd=("head_mae", "std"),
        )
        .reset_index()
    )
    summary.to_csv(output / "FigS1-summary.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for mode, label, color in [("single", "Single", "#55A868"), ("multi", "Multi", "#DD8452")]:
        for scope, style in [("train", "--"), ("test", "-")]:
            d = summary[(summary["mode"] == mode) & (summary.scope == scope)].sort_values("fraction")
            for ax, field in zip(axes, ["pc1", "mae"]):
                ax.plot(
                    100 * d.fraction,
                    d[field + "_mean"],
                    style,
                    marker="o",
                    ms=4,
                    color=color,
                    alpha=0.65 if scope == "train" else 1,
                    label=f"{label} {scope}",
                )
                if scope == "test":
                    ax.fill_between(
                        100 * d.fraction,
                        d[field + "_mean"] - d[field + "_sd"],
                        d[field + "_mean"] + d[field + "_sd"],
                        color=color,
                        alpha=0.15,
                    )
    fractions = sorted(summary.fraction.unique())
    sizes = summary.groupby("fraction").agg(lo=("n_train_min", "min"), hi=("n_train_max", "max"))
    labels = [
        f"{100 * f:g}%\n{sizes.loc[f, 'lo']:.0f}–{sizes.loc[f, 'hi']:.0f}"
        if sizes.loc[f, "lo"] != sizes.loc[f, "hi"]
        else f"{100 * f:g}%\n{sizes.loc[f, 'lo']:.0f}"
        for f in fractions
    ]
    for ax in axes:
        ax.set_xticks(np.array(fractions) * 100, labels)
        ax.set_xlabel("Training fraction / image count (validation labels fixed separately)")
        ax.grid(alpha=0.18)
        soften_axes(ax)
    axes[0].set(title="PC1 association with five source traits", ylabel="Mean |Pearson r|", ylim=(0.8, 1.0))
    axes[1].set(title="Fresh-weight prediction", ylabel="MAE (g / plant)", ylim=(0, None))
    axes[0].legend(frameon=False, ncol=2, fontsize=9, loc="lower right")
    fig.suptitle("Frozen DINO · source-projector scaling · five held-out folds", fontsize=14)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"FigS1.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument(
        "--train", action="store_true", help="Train the 50 scaling models with the shared MLP trainer."
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / "weights/scaling")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_global_seed(42)
    data = load_dataset("Lettuce", config=args.config, targets=TRAITS)
    if args.train:
        for split in make_splits(data.metadata.group.to_numpy(), FRACTIONS, 42):
            for fraction in FRACTIONS:
                ids = split["subsets"][str(fraction)]
                for mode, targets in [("single", TRAITS[:1]), ("multi", TRAITS)]:
                    _, saved = train_mlp(
                        data.features,
                        data.targets[:, : len(targets)],
                        targets,
                        updates=300,
                        seed=split["seed"],
                        train_indices=ids,
                        validation_indices=split["validation"],
                        device=args.device,
                    )
                    saved.update(test_indices=split["test"], fraction=fraction, fold=split["fold"], mode=mode)
                    save_checkpoint(saved, args.model_dir / f"{mode}-fold{split['fold']}-n{len(ids)}.pth")
        directory = args.model_dir
    else:
        directory = args.checkpoints or model_paths(args.models)["source"]["scaling"]
    rows = []
    predictions = []
    for path in sorted(directory.glob("*.pth")):
        model, saved = load_model(path, args.device)
        saved["target_names"] = saved.get("target_names", saved.get("targets"))
        match = re.fullmatch(r"(single|multi)-fold(\d+)-n(\d+)", path.stem)
        mode, fold = match[1], int(match[2])
        pool_size = len(data) - len(saved["validation_indices"]) - len(saved["test_indices"])
        n_train = len(saved["train_indices"])
        fraction = saved.get("fraction", min(FRACTIONS, key=lambda f: abs(round(f * pool_size) - n_train)))
        identity = dict(
            mode=mode,
            fold=fold,
            seed=40 + fold,
            fraction=fraction,
            n_train=n_train,
            n_validation=len(saved["validation_indices"]),
            n_test=len(saved["test_indices"]),
            selected_step=saved["best_step"],
        )
        result, prediction = evaluate(
            model, saved, data.features, data.targets, identity, torch.device(args.device)
        )
        rows.extend(result)
        predictions.append(prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "FigS1-trait-metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "FigS1-predictions.csv", index=False)
    plot(frame, args.output_dir)


if __name__ == "__main__":
    main()
