"""Table 1: five shared subclass-stratified bootstrap replicates of backbone DeltaCorr."""

import numpy as np

from src.analysis import fit_pca, weighted_delta_corr_from_arrays
from src.dataloader import load_features, load_metadata
from src.evaluation import BACKBONES
from src.models import set_global_seed
from src.tables import save_table, table_parser


def main():
    parser = table_parser(__doc__)
    parser.add_argument("--datasets", nargs="+", default=["Tomato", "Lettuce"])
    args = parser.parse_args()
    set_global_seed(42)
    rows = []
    for dataset in args.datasets:
        reference = load_metadata(dataset, config=args.config)
        groups = reference.group.to_numpy()
        keys = reference.key.to_numpy()
        draws = {}
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            draws[seed] = np.concatenate(
                [
                    rng.choice(keys[groups == g], size=np.sum(groups == g), replace=True)
                    for g in sorted(set(groups))
                ]
            )
        backbones = {}
        for model, encoder in BACKBONES:
            meta = load_metadata(dataset, encoder, args.config)
            meta = meta[meta.key.isin(keys)].reset_index(drop=True)
            backbones[model] = (
                meta.set_index("key", drop=False),
                load_features(meta, "raw"),
                {k: i for i, k in enumerate(meta.key)},
            )
        traits = (
            ["height", "fw_plant", "leaf_area"]
            if dataset == "Tomato"
            else ["FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"]
        )
        for trait in traits:
            for model, _ in BACKBONES:
                meta, features, indices = backbones[model]
                for seed in args.seeds:
                    sampled = [k for k in draws[seed] if k in indices and np.isfinite(meta.loc[k, trait])]
                    selected = meta.loc[sampled]
                    xy, _, _ = fit_pca(features[[indices[k] for k in sampled]])
                    value = weighted_delta_corr_from_arrays(
                        xy[:, 0], selected[trait].to_numpy(), selected.group.to_numpy()
                    )
                    rows.append(
                        dict(dataset=dataset, target=trait, model=model, seed=seed, delta_corr_abs=abs(value))
                    )
    save_table(rows, "Table1", args.output_dir, ["dataset", "target", "model"], ["delta_corr_abs"])


if __name__ == "__main__":
    main()
