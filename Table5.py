"""Table 5: tomato DeltaCorr from the K=70 transfer checkpoints, across five SSA seeds."""

from src.analysis import fit_pca, weighted_delta_corr_from_arrays
from src.dataloader import load_dataset, load_features
from src.evaluation import checkpoint, hidden_at
from src.models import model_paths, set_global_seed
from src.tables import save_table, table_parser


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    data = load_dataset("Tomato", config=args.config)
    paths = model_paths(args.models)
    baseline = {"DINO latent": load_features(data.metadata, "raw")}
    for branch in ["single", "multi"]:
        baseline[branch.title()], _ = hidden_at(paths["source"][branch], data.features, args.device)
    rows = []
    for seed in args.seeds:
        features = {"DINO latent": baseline["DINO latent"]}
        for branch in ["single", "multi"]:
            features[branch.title()] = baseline[branch.title()]
            features[branch.title() + " + SSA"], _ = hidden_at(
                checkpoint(paths, "Tomato", branch + "_runs", seed=seed), data.features, args.device
            )
        for method, values in features.items():
            for trait in data.target_names:
                xy, _, _ = fit_pca(values)
                value = weighted_delta_corr_from_arrays(
                    xy[:, 0], data.metadata[trait].to_numpy(), data.metadata.group.to_numpy()
                )
                rows.append(dict(method=method, target_metric=trait, seed=seed, delta_corr_abs=abs(value)))
    save_table(rows, "Table5", args.output_dir, ["method", "target_metric"], ["delta_corr_abs"])


if __name__ == "__main__":
    main()
