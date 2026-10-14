"""Table 3: held-out lettuce DeltaCorr for DINO, source projectors and image baselines."""

from src.analysis import weighted_delta_corr_from_arrays
from src.dataloader import load_dataset
from src.evaluation import checkpoint, hidden_at, projected_split, source_split
from src.features import image_features
from src.models import model_paths, set_global_seed
from src.tables import save_table, table_parser


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    data = load_dataset("Lettuce", config=args.config)
    paths = model_paths(args.models)
    baseline = image_features("Lettuce", data.metadata, args.config)
    rows = []
    for seed in args.seeds:
        features = {"DINOv2": data.features}
        for branch in ["single", "multi"]:
            features[branch.title()], payload = hidden_at(
                checkpoint(paths, "source", branch + "_runs", hidden_dim=256, seed=seed),
                data.features,
                args.device,
            )
        features["Baseline"] = baseline
        train, val = source_split(payload, len(data), seed)
        for method, values in features.items():
            _, pc1 = projected_split(values, train, val)
            meta = data.metadata.iloc[val]
            for trait in data.target_names:
                score = weighted_delta_corr_from_arrays(pc1, meta[trait].to_numpy(), meta.group.to_numpy())
                rows.append(dict(model=method, target=trait, seed=seed, delta_corr_abs=abs(score)))
    save_table(rows, "Table3", args.output_dir, ["target", "model"], ["delta_corr_abs"])


if __name__ == "__main__":
    main()
