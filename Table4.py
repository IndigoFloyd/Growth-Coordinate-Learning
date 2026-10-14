"""Table 4: proxy-only, PC1-only and combined linear probes on lettuce validation splits."""

from src.dataloader import load_dataset
from src.evaluation import (
    checkpoint,
    hidden_at,
    projected_split,
    proxy_rows,
    source_split,
)
from src.features import PROXY_FEATURES, image_features
from src.models import model_paths, set_global_seed
from src.tables import save_table, table_parser


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    traits = ["FreshWeightShoot", "Height", "Diameter", "LeafArea", "DryWeightShoot"]
    data = load_dataset("Lettuce", config=args.config, targets=traits)
    paths = model_paths(args.models)
    proxies = image_features("Lettuce", data.metadata, args.config, PROXY_FEATURES)
    rows = []
    for seed in args.seeds:
        features = {"DINOv2": data.features}
        for branch in ["single", "multi"]:
            features[branch.title()], payload = hidden_at(
                checkpoint(paths, "source", branch + "_runs", hidden_dim=256, seed=seed),
                data.features,
                args.device,
            )
        train, val = source_split(payload, len(data), seed)
        for method, values in features.items():
            pc1_train, pc1_val = projected_split(values, train, val)
            rows.extend(
                proxy_rows(
                    pc1_train,
                    pc1_val,
                    proxies[train],
                    proxies[val],
                    data.targets[train],
                    data.targets[val],
                    traits,
                    seed=seed,
                    method=method,
                )
            )
    save_table(rows, "Table4", args.output_dir, ["target", "method", "model_type"], ["r2"])


if __name__ == "__main__":
    main()
