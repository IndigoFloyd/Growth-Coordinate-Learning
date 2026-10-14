"""Table 6: proxy-augmented tomato regression with transductive unlabeled SSA adaptation."""

from pathlib import Path

import numpy as np

from src.dataloader import load_dataset
from src.evaluation import (
    checkpoint,
    hidden_at,
    projected_split,
    proxy_rows,
    stratified_split,
)
from src.features import PROXY_FEATURES, image_features
from src.models import model_paths, set_global_seed
from src.tables import save_table, table_parser


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    data = load_dataset("Tomato", config=args.config)
    paths = model_paths(args.models)
    proxies = image_features("Tomato", data.metadata, args.config, PROXY_FEATURES)
    groups = np.asarray([Path(key).stem.split("_", 1)[0] for key in data.metadata.key])
    baseline = {}
    for branch in ["single", "multi"]:
        baseline[branch.title()], _ = hidden_at(paths["source"][branch], data.features, args.device)
    rows = []
    for seed in args.seeds:
        train, val = stratified_split(groups, seed, shuffle_result=True)
        features = {"DINOv2": data.features}
        for branch in ["single", "multi"]:
            features[branch.title()] = baseline[branch.title()]
            features[branch.title() + " + SSA"], _ = hidden_at(
                checkpoint(paths, "Tomato", branch + "_proxy_runs", seed=seed), data.features, args.device
            )
        for method, values in features.items():
            pc1_train, pc1_val = projected_split(values, train, val, apply_to_train=False)
            rows.extend(
                proxy_rows(
                    pc1_train,
                    pc1_val,
                    proxies[train],
                    proxies[val],
                    data.targets[train],
                    data.targets[val],
                    data.target_names,
                    seed=seed,
                    method=method,
                )
            )
    save_table(rows, "Table6", args.output_dir, ["method", "target", "model_type"], ["r2"])


if __name__ == "__main__":
    main()
