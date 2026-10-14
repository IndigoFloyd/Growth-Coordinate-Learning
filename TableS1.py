"""Table S1: validation-head performance for the 64/128/256-dimensional source models."""

import numpy as np
import pandas as pd
import torch

from src.analysis import regression_metrics
from src.dataloader import load_dataset
from src.evaluation import checkpoint, source_split
from src.mlp import forward_outputs
from src.models import load_model, model_paths, set_global_seed
from src.tables import save_table, table_parser


def main():
    parser = table_parser(__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[64, 128, 256])
    args = parser.parse_args()
    set_global_seed(42)
    data = load_dataset("Lettuce", config=args.config)
    paths = model_paths(args.models)
    rows = []
    traits = []
    for mode in ["single", "multi"]:
        for dimension in args.dimensions:
            for seed in args.seeds:
                model, payload = load_model(
                    checkpoint(paths, "source", mode + "_runs", hidden_dim=dimension, seed=seed), args.device
                )
                _, val = source_split(payload, len(data), seed)
                targets = payload["target_names"]
                truth = data.metadata[targets].to_numpy(dtype=np.float32)[val]
                mean = torch.tensor(payload["target_mean"], device=args.device, dtype=torch.float32)
                std = torch.tensor(payload["target_std"], device=args.device, dtype=torch.float32)
                predictions = []
                with torch.no_grad():
                    for start in range(0, len(val), 128):
                        inputs = data.x[val[start : start + 128]].to(args.device)
                        predictions.append(
                            (forward_outputs(model, inputs, targets) * std + mean).cpu().numpy()
                        )
                metrics = regression_metrics(truth, np.concatenate(predictions))
                rows.append(
                    dict(
                        mode=mode,
                        hidden_dim=dimension,
                        seed=seed,
                        **{k: float(np.mean(v)) for k, v in metrics.items()},
                    )
                )
                traits.extend(
                    dict(
                        mode=mode,
                        hidden_dim=dimension,
                        seed=seed,
                        target=name,
                        **{k: float(v[i]) for k, v in metrics.items()},
                    )
                    for i, name in enumerate(targets)
                )
    save_table(rows, "TableS1", args.output_dir, ["mode", "hidden_dim"], ["r2", "mae", "rmse"])
    pd.DataFrame(traits).to_csv(args.output_dir / "TableS1-traits.csv", index=False)


if __name__ == "__main__":
    main()
