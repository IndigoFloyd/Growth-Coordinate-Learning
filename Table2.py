"""Table 2: transductive PC1 identity probe, using the original stratified label splits."""

import re

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.analysis import fit_pca
from src.dataloader import load_features, load_metadata
from src.evaluation import BACKBONES, stratified_split
from src.models import set_global_seed
from src.tables import save_table, table_parser


def main():
    parser = table_parser(__doc__)
    parser.add_argument("--datasets", nargs="+", default=["Tomato", "Komatsuna", "Lettuce"])
    args = parser.parse_args()
    set_global_seed(42)
    rows = []
    for dataset in args.datasets:
        metadata = {
            model: load_metadata(dataset, encoder, args.config).set_index("key", drop=False)
            for model, encoder in BACKBONES
        }
        keys = sorted(set.intersection(*(set(frame.index) for frame in metadata.values())))
        reference = metadata["DINOv2"].loc[keys]
        groups = reference.group.to_numpy()
        if dataset == "Komatsuna":
            groups = np.asarray([re.search(r"(\d{2})_(\d{3})_(\d{2})", key)[1] for key in keys])
        splits = {seed: stratified_split(groups, seed) for seed in args.seeds}
        for model, _ in BACKBONES:
            features = load_features(metadata[model].loc[keys], "raw")
            xy, _, _ = fit_pca(features)
            pc1 = xy[:, :1]
            for seed, (train, val) in splits.items():
                probe = LogisticRegression(max_iter=1000, random_state=seed)
                probe.fit(pc1[train], groups[train])
                rows.append(
                    dict(
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        n_train=len(train),
                        n_val=len(val),
                        pca_fit_scope="full_dataset_transductive",
                        accuracy=float(np.mean(probe.predict(pc1[val]) == groups[val])),
                    )
                )
    save_table(rows, "Table2", args.output_dir, ["dataset", "model"], ["accuracy"])


if __name__ == "__main__":
    main()
