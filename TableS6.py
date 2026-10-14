"""Table S6: GroMo25 coverage calculated from annotations and available source images."""

import argparse
from pathlib import Path

import pandas as pd

from src.dataloader import CONFIG, load_metadata
from src.models import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--datasets", nargs="+", default=["mustard", "okra", "radish", "wheat"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows = []
    for crop in args.datasets:
        meta = load_metadata(crop, config=args.config)
        rows.append(
            dict(
                species=crop,
                images=len(meta),
                plants=meta.plant.nunique(),
                plant_days=len(meta[["plant", "day"]].drop_duplicates()),
                views=len(meta[["plant", "level", "angle"]].drop_duplicates()),
            )
        )
    frame = pd.DataFrame(rows)
    total = dict(species="Total", **frame.select_dtypes("number").sum().to_dict())
    frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "TableS6.csv", index=False)
    print(args.output_dir / "TableS6.csv")


if __name__ == "__main__":
    main()
