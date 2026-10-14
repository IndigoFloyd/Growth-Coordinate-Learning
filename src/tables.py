"""Command options and numeric CSV output shared by manuscript table scripts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .dataloader import CONFIG
from .models import ROOT


def table_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(40, 45)))
    return parser


def save_table(rows, name, output, keys, metrics, ddof=1):
    """Keep replicate rows as well as the mean, SD and range used in the table."""
    frame = pd.DataFrame(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{name}-details.csv", index=False)
    summary = []
    for identity, group in frame.groupby(keys, sort=False, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        row = dict(zip(keys, identity))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row.update(
                {
                    metric + "_mean": float(np.mean(values)),
                    metric + "_std": float(np.std(values, ddof=ddof)) if len(values) > ddof else 0.0,
                    metric + "_min": float(np.min(values)),
                    metric + "_max": float(np.max(values)),
                    metric + "_n": len(values),
                }
            )
        summary.append(row)
    result = pd.DataFrame(summary)
    result.to_csv(output / f"{name}.csv", index=False)
    print(output / f"{name}.csv")
    return result
