"""FigS3: individual LAB-Lettuce trajectories, current pre-disruption fitting protocol."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt

from src.dataloader import CONFIG
from src.models import set_global_seed
from src.plotting import soften_axes
from src.trajectories import (
    SPLIT_DATE,
    before_normalize,
    compute_trajectories,
    time_colors,
)

ROOT = Path(__file__).resolve().parent
SELECTED_CAMS = [f"Cam{i}" for i in (1, 5, 9, 13, 17, 21)]
METHOD_ORDER = ["DINO", "Single", "Single+SSA", "Multi", "Multi+SSA"]
FIGURE_WIDTH_IN = 7.2
DPI = 300
TITLE_FONTSIZE = 8.5
LABEL_FONTSIZE = 7.5
TICK_FONTSIZE = 6.5


def selected_units(samples: List[object]) -> List[str]:
    counts: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        item = counts.setdefault(sample.unit_id, {"cam": sample.cam, "before": 0, "after": 0, "total": 0})
        item["total"] += 1
        if sample.timestamp.date() < SPLIT_DATE:
            item["before"] += 1
        else:
            item["after"] += 1
    picked: List[str] = []
    for cam in SELECTED_CAMS:
        candidates = [
            (unit_id, meta["total"])
            for unit_id, meta in counts.items()
            if meta["cam"] == cam and meta["before"] >= 20 and meta["after"] >= 10
        ]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        if candidates:
            picked.append(candidates[0][0])
    return picked


def short_unit_label(unit_id: str) -> str:
    layer, side, grid, cam = unit_id.split("|")
    return f"L{layer} {side[0].upper()} {grid} {cam}"


def plot_unit_trends(
    samples: List[object], coords_by_method: Dict[str, np.ndarray], output: Path
) -> List[str]:
    units = selected_units(samples)
    fig, axes = plt.subplots(
        len(units),
        len(METHOD_ORDER),
        figsize=(FIGURE_WIDTH_IN, 8.2),
        sharex=True,
        sharey=True,
        dpi=DPI,
        squeeze=False,
    )
    split_dt = datetime.combine(SPLIT_DATE, datetime.min.time())

    all_values = []
    for method_name in METHOD_ORDER:
        coords = coords_by_method[method_name]
        for unit_id in units:
            indices = [idx for idx, sample in enumerate(samples) if sample.unit_id == unit_id]
            indices.sort(key=lambda idx: samples[idx].timestamp)
            timestamps = [samples[idx].timestamp for idx in indices]
            before_mask = np.array([ts.date() < SPLIT_DATE for ts in timestamps], dtype=bool)
            if before_mask.sum() >= 3:
                all_values.extend(before_normalize(coords[indices, 0], before_mask).tolist())

    y_min = float(np.nanmin(all_values)) if all_values else -2.0
    y_max = float(np.nanmax(all_values)) if all_values else 2.0
    pad = 0.08 * max(1.0, y_max - y_min)

    for col_idx, method_name in enumerate(METHOD_ORDER):
        coords = coords_by_method[method_name]
        for row_idx, unit_id in enumerate(units):
            ax = axes[row_idx, col_idx]
            indices = [idx for idx, sample in enumerate(samples) if sample.unit_id == unit_id]
            indices.sort(key=lambda idx: samples[idx].timestamp)
            timestamps = [samples[idx].timestamp for idx in indices]
            before_mask = np.array([ts.date() < SPLIT_DATE for ts in timestamps], dtype=bool)
            pc1 = before_normalize(coords[indices, 0], before_mask)
            colors = time_colors(timestamps)
            ax.scatter(timestamps, pc1, s=5.5, c=colors, edgecolors="none", alpha=0.78)
            ax.axvline(split_dt, color="#707477", linestyle="--", linewidth=0.6)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.tick_params(labelsize=TICK_FONTSIZE, width=0.55, length=2.5)
            soften_axes(ax)
            if row_idx == 0:
                ax.set_title(method_name, fontsize=TITLE_FONTSIZE, pad=3)
            if row_idx == len(units) - 1:
                ax.tick_params(axis="x", rotation=35, labelbottom=True)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if col_idx == 0:
                ax.annotate(
                    short_unit_label(unit_id).replace(" Cam", "\nCam"),
                    xy=(-0.46, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=LABEL_FONTSIZE,
                    annotation_clip=False,
                )

    fig.suptitle("Single-plant crop PC1 trends by unit (before-normalized)", fontsize=9.5, y=0.995)
    fig.text(0.52, 0.025, "Date", ha="center", fontsize=LABEL_FONTSIZE)
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.075, top=0.955, wspace=0.16, hspace=0.18)
    fig.savefig(output / "FigS3.png", bbox_inches="tight", pad_inches=0.03, dpi=DPI)
    fig.savefig(output / "FigS3.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_global_seed(42)
    samples, coordinates = compute_trajectories("LAB-crop", args.config, args.models, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_unit_trends(samples, coordinates, args.output_dir)
    np.savez_compressed(args.output_dir / "FigS3-coordinates.npz", **coordinates)


if __name__ == "__main__":
    main()
