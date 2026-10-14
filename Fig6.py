"""Compute mean LAB trajectories; photographs are assembled separately by the author."""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.dataloader import CONFIG
from src.models import ROOT, set_global_seed
from src.plotting import save, soften_axes
from src.trajectories import METHOD_ORDER, SPLIT_DATE, compute_trajectories

TICK_FONTSIZE = 6.5
SPLIT_COLOR = "#707477"


def rounded_hour(timestamp: datetime) -> datetime:
    base = timestamp.replace(minute=0, second=0, microsecond=0)
    if timestamp.minute >= 30:
        base += timedelta(hours=1)
    return base


def entity_series(
    samples: List[object],
    coords: np.ndarray,
    *,
    entity_attr: str,
    split_date,
    trend_degree: int,
) -> Tuple[Dict[str, Dict[datetime, float]], Dict[str, Dict[datetime, float]]]:
    indices_by_entity: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        indices_by_entity[str(getattr(sample, entity_attr))].append(idx)

    observed_output: Dict[str, Dict[datetime, float]] = {}
    expected_output: Dict[str, Dict[datetime, float]] = {}
    for entity, indices in indices_by_entity.items():
        indices.sort(key=lambda idx: samples[idx].timestamp)
        timestamps = [samples[idx].timestamp for idx in indices]
        before_mask = np.asarray([timestamp.date() < split_date for timestamp in timestamps], dtype=bool)
        if int(before_mask.sum()) < 3:
            continue
        values = np.asarray(coords[indices, 0], dtype=float)
        before_values = values[before_mask]
        std = float(np.std(before_values))
        if std < 1e-8:
            continue
        normalized = (values - float(np.mean(before_values))) / std
        x0 = timestamps[0].timestamp()
        x_days = np.asarray([(timestamp.timestamp() - x0) / 86400.0 for timestamp in timestamps], dtype=float)
        degree = min(int(trend_degree), int(before_mask.sum()) - 1)
        coefficients = np.polyfit(x_days[before_mask], normalized[before_mask], deg=degree)
        expected = np.polyval(coefficients, x_days)

        observed_hourly: Dict[datetime, List[float]] = defaultdict(list)
        expected_hourly: Dict[datetime, List[float]] = defaultdict(list)
        for timestamp, observed_value, expected_value in zip(timestamps, normalized, expected):
            hour = rounded_hour(timestamp)
            if np.isfinite(observed_value):
                observed_hourly[hour].append(float(observed_value))
            if np.isfinite(expected_value):
                expected_hourly[hour].append(float(expected_value))
        observed_output[entity] = {
            timestamp: float(np.mean(values_at_hour)) for timestamp, values_at_hour in observed_hourly.items()
        }
        expected_output[entity] = {
            timestamp: float(np.mean(values_at_hour)) for timestamp, values_at_hour in expected_hourly.items()
        }
    return observed_output, expected_output


def aggregate_series(
    series_by_entity: Dict[str, Dict[datetime, float]],
) -> List[Dict[str, object]]:
    values_by_time: Dict[datetime, List[float]] = defaultdict(list)
    for series in series_by_entity.values():
        for timestamp, value in series.items():
            values_by_time[timestamp].append(value)

    rows: List[Dict[str, object]] = []
    for timestamp in sorted(values_by_time):
        values = np.asarray(values_by_time[timestamp], dtype=float)
        count = int(values.size)
        std = float(np.std(values, ddof=1)) if count > 1 else 0.0
        rows.append(
            {
                "timestamp": timestamp,
                "mean": float(np.mean(values)),
                "std": std,
                "sem": std / np.sqrt(count) if count > 0 else float("nan"),
                "entity_count": count,
            }
        )
    return rows


def plot_mean_curve(
    ax,
    observed_rows: List[Dict[str, object]],
    expected_rows: List[Dict[str, object]],
    split_date,
) -> None:
    timestamps = np.asarray([row["timestamp"] for row in observed_rows], dtype=object)
    means = np.asarray([row["mean"] for row in observed_rows], dtype=float)
    stds = np.asarray([row["std"] for row in observed_rows], dtype=float)
    before = np.asarray([timestamp.date() < split_date for timestamp in timestamps], dtype=bool)
    after = ~before

    for mask, cmap_name in ((before, "Blues"), (after, "Reds")):
        if not np.any(mask):
            continue
        segment_times = timestamps[mask]
        segment_means = means[mask]
        segment_stds = stds[mask]
        cmap = plt.get_cmap(cmap_name)
        band_color = cmap(0.58)
        ax.fill_between(
            segment_times,
            segment_means - segment_stds,
            segment_means + segment_stds,
            color=band_color,
            alpha=0.22,
            linewidth=0,
        )
        point_colors = cmap(np.linspace(0.42, 0.90, len(segment_times)))
        ax.plot(
            segment_times,
            segment_means,
            color=cmap(0.68),
            linewidth=0.75,
            alpha=0.72,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )
        ax.scatter(
            segment_times,
            segment_means,
            s=5.2,
            c=point_colors,
            marker="o",
            edgecolors="none",
            linewidths=0,
            alpha=0.92,
            zorder=3,
        )

    expected_by_time = {row["timestamp"]: row for row in expected_rows}
    post_times = timestamps[after]
    post_observed = means[after]
    matched_times = [timestamp for timestamp in post_times if timestamp in expected_by_time]
    if matched_times:
        matched_observed = np.asarray(
            [post_observed[idx] for idx, timestamp in enumerate(post_times) if timestamp in expected_by_time],
            dtype=float,
        )
        matched_expected = np.asarray(
            [float(expected_by_time[timestamp]["mean"]) for timestamp in matched_times],
            dtype=float,
        )
        matched_expected_std = np.asarray(
            [float(expected_by_time[timestamp]["std"]) for timestamp in matched_times],
            dtype=float,
        )
        ax.fill_between(
            matched_times,
            matched_expected - matched_expected_std,
            matched_expected + matched_expected_std,
            color="#6F7377",
            alpha=0.13,
            linewidth=0,
            zorder=0,
        )
        ax.fill_between(
            matched_times,
            matched_observed,
            matched_expected,
            color="#8A8D91",
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            matched_times,
            matched_expected,
            color="#55595D",
            linestyle="--",
            linewidth=0.9,
            zorder=4,
        )

    split_dt = datetime.combine(split_date, datetime.min.time())
    ax.axvline(split_dt, color=SPLIT_COLOR, linestyle="--", linewidth=0.65)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(labelsize=TICK_FONTSIZE, width=0.55, length=2.5)
    soften_axes(ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", type=Path, default=ROOT / "data/models.yml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_global_seed(42)
    fig, axes = plt.subplots(2, 5, figsize=(12, 4.9), sharey=True)
    rows = []
    for row, (dataset, label, attr) in enumerate(
        [("LAB-crop", "Single-plant crops", "unit_id"), ("LAB-fullimage", "Full images", "cam")]
    ):
        samples, coordinates = compute_trajectories(dataset, args.config, args.models, args.device)
        for column, method in enumerate(METHOD_ORDER):
            observed, expected = entity_series(
                samples, coordinates[method], entity_attr=attr, split_date=SPLIT_DATE, trend_degree=2
            )
            obs, exp = aggregate_series(observed), aggregate_series(expected)
            plot_mean_curve(axes[row, column], obs, exp, SPLIT_DATE)
            if row == 0:
                axes[row, column].set_title("DINOv2" if method == "DINO" else method, fontsize=10)
            if column == 0:
                axes[row, column].set_ylabel(f"{label} (n={len(observed)})", fontsize=9)
            for category, series in [("observed", obs), ("expected", exp)]:
                rows.extend(
                    dict(dataset=dataset, method=method, series=category, **record) for record in series
                )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.93, hspace=0.22, wspace=0.18)
    save(fig, "Fig6", args.output_dir)
    pd.DataFrame(rows).to_csv(args.output_dir / "Fig6-curves.csv", index=False)


if __name__ == "__main__":
    main()
