"""Table 9: full-image rollback after irrigation interruption; SD is across cameras (ddof=0)."""

from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.models import set_global_seed
from src.tables import save_table, table_parser
from src.trajectories import (
    METHOD_ORDER,
    SPLIT_DATE,
    before_normalize,
    compute_trajectories,
)


def mean_finite(values: Iterable[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def fit_before_curve(
    timestamps: List[datetime],
    pc1: np.ndarray,
    before_mask: np.ndarray,
    requested_degree: int,
    min_before: int,
) -> tuple[np.ndarray, int, float]:
    before_indices = np.where(before_mask)[0]
    if before_indices.size < min_before:
        return np.full(pc1.shape, np.nan, dtype=float), -1, float("nan")

    x0 = timestamps[int(before_indices[0])].timestamp()
    x = np.asarray([(timestamp.timestamp() - x0) / 86400.0 for timestamp in timestamps], dtype=float)
    x_before = x[before_mask]
    y_before = pc1[before_mask]
    degree = min(max(1, int(requested_degree)), int(before_indices.size) - 1)
    coeffs = np.polyfit(x_before, y_before, deg=degree)
    fitted = np.polyval(coeffs, x)

    before_corr = float("nan")
    fitted_before = fitted[before_mask]
    if np.std(fitted_before) > 1e-9 and np.std(y_before) > 1e-9:
        before_corr = float(np.corrcoef(fitted_before, y_before)[0, 1])
    return fitted.astype(float), degree, before_corr


def build_rows(
    samples: List[object],
    coords_by_method: Dict[str, np.ndarray],
    split_date,
    fit_degree: int,
    min_before: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    image_rows: List[Dict[str, object]] = []
    cam_rows: List[Dict[str, object]] = []

    cams = sorted({sample.cam for sample in samples}, key=lambda x: int(x.replace("IPCam", "")))
    for method_name in METHOD_ORDER:
        coords = coords_by_method[method_name]
        for cam in cams:
            indices = [idx for idx, sample in enumerate(samples) if sample.cam == cam]
            indices.sort(key=lambda idx: samples[idx].timestamp)
            timestamps = [samples[idx].timestamp for idx in indices]
            before_mask = np.asarray([timestamp.date() < split_date for timestamp in timestamps], dtype=bool)
            after_mask = ~before_mask
            if before_mask.sum() < min_before or after_mask.sum() == 0:
                continue

            pc1 = before_normalize(coords[indices, 0], before_mask).astype(float)
            fitted, actual_degree, fit_corr = fit_before_curve(
                timestamps, pc1, before_mask, fit_degree, min_before
            )
            before_values = pc1[before_mask]
            before_range = float(np.max(before_values) - np.min(before_values))
            if before_range < 1e-8:
                before_range = float("nan")

            drops: List[float] = []
            amplitudes: List[float] = []
            for local_idx, sample_idx in enumerate(indices):
                phase = "before" if before_mask[local_idx] else "after"
                drop = float("nan")
                amplitude = float("nan")
                if phase == "after":
                    drop = float(fitted[local_idx] - pc1[local_idx])
                    amplitude = float(drop / before_range) if np.isfinite(before_range) else float("nan")
                    drops.append(drop)
                    amplitudes.append(amplitude)
                image_rows.append(
                    {
                        "method": method_name,
                        "cam": cam,
                        "timestamp": samples[sample_idx].timestamp.isoformat(),
                        "phase": phase,
                        "pc1": float(pc1[local_idx]),
                        "fitted_pc1": float(fitted[local_idx]),
                        "residual_observed_minus_fit": float(pc1[local_idx] - fitted[local_idx]),
                        "drop_from_fit": drop,
                        "drop_amplitude_vs_before_range": amplitude,
                    }
                )

            cam_rows.append(
                {
                    "method": method_name,
                    "cam": cam,
                    "n_total": len(indices),
                    "n_before": int(before_mask.sum()),
                    "n_after": int(after_mask.sum()),
                    "fit_degree": actual_degree,
                    "before_fit_corr": fit_corr,
                    "before_pc1_min": float(np.min(before_values)),
                    "before_pc1_max": float(np.max(before_values)),
                    "before_pc1_range": before_range,
                    "normal_after_mean": mean_finite(fitted[after_mask]),
                    "actual_after_mean": mean_finite(pc1[after_mask]),
                    "mean_after_drop_from_fit": mean_finite(drops),
                    "mean_after_drop_amplitude": mean_finite(amplitudes),
                }
            )
    return image_rows, cam_rows


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    samples, coordinates = compute_trajectories("LAB-fullimage", args.config, args.models, args.device)
    images, cameras = build_rows(samples, coordinates, SPLIT_DATE, 2, 3)
    save_table(
        cameras,
        "Table9",
        args.output_dir,
        ["method"],
        ["before_fit_corr", "mean_after_drop_amplitude"],
        ddof=0,
    )
    pd.DataFrame(images).to_csv(args.output_dir / "Table9-observations.csv", index=False)


if __name__ == "__main__":
    main()
