"""Table S5: crop/full-image trajectory correlations, averaged within cameras before across-camera SD."""

from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.models import set_global_seed
from src.tables import save_table, table_parser
from src.trajectories import (
    SPLIT_DATE,
    before_normalize,
    compute_trajectories,
)


def pearson_corr(xs: Iterable[float], ys: Iterable[float], min_shared: int) -> float:
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < min_shared or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_full_curves(
    samples: List[object],
    coords_by_method: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Dict[datetime, float]]]:
    curves: Dict[str, Dict[str, Dict[datetime, float]]] = {}
    for method_name, coords in coords_by_method.items():
        curves[method_name] = {}
        cams = sorted({sample.cam for sample in samples}, key=lambda x: int(x.replace("IPCam", "")))
        for cam in cams:
            indices = [idx for idx, sample in enumerate(samples) if sample.cam == cam]
            indices.sort(key=lambda idx: samples[idx].timestamp)
            timestamps = [samples[idx].timestamp for idx in indices]
            before_mask = np.asarray([ts.date() < SPLIT_DATE for ts in timestamps], dtype=bool)
            if before_mask.sum() < 1:
                continue
            pc1 = before_normalize(coords[indices, 0], before_mask)
            curves[method_name][cam] = {timestamp: float(value) for timestamp, value in zip(timestamps, pc1)}
    return curves


def build_crop_curves(
    samples: List[object],
    coords_by_method: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Tuple[str, Dict[datetime, float]]]]:
    curves: Dict[str, Dict[str, Tuple[str, Dict[datetime, float]]]] = {}
    for method_name, coords in coords_by_method.items():
        curves[method_name] = {}
        unit_ids = sorted({sample.unit_id for sample in samples}, key=lambda x: (x.split("|")[3], x))
        for unit_id in unit_ids:
            indices = [idx for idx, sample in enumerate(samples) if sample.unit_id == unit_id]
            indices.sort(key=lambda idx: samples[idx].timestamp)
            timestamps = [samples[idx].timestamp for idx in indices]
            before_mask = np.asarray([ts.date() < SPLIT_DATE for ts in timestamps], dtype=bool)
            if before_mask.sum() < 1:
                continue
            pc1 = before_normalize(coords[indices, 0], before_mask)
            cam = samples[indices[0]].cam
            curves[method_name][unit_id] = (
                cam,
                {timestamp: float(value) for timestamp, value in zip(timestamps, pc1)},
            )
    return curves


def full_cam_name(crop_cam: str) -> str:
    return f"IP{crop_cam}" if crop_cam.startswith("Cam") else crop_cam


def compare_curves(
    crop_curves: Dict[str, Dict[str, Tuple[str, Dict[datetime, float]]]],
    full_curves: Dict[str, Dict[str, Dict[datetime, float]]],
    split_date,
    min_shared: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method_name in crop_curves:
        for unit_id, (crop_cam, crop_curve) in crop_curves[method_name].items():
            matching_full_cam = full_cam_name(crop_cam)
            full_curve = full_curves.get(method_name, {}).get(matching_full_cam)
            if full_curve is None:
                continue
            shared = sorted(set(crop_curve) & set(full_curve))
            before = [timestamp for timestamp in shared if timestamp.date() < split_date]
            after = [timestamp for timestamp in shared if timestamp.date() >= split_date]

            def _corr(timestamps: List[datetime]) -> float:
                return pearson_corr(
                    [crop_curve[timestamp] for timestamp in timestamps],
                    [full_curve[timestamp] for timestamp in timestamps],
                    min_shared=min_shared,
                )

            rows.append(
                {
                    "method": method_name,
                    "cam": crop_cam,
                    "full_cam": matching_full_cam,
                    "unit_id": unit_id,
                    "n_shared": len(shared),
                    "n_before": len(before),
                    "n_after": len(after),
                    "corr_all": _corr(shared),
                    "corr_before": _corr(before),
                    "corr_after": _corr(after),
                }
            )
    return rows


def main():
    args = table_parser(__doc__).parse_args()
    set_global_seed(42)
    full_samples, full = compute_trajectories("LAB-fullimage", args.config, args.models, args.device)
    crop_samples, crop = compute_trajectories("LAB-crop", args.config, args.models, args.device)
    rows = compare_curves(
        build_crop_curves(crop_samples, crop), build_full_curves(full_samples, full), SPLIT_DATE, 3
    )
    units = pd.DataFrame(rows)
    cameras = (
        units.groupby(["method", "cam"], sort=False)[["corr_all", "corr_before", "corr_after"]]
        .mean()
        .reset_index()
    )
    save_table(
        cameras, "TableS5", args.output_dir, ["method"], ["corr_all", "corr_before", "corr_after"], ddof=0
    )
    units.to_csv(args.output_dir / "TableS5-units.csv", index=False)


if __name__ == "__main__":
    main()
