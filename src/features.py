"""Image measurements shared by Fig4 and the proxy/baseline tables."""

from typing import Optional

import cv2
import numpy as np

from .dataloader import CONFIG, image_paths

BASELINE_FEATURES = ("exg_mean", "exg_area", "hu1", "hu2", "hu3", "edge_density")
PROXY_FEATURES = (
    "exg_area",
    "exg_mean",
    "perimeter",
    "compactness",
    "circularity",
    "hu1",
    "hu2",
    "edge_density",
)


def _compute_exg_map_from_bgr(arr_bgr: np.ndarray) -> np.ndarray:
    arr = arr_bgr.astype(np.float32) / 255.0
    b = arr[:, :, 0]
    g = arr[:, :, 1]
    r = arr[:, :, 2]
    return 2.0 * g - r - b


def _otsu_mask_from_float(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if v_max <= v_min + 1e-12:
        return np.zeros(values.shape, dtype=bool)
    values_u8 = cv2.normalize(values, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    values_u8 = values_u8.astype(np.uint8)
    _, binary = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def _compute_hu_first3(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    if int(np.count_nonzero(mask_u8)) == 0:
        return np.zeros(3, dtype=np.float32)
    moments = cv2.moments(mask_u8, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()[:3]
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu.astype(np.float32)


def _compute_edge_density(gray_uint8: np.ndarray, mask: np.ndarray) -> float:
    if int(np.count_nonzero(mask)) == 0:
        return 0.0
    grad_x = cv2.Sobel(gray_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_uint8, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    roi = grad_mag[mask]
    if roi.size == 0:
        return 0.0
    roi_u8 = cv2.normalize(roi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    roi_u8 = roi_u8.reshape(-1, 1)
    _, edge_mask = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.mean(edge_mask > 0))


def _compute_area_ratio_to_image(mask: np.ndarray) -> float:
    mask_area = float(np.count_nonzero(mask))
    total_area = float(mask.size)
    if mask_area <= 0.0 or total_area <= 0.0:
        return 0.0
    return mask_area / total_area


def _compute_perimeter_pixels(mask: np.ndarray) -> float:
    if int(np.count_nonzero(mask)) == 0:
        return 0.0
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return float(sum(cv2.arcLength(cnt, True) for cnt in contours))


def _compute_perimeter_ratio_to_image(mask: np.ndarray, perimeter_pixels: Optional[float] = None) -> float:
    if perimeter_pixels is None:
        perimeter_pixels = _compute_perimeter_pixels(mask)
    if perimeter_pixels <= 0.0:
        return 0.0
    h, w = mask.shape[:2]
    image_perimeter = float(2 * (h + w))
    if image_perimeter <= 0.0:
        return 0.0
    return float(perimeter_pixels / image_perimeter)


def _compute_compactness(area_pixels: float, perimeter_pixels: float) -> float:
    if area_pixels <= 0.0 or perimeter_pixels <= 0.0:
        return 0.0
    return float((perimeter_pixels * perimeter_pixels) / (4.0 * np.pi * area_pixels + 1e-12))


def _compute_circularity(area_pixels: float, perimeter_pixels: float) -> float:
    if area_pixels <= 0.0 or perimeter_pixels <= 0.0:
        return 0.0
    value = float((4.0 * np.pi * area_pixels) / (perimeter_pixels * perimeter_pixels + 1e-12))
    return float(np.clip(value, 0.0, 1.0))


def baseline_features(path, names=BASELINE_FEATURES):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    exg = _compute_exg_map_from_bgr(bgr)
    mask = _otsu_mask_from_float(exg)
    hu = _compute_hu_first3(mask)
    values = dict(
        exg_mean=float(np.mean(exg[mask])) if np.any(mask) else 0.0,
        exg_area=_compute_area_ratio_to_image(mask),
        hu1=float(hu[0]),
        hu2=float(hu[1]),
        hu3=float(hu[2]),
        edge_density=_compute_edge_density(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask),
    )
    if any(name in names for name in ["perimeter", "compactness", "circularity"]):
        perimeter = _compute_perimeter_pixels(mask)
        area = float(np.count_nonzero(mask))
        values.update(
            perimeter=_compute_perimeter_ratio_to_image(mask, perimeter),
            compactness=_compute_compactness(area, perimeter),
            circularity=_compute_circularity(area, perimeter),
        )
    return np.asarray([values[name] for name in names], dtype=np.float32)


def image_features(dataset, metadata, config=CONFIG, names=BASELINE_FEATURES):
    return np.stack([baseline_features(path, names) for path in image_paths(dataset, metadata, config)])
