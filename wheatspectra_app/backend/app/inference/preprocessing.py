"""
PhytoSpectra-VLM / WheatSpectraChat
Exact port of the corrected leaf-mask preprocessing algorithm from Notebook 01B.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import ndimage
from PIL import Image

from app.core.constants import (
    EXPECTED_RAW_SHAPE,
    EXPECTED_HWB_SHAPE,
    MASK_SUPPORT_THRESHOLD,
    MIN_COMPONENT_SIZE,
    FILL_HOLES,
    UINT16_MAX,
)


def validate_and_canonicalize_cube(cube: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Validates input HSI cube shape and converts to canonical (48, 352, 204) float32 in [0, 1].

    Accepts:
        - Raw TIFF: (204, 48, 352) [Bands x Height x Width]
        - Canonical: (48, 352, 204) [Height x Width x Bands]

    Raises:
        ValueError if the shape is unsupported.
    """
    orig_shape = cube.shape

    if cube.ndim != 3:
        raise ValueError(
            f"Unsupported HSI dimension {cube.ndim}. Expected 3D cube with "
            f"shape {EXPECTED_RAW_SHAPE} (Bands, H, W) or {EXPECTED_HWB_SHAPE} (H, W, Bands)."
        )

    if orig_shape == EXPECTED_RAW_SHAPE:
        # Move bands from axis 0 to axis -1: (204, 48, 352) -> (48, 352, 204)
        cube = np.moveaxis(cube, 0, -1)
    elif orig_shape == EXPECTED_HWB_SHAPE:
        # Already canonical
        pass
    else:
        raise ValueError(
            f"Unsupported HSI shape {orig_shape}. Expected {EXPECTED_RAW_SHAPE} "
            f"(204 bands, 48 height, 352 width) or {EXPECTED_HWB_SHAPE} (48, 352, 204)."
        )

    # Convert uint16 sensor values to unit float32 in [0, 1]
    if cube.dtype == np.uint16 or np.issubdtype(cube.dtype, np.integer):
        cube_unit = cube.astype(np.float32) / UINT16_MAX
    else:
        cube_unit = cube.astype(np.float32)
        if cube_unit.max() > 1.0:
            cube_unit /= UINT16_MAX

    return cube_unit, orig_shape


def spectral_support_map(cube: np.ndarray) -> np.ndarray:
    """
    Fraction of non-zero spectral bands at each spatial pixel.
    cube: (48, 352, 204)
    """
    return np.mean(cube > 0, axis=-1)


def remove_small_components(mask: np.ndarray, min_size: int = MIN_COMPONENT_SIZE) -> Tuple[np.ndarray, int, int]:
    """
    Keep every connected foreground component with at least min_size pixels.
    IMPORTANT: This intentionally does NOT keep only the single largest component.
    """
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool), 0, 0

    counts = np.bincount(labels.ravel())
    keep_labels = [
        label_id
        for label_id in range(1, len(counts))
        if counts[label_id] >= min_size
    ]

    cleaned = np.isin(labels, keep_labels)
    return cleaned.astype(bool), n, len(keep_labels)


def fill_component_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes while preserving real gaps between disconnected components.
    """
    labels, n = ndimage.label(mask)
    output = np.zeros_like(mask, dtype=bool)

    for label_id in range(1, n + 1):
        component = labels == label_id
        output |= ndimage.binary_fill_holes(component)

    return output


def mask_to_bbox(mask: np.ndarray, padding: int = 2) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding box around ALL retained foreground components.
    Returns (x1, y1, x2, y2) with exclusive x2/y2.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + 1 + padding, mask.shape[1])
    y2 = min(int(ys.max()) + 1 + padding, mask.shape[0])

    return x1, y1, x2, y2


def saturation_statistics(cube: np.ndarray, leaf_mask: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Computes saturation metrics on unit-scale cube [0, 1].
    """
    pixels = cube[leaf_mask]
    if pixels.size == 0:
        return {
            "sat_fraction": None,
            "near_sat_fraction": None,
        }

    return {
        "sat_fraction": float((pixels >= 1.0).mean()),
        "near_sat_fraction": float((pixels >= 0.99).mean()),
    }


def create_leaf_mask(
    cube: np.ndarray,
    support_threshold: float = MASK_SUPPORT_THRESHOLD,
    min_component_size: int = MIN_COMPONENT_SIZE,
    fill_holes: bool = FILL_HOLES,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes corrected leaf mask on the fly from canonical unit HSI cube.
    """
    support = spectral_support_map(cube)
    raw_mask = support >= support_threshold

    cleaned, raw_n_components, kept_n_components = remove_small_components(
        raw_mask,
        min_size=min_component_size,
    )

    if fill_holes:
        cleaned = fill_component_holes(cleaned)

    stats = {
        "raw_foreground_pixels": int(raw_mask.sum()),
        "final_foreground_pixels": int(cleaned.sum()),
        "raw_components": int(raw_n_components),
        "kept_components": int(kept_n_components),
    }

    if raw_mask.sum() > 0:
        stats["retained_foreground_ratio"] = float(cleaned.sum() / raw_mask.sum())
    else:
        stats["retained_foreground_ratio"] = None

    stats["foreground_fraction"] = float(cleaned.mean())

    bbox = mask_to_bbox(cleaned)
    stats["bbox"] = list(bbox) if bbox else None

    sat = saturation_statistics(cube, cleaned)
    stats.update(sat)

    return cleaned.astype(bool), support, stats


def robust_scale_channel(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Robust 1st-99th percentile scaling on foreground pixels.
    """
    values = channel[mask]
    if values.size == 0:
        return np.zeros_like(channel, dtype=np.float32)

    low = float(np.percentile(values, 1))
    high = float(np.percentile(values, 99))

    if high <= low:
        scaled = np.zeros_like(channel, dtype=np.float32)
    else:
        scaled = np.clip((channel - low) / (high - low), 0.0, 1.0)

    scaled[~mask] = 0.0
    return scaled


def create_false_color_image(
    cube: np.ndarray,
    mask: np.ndarray,
    bands: Tuple[int, int, int] = (40, 100, 160),
) -> Image.Image:
    """
    Creates HSI false-color composite from fixed bands (40, 100, 160).
    Labelled as 'HSI false-color composite', NOT RGB.
    """
    channels = [
        robust_scale_channel(cube[:, :, band_index], mask)
        for band_index in bands
    ]
    array = np.stack(channels, axis=-1)
    uint8_img = (array * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_img)
