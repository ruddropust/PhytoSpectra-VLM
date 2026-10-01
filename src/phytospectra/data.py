"""HyperLeaf2024 loading, deterministic masks, normalization, and datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage
from torch.utils.data import Dataset

TARGETS = ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar")
REGRESSION_TARGETS = TARGETS[:3]
FERTILIZER_CLASSES = (0.0, 0.5, 1.0)
CULTIVAR_CLASSES = ("Heerup", "Kvium", "Rembrandt", "Sheriff")
IMAGE_SHAPE = (48, 352, 204)


@dataclass(frozen=True)
class PreprocessSpec:
    support_threshold: float = 0.75
    min_component_size: int = 20
    fill_holes: bool = True
    scale_divisor: float = 65535.0


STAGE42D_AUGMENTATION = {
    "translation_probability": 0.5,
    "max_dx": 8,
    "max_dy": 2,
    "gain_probability": 0.5,
    "gain_low": 0.98,
    "gain_high": 1.02,
    "high_saturation_threshold": 0.05,
    "noise_probability": 0.5,
    "noise_sigma_max": 0.002,
    "smooth_spectral_jitter_probability": 0.35,
    "smooth_spectral_jitter_amp_max": 0.01,
    "smooth_spectral_jitter_sigma_bands": 8.0,
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _zero_fill_shift(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    output = np.zeros_like(array)
    height, width = array.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx) if dx >= 0 else width
    sy0, sy1 = max(0, -dy), min(height, height - dy) if dy >= 0 else height
    dx0, dy0 = max(0, dx), max(0, dy)
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    if sx1 > sx0 and sy1 > sy0:
        output[dy0:dy1, dx0:dx1, ...] = array[sy0:sy1, sx0:sx1, ...]
    return output


def apply_stage42d_train_augmentation(
    cube: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    settings: Mapping[str, Any] = STAGE42D_AUGMENTATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact train-only augmentation sequence frozen by historical Stage-42D."""
    output, output_mask = cube.copy(), mask.copy()
    if rng.random() < float(settings["gain_probability"]):
        foreground = output[output_mask]
        saturation = float((foreground >= 1.0).mean()) if foreground.size else 0.0
        high = 1.0 if saturation > float(settings["high_saturation_threshold"]) else float(settings["gain_high"])
        output[output_mask] *= float(rng.uniform(float(settings["gain_low"]), high))
        output = np.clip(output, 0.0, 1.0)
    if rng.random() < float(settings["noise_probability"]):
        sigma = float(rng.uniform(0.0, float(settings["noise_sigma_max"])))
        if sigma > 0 and output_mask.any():
            noise = rng.normal(0.0, sigma, size=output.shape).astype(np.float32)
            output[output_mask] += noise[output_mask]
            output = np.clip(output, 0.0, 1.0)
    if rng.random() < float(settings["smooth_spectral_jitter_probability"]):
        raw = rng.normal(0.0, 1.0, output.shape[-1]).astype(np.float32)
        curve = ndimage.gaussian_filter1d(raw, sigma=float(settings["smooth_spectral_jitter_sigma_bands"]), mode="reflect")
        curve -= curve.mean()
        if curve.std() > 1e-8:
            curve /= curve.std()
        amplitude = float(rng.uniform(0.0, float(settings["smooth_spectral_jitter_amp_max"])))
        factors = np.clip(1.0 + amplitude * curve, 0.95, 1.05).astype(np.float32)
        if output_mask.any():
            output[output_mask] *= factors[None]
        output = np.clip(output, 0.0, 1.0)
    if rng.random() < float(settings["translation_probability"]):
        box = _bbox(output_mask)
        if box is not None:
            x0, y0, x1, y1 = box
            max_dx, max_dy = int(settings["max_dx"]), int(settings["max_dy"])
            left, right = min(max_dx, x0), min(max_dx, output.shape[1] - x1)
            up, down = min(max_dy, y0), min(max_dy, output.shape[0] - y1)
            dx, dy = int(rng.integers(-left, right + 1)), int(rng.integers(-up, down + 1))
            output = _zero_fill_shift(output, dx, dy)
            output_mask = _zero_fill_shift(output_mask.astype(np.uint8), dx, dy).astype(bool)
    output[~output_mask] = 0.0
    return output.astype(np.float32, copy=False), output_mask


def load_metadata(raw_data: str | Path) -> pd.DataFrame:
    path = Path(raw_data) / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing HyperLeaf metadata: {path}")
    frame = pd.read_csv(path, dtype={"ImageId": str})
    required = {"ImageId", *REGRESSION_TARGETS, "Fertilizer", *CULTIVAR_CLASSES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame["ImageId"] = frame["ImageId"].str.zfill(5)
    if frame["ImageId"].duplicated().any():
        raise ValueError("ImageId values must be unique")
    one_hot = frame.loc[:, CULTIVAR_CLASSES].apply(pd.to_numeric, errors="coerce")
    valid = one_hot.isin((0, 1)).all(axis=1) & one_hot.sum(axis=1).eq(1)
    if not valid.all():
        raise ValueError("Cultivar columns are not valid one-hot labels")
    frame["Cultivar"] = one_hot.idxmax(axis=1)
    for target in (*REGRESSION_TARGETS, "Fertilizer"):
        frame[target] = pd.to_numeric(frame[target], errors="raise")
    if len(frame) != 1590:
        raise ValueError(f"Expected 1590 training images, observed {len(frame)}")
    frame["ImagePath"] = frame["ImageId"].map(lambda value: str(Path(raw_data) / "images" / f"{value}.tiff"))
    missing_images = [path for path in frame["ImagePath"] if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} TIFF files; first={missing_images[0]}")
    return frame


def load_cube(path: str | Path) -> np.ndarray:
    raw = tifffile.imread(path)
    if raw.dtype != np.uint16:
        raise TypeError(f"Expected uint16 HSI, observed {raw.dtype}")
    if raw.shape == (204, 48, 352):
        raw = np.moveaxis(raw, 0, -1)
    if raw.shape != IMAGE_SHAPE:
        raise ValueError(f"Expected HSI {IMAGE_SHAPE}, observed {raw.shape}")
    cube = np.ascontiguousarray(raw.astype(np.float32) / np.float32(65535.0))
    if not np.isfinite(cube).all() or cube.min() < 0 or cube.max() > 1:
        raise ValueError("Invalid unit-scaled HSI values")
    return cube


def create_leaf_mask(cube: np.ndarray, spec: PreprocessSpec = PreprocessSpec()) -> np.ndarray:
    support = np.mean(cube > 0, axis=-1) >= spec.support_threshold
    labels, count = ndimage.label(support)
    sizes = np.bincount(labels.ravel()) if count else np.zeros(1, dtype=int)
    keep = np.flatnonzero(sizes >= spec.min_component_size)
    keep = keep[keep != 0]
    mask = np.isin(labels, keep)
    if spec.fill_holes:
        result = np.zeros_like(mask, dtype=bool)
        labels, count = ndimage.label(mask)
        for index in range(1, count + 1):
            result |= ndimage.binary_fill_holes(labels == index)
        mask = result
    if not mask.any():
        raise ValueError("Leaf mask is empty")
    return mask.astype(bool)


def prepare_images(frame: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, row in frame.iterrows():
        cube = load_cube(row["ImagePath"])
        mask = create_leaf_mask(cube)
        mask_path = mask_dir / f"{row['ImageId']}.npy"
        np.save(mask_path, mask.astype(np.uint8), allow_pickle=False)
        rows.append({
            **row.to_dict(),
            "MaskPath": str(mask_path.resolve()),
            "ForegroundPixels": int(mask.sum()),
            "ForegroundFraction": float(mask.mean()),
        })
        if (position + 1) % 100 == 0:
            print(f"prepare {position + 1}/{len(frame)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "dataset_index.csv", index=False)
    return result


def fit_normalization(frame: pd.DataFrame, fold: int, output_dir: str | Path) -> dict[str, Any]:
    train = frame.loc[frame["Fold"] != int(fold)]
    total = np.zeros(204, dtype=np.float64)
    total_sq = np.zeros(204, dtype=np.float64)
    count = 0
    for _, row in train.iterrows():
        cube = load_cube(row["ImagePath"])
        mask = np.load(row["MaskPath"], allow_pickle=False).astype(bool)
        values = cube[mask].astype(np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    target_stats = {
        target: {"mean": float(train[target].mean()), "std": float(train[target].std(ddof=0))}
        for target in REGRESSION_TARGETS
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"fold_{fold}_band_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32), foreground_pixel_count=np.int64(count), n_images=np.int64(len(train)))
    from .io import write_json
    write_json(output_dir / f"fold_{fold}_target_stats.json", target_stats)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "targets": target_stats}


def load_normalization(directory: str | Path, fold: int) -> dict[str, Any]:
    directory = Path(directory)
    with np.load(directory / f"fold_{fold}_band_stats.npz") as payload:
        mean, std = payload["mean"], payload["std"]
    from .io import read_json
    return {"mean": mean, "std": std, "targets": read_json(directory / f"fold_{fold}_target_stats.json")}


class HyperLeafDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        normalization: Mapping[str, Any],
        *,
        augment: bool = False,
        base_seed: int = 2026,
    ):
        self.frame = frame.sort_values("ImageId").reset_index(drop=True)
        self.normalization = normalization
        self.augment = bool(augment)
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        cube = load_cube(row["ImagePath"])
        mask = np.load(row["MaskPath"], allow_pickle=False).astype(bool)
        cube[~mask] = 0.0
        if self.augment:
            image_id = int(str(row["ImageId"]))
            rng = np.random.default_rng(
                self.base_seed + self.epoch * 1_000_003 + image_id * 1009 + index * 17
            )
            cube, mask = apply_stage42d_train_augmentation(cube, mask, rng)
        normalized = np.zeros_like(cube, dtype=np.float32)
        normalized[mask] = (cube[mask] - self.normalization["mean"]) / self.normalization["std"]
        regression = np.asarray([
            (float(row[target]) - self.normalization["targets"][target]["mean"]) /
            self.normalization["targets"][target]["std"]
            for target in REGRESSION_TARGETS
        ], dtype=np.float32)
        return {
            "image_id": str(row["ImageId"]),
            "hsi": torch.from_numpy(np.moveaxis(normalized, -1, 0).copy()),
            "leaf_mask": torch.from_numpy(mask.copy()),
            "regression_targets": torch.from_numpy(regression),
            "regression_targets_raw": torch.tensor([float(row[t]) for t in REGRESSION_TARGETS]),
            "fertilizer_label": torch.tensor(FERTILIZER_CLASSES.index(float(row["Fertilizer"])), dtype=torch.long),
            "cultivar_label": torch.tensor(CULTIVAR_CLASSES.index(str(row["Cultivar"])), dtype=torch.long),
        }
