"""
PhytoSpectra-VLM / WheatSpectraChat
Normalization and target de-normalization artifact management.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

from app.config import settings
from app.core.constants import N_BANDS, REGRESSION_TARGETS


class NormalizationManager:
    def __init__(self, norm_dir: Path = settings.NORMALIZATION_DIR):
        self.norm_dir = norm_dir
        self.fold_band_stats: Dict[int, Dict[str, np.ndarray]] = {}
        self.fold_target_stats: Dict[int, Dict[str, Dict[str, float]]] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        for fold in range(5):
            npz_path = self.norm_dir / f"fold_{fold}_band_stats.npz"
            json_path = self.norm_dir / f"fold_{fold}_target_stats.json"

            if not npz_path.exists():
                raise FileNotFoundError(f"Missing required normalization artifact: {npz_path}")
            if not json_path.exists():
                raise FileNotFoundError(f"Missing required target stats artifact: {json_path}")

            data = np.load(npz_path)
            self.fold_band_stats[fold] = {
                "mean": data["mean"].astype(np.float32),
                "std": data["std"].astype(np.float32),
            }

            with open(json_path, "r", encoding="utf-8") as f:
                self.fold_target_stats[fold] = json.load(f)

    def normalize_cube(
        self,
        cube_unit: np.ndarray,
        mask: np.ndarray,
        fold: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes canonical unit cube (48, 352, 204) with fold-specific band statistics.
        Returns:
            x: torch.Tensor [1, 204, 48, 352]
            mask_tensor: torch.Tensor [1, 48, 352]
        """
        stats = self.fold_band_stats[fold]
        mean = stats["mean"]
        std = stats["std"]

        normalized = np.zeros_like(cube_unit, dtype=np.float32)
        normalized[mask] = (cube_unit[mask] - mean[None, :]) / std[None, :]

        # Shape: (204, 48, 352)
        chw = np.moveaxis(normalized, -1, 0)
        x = torch.from_numpy(chw).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return x, mask_tensor

    def inverse_standardize_target(
        self,
        value_z: float,
        target_name: str,
        fold: int,
    ) -> float:
        """
        De-standardizes a single regression z-score into natural scientific value.
        """
        stats = self.fold_target_stats[fold][target_name]
        return float(value_z * stats["std"] + stats["mean"])

    def inverse_standardize_batch(
        self,
        z_scores: np.ndarray,
        fold: int,
    ) -> Dict[str, float]:
        """
        z_scores: array of shape (4,) corresponding to REGRESSION_TARGETS.
        """
        results = {}
        for i, target in enumerate(REGRESSION_TARGETS):
            results[target] = self.inverse_standardize_target(
                float(z_scores[i]), target, fold
            )
        return results
