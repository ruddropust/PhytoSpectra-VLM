"""Phenotype-conditioned visual evidence encoding.

This module converts PCTG output for one scientific task into a target-specific
visual representation. It selects a connected spatial focus region, preserves
the complete spatial sensitivity map, and carries the selected spectral tokens
without asserting physical wavelengths.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.constants import (
    N_SPATIAL_TOKENS,
    PHENOTYPE_TO_INDEX,
    SPATIAL_PATCH,
    WIDTH,
)
from app.inference.preprocessing import mask_to_bbox


@dataclass
class TargetVisualEncoding:
    target: str
    evidence_available: bool
    spatial_scores: np.ndarray
    normalized_spatial_map: np.ndarray
    selected_tokens: List[Dict[str, Any]]
    focus_tokens: List[Dict[str, Any]]
    focus_box: Optional[List[int]]
    selected_spectral_indices: List[int]
    spectral_scores: Optional[np.ndarray]


class PhenotypeConditionedVisualEncoder:
    """Encode one requested phenotype into spatial and spectral visual evidence."""

    @staticmethod
    def _valid_box(box: Any, width: int, height: int) -> Optional[List[int]]:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        x0, y0, x1, y1 = [int(value) for value in box]
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))
        return [x0, y0, x1, y1]

    @staticmethod
    def _foreground_fraction(box: List[int], mask: np.ndarray) -> float:
        x0, y0, x1, y1 = box
        patch = mask[y0:y1, x0:x1]
        return float(patch.mean()) if patch.size else 0.0

    def _normalize_tokens(
        self,
        spatial_tokens: List[Dict[str, Any]],
        mask: np.ndarray,
    ) -> List[Dict[str, Any]]:
        height, width = mask.shape
        normalized_tokens: List[Dict[str, Any]] = []
        for token in spatial_tokens:
            if not isinstance(token, dict):
                continue
            box = self._valid_box(token.get("pixel_box"), width, height)
            if not box:
                continue
            normalized = dict(token)
            normalized["pixel_box"] = box
            normalized["grid_row"] = int(token.get("grid_row", box[1] // SPATIAL_PATCH[0]))
            normalized["grid_col"] = int(token.get("grid_col", box[0] // SPATIAL_PATCH[1]))
            normalized["score"] = float(token.get("score", 0.0))
            normalized["foreground_fraction"] = self._foreground_fraction(box, mask)
            normalized_tokens.append(normalized)
        return normalized_tokens

    @staticmethod
    def _strongest_connected_component(
        tokens: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not tokens:
            return []
        foreground_tokens = [token for token in tokens if token["foreground_fraction"] > 0.0]
        if foreground_tokens:
            tokens = foreground_tokens
        by_position = {
            (token["grid_row"], token["grid_col"]): token
            for token in tokens
        }
        remaining = set(by_position)
        components: List[List[Dict[str, Any]]] = []
        while remaining:
            start = remaining.pop()
            stack = [start]
            positions = [start]
            while stack:
                row, col = stack.pop()
                neighbors = {
                    (row + dr, col + dc)
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if dr != 0 or dc != 0
                }
                connected = neighbors & remaining
                remaining -= connected
                stack.extend(connected)
                positions.extend(connected)
            components.append([by_position[position] for position in positions])
        return max(
            components,
            key=lambda component: (
                max(token["score"] for token in component),
                sum(max(0.0, token["score"]) for token in component),
                sum(token["foreground_fraction"] for token in component),
            ),
        )

    @staticmethod
    def _union_box(
        boxes: List[List[int]],
        width: int,
        height: int,
        padding: int = 6,
    ) -> Optional[List[int]]:
        if not boxes:
            return None
        return [
            max(0, min(box[0] for box in boxes) - padding),
            max(0, min(box[1] for box in boxes) - padding),
            min(width, max(box[2] for box in boxes) + padding),
            min(height, max(box[3] for box in boxes) + padding),
        ]

    @staticmethod
    def _normalize_spatial_scores(scores: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)
        low = float(clean.min())
        high = float(clean.max())
        if high <= low:
            normalized = np.zeros_like(clean)
        else:
            normalized = (clean - low) / (high - low)
        grid_width = WIDTH // SPATIAL_PATCH[1]
        return normalized.reshape(N_SPATIAL_TOKENS // grid_width, grid_width)

    def encode(
        self,
        pctg_result: Dict[str, Any],
        target: str,
        mask: np.ndarray,
    ) -> TargetVisualEncoding:
        target_index = PHENOTYPE_TO_INDEX.get(target, 0)
        evidence = pctg_result.get("evidence", {}).get(target, {})
        spatial = evidence.get("spatial", {}) if isinstance(evidence, dict) else {}
        spectral = evidence.get("spectral", {}) if isinstance(evidence, dict) else {}
        selected_tokens = self._normalize_tokens(spatial.get("selected_tokens", []), mask)
        focus_tokens = self._strongest_connected_component(selected_tokens)

        focus_box = self._union_box(
            [token["pixel_box"] for token in focus_tokens],
            mask.shape[1],
            mask.shape[0],
        )
        if not focus_box:
            leaf_box = mask_to_bbox(mask)
            if leaf_box:
                focus_box = self._union_box(
                    [list(leaf_box)],
                    mask.shape[1],
                    mask.shape[0],
                )

        all_spatial_scores = np.asarray(pctg_result.get("spatial_scores", []))
        spatial_available = (
            all_spatial_scores.ndim == 2
            and target_index < all_spatial_scores.shape[0]
            and all_spatial_scores.shape[1] == N_SPATIAL_TOKENS
        )
        spatial_scores = (
            np.asarray(all_spatial_scores[target_index], dtype=np.float32)
            if spatial_available
            else np.zeros(N_SPATIAL_TOKENS, dtype=np.float32)
        )

        all_spectral_scores = np.asarray(pctg_result.get("spectral_scores", []))
        spectral_scores = (
            np.asarray(all_spectral_scores[target_index], dtype=np.float32)
            if all_spectral_scores.ndim == 2 and target_index < all_spectral_scores.shape[0]
            else None
        )
        selected_spectral_indices = [
            int(index)
            for index in spectral.get("selected_token_indices", [])
            if isinstance(index, (int, np.integer))
        ]

        return TargetVisualEncoding(
            target=target,
            evidence_available=bool(selected_tokens and spatial_available),
            spatial_scores=spatial_scores,
            normalized_spatial_map=self._normalize_spatial_scores(spatial_scores),
            selected_tokens=selected_tokens,
            focus_tokens=focus_tokens,
            focus_box=focus_box,
            selected_spectral_indices=selected_spectral_indices,
            spectral_scores=spectral_scores,
        )
