"""
PhytoSpectra-VLM / WheatSpectraChat
Exact port of PCTG (Phenotype-Conditioned Token-Counterfactual Grounder) from Notebook 07D.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from app.config import settings
from app.core.constants import (
    HEIGHT,
    WIDTH,
    SPATIAL_PATCH,
    N_SPATIAL_TOKENS,
    N_SPECTRAL_TOKENS,
    PHENOTYPES,
    PHENOTYPE_TO_INDEX,
    DISCLAIMER_GROUNDING,
    DISCLAIMER_SPECTRAL,
)
from app.inference.phytocore_ensemble import PhytoCoreTokenCounterfactual
from app.inference.normalization import NormalizationManager


class PCTGManager:
    def __init__(
        self,
        pctg_dir: Path = settings.PCTG_DIR,
        norm_manager: Optional[NormalizationManager] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.pctg_dir = pctg_dir
        self.norm_manager = norm_manager or NormalizationManager()
        self.device = device
        self.selected_budgets: Dict[str, float] = {}
        self._load_budgets()

    def _load_budgets(self) -> None:
        budget_path = self.pctg_dir / "pctg_selected_budgets.json"
        if not budget_path.exists():
            raise FileNotFoundError(f"Missing PCTG budget artifact: {budget_path}")
        with open(budget_path, "r", encoding="utf-8") as f:
            self.selected_budgets = json.load(f)

    @torch.inference_mode()
    def compute_pctg_for_image(
        self,
        model: PhytoCoreTokenCounterfactual,
        cube_unit: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Computes PCTG single-token counterfactual attribution maps for all 5 phenotypes.
        Uses reference Fold 0 normalization.
        """
        x, mask_t = self.norm_manager.normalize_cube(cube_unit, mask, fold=0)
        x = x.to(self.device)
        mask_t = mask_t.to(self.device)

        # Baseline forward pass
        encoded = model.encode(x, mask_t)
        base_out = model.predict_from_encoded(encoded, return_attention=True)

        orig_regression = base_out["regression"].cpu()
        orig_cult_logits = base_out["cultivar_logits"].cpu()
        orig_cult_probs = torch.softmax(orig_cult_logits[0], dim=0)
        orig_cult_class = int(orig_cult_probs.argmax())

        # ---------------- Spatial token ablations ----------------
        spatial_masks = torch.ones(
            (N_SPATIAL_TOKENS, N_SPATIAL_TOKENS), dtype=torch.float32, device=self.device
        )
        spatial_masks.fill_diagonal_(0.0)

        # Predict with batching
        spatial_out = self._predict_variants(
            model, encoded, spatial_masks=spatial_masks, batch_size=32
        )
        spatial_changes = self._compute_changes(
            orig_regression, orig_cult_probs, orig_cult_class, spatial_out
        )
        # Shape: [5, 132] (phenotypes x tokens)
        spatial_scores = spatial_changes.T

        # ---------------- Spectral token ablations ----------------
        spectral_masks = torch.ones(
            (N_SPECTRAL_TOKENS, N_SPECTRAL_TOKENS), dtype=torch.float32, device=self.device
        )
        spectral_masks.fill_diagonal_(0.0)

        spectral_out = self._predict_variants(
            model, encoded, spectral_masks=spectral_masks, batch_size=32
        )
        spectral_changes = self._compute_changes(
            orig_regression, orig_cult_probs, orig_cult_class, spectral_out
        )
        # Shape: [5, 50] (phenotypes x tokens)
        spectral_scores = spectral_changes.T

        # Extract per-phenotype evidence
        evidence_by_phenotype: Dict[str, Dict[str, Any]] = {}
        for p_idx, phenotype in enumerate(PHENOTYPES):
            spatial_budget = self.selected_budgets.get(f"{phenotype}|Spatial", 0.1)
            spectral_budget = self.selected_budgets.get(f"{phenotype}|Spectral", 0.3)

            k_spatial = max(1, int(round(spatial_budget * N_SPATIAL_TOKENS)))
            k_spectral = max(1, int(round(spectral_budget * N_SPECTRAL_TOKENS)))

            p_spatial_scores = spatial_scores[p_idx]
            p_spectral_scores = spectral_scores[p_idx]

            topk_spatial_indices = list(np.argsort(-p_spatial_scores)[:k_spatial])
            topk_spectral_indices = list(np.argsort(-p_spectral_scores)[:k_spectral])

            # Build spatial token descriptions with bounding boxes
            spatial_tokens_info = []
            grid_w = WIDTH // SPATIAL_PATCH[1]  # 22
            for t_idx in topk_spatial_indices:
                r = int(t_idx // grid_w)
                c = int(t_idx % grid_w)
                y0 = r * SPATIAL_PATCH[0]
                y1 = min(HEIGHT, (r + 1) * SPATIAL_PATCH[0])
                x0 = c * SPATIAL_PATCH[1]
                x1 = min(WIDTH, (c + 1) * SPATIAL_PATCH[1])

                spatial_tokens_info.append({
                    "token_index": int(t_idx),
                    "grid_row": r,
                    "grid_col": c,
                    "pixel_box": [x0, y0, x1, y1],
                    "score": float(p_spatial_scores[t_idx]),
                })

            # Evaluate counterfactual comprehensiveness and sufficiency
            spatial_comp, spatial_suff = self._evaluate_subsets(
                model, encoded, orig_regression, orig_cult_probs, orig_cult_class,
                p_idx, domain="Spatial", topk_indices=topk_spatial_indices, n_tokens=N_SPATIAL_TOKENS,
            )
            spectral_comp, spectral_suff = self._evaluate_subsets(
                model, encoded, orig_regression, orig_cult_probs, orig_cult_class,
                p_idx, domain="Spectral", topk_indices=topk_spectral_indices, n_tokens=N_SPECTRAL_TOKENS,
            )

            # Verification strength
            spatial_strength = "STRONG" if spatial_comp > 0 and spatial_suff > 0 else "MODERATE"
            spectral_strength = "STRONG" if spectral_comp > 0 and spectral_suff > 0 else "MODERATE"

            evidence_by_phenotype[phenotype] = {
                "spatial": {
                    "selected_tokens": spatial_tokens_info,
                    "selected_token_indices": [int(i) for i in topk_spatial_indices],
                    "budget": spatial_budget,
                    "validation_strength": spatial_strength,
                    "comprehensiveness_gain": spatial_comp,
                    "sufficiency_advantage": spatial_suff,
                },
                "spectral": {
                    "selected_token_indices": [int(i) for i in topk_spectral_indices],
                    "budget": spectral_budget,
                    "validation_strength": spectral_strength,
                    "comprehensiveness_gain": spectral_comp,
                    "sufficiency_advantage": spectral_suff,
                    "disclaimer": DISCLAIMER_SPECTRAL,
                },
            }

        return {
            "spatial_scores": spatial_scores,
            "spectral_scores": spectral_scores,
            "evidence": evidence_by_phenotype,
            "disclaimer": DISCLAIMER_GROUNDING,
        }

    def _predict_variants(
        self,
        model: PhytoCoreTokenCounterfactual,
        encoded: Dict[str, torch.Tensor],
        spatial_masks: Optional[torch.Tensor] = None,
        spectral_masks: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        n = spatial_masks.shape[0] if spatial_masks is not None else spectral_masks.shape[0]
        reg_outs = []
        cult_outs = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bs = end - start

            batch_encoded = {
                "spatial_tokens": encoded["spatial_tokens"].repeat(bs, 1, 1),
                "spectral_tokens": encoded["spectral_tokens"].repeat(bs, 1, 1),
                "spatial_padding_mask": encoded["spatial_padding_mask"].repeat(bs, 1),
            }
            s_mask = spatial_masks[start:end] if spatial_masks is not None else None
            l_mask = spectral_masks[start:end] if spectral_masks is not None else None

            out = model.predict_from_encoded(
                batch_encoded,
                spatial_keep_mask=s_mask,
                spectral_keep_mask=l_mask,
            )
            reg_outs.append(out["regression"].cpu())
            cult_outs.append(out["cultivar_logits"].cpu())

        return {
            "regression": torch.cat(reg_outs, dim=0),
            "cultivar_logits": torch.cat(cult_outs, dim=0),
        }

    def _compute_changes(
        self,
        orig_reg: torch.Tensor,
        orig_cult_probs: torch.Tensor,
        orig_cult_class: int,
        variant_out: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        # [N_variants, 4]
        reg_change = torch.abs(variant_out["regression"] - orig_reg)
        # Cultivar probability change
        variant_cult_probs = torch.softmax(variant_out["cultivar_logits"], dim=1)
        cult_change = torch.abs(
            variant_cult_probs[:, orig_cult_class] - orig_cult_probs[orig_cult_class]
        )[:, None]

        changes = torch.cat([reg_change, cult_change], dim=1).numpy()
        return changes

    def _evaluate_subsets(
        self,
        model: PhytoCoreTokenCounterfactual,
        encoded: Dict[str, torch.Tensor],
        orig_reg: torch.Tensor,
        orig_cult_probs: torch.Tensor,
        orig_cult_class: int,
        phenotype_idx: int,
        domain: str,
        topk_indices: List[int],
        n_tokens: int,
    ) -> Tuple[float, float]:
        """
        Evaluates counterfactual comprehensiveness (change when top-k removed)
        and sufficiency (error when only top-k kept).
        """
        candidate_mask = torch.zeros(n_tokens, dtype=torch.float32, device=self.device)
        candidate_mask[topk_indices] = 1.0

        remove_keep_mask = (1.0 - candidate_mask)[None]
        keep_only_mask = candidate_mask[None]
        combined_masks = torch.cat([remove_keep_mask, keep_only_mask], dim=0)

        if domain == "Spatial":
            out = self._predict_variants(model, encoded, spatial_masks=combined_masks)
        else:
            out = self._predict_variants(model, encoded, spectral_masks=combined_masks)

        changes = self._compute_changes(orig_reg, orig_cult_probs, orig_cult_class, out)
        comprehensiveness = float(changes[0, phenotype_idx])
        sufficiency_error = float(changes[1, phenotype_idx])

        # Sufficiency advantage = baseline drop when kept only is small error
        suff_adv = float(max(0.0, 0.5 - sufficiency_error))
        return comprehensiveness, suff_adv
