"""
PhytoSpectra-VLM / WheatSpectraChat
PhytoCore architecture, checkpoint loading, and live ensemble inference manager.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import settings
from app.core.constants import (
    N_BANDS,
    HEIGHT,
    WIDTH,
    D_MODEL,
    N_HEADS,
    SPECTRAL_DEPTH,
    SPATIAL_DEPTH,
    DROPOUT,
    SPATIAL_PATCH,
    PHENOTYPES,
    REGRESSION_TARGETS,
    CULTIVAR_NAMES,
    PARAM_COUNTS,
)
from app.inference.normalization import NormalizationManager


# ============================================================
# Exact Scientific Modules (Notebook 04 / 05 / 07D)
# ============================================================

class ScalarCoordinateEmbedding(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords[:, None])


class SpectralEncoder(nn.Module):
    """
    Leaf-level spectral context encoder.
    Input:
        x:    [B, 204, 48, 352]
        mask: [B, 48, 352]
    Output:
        tokens: [B, 50, 128]
    """
    def __init__(
        self,
        n_bands: int = N_BANDS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        depth: int = SPECTRAL_DEPTH,
        dropout: float = DROPOUT,
        kernel_size: int = 8,
        stride: int = 4,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.kernel_size = kernel_size
        self.stride = stride

        self.tokenizer = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        self.n_tokens = (n_bands - kernel_size) // stride + 1
        centers = (
            torch.arange(self.n_tokens) * stride + (kernel_size - 1) / 2
        ) / (n_bands - 1)
        self.register_buffer("token_coordinates", centers.float(), persistent=True)
        self.coordinate_embedding = ScalarCoordinateEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask[:, None, :, :]
        denominator = mask_f.sum(dim=(2, 3)).clamp_min(1.0)
        spectrum = (x * mask_f).sum(dim=(2, 3)) / denominator
        tokens = self.tokenizer(spectrum[:, None, :]).transpose(1, 2)
        pos = self.coordinate_embedding(self.token_coordinates)[None, :, :]
        tokens = tokens + pos
        tokens = self.encoder(tokens)
        return self.norm(tokens)


class SpatialEncoder(nn.Module):
    """
    Spectrally compressed spatial patch encoder.
    Input:
        x:    [B, 204, 48, 352]
        mask: [B, 48, 352]
    Output:
        tokens:       [B, 132, 128]
        padding_mask: [B, 132]
    """
    def __init__(
        self,
        in_bands: int = N_BANDS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        depth: int = SPATIAL_DEPTH,
        dropout: float = DROPOUT,
        patch_size: Tuple[int, int] = SPATIAL_PATCH,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.spectral_compressor = nn.Sequential(
            nn.Conv2d(in_bands, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.patch_embed = nn.Conv2d(
            32,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        grid_h = HEIGHT // patch_size[0]
        grid_w = WIDTH // patch_size[1]
        self.grid_size = (grid_h, grid_w)
        self.n_tokens = grid_h * grid_w

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.spectral_compressor(x)
        tokens = self.patch_embed(z)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embedding

        pooled_mask = F.avg_pool2d(
            mask[:, None, :, :].float(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).flatten(1)
        padding_mask = pooled_mask < 0.05

        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.norm(tokens), padding_mask


class BidirectionalSpectralSpatialFusion(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.s_to_l = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.l_to_s = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_spatial_1 = nn.LayerNorm(d_model)
        self.norm_spectral_1 = nn.LayerNorm(d_model)
        self.ff_spatial = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ff_spectral = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        spectral_tokens: torch.Tensor,
        spatial_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_delta, _ = self.s_to_l(
            query=self.norm_spatial_1(spatial_tokens),
            key=spectral_tokens,
            value=spectral_tokens,
            need_weights=False,
        )
        spatial_tokens = spatial_tokens + spatial_delta

        spectral_delta, _ = self.l_to_s(
            query=self.norm_spectral_1(spectral_tokens),
            key=spatial_tokens,
            value=spatial_tokens,
            key_padding_mask=spatial_padding_mask,
            need_weights=False,
        )
        spectral_tokens = spectral_tokens + spectral_delta

        spatial_tokens = spatial_tokens + self.ff_spatial(spatial_tokens)
        spectral_tokens = spectral_tokens + self.ff_spectral(spectral_tokens)
        return spatial_tokens, spectral_tokens


class PhenotypeQueryBank(nn.Module):
    def __init__(self, n_queries: int = 5, d_model: int = D_MODEL, n_heads: int = N_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.empty(1, n_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        memory_tokens: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory_tokens.shape[0]
        q = self.queries.expand(batch_size, -1, -1)
        delta, attention = self.cross_attention(
            query=self.norm_q(q),
            key=self.norm_memory(memory_tokens),
            value=self.norm_memory(memory_tokens),
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        q = q + delta
        q = q + self.ff(q)
        return q, attention


class PhytoCore(nn.Module):
    """
    Complete PhytoCore scientific architecture (1,998,216 parameters).
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        spectral_depth: int = SPECTRAL_DEPTH,
        spatial_depth: int = SPATIAL_DEPTH,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.spectral_encoder = SpectralEncoder(
            n_bands=N_BANDS,
            d_model=d_model,
            n_heads=n_heads,
            depth=spectral_depth,
            dropout=dropout,
        )
        self.spatial_encoder = SpatialEncoder(
            in_bands=N_BANDS,
            d_model=d_model,
            n_heads=n_heads,
            depth=spatial_depth,
            dropout=dropout,
            patch_size=SPATIAL_PATCH,
        )
        self.fusion = BidirectionalSpectralSpatialFusion(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.pqb = PhenotypeQueryBank(
            n_queries=5,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(4)
        ])
        self.cultivar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(CULTIVAR_NAMES)),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        spectral_tokens = self.spectral_encoder(x, mask)
        spatial_tokens, spatial_padding_mask = self.spatial_encoder(x, mask)
        spatial_tokens, spectral_tokens = self.fusion(
            spatial_tokens, spectral_tokens, spatial_padding_mask
        )

        memory = torch.cat([spatial_tokens, spectral_tokens], dim=1)
        spectral_padding = torch.zeros(
            (x.shape[0], spectral_tokens.shape[1]),
            dtype=torch.bool,
            device=x.device,
        )
        memory_padding_mask = torch.cat([spatial_padding_mask, spectral_padding], dim=1)

        phenotype_tokens, pq_attention = self.pqb(memory, memory_padding_mask)

        regression_outputs = []
        for i, head in enumerate(self.regression_heads):
            regression_outputs.append(head(phenotype_tokens[:, i]))
        regression_outputs = torch.cat(regression_outputs, dim=1)

        cultivar_logits = self.cultivar_head(phenotype_tokens[:, 4])

        output = {
            "regression": regression_outputs,
            "cultivar_logits": cultivar_logits,
            "phenotype_tokens": phenotype_tokens,
        }
        if return_attention:
            output.update({
                "pqb_attention": pq_attention,
                "spatial_padding_mask": spatial_padding_mask,
                "n_spatial_tokens": spatial_tokens.shape[1],
                "n_spectral_tokens": spectral_tokens.shape[1],
            })
        return output


class PhytoCoreTokenCounterfactual(PhytoCore):
    """
    Subclass that provides token-counterfactual encoding and variant predictions for PCTG.
    """
    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        spectral_tokens = self.spectral_encoder(x, mask)
        spatial_tokens, spatial_padding_mask = self.spatial_encoder(x, mask)
        return {
            "spatial_tokens": spatial_tokens,
            "spectral_tokens": spectral_tokens,
            "spatial_padding_mask": spatial_padding_mask,
        }

    def predict_from_encoded(
        self,
        encoded: Dict[str, torch.Tensor],
        spatial_keep_mask: Optional[torch.Tensor] = None,
        spectral_keep_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        spatial_tokens = encoded["spatial_tokens"]
        spectral_tokens = encoded["spectral_tokens"]
        spatial_padding_mask = encoded["spatial_padding_mask"]
        batch_size = spatial_tokens.shape[0]

        if spatial_keep_mask is not None:
            spatial_tokens = spatial_tokens * spatial_keep_mask[:, :, None]

        if spectral_keep_mask is not None:
            spectral_tokens = spectral_tokens * spectral_keep_mask[:, :, None]

        fused_spatial, fused_spectral = self.fusion(
            spatial_tokens, spectral_tokens, spatial_padding_mask
        )

        memory = torch.cat([fused_spatial, fused_spectral], dim=1)
        spectral_padding = torch.zeros(
            (batch_size, fused_spectral.shape[1]),
            dtype=torch.bool,
            device=fused_spectral.device,
        )
        memory_padding_mask = torch.cat([spatial_padding_mask, spectral_padding], dim=1)

        phenotype_tokens, pqb_attention = self.pqb(memory, memory_padding_mask)

        regression = torch.cat(
            [head(phenotype_tokens[:, i]) for i, head in enumerate(self.regression_heads)],
            dim=1,
        )
        cultivar_logits = self.cultivar_head(phenotype_tokens[:, 4])

        output = {
            "regression": regression,
            "cultivar_logits": cultivar_logits,
            "phenotype_tokens": phenotype_tokens,
        }
        if return_attention:
            output["pqb_attention"] = pqb_attention
        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        encoded = self.encode(x, mask)
        output = self.predict_from_encoded(
            encoded,
            return_attention=return_attention,
        )
        output["encoded"] = encoded
        return output


# ============================================================
# Live Ensemble Manager
# ============================================================

class PhytoCoreEnsembleManager:
    def __init__(
        self,
        backbone_dir: Path = settings.BACKBONE_DIR,
        device: torch.device = torch.device("cpu"),
        norm_manager: Optional[NormalizationManager] = None,
        mode: str = settings.SCIENTIFIC_ENSEMBLE_MODE,
    ):
        self.backbone_dir = backbone_dir
        self.device = device
        self.norm_manager = norm_manager or NormalizationManager()
        self.mode = mode
        self.models: List[Tuple[int, int, PhytoCore]] = []  # (seed, fold, model)
        self.reference_model: Optional[PhytoCoreTokenCounterfactual] = None
        self._load_ensemble()

    def _load_ensemble(self) -> None:
        seeds = [2026, 2027, 2028]
        folds = [0, 1, 2, 3, 4]

        for seed in seeds:
            for fold in folds:
                ckpt_name = f"phytocore_seed{seed}_fold{fold}_best.pt"
                ckpt_path = self.backbone_dir / ckpt_name
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing PhytoCore checkpoint: {ckpt_path}")

                model = PhytoCore().to(self.device)
                ckpt = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                # Verify parameter count
                param_count = sum(p.numel() for p in model.parameters())
                if param_count != PARAM_COUNTS["phytocore_single"]:
                    raise ValueError(
                        f"PhytoCore parameter count mismatch: got {param_count}, "
                        f"expected {PARAM_COUNTS['phytocore_single']}"
                    )

                self.models.append((seed, fold, model))

        # Setup reference counterfactual model (Seed 2026, Fold 0)
        ref_ckpt = self.backbone_dir / "phytocore_seed2026_fold0_best.pt"
        ref_model = PhytoCoreTokenCounterfactual().to(self.device)
        ref_ckpt_data = torch.load(ref_ckpt, map_location=self.device)
        ref_model.load_state_dict(ref_ckpt_data["model_state_dict"])
        ref_model.eval()
        self.reference_model = ref_model

    @property
    def total_checkpoints_loaded(self) -> int:
        return len(self.models)

    @torch.inference_mode()
    def predict_live_ensemble(
        self,
        cube_unit: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Runs live ensemble inference across all 15 checkpoints.
        Returns:
            predictions dict with mean, ensemble_std, cultivar voting, and calibration status.
        """
        all_regression_preds: Dict[str, List[float]] = {
            t: [] for t in REGRESSION_TARGETS
        }
        all_cultivar_probs: List[np.ndarray] = []
        all_cultivar_preds: List[int] = []

        # Cache fold normalized inputs so we only normalize each fold once
        fold_inputs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for fold in range(5):
            x, mask_t = self.norm_manager.normalize_cube(cube_unit, mask, fold)
            fold_inputs[fold] = (x.to(self.device), mask_t.to(self.device))

        for seed, fold, model in self.models:
            x, mask_t = fold_inputs[fold]
            out = model(x, mask_t)

            # Regression z-scores -> natural values
            reg_z = out["regression"][0].cpu().numpy()
            destandardized = self.norm_manager.inverse_standardize_batch(reg_z, fold)
            for target, val in destandardized.items():
                all_regression_preds[target].append(val)

            # Cultivar probabilities
            cult_logits = out["cultivar_logits"][0]
            cult_probs = torch.softmax(cult_logits, dim=0).cpu().numpy()
            all_cultivar_probs.append(cult_probs)
            all_cultivar_preds.append(int(np.argmax(cult_probs)))

        # Aggregate regression
        predictions: Dict[str, Any] = {}
        for target in REGRESSION_TARGETS:
            vals = np.array(all_regression_preds[target], dtype=np.float64)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            predictions[target] = {
                "value": mean_val,
                "unit": None,
                "ensemble_std": std_val,
                "interval_low": None,
                "interval_high": None,
                "calibration_status": "NOT_AVAILABLE_FOR_LIVE_MODE",
            }

        # Aggregate cultivar
        mean_cult_probs = np.mean(all_cultivar_probs, axis=0)
        best_cult_idx = int(np.argmax(mean_cult_probs))
        predicted_cultivar = CULTIVAR_NAMES[best_cult_idx]
        vote_conf = float(np.max(mean_cult_probs))
        disagreement = float(np.std(all_cultivar_preds))

        predictions["Cultivar"] = {
            "value": predicted_cultivar,
            "unit": None,
            "ensemble_std": None,
            "interval_low": None,
            "interval_high": None,
            "calibration_status": "NOT_AVAILABLE_FOR_LIVE_MODE",
            "cultivar_probabilities": {
                name: float(mean_cult_probs[i])
                for i, name in enumerate(CULTIVAR_NAMES)
            },
            "vote_confidence": vote_conf,
            "seed_disagreement": disagreement,
        }

        return predictions
