"""PhytoSpectra-VLM: the current complete model in one readable Python file.

This is the authoritative Weighted-CE PhytoCore and PhytoSpectra-VLM
architecture for the clean research_v3 implementation. State-dict names of the
neural layers are preserved; no legacy package is imported.

Sections
--------
1. PhytoCore: spectral / spatial encoders, fusion, phenotype queries and heads.
2. Global-local visual bridge: question adapter, target conditioning, resampler.
3. PhytoSpectraVLM: continuous visual prefix -> pretrained Qwen + LoRA.
4. Target counterfactual grounding, ROI selection, all-band crop / resize.
5. Loss, EMA, assistant-only VQA collation.
6. Optional deterministic evidence renderer (mark, heatmap, crop, 4x zoom).
7. Normalization, input preparation, trusted checkpoint / EMA bundle loading.
8. CPU-only architecture summary and smoke test.

Quick start (PowerShell)
-----------------------
python -m phytospectra.model
python -m phytospectra.model --smoke-test

From Python / Jupyter:
    from phytospectra.model import load_trained_model, prepare_grounded_visual_inputs
    model, tokenizer, manifest = load_trained_model(
        "E:/Habib/PhytoChat/artifacts/30_idcv_phytospectra_vlm", device="cpu")
    # This explicit call loads the real pretrained language model; the CLI does not.

Dependencies: torch, numpy. Real Qwen loading additionally uses transformers,
peft and accelerate. Optional visualization uses Pillow and matplotlib.
The pretrained Qwen implementation/weights remain external library/model assets;
billions of weight values are not embedded in a Python source file.

Input contract
--------------
global_hsi: float32 [B,204,48,352], TRAINING-stat normalized, background zero.
global_leaf_mask: bool [B,48,352].
local_hsi: float32 [B,2,204,48,352], crops of UNIT-SCALED HSI, THEN normalized.
local_leaf_masks: bool [B,2,48,352]; local_view_mask: bool [B,2].
local_target_indices: int64 [B,2]; primary_target_index: int64 [B].
Targets: GrainWeight=0, Gsw=1, PhiPS2=2, Fertilizer=3, Cultivar=4, GENERAL=5.
input_ids / attention_mask / labels: [B,T]; prompt and padding labels = -100.
question_input_ids / question_attention_mask: [B,Q], QUESTION ONLY, no answer.
forward() returns loss, logits, structured predictions, 16 visual-prefix tokens.
generate() returns raw generated token IDs; use tokenizer.batch_decode(...).

Regression outputs are standardized (not physical measurements). Inverse-transform
with the matching training-fold statistics. Fertilizer is a 3-class classifier;
Cultivar is a 4-class classifier. Class order must come from the training assets.
Evidence maps are model sensitivity, not biological causality or wavelength nm.
The optional renderer is deterministic; Qwen does NOT generate image pixels.
Raw generated text is NOT automatically verified. The chatbot verifier remains
separate and is deliberately not relabeled as part of the learned VLM.
Import/default CLI never loads checkpoints, downloads Qwen, or selects a GPU.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONSOLIDATED_FROM = (
    "models/phytocore.py", "models/visual_token_bridge.py",
    "models/phytospectra_vlm.py", "models/roi_cropper.py",
    "evaluation/counterfactual_evidence.py", "evaluation/visual_evidence.py",
    "training/losses.py", "training/trainable_ema.py", "qa/collator.py",
)


# ============================================================================
# 1. PHYTOCORE: trainable perception, frozen during language alignment
# ============================================================================

@dataclass(frozen=True)
class PhytoCoreConfig:
    """Validated architecture values required to construct PhytoCore."""

    input_bands: int = 204
    image_height: int = 48
    image_width: int = 352
    d_model: int = 128
    attention_heads: int = 4
    spectral_depth: int = 2
    spatial_depth: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.1
    spectral_kernel_size: int = 8
    spectral_stride: int = 4
    compressed_channels: int = 32
    patch_size: tuple[int, int] = (8, 16)
    minimum_foreground_fraction: float = 0.05
    phenotype_queries: tuple[str, ...] = (
        "GrainWeight",
        "Gsw",
        "PhiPS2",
        "Fertilizer",
        "Cultivar",
    )
    regression_targets: tuple[str, ...] = ("GrainWeight", "Gsw", "PhiPS2")
    fertilizer_classes: int = 3
    cultivar_classes: int = 4
    expected_parameter_count: int | None = 1_998_346

    @classmethod
    def from_configs(
        cls,
        model_config: Mapping[str, Any],
        data_config: Mapping[str, Any] | None = None,
    ) -> "PhytoCoreConfig":
        """Create a typed configuration from the YAML mappings."""

        model = model_config.get("model", model_config)
        spectral = model["spectral_tokenizer"]
        spatial = model["spatial_tokenizer"]
        heads = model["heads"]
        if data_config is None:
            height, width = 48, 352
        else:
            dataset = data_config.get("dataset", data_config)
            height, width, _bands = (
                int(value) for value in dataset["image_shape_hwc"]
            )
            if _bands != int(model["input_bands"]):
                raise ValueError("Data and model configurations disagree on band count")
        return cls(
            input_bands=int(model["input_bands"]),
            image_height=height,
            image_width=width,
            d_model=int(model["d_model"]),
            attention_heads=int(model["attention_heads"]),
            spectral_depth=int(model["spectral_depth"]),
            spatial_depth=int(model["spatial_depth"]),
            feedforward_dim=int(model["feedforward_dim"]),
            dropout=float(model["dropout"]),
            spectral_kernel_size=int(spectral["kernel_size"]),
            spectral_stride=int(spectral["stride"]),
            compressed_channels=int(spatial["compressed_channels"]),
            patch_size=tuple(int(value) for value in spatial["patch_size"]),
            minimum_foreground_fraction=float(
                spatial["minimum_foreground_fraction"]
            ),
            phenotype_queries=tuple(str(value) for value in model["phenotype_queries"]),
            regression_targets=tuple(str(value) for value in heads["regression"]),
            fertilizer_classes=int(heads["fertilizer_classes"]),
            cultivar_classes=int(heads["cultivar_classes"]),
            expected_parameter_count=(
                int(model["expected_parameter_count"])
                if model.get("expected_parameter_count") is not None
                else None
            ),
        )

    def validate(self) -> None:
        if self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.compressed_channels % 8 != 0:
            raise ValueError("compressed_channels must be divisible by 8")
        if self.image_height % self.patch_size[0] != 0:
            raise ValueError("Image height must be divisible by spatial patch height")
        if self.image_width % self.patch_size[1] != 0:
            raise ValueError("Image width must be divisible by spatial patch width")
        if len(self.phenotype_queries) != 5:
            raise ValueError("PhytoCore requires exactly five phenotype queries")
        expected_queries = (*self.regression_targets, "Fertilizer", "Cultivar")
        if self.phenotype_queries != expected_queries:
            raise ValueError(
                "Phenotype query order must be regression targets, Fertilizer, Cultivar"
            )
        if self.fertilizer_classes < 2 or self.cultivar_classes < 2:
            raise ValueError("Classification heads require at least two classes")
        if not 0.0 <= self.minimum_foreground_fraction <= 1.0:
            raise ValueError("minimum_foreground_fraction must lie in [0, 1]")


class ScalarCoordinateEmbedding(nn.Module):
    """Map a normalized scalar spectral coordinate to model dimension."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.net(coordinates[:, None])


def _transformer_encoder(
    d_model: int,
    heads: int,
    feedforward_dim: int,
    dropout: float,
    depth: int,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=depth,
        enable_nested_tensor=False,
    )


class SpectralEncoder(nn.Module):
    """Encode the foreground-mean leaf spectrum into spectral tokens."""

    def __init__(self, config: PhytoCoreConfig) -> None:
        super().__init__()
        self.input_bands = config.input_bands
        self.tokenizer = nn.Conv1d(
            1,
            config.d_model,
            kernel_size=config.spectral_kernel_size,
            stride=config.spectral_stride,
        )
        self.n_tokens = (
            (config.input_bands - config.spectral_kernel_size)
            // config.spectral_stride
            + 1
        )
        centers = (
            torch.arange(self.n_tokens) * config.spectral_stride
            + (config.spectral_kernel_size - 1) / 2
        ) / (config.input_bands - 1)
        self.register_buffer("token_coordinates", centers.float(), persistent=True)
        self.coordinate_embedding = ScalarCoordinateEmbedding(config.d_model)
        self.encoder = _transformer_encoder(
            config.d_model,
            config.attention_heads,
            config.feedforward_dim,
            config.dropout,
            config.spectral_depth,
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, hsi: torch.Tensor, leaf_mask: torch.Tensor) -> torch.Tensor:
        mask = leaf_mask[:, None].to(dtype=hsi.dtype)
        denominator = mask.sum(dim=(2, 3)).clamp_min(1.0)
        spectrum = (hsi * mask).sum(dim=(2, 3)) / denominator
        tokens = self.tokenizer(spectrum[:, None]).transpose(1, 2)
        coordinates = self.coordinate_embedding(self.token_coordinates)[None]
        return self.norm(self.encoder(tokens + coordinates))


class SpatialEncoder(nn.Module):
    """Compress bands and tokenize the 2-D leaf support into spatial patches."""

    def __init__(self, config: PhytoCoreConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.minimum_foreground_fraction = config.minimum_foreground_fraction
        self.grid_size = (
            config.image_height // config.patch_size[0],
            config.image_width // config.patch_size[1],
        )
        self.n_tokens = self.grid_size[0] * self.grid_size[1]
        self.spectral_compressor = nn.Sequential(
            nn.Conv2d(
                config.input_bands,
                config.compressed_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, config.compressed_channels),
            nn.GELU(),
        )
        self.patch_embed = nn.Conv2d(
            config.compressed_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.n_tokens, config.d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.encoder = _transformer_encoder(
            config.d_model,
            config.attention_heads,
            config.feedforward_dim,
            config.dropout,
            config.spatial_depth,
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hsi: torch.Tensor,
        leaf_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed = self.spectral_compressor(hsi)
        tokens = self.patch_embed(compressed).flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.n_tokens:
            raise ValueError(
                f"Spatial token count changed: expected {self.n_tokens}, "
                f"observed {tokens.shape[1]}"
            )
        pooled_mask = F.avg_pool2d(
            leaf_mask[:, None].float(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).flatten(1)
        padding_mask = pooled_mask < self.minimum_foreground_fraction

        # PyTorch attention cannot safely receive a sample where every key is
        # padded.  A very thin valid leaf can occupy less than 5% of all its
        # patches, so retain its strongest-support patch deterministically.
        all_padded = padding_mask.all(dim=1)
        if all_padded.any():
            strongest = pooled_mask.argmax(dim=1)
            padding_mask = padding_mask.clone()
            padding_mask[all_padded, strongest[all_padded]] = False

        encoded = self.encoder(
            tokens + self.pos_embedding,
            src_key_padding_mask=padding_mask,
        )
        return self.norm(encoded), padding_mask


class BidirectionalSpectralSpatialFusion(nn.Module):
    """Exchange information between spatial and spectral token sequences."""

    def __init__(self, config: PhytoCoreConfig) -> None:
        super().__init__()
        attention = {
            "embed_dim": config.d_model,
            "num_heads": config.attention_heads,
            "dropout": config.dropout,
            "batch_first": True,
        }
        self.spatial_queries_spectral = nn.MultiheadAttention(**attention)
        self.spectral_queries_spatial = nn.MultiheadAttention(**attention)
        self.norm_spatial = nn.LayerNorm(config.d_model)
        self.norm_spectral = nn.LayerNorm(config.d_model)
        self.ff_spatial = _feedforward(config)
        self.ff_spectral = _feedforward(config)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        spectral_tokens: torch.Tensor,
        spatial_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_delta, _ = self.spatial_queries_spectral(
            query=self.norm_spatial(spatial_tokens),
            key=spectral_tokens,
            value=spectral_tokens,
            need_weights=False,
        )
        spatial_tokens = spatial_tokens + spatial_delta
        spectral_delta, _ = self.spectral_queries_spatial(
            query=self.norm_spectral(spectral_tokens),
            key=spatial_tokens,
            value=spatial_tokens,
            key_padding_mask=spatial_padding_mask,
            need_weights=False,
        )
        spectral_tokens = spectral_tokens + spectral_delta
        return (
            spatial_tokens + self.ff_spatial(spatial_tokens),
            spectral_tokens + self.ff_spectral(spectral_tokens),
        )


class PhenotypeQueryBank(nn.Module):
    """Learn one query token for each scientific prediction task."""

    def __init__(self, config: PhytoCoreConfig) -> None:
        super().__init__()
        self.queries = nn.Parameter(
            torch.empty(1, len(config.phenotype_queries), config.d_model)
        )
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_query = nn.LayerNorm(config.d_model)
        self.norm_memory = nn.LayerNorm(config.d_model)
        self.ff = _feedforward(config)

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.queries.expand(memory.shape[0], -1, -1)
        normalized_memory = self.norm_memory(memory)
        delta, attention = self.cross_attention(
            query=self.norm_query(query),
            key=normalized_memory,
            value=normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + delta
        return query + self.ff(query), attention


def _feedforward(config: PhytoCoreConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(config.d_model),
        nn.Linear(config.d_model, config.feedforward_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.feedforward_dim, config.d_model),
    )


def _prediction_head(d_model: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model // 2, output_dim),
    )


class PhytoCore(nn.Module):
    """Spectral-spatial phenotype model with three regression and two class tasks."""

    def __init__(self, config: PhytoCoreConfig | None = None) -> None:
        super().__init__()
        self.config = config or PhytoCoreConfig()
        self.config.validate()
        self.spectral_encoder = SpectralEncoder(self.config)
        self.spatial_encoder = SpatialEncoder(self.config)
        self.fusion = BidirectionalSpectralSpatialFusion(self.config)
        self.phenotype_query_bank = PhenotypeQueryBank(self.config)
        self.regression_heads = nn.ModuleList(
            [
                _prediction_head(self.config.d_model, 1, self.config.dropout)
                for _ in self.config.regression_targets
            ]
        )
        self.fertilizer_head = _prediction_head(
            self.config.d_model,
            self.config.fertilizer_classes,
            self.config.dropout,
        )
        self.cultivar_head = _prediction_head(
            self.config.d_model,
            self.config.cultivar_classes,
            self.config.dropout,
        )

    def encode(
        self,
        hsi: torch.Tensor,
        leaf_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode cube and mask before cross-modal fusion.

        Keeping this interface separate enables later crop/zoom and token
        counterfactual evidence without changing the trained prediction heads.
        """

        self._validate_input(hsi, leaf_mask)
        spectral_tokens = self.spectral_encoder(hsi, leaf_mask)
        spatial_tokens, spatial_padding_mask = self.spatial_encoder(hsi, leaf_mask)
        return {
            "spectral_tokens": spectral_tokens,
            "spatial_tokens": spatial_tokens,
            "spatial_padding_mask": spatial_padding_mask,
        }

    def predict_from_encoded(
        self,
        encoded: Mapping[str, torch.Tensor],
        spatial_keep_mask: torch.Tensor | None = None,
        spectral_keep_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        """Fuse encoded tokens and make task predictions.

        Optional keep masks are multiplicative counterfactual interventions;
        they are intended for evidence generation, not ordinary training.
        """

        fused = self.fuse_encoded(
            encoded,
            spatial_keep_mask=spatial_keep_mask,
            spectral_keep_mask=spectral_keep_mask,
        )
        return self.predict_from_fused(fused, return_attention=return_attention)

    def fuse_encoded(
        self,
        encoded: Mapping[str, torch.Tensor],
        spatial_keep_mask: torch.Tensor | None = None,
        spectral_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse spectral and spatial tokens into reusable visual memory.

        The prediction heads and the VLM connector consume this same memory.
        This keeps the structured measurements and language-visible evidence
        tied to one shared, trained HSI representation.
        """

        spatial = encoded["spatial_tokens"]
        spectral = encoded["spectral_tokens"]
        spatial_padding = encoded["spatial_padding_mask"]
        if spatial_keep_mask is not None:
            if spatial_keep_mask.shape != spatial.shape[:2]:
                raise ValueError("spatial_keep_mask must match spatial token shape")
            spatial = spatial * spatial_keep_mask[:, :, None].to(spatial.dtype)
        if spectral_keep_mask is not None:
            if spectral_keep_mask.shape != spectral.shape[:2]:
                raise ValueError("spectral_keep_mask must match spectral token shape")
            spectral = spectral * spectral_keep_mask[:, :, None].to(spectral.dtype)
        spatial, spectral = self.fusion(spatial, spectral, spatial_padding)
        spectral_padding = torch.zeros(
            spectral.shape[:2], dtype=torch.bool, device=spectral.device
        )
        memory = torch.cat((spatial, spectral), dim=1)
        memory_padding = torch.cat((spatial_padding, spectral_padding), dim=1)
        return {
            "spatial_tokens": spatial,
            "spectral_tokens": spectral,
            "spatial_padding_mask": spatial_padding,
            "memory": memory,
            "memory_padding_mask": memory_padding,
        }

    def predict_from_fused(
        self,
        fused: Mapping[str, torch.Tensor],
        return_attention: bool = False,
    ) -> dict[str, Any]:
        """Apply phenotype queries and prediction heads to fused memory."""

        memory = fused["memory"]
        memory_padding = fused["memory_padding_mask"]
        phenotype_tokens, attention = self.phenotype_query_bank(
            memory,
            memory_padding,
        )
        regression = torch.cat(
            [
                head(phenotype_tokens[:, index])
                for index, head in enumerate(self.regression_heads)
            ],
            dim=1,
        )
        fertilizer_index = len(self.regression_heads)
        cultivar_index = fertilizer_index + 1
        output: dict[str, Any] = {
            "regression": regression,
            "fertilizer_logits": self.fertilizer_head(
                phenotype_tokens[:, fertilizer_index]
            ),
            "cultivar_logits": self.cultivar_head(
                phenotype_tokens[:, cultivar_index]
            ),
            "phenotype_tokens": phenotype_tokens,
        }
        if return_attention:
            output.update(
                {
                    "pqb_attention": attention,
                    "spatial_padding_mask": fused["spatial_padding_mask"],
                    "memory_padding_mask": memory_padding,
                    "n_spatial_tokens": fused["spatial_tokens"].shape[1],
                    "n_spectral_tokens": fused["spectral_tokens"].shape[1],
                }
            )
        return output

    def forward(
        self,
        hsi: torch.Tensor,
        leaf_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        return self.predict_from_encoded(
            self.encode(hsi, leaf_mask),
            return_attention=return_attention,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_input(self, hsi: torch.Tensor, leaf_mask: torch.Tensor) -> None:
        expected_hsi = (
            self.config.input_bands,
            self.config.image_height,
            self.config.image_width,
        )
        if hsi.ndim != 4 or tuple(hsi.shape[1:]) != expected_hsi:
            raise ValueError(
                f"Expected HSI [B, {expected_hsi[0]}, {expected_hsi[1]}, "
                f"{expected_hsi[2]}], observed {tuple(hsi.shape)}"
            )
        if leaf_mask.ndim != 3 or leaf_mask.shape != hsi.shape[:1] + hsi.shape[2:]:
            raise ValueError("leaf_mask must have shape [B, H, W] matching HSI")
        if not hsi.is_floating_point():
            raise TypeError("HSI input must be floating point")


def build_phytocore(
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any] | None = None,
    validate_parameter_count: bool = True,
) -> PhytoCore:
    """Build PhytoCore and enforce the paper architecture parameter count."""

    config = PhytoCoreConfig.from_configs(model_config, data_config)
    model = PhytoCore(config)
    expected = config.expected_parameter_count
    observed = model.parameter_count()
    if validate_parameter_count and expected is not None and observed != expected:
        raise RuntimeError(
            f"PhytoCore parameter count mismatch: expected {expected:,}, "
            f"observed {observed:,}"
        )
    return model


def parameter_table(model: nn.Module) -> list[dict[str, int | str]]:
    """Return trainable parameter counts grouped by top-level module."""

    counts: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            top_level = name.split(".", maxsplit=1)[0]
            counts[top_level] = counts.get(top_level, 0) + parameter.numel()
    return [
        {"module": name, "trainable_parameters": count}
        for name, count in counts.items()
    ]


# ============================================================================
# 2. QUESTION / TARGET-CONDITIONED GLOBAL-LOCAL VISUAL BRIDGE
# ============================================================================

@dataclass(frozen=True)
class VisualTokenBridgeConfig:
    vision_dim: int = 128
    language_dim: int = 1536
    attention_heads: int = 4
    prefix_tokens: int = 16
    resampler_layers: int = 2
    question_adapter_layers: int = 2
    feedforward_dim: int = 512
    max_local_views: int = 2
    dropout: float = 0.1
    target_names: tuple[str, ...] = (
        "GrainWeight",
        "Gsw",
        "PhiPS2",
        "Fertilizer",
        "Cultivar",
        "GENERAL",
    )
    freeze_vision_tower: bool = True
    use_roi_geometry: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VisualTokenBridgeConfig":
        section = config.get("model", config)
        return cls(
            vision_dim=int(section["vision_dim"]),
            language_dim=int(section["language_dim"]),
            attention_heads=int(section["cross_modal_attention_heads"]),
            prefix_tokens=int(section["evidence_resampler_tokens"]),
            resampler_layers=int(section["evidence_resampler_layers"]),
            question_adapter_layers=int(section["text_adapter_layers"]),
            feedforward_dim=int(section["feedforward_dim"]),
            max_local_views=int(section["max_local_views"]),
            dropout=float(section["dropout"]),
            target_names=tuple(str(value) for value in section["target_order"]),
            freeze_vision_tower=bool(
                section["freeze_phytocore_during_language_training"]
            ),
            use_roi_geometry=bool(section.get("use_roi_geometry", True)),
        )

    def validate(self) -> None:
        if self.vision_dim % self.attention_heads != 0:
            raise ValueError("vision_dim must be divisible by attention_heads")
        if min(
            self.language_dim,
            self.prefix_tokens,
            self.resampler_layers,
            self.question_adapter_layers,
            self.feedforward_dim,
            self.max_local_views,
        ) <= 0:
            raise ValueError("Bridge dimensions and depths must be positive")
        if len(self.target_names) < 2 or self.target_names[-1] != "GENERAL":
            raise ValueError("target_names must end with the GENERAL target")
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target_names must be unique")


class QuestionTokenAdapter(nn.Module):
    """Compress native Qwen token embeddings into the visual token dimension."""

    def __init__(self, config: VisualTokenBridgeConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.language_dim)
        self.project = nn.Linear(config.language_dim, config.vision_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.vision_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.question_adapter_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.vision_dim)

    def forward(
        self,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        padding = ~question_attention_mask.bool()
        tokens = self.project(self.input_norm(question_embeddings))
        return self.output_norm(
            self.encoder(tokens, src_key_padding_mask=padding)
        )


class ConditionedResamplerLayer(nn.Module):
    """Update learned prefix queries from question tokens and visual memory."""

    def __init__(self, config: VisualTokenBridgeConfig) -> None:
        super().__init__()
        attention = {
            "embed_dim": config.vision_dim,
            "num_heads": config.attention_heads,
            "dropout": config.dropout,
            "batch_first": True,
        }
        self.question_attention = nn.MultiheadAttention(**attention)
        self.visual_attention = nn.MultiheadAttention(**attention)
        self.self_attention = nn.MultiheadAttention(**attention)
        self.norm_latent_question = nn.LayerNorm(config.vision_dim)
        self.norm_question = nn.LayerNorm(config.vision_dim)
        self.norm_latent_visual = nn.LayerNorm(config.vision_dim)
        self.norm_visual = nn.LayerNorm(config.vision_dim)
        self.norm_latent_self = nn.LayerNorm(config.vision_dim)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(config.vision_dim),
            nn.Linear(config.vision_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.vision_dim),
        )

    def forward(
        self,
        latents: torch.Tensor,
        question_tokens: torch.Tensor,
        question_padding_mask: torch.Tensor,
        visual_memory: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        return_attention: bool,
        use_question_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if use_question_conditioning:
            delta, _ = self.question_attention(
                query=self.norm_latent_question(latents),
                key=self.norm_question(question_tokens),
                value=self.norm_question(question_tokens),
                key_padding_mask=question_padding_mask,
                need_weights=False,
            )
            latents = latents + delta
        delta, visual_attention = self.visual_attention(
            query=self.norm_latent_visual(latents),
            key=self.norm_visual(visual_memory),
            value=self.norm_visual(visual_memory),
            key_padding_mask=visual_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        latents = latents + delta
        normalized = self.norm_latent_self(latents)
        delta, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            need_weights=False,
        )
        latents = latents + delta
        return latents + self.feedforward(latents), visual_attention


class TargetConditionedGlobalLocalBridge(nn.Module):
    """Produce Qwen prefix tokens from global HSI, ROI crops, and a question."""

    def __init__(
        self,
        vision_tower: PhytoCore,
        config: VisualTokenBridgeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or VisualTokenBridgeConfig()
        self.config.validate()
        if vision_tower.config.d_model != self.config.vision_dim:
            raise ValueError("PhytoCore d_model and bridge vision_dim disagree")
        self.vision_tower = vision_tower
        self.question_adapter = QuestionTokenAdapter(self.config)
        self.prefix_queries = nn.Parameter(
            torch.empty(1, self.config.prefix_tokens, self.config.vision_dim)
        )
        self.target_embedding = nn.Embedding(
            len(self.config.target_names), self.config.vision_dim
        )
        self.view_type_embedding = nn.Embedding(2, self.config.vision_dim)
        self.local_view_embedding = nn.Embedding(
            self.config.max_local_views, self.config.vision_dim
        )
        self.resampler = nn.ModuleList(
            [
                ConditionedResamplerLayer(self.config)
                for _ in range(self.config.resampler_layers)
            ]
        )
        self.language_projector = nn.Sequential(
            nn.LayerNorm(self.config.vision_dim),
            nn.Linear(self.config.vision_dim, self.config.language_dim),
        )
        self.roi_geometry = (
            nn.Sequential(
                nn.Linear(4, self.config.vision_dim), nn.GELU(),
                nn.Linear(self.config.vision_dim, self.config.vision_dim),
            )
            if self.config.use_roi_geometry else None
        )
        nn.init.trunc_normal_(self.prefix_queries, std=0.02)
        if self.config.freeze_vision_tower:
            for parameter in self.vision_tower.parameters():
                parameter.requires_grad_(False)
            self.vision_tower.eval()

    def train(self, mode: bool = True) -> "TargetConditionedGlobalLocalBridge":
        super().train(mode)
        if self.config.freeze_vision_tower:
            self.vision_tower.eval()
        return self

    def forward(
        self,
        global_hsi: torch.Tensor,
        global_leaf_mask: torch.Tensor,
        local_hsi: torch.Tensor,
        local_leaf_masks: torch.Tensor,
        local_view_mask: torch.Tensor,
        local_target_indices: torch.Tensor,
        primary_target_index: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor,
        return_attention: bool = False,
        ablation_mode: str = "full",
        roi_boxes_xyxy: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(
            global_hsi,
            global_leaf_mask,
            local_hsi,
            local_leaf_masks,
            local_view_mask,
            local_target_indices,
            primary_target_index,
            question_embeddings,
            question_attention_mask,
        )
        allowed_ablations = {
            "full",
            "global_only",
            "local_only",
            "no_question_conditioning",
        }
        if ablation_mode not in allowed_ablations:
            raise ValueError(
                f"Unknown ablation_mode={ablation_mode}; expected "
                f"{sorted(allowed_ablations)}"
            )
        if ablation_mode == "local_only" and not local_view_mask.bool().any(dim=1).all():
            raise ValueError("local_only ablation requires an active local view per sample")
        batch, views = local_view_mask.shape
        global_fused = self.vision_tower.fuse_encoded(
            self.vision_tower.encode(global_hsi, global_leaf_mask)
        )
        structured = self.vision_tower.predict_from_fused(global_fused)
        global_memory = global_fused["memory"] + self.view_type_embedding.weight[0]
        visual_parts = [global_memory]
        global_padding = global_fused["memory_padding_mask"]
        if ablation_mode == "local_only":
            global_padding = torch.ones_like(global_padding)
        padding_parts = [global_padding]

        flat_hsi = local_hsi.reshape(
            batch * views, *local_hsi.shape[2:]
        )
        flat_masks = local_leaf_masks.reshape(
            batch * views, *local_leaf_masks.shape[2:]
        )
        local_fused = self.vision_tower.fuse_encoded(
            self.vision_tower.encode(flat_hsi, flat_masks)
        )
        token_count = local_fused["memory"].shape[1]
        local_memory = local_fused["memory"].reshape(
            batch, views, token_count, self.config.vision_dim
        )
        local_padding = local_fused["memory_padding_mask"].reshape(
            batch, views, token_count
        )
        target_bias = self.target_embedding(local_target_indices)[:, :, None]
        view_ids = torch.arange(views, device=local_hsi.device)
        view_bias = self.local_view_embedding(view_ids)[None, :, None]
        local_memory = (
            local_memory
            + target_bias
            + view_bias
            + self.view_type_embedding.weight[1]
        )
        if self.roi_geometry is not None:
            if roi_boxes_xyxy is None or roi_boxes_xyxy.shape != (batch, views, 4):
                raise ValueError("Geometry bridge requires ROI boxes [B,V,4]")
            height, width = global_hsi.shape[-2:]
            boxes = roi_boxes_xyxy.to(device=local_memory.device, dtype=torch.float32)
            active_boxes = boxes[local_view_mask.bool()]
            if not torch.isfinite(boxes).all() or (active_boxes < 0).any():
                raise ValueError("ROI coordinates must be finite and nonnegative")
            if active_boxes.numel() and (
                (active_boxes[:, 2] > width).any() or (active_boxes[:, 3] > height).any()
                or (active_boxes[:, 0] >= active_boxes[:, 2]).any()
                or (active_boxes[:, 1] >= active_boxes[:, 3]).any()
            ):
                raise ValueError("Active ROI is empty or outside the original image")
            scale = boxes.new_tensor([width, height, width, height])
            geometry = self.roi_geometry((boxes / scale).to(local_memory.dtype))
            local_memory = local_memory + geometry[:, :, None] * local_view_mask[:, :, None, None]
        effective_local_view_mask = local_view_mask.bool()
        if ablation_mode == "global_only":
            effective_local_view_mask = torch.zeros_like(effective_local_view_mask)
        local_padding = local_padding | (~effective_local_view_mask)[:, :, None]
        visual_parts.append(local_memory.flatten(1, 2))
        padding_parts.append(local_padding.flatten(1, 2))
        visual_memory = torch.cat(visual_parts, dim=1)
        visual_padding = torch.cat(padding_parts, dim=1)

        question_tokens = self.question_adapter(
            question_embeddings, question_attention_mask
        )
        latents = self.prefix_queries.expand(batch, -1, -1)
        latents = latents + self.target_embedding(primary_target_index)[:, None]
        attention: torch.Tensor | None = None
        for layer in self.resampler:
            latents, attention = layer(
                latents,
                question_tokens,
                ~question_attention_mask.bool(),
                visual_memory,
                visual_padding,
                return_attention,
                use_question_conditioning=(
                    ablation_mode != "no_question_conditioning"
                ),
            )
        prefix = self.language_projector(latents)
        output: dict[str, Any] = {
            "prefix_embeddings": prefix,
            "prefix_attention_mask": torch.ones(
                prefix.shape[:2], dtype=torch.long, device=prefix.device
            ),
            "structured_predictions": structured,
            "visual_padding_mask": visual_padding,
            "visual_token_count": visual_memory.shape[1],
            "ablation_mode": ablation_mode,
        }
        if return_attention:
            output["resampler_visual_attention"] = attention
        return output

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _validate_inputs(
        self,
        global_hsi: torch.Tensor,
        global_leaf_mask: torch.Tensor,
        local_hsi: torch.Tensor,
        local_leaf_masks: torch.Tensor,
        local_view_mask: torch.Tensor,
        local_target_indices: torch.Tensor,
        primary_target_index: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> None:
        batch = global_hsi.shape[0]
        if global_hsi.ndim != 4 or global_leaf_mask.ndim != 3:
            raise ValueError("Global HSI/mask must be [B,C,H,W] and [B,H,W]")
        if local_hsi.ndim != 5 or local_leaf_masks.ndim != 4:
            raise ValueError("Local HSI/masks must be [B,V,C,H,W] and [B,V,H,W]")
        views = local_hsi.shape[1]
        if views != self.config.max_local_views:
            raise ValueError(
                f"Expected {self.config.max_local_views} local slots, got {views}"
            )
        if local_hsi.shape[:2] != (batch, views):
            raise ValueError("Local HSI batch does not match global HSI")
        if local_view_mask.shape != (batch, views):
            raise ValueError("local_view_mask must have shape [B,V]")
        if local_target_indices.shape != (batch, views):
            raise ValueError("local_target_indices must have shape [B,V]")
        if primary_target_index.shape != (batch,):
            raise ValueError("primary_target_index must have shape [B]")
        if question_embeddings.ndim != 3 or question_embeddings.shape[0] != batch:
            raise ValueError("question_embeddings must have shape [B,T,language_dim]")
        if question_embeddings.shape[-1] != self.config.language_dim:
            raise ValueError("Question embedding dimension does not match language_dim")
        if question_attention_mask.shape != question_embeddings.shape[:2]:
            raise ValueError("question_attention_mask must have shape [B,T]")
        if not question_attention_mask.bool().any(dim=1).all():
            raise ValueError("Every question must contain at least one visible token")
        target_count = len(self.config.target_names)
        if torch.any(local_target_indices < 0) or torch.any(
            local_target_indices >= target_count
        ):
            raise ValueError("local_target_indices contains an unknown target")
        if torch.any(primary_target_index < 0) or torch.any(
            primary_target_index >= target_count
        ):
            raise ValueError("primary_target_index contains an unknown target")


def prepend_visual_prefix(
    prefix_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_attention_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Create Qwen ``inputs_embeds`` and mask loss on visual prefix positions."""

    if prefix_embeddings.ndim != 3 or text_embeddings.ndim != 3:
        raise ValueError("Prefix and text embeddings must both be rank-3")
    if prefix_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError("Prefix and text batch sizes disagree")
    if prefix_embeddings.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("Prefix and text embedding dimensions disagree")
    if text_attention_mask.shape != text_embeddings.shape[:2]:
        raise ValueError("text_attention_mask must match text tokens")
    batch, prefix_length = prefix_embeddings.shape[:2]
    prefix_mask = torch.ones(
        (batch, prefix_length),
        dtype=text_attention_mask.dtype,
        device=text_attention_mask.device,
    )
    output = {
        "inputs_embeds": torch.cat((prefix_embeddings, text_embeddings), dim=1),
        "attention_mask": torch.cat((prefix_mask, text_attention_mask), dim=1),
    }
    if labels is not None:
        if labels.shape != text_attention_mask.shape:
            raise ValueError("labels must match text_attention_mask")
        ignored = torch.full(
            (batch, prefix_length),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        output["labels"] = torch.cat((ignored, labels), dim=1)
    return output


# ============================================================================
# 3. COMPLETE LEARNED VLM: VISUAL PREFIX + QWEN-LORA
# ============================================================================

@dataclass(frozen=True)
class QwenAlignmentConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    language_dim: int = 1536
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    local_files_only: bool = True
    trust_remote_code: bool = False
    gradient_checkpointing: bool = True

    @classmethod
    def from_configs(
        cls,
        model_config: Mapping[str, Any],
        alignment_config: Mapping[str, Any],
    ) -> "QwenAlignmentConfig":
        model = model_config.get("model", model_config)
        language = alignment_config.get("language", alignment_config)
        lora = model["qwen"]["lora"]
        return cls(
            model_name=str(model["language_model"]),
            language_dim=int(model["language_dim"]),
            lora_rank=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            lora_target_modules=tuple(str(value) for value in lora["target_modules"]),
            local_files_only=bool(language["local_files_only"]),
            trust_remote_code=bool(language["trust_remote_code"]),
            gradient_checkpointing=bool(language["gradient_checkpointing"]),
        )


class PhytoSpectraVLM(nn.Module):
    """Inject question-conditioned HSI prefix tokens into a causal Qwen model."""

    def __init__(
        self,
        bridge: TargetConditionedGlobalLocalBridge,
        language_model: nn.Module,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.language_model = language_model
        hidden_size = int(getattr(self.language_model.config, "hidden_size"))
        if hidden_size != self.bridge.config.language_dim:
            raise ValueError(
                f"Qwen hidden size {hidden_size} does not match bridge "
                f"language_dim {self.bridge.config.language_dim}"
            )

    def forward(
        self,
        global_hsi: torch.Tensor,
        global_leaf_mask: torch.Tensor,
        local_hsi: torch.Tensor,
        local_leaf_masks: torch.Tensor,
        local_view_mask: torch.Tensor,
        local_target_indices: torch.Tensor,
        primary_target_index: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        roi_boxes_xyxy: torch.Tensor,
        return_bridge_attention: bool = False,
        ablation_mode: str = "full",
    ) -> dict[str, Any]:
        embedding_layer = self.language_model.get_input_embeddings()
        question_embeddings = embedding_layer(question_input_ids)
        # Qwen may run in bfloat16 while the compact visual bridge remains
        # float32.  LayerNorm requires its input and parameters to agree.
        question_dtype = next(self.bridge.question_adapter.parameters()).dtype
        question_embeddings = question_embeddings.to(question_dtype)
        bridge_output = self.bridge(
            global_hsi=global_hsi,
            global_leaf_mask=global_leaf_mask,
            local_hsi=local_hsi,
            local_leaf_masks=local_leaf_masks,
            local_view_mask=local_view_mask,
            local_target_indices=local_target_indices,
            primary_target_index=primary_target_index,
            question_embeddings=question_embeddings,
            question_attention_mask=question_attention_mask,
            return_attention=return_bridge_attention,
            ablation_mode=ablation_mode,
            roi_boxes_xyxy=roi_boxes_xyxy,
        )
        text_embeddings = embedding_layer(input_ids)
        prefix = bridge_output["prefix_embeddings"].to(text_embeddings.dtype)
        combined = prepend_visual_prefix(
            prefix,
            text_embeddings,
            attention_mask,
            labels,
        )
        language_output = self.language_model(
            inputs_embeds=combined["inputs_embeds"],
            attention_mask=combined["attention_mask"],
            labels=combined["labels"],
            use_cache=False,
            return_dict=True,
        )
        output: dict[str, Any] = {
            "loss": language_output.loss,
            "logits": language_output.logits,
            "structured_predictions": bridge_output["structured_predictions"],
            "prefix_embeddings": prefix,
            "combined_attention_mask": combined["attention_mask"],
            "visual_token_count": bridge_output["visual_token_count"],
        }
        if return_bridge_attention:
            output["resampler_visual_attention"] = bridge_output[
                "resampler_visual_attention"
            ]
        return output

    @torch.inference_mode()
    def generate(
        self,
        global_hsi: torch.Tensor,
        global_leaf_mask: torch.Tensor,
        local_hsi: torch.Tensor,
        local_leaf_masks: torch.Tensor,
        local_view_mask: torch.Tensor,
        local_target_indices: torch.Tensor,
        primary_target_index: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        roi_boxes_xyxy: torch.Tensor,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: int,
        ablation_mode: str = "full",
    ) -> torch.Tensor:
        """Generate from visual-prefix + prompt embeddings without answer leakage."""

        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        embedding_layer = self.language_model.get_input_embeddings()
        question_embeddings = embedding_layer(question_input_ids)
        question_dtype = next(self.bridge.question_adapter.parameters()).dtype
        question_embeddings = question_embeddings.to(question_dtype)
        bridge_output = self.bridge(
            global_hsi=global_hsi,
            global_leaf_mask=global_leaf_mask,
            local_hsi=local_hsi,
            local_leaf_masks=local_leaf_masks,
            local_view_mask=local_view_mask,
            local_target_indices=local_target_indices,
            primary_target_index=primary_target_index,
            question_embeddings=question_embeddings,
            question_attention_mask=question_attention_mask,
            ablation_mode=ablation_mode,
            roi_boxes_xyxy=roi_boxes_xyxy,
        )
        text_embeddings = embedding_layer(input_ids)
        prefix = bridge_output["prefix_embeddings"].to(text_embeddings.dtype)
        combined = prepend_visual_prefix(prefix, text_embeddings, attention_mask)
        return self.language_model.generate(
            inputs_embeds=combined["inputs_embeds"],
            attention_mask=combined["attention_mask"],
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=int(pad_token_id),
            eos_token_id=int(eos_token_id),
            use_cache=True,
        )

    def trainable_parameter_breakdown(self) -> dict[str, int]:
        vision = sum(
            parameter.numel()
            for parameter in self.bridge.vision_tower.parameters()
            if parameter.requires_grad
        )
        bridge = sum(
            parameter.numel()
            for name, parameter in self.bridge.named_parameters()
            if parameter.requires_grad and not name.startswith("vision_tower.")
        )
        language = sum(
            parameter.numel()
            for parameter in self.language_model.parameters()
            if parameter.requires_grad
        )
        return {
            "vision_tower": vision,
            "visual_bridge": bridge,
            "qwen_lora": language,
            "total": vision + bridge + language,
        }


def load_qwen_tokenizer(config: QwenAlignmentConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_qwen_lora(
    config: QwenAlignmentConfig,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the frozen Qwen base and attach the trainable LoRA matrices."""

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
        attn_implementation="sdpa",
    )
    if int(base.config.hidden_size) != config.language_dim:
        raise ValueError(
            f"Configured language_dim={config.language_dim}, but Qwen uses "
            f"{base.config.hidden_size}"
        )
    base.config.use_cache = not config.gradient_checkpointing
    if config.gradient_checkpointing:
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            base.gradient_checkpointing_enable()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
    )
    return get_peft_model(base, lora)


def build_phytospectra_vlm(
    vision_tower: PhytoCore,
    model_config: Mapping[str, Any],
    alignment_config: Mapping[str, Any],
    dtype: torch.dtype = torch.bfloat16,
    language_model: nn.Module | None = None,
) -> tuple[PhytoSpectraVLM, QwenAlignmentConfig]:
    bridge_config = VisualTokenBridgeConfig.from_config(model_config)
    qwen_config = QwenAlignmentConfig.from_configs(model_config, alignment_config)
    bridge = TargetConditionedGlobalLocalBridge(vision_tower, bridge_config)
    qwen = language_model or load_qwen_lora(qwen_config, dtype=dtype)
    return PhytoSpectraVLM(bridge, qwen), qwen_config


def build_idcv_classification_phytospectra_vlm(
    checkpoint_path: str,
    model_config: Mapping[str, Any],
    alignment_config: Mapping[str, Any],
    dtype: torch.dtype = torch.bfloat16,
    language_model: nn.Module | None = None,
) -> tuple[PhytoSpectraVLM, QwenAlignmentConfig, dict[str, Any]]:
    """Build the active VLM with the selected 04B Weighted-CE PhytoCore."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    vision_tower = PhytoCore()
    vision_tower.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = dict(checkpoint.get("metadata", {}))
    if metadata.get("fertilizer_formulation") != "WEIGHTED_CE":
        raise ValueError("VLM requires the selected Weighted-CE Fertilizer head")
    if vision_tower.parameter_count() != 1_998_346:
        raise ValueError("Weighted-CE PhytoCore parameter contract changed")
    model, qwen_config = build_phytospectra_vlm(
        vision_tower=vision_tower,
        model_config=model_config,
        alignment_config=alignment_config,
        dtype=dtype,
        language_model=language_model,
    )
    if any(parameter.requires_grad for parameter in model.bridge.vision_tower.parameters()):
        raise RuntimeError("PhytoCore must remain frozen during VLM alignment")
    return model, qwen_config, metadata


# ============================================================================
# 4a. MODEL-LEVEL COUNTERFACTUAL GROUNDING
# ============================================================================

TARGETS = ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar")


@dataclass
class ModelEvidenceState:
    encoded: dict[str, torch.Tensor]
    baseline: dict[str, torch.Tensor]
    spatial_scores: np.ndarray
    spectral_scores: np.ndarray
    pqb_spatial_attention: np.ndarray
    pqb_spectral_attention: np.ndarray
    spatial_padding_mask: np.ndarray


@dataclass(frozen=True)
class TargetSelection:
    target: str
    target_index: int
    spatial_indices: tuple[int, ...]
    spectral_indices: tuple[int, ...]
    focus_spatial_indices: tuple[int, ...]
    selected_patch_boxes: tuple[tuple[int, int, int, int], ...]
    focus_patch_boxes: tuple[tuple[int, int, int, int], ...]
    roi_box_xyxy: tuple[int, int, int, int]
    comprehensiveness_spatial: float
    sufficiency_error_spatial: float
    comprehensiveness_spectral: float
    sufficiency_error_spectral: float


@dataclass(frozen=True)
class EnsembleEvidence:
    targets: tuple[str, ...]
    regression_z: np.ndarray
    fertilizer_probabilities: np.ndarray
    cultivar_probabilities: np.ndarray
    spatial_scores: np.ndarray
    spectral_scores: np.ndarray
    pqb_spatial_attention: np.ndarray
    pqb_spectral_attention: np.ndarray
    selections: tuple[TargetSelection, ...]
    model_count: int


class TokenCounterfactualGrounder:
    """Generate task-specific PCTG evidence from one OOF model ensemble."""

    def __init__(
        self,
        spatial_top_k: int = 13,
        spectral_top_k: int = 15,
        variant_batch_size: int = 32,
        roi_margin_patches: int = 1,
    ) -> None:
        if spatial_top_k <= 0 or spectral_top_k <= 0:
            raise ValueError("Evidence top-k values must be positive")
        if variant_batch_size <= 0:
            raise ValueError("variant_batch_size must be positive")
        if roi_margin_patches < 0:
            raise ValueError("roi_margin_patches cannot be negative")
        self.spatial_top_k = int(spatial_top_k)
        self.spectral_top_k = int(spectral_top_k)
        self.variant_batch_size = int(variant_batch_size)
        self.roi_margin_patches = int(roi_margin_patches)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TokenCounterfactualGrounder":
        section = config.get("evidence", config)
        if section["method"] != "single_token_zero_ablation":
            raise ValueError("Only single-token zero ablation is implemented")
        if tuple(section["targets"]) != TARGETS:
            raise ValueError("Evidence target order must match PhytoCore query order")
        return cls(
            spatial_top_k=int(section["spatial_top_k"]),
            spectral_top_k=int(section["spectral_top_k"]),
            variant_batch_size=int(section["variant_batch_size"]),
            roi_margin_patches=int(section["roi_margin_patches"]),
        )

    @torch.inference_mode()
    def ground_ensemble(
        self,
        models: Sequence[PhytoCore],
        hsi: torch.Tensor,
        leaf_mask: torch.Tensor,
    ) -> EnsembleEvidence:
        """Average OOF seed-model attribution, then select common evidence."""

        if not models:
            raise ValueError("At least one OOF model is required")
        if hsi.shape[0] != 1 or leaf_mask.shape[0] != 1:
            raise ValueError("Evidence generation currently requires batch size one")
        states = [self.compute_model_state(model, hsi, leaf_mask) for model in models]
        spatial_scores = np.mean(
            np.stack([state.spatial_scores for state in states]), axis=0
        ).astype(np.float32)
        spectral_scores = np.mean(
            np.stack([state.spectral_scores for state in states]), axis=0
        ).astype(np.float32)
        spatial_attention = np.mean(
            np.stack([state.pqb_spatial_attention for state in states]), axis=0
        ).astype(np.float32)
        spectral_attention = np.mean(
            np.stack([state.pqb_spectral_attention for state in states]), axis=0
        ).astype(np.float32)
        regression_z = np.mean(
            np.stack(
                [state.baseline["regression"][0].detach().float().cpu().numpy() for state in states]
            ),
            axis=0,
        ).astype(np.float32)
        fertilizer_probabilities = np.mean(
            np.stack(
                [
                    torch.softmax(state.baseline["fertilizer_logits"][0], dim=0)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for state in states
                ]
            ),
            axis=0,
        ).astype(np.float32)
        cultivar_probabilities = np.mean(
            np.stack(
                [
                    torch.softmax(state.baseline["cultivar_logits"][0], dim=0)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for state in states
                ]
            ),
            axis=0,
        ).astype(np.float32)
        config = models[0].config
        for model in models[1:]:
            if model.config != config:
                raise ValueError("All OOF ensemble models must share one architecture")
        valid_spatial = ~np.logical_or.reduce(
            [state.spatial_padding_mask for state in states]
        )
        selections_without_metrics = [
            self._select_target(
                target_index=index,
                target=target,
                spatial_scores=spatial_scores[index],
                spectral_scores=spectral_scores[index],
                valid_spatial=valid_spatial,
                image_height=config.image_height,
                image_width=config.image_width,
                patch_size=config.patch_size,
            )
            for index, target in enumerate(TARGETS)
        ]
        per_model_metrics = [
            self._evaluate_selections(model, state, selections_without_metrics)
            for model, state in zip(models, states)
        ]
        selections = []
        for target_index, selection in enumerate(selections_without_metrics):
            metrics = np.mean(
                np.asarray(
                    [values[target_index] for values in per_model_metrics],
                    dtype=np.float64,
                ),
                axis=0,
            )
            selections.append(
                TargetSelection(
                    **selection,
                    comprehensiveness_spatial=float(metrics[0]),
                    sufficiency_error_spatial=float(metrics[1]),
                    comprehensiveness_spectral=float(metrics[2]),
                    sufficiency_error_spectral=float(metrics[3]),
                )
            )
        return EnsembleEvidence(
            targets=TARGETS,
            regression_z=regression_z,
            fertilizer_probabilities=fertilizer_probabilities,
            cultivar_probabilities=cultivar_probabilities,
            spatial_scores=spatial_scores,
            spectral_scores=spectral_scores,
            pqb_spatial_attention=spatial_attention,
            pqb_spectral_attention=spectral_attention,
            selections=tuple(selections),
            model_count=len(models),
        )

    @torch.inference_mode()
    def compute_model_state(
        self,
        model: PhytoCore,
        hsi: torch.Tensor,
        leaf_mask: torch.Tensor,
    ) -> ModelEvidenceState:
        model.eval()
        encoded = model.encode(hsi, leaf_mask)
        baseline = model.predict_from_encoded(encoded, return_attention=True)
        n_spatial = int(encoded["spatial_tokens"].shape[1])
        n_spectral = int(encoded["spectral_tokens"].shape[1])
        spatial_masks = torch.ones(
            n_spatial, n_spatial, device=hsi.device, dtype=hsi.dtype
        )
        spatial_masks.fill_diagonal_(0.0)
        spectral_masks = torch.ones(
            n_spectral, n_spectral, device=hsi.device, dtype=hsi.dtype
        )
        spectral_masks.fill_diagonal_(0.0)
        spatial_output = self._predict_variants(
            model, encoded, spatial_masks=spatial_masks
        )
        spectral_output = self._predict_variants(
            model, encoded, spectral_masks=spectral_masks
        )
        spatial_scores = self._phenotype_changes(baseline, spatial_output).T
        spectral_scores = self._phenotype_changes(baseline, spectral_output).T
        padding = encoded["spatial_padding_mask"][0].detach().cpu().numpy().astype(bool)
        spatial_scores[:, padding] = 0.0
        attention = (
            baseline["pqb_attention"][0]
            .mean(dim=0)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        return ModelEvidenceState(
            encoded=encoded,
            baseline={
                key: value
                for key, value in baseline.items()
                if key in {"regression", "fertilizer_logits", "cultivar_logits"}
            },
            spatial_scores=spatial_scores.astype(np.float32),
            spectral_scores=spectral_scores.astype(np.float32),
            pqb_spatial_attention=attention[:, :n_spatial].astype(np.float32),
            pqb_spectral_attention=attention[:, n_spatial:].astype(np.float32),
            spatial_padding_mask=padding,
        )

    def _predict_variants(
        self,
        model: PhytoCore,
        encoded: Mapping[str, torch.Tensor],
        spatial_masks: torch.Tensor | None = None,
        spectral_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (spatial_masks is None) == (spectral_masks is None):
            raise ValueError("Provide exactly one token-mask domain")
        masks = spatial_masks if spatial_masks is not None else spectral_masks
        assert masks is not None
        output_parts: dict[str, list[torch.Tensor]] = {
            "regression": [],
            "fertilizer_logits": [],
            "cultivar_logits": [],
        }
        for start in range(0, masks.shape[0], self.variant_batch_size):
            end = min(start + self.variant_batch_size, masks.shape[0])
            count = end - start
            batch_encoded = {
                "spatial_tokens": encoded["spatial_tokens"].repeat(count, 1, 1),
                "spectral_tokens": encoded["spectral_tokens"].repeat(count, 1, 1),
                "spatial_padding_mask": encoded["spatial_padding_mask"].repeat(count, 1),
            }
            output = model.predict_from_encoded(
                batch_encoded,
                spatial_keep_mask=(masks[start:end] if spatial_masks is not None else None),
                spectral_keep_mask=(masks[start:end] if spectral_masks is not None else None),
            )
            for key in output_parts:
                output_parts[key].append(output[key])
        return {key: torch.cat(values, dim=0) for key, values in output_parts.items()}

    @staticmethod
    def _phenotype_changes(
        baseline: Mapping[str, torch.Tensor],
        variants: Mapping[str, torch.Tensor],
    ) -> np.ndarray:
        regression_change = torch.abs(
            variants["regression"] - baseline["regression"]
        )
        classification_changes = []
        for key in ("fertilizer_logits", "cultivar_logits"):
            baseline_probability = torch.softmax(baseline[key][0], dim=0)
            baseline_class = int(baseline_probability.argmax())
            variant_probability = torch.softmax(variants[key], dim=1)
            classification_changes.append(
                torch.abs(
                    variant_probability[:, baseline_class]
                    - baseline_probability[baseline_class]
                )[:, None]
            )
        return (
            torch.cat([regression_change, *classification_changes], dim=1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    def _select_target(
        self,
        target_index: int,
        target: str,
        spatial_scores: np.ndarray,
        spectral_scores: np.ndarray,
        valid_spatial: np.ndarray,
        image_height: int,
        image_width: int,
        patch_size: tuple[int, int],
    ) -> dict[str, Any]:
        spatial_indices = _stable_top_k(
            spatial_scores, self.spatial_top_k, allowed=valid_spatial
        )
        spectral_indices = _stable_top_k(spectral_scores, self.spectral_top_k)
        grid_width = image_width // patch_size[1]
        focus_indices = _strongest_connected_component(
            spatial_indices,
            spatial_scores,
            grid_width,
        )
        selected_boxes = tuple(
            _token_box(index, grid_width, patch_size, image_height, image_width)
            for index in spatial_indices
        )
        focus_boxes = tuple(
            _token_box(index, grid_width, patch_size, image_height, image_width)
            for index in focus_indices
        )
        roi_box = _patch_aligned_roi(
            focus_indices or spatial_indices,
            grid_width,
            patch_size,
            image_height,
            image_width,
            self.roi_margin_patches,
        )
        return {
            "target": target,
            "target_index": int(target_index),
            "spatial_indices": tuple(int(value) for value in spatial_indices),
            "spectral_indices": tuple(int(value) for value in spectral_indices),
            "focus_spatial_indices": tuple(int(value) for value in focus_indices),
            "selected_patch_boxes": selected_boxes,
            "focus_patch_boxes": focus_boxes,
            "roi_box_xyxy": roi_box,
        }

    def _evaluate_selections(
        self,
        model: PhytoCore,
        state: ModelEvidenceState,
        selections: Sequence[Mapping[str, Any]],
    ) -> list[tuple[float, float, float, float]]:
        n_spatial = state.encoded["spatial_tokens"].shape[1]
        n_spectral = state.encoded["spectral_tokens"].shape[1]
        device = state.encoded["spatial_tokens"].device
        spatial_masks = []
        spectral_masks = []
        for selection in selections:
            spatial_selected = torch.zeros(n_spatial, device=device)
            spatial_selected[list(selection["spatial_indices"])] = 1.0
            spatial_masks.extend((1.0 - spatial_selected, spatial_selected))
            spectral_selected = torch.zeros(n_spectral, device=device)
            spectral_selected[list(selection["spectral_indices"])] = 1.0
            spectral_masks.extend((1.0 - spectral_selected, spectral_selected))
        spatial_output = self._predict_variants(
            model, state.encoded, spatial_masks=torch.stack(spatial_masks)
        )
        spectral_output = self._predict_variants(
            model, state.encoded, spectral_masks=torch.stack(spectral_masks)
        )
        spatial_change = self._phenotype_changes(state.baseline, spatial_output)
        spectral_change = self._phenotype_changes(state.baseline, spectral_output)
        return [
            (
                float(spatial_change[2 * index, index]),
                float(spatial_change[2 * index + 1, index]),
                float(spectral_change[2 * index, index]),
                float(spectral_change[2 * index + 1, index]),
            )
            for index in range(len(selections))
        ]


def summarize_predictions(
    evidence: EnsembleEvidence,
    target_statistics: Mapping[str, Mapping[str, float]],
    fertilizer_values: Sequence[float],
    cultivar_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    regression_raw = inverse_standardize_targets(
        evidence.regression_z[None], target_statistics
    )[0]
    result = {
        target: {
            "task": "regression",
            "value": float(regression_raw[index]),
            "standardized_value": float(evidence.regression_z[index]),
        }
        for index, target in enumerate(TARGETS[:3])
    }
    fertilizer_index = int(evidence.fertilizer_probabilities.argmax())
    cultivar_index = int(evidence.cultivar_probabilities.argmax())
    result["Fertilizer"] = {
        "task": "classification",
        "value": float(fertilizer_values[fertilizer_index]),
        "class_index": fertilizer_index,
        "confidence": float(evidence.fertilizer_probabilities[fertilizer_index]),
        "probabilities": evidence.fertilizer_probabilities.tolist(),
    }
    result["Cultivar"] = {
        "task": "classification",
        "value": str(cultivar_names[cultivar_index]),
        "class_index": cultivar_index,
        "confidence": float(evidence.cultivar_probabilities[cultivar_index]),
        "probabilities": evidence.cultivar_probabilities.tolist(),
    }
    return result


def _stable_top_k(
    scores: np.ndarray,
    k: int,
    allowed: np.ndarray | None = None,
) -> tuple[int, ...]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    candidates = np.arange(len(values))
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool).reshape(-1)
        if allowed.shape != values.shape:
            raise ValueError("Allowed-token mask shape differs from scores")
        candidates = candidates[allowed]
    if len(candidates) == 0:
        candidates = np.arange(len(values))
    order = np.lexsort((candidates, -values[candidates]))
    return tuple(int(value) for value in candidates[order[: min(k, len(candidates))]])


def _strongest_connected_component(
    indices: Sequence[int],
    scores: np.ndarray,
    grid_width: int,
) -> tuple[int, ...]:
    remaining = set(int(value) for value in indices)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = [start]
        while stack:
            token = stack.pop()
            row, column = divmod(token, grid_width)
            neighbor_positions = {
                (row + dr) * grid_width + column + dc
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if dr != 0 or dc != 0
                if row + dr >= 0 and 0 <= column + dc < grid_width
            }
            connected = sorted(neighbor_positions & remaining)
            remaining.difference_update(connected)
            stack.extend(connected)
            component.extend(connected)
        components.append(component)
    if not components:
        return ()
    best = max(
        components,
        key=lambda component: (
            float(np.asarray(scores)[component].sum()),
            len(component),
            -min(component),
        ),
    )
    return tuple(sorted(best))


def _token_box(
    index: int,
    grid_width: int,
    patch_size: tuple[int, int],
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    row, column = divmod(int(index), int(grid_width))
    y0 = row * patch_size[0]
    x0 = column * patch_size[1]
    return (
        x0,
        y0,
        min(image_width, x0 + patch_size[1]),
        min(image_height, y0 + patch_size[0]),
    )


def _patch_aligned_roi(
    indices: Sequence[int],
    grid_width: int,
    patch_size: tuple[int, int],
    image_height: int,
    image_width: int,
    margin_patches: int,
) -> tuple[int, int, int, int]:
    if not indices:
        return (0, 0, image_width, image_height)
    positions = [divmod(int(index), int(grid_width)) for index in indices]
    rows = [value[0] for value in positions]
    columns = [value[1] for value in positions]
    grid_height = image_height // patch_size[0]
    row0 = max(0, min(rows) - margin_patches)
    row1 = min(grid_height, max(rows) + 1 + margin_patches)
    column0 = max(0, min(columns) - margin_patches)
    column1 = min(grid_width, max(columns) + 1 + margin_patches)
    return (
        column0 * patch_size[1],
        row0 * patch_size[0],
        column1 * patch_size[1],
        row1 * patch_size[0],
    )


# ============================================================================
# 4b. ALL-BAND LOCAL HSI CROP / RESIZE
# ============================================================================

def validate_roi_box(
    box_xyxy: Sequence[int],
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    if len(box_xyxy) != 4:
        raise ValueError("ROI box must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = (int(value) for value in box_xyxy)
    if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
        raise ValueError(
            f"ROI {tuple(box_xyxy)} lies outside image {(image_height, image_width)}"
        )
    return x0, y0, x1, y1


def extract_local_hsi_view(
    cube_unit_hwc: np.ndarray,
    leaf_mask: np.ndarray,
    box_xyxy: Sequence[int],
    output_shape_hw: tuple[int, int] = (48, 352),
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a 204-band ROI and resize it for the shared local HSI encoder.

    Bilinear interpolation is applied independently to every spectral band;
    nearest interpolation is used for the binary mask. No wavelength shift or
    band synthesis is performed. Background is restored to exact zero.
    """

    cube = np.asarray(cube_unit_hwc, dtype=np.float32)
    mask = np.asarray(leaf_mask, dtype=bool)
    if cube.ndim != 3 or mask.shape != cube.shape[:2]:
        raise ValueError("Expected cube [H, W, B] and matching leaf mask")
    if len(output_shape_hw) != 2 or min(output_shape_hw) <= 0:
        raise ValueError("output_shape_hw must contain two positive values")
    x0, y0, x1, y1 = validate_roi_box(box_xyxy, cube.shape[0], cube.shape[1])
    crop_cube = cube[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    cube_tensor = torch.from_numpy(np.moveaxis(crop_cube, -1, 0).copy())[None]
    mask_tensor = torch.from_numpy(crop_mask.astype(np.float32))[None, None]
    resized_cube = F.interpolate(
        cube_tensor,
        size=tuple(int(value) for value in output_shape_hw),
        mode="bilinear",
        align_corners=False,
    )[0]
    resized_mask = F.interpolate(
        mask_tensor,
        size=tuple(int(value) for value in output_shape_hw),
        mode="nearest",
    )[0, 0].bool()
    resized_cube[:, ~resized_mask] = 0.0
    output_cube = np.moveaxis(resized_cube.numpy(), 0, -1).astype(np.float32)
    return output_cube, resized_mask.numpy()


# ============================================================================
# 5a. PHYTOCORE MULTI-TASK TRAINING LOSS
# ============================================================================

REGRESSION_TARGETS = ("GrainWeight", "Gsw", "PhiPS2")


def inverse_sqrt_class_weights(
    labels: torch.Tensor | Sequence[int],
    n_classes: int,
) -> torch.Tensor:
    """Return inverse-square-root class weights normalized to mean one.

    Every training partition must contain all fertilizer classes; silently
    assigning a weight to a missing class would hide an invalid split.
    """

    values = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    if values.numel() == 0:
        raise ValueError("Cannot calculate class weights from no labels")
    if int(values.min()) < 0 or int(values.max()) >= int(n_classes):
        raise ValueError("Class labels lie outside the configured range")
    counts = torch.bincount(values, minlength=int(n_classes)).to(torch.float64)
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training partition is missing classes: {missing}")
    weights = counts.rsqrt()
    weights = weights / weights.mean()
    return weights.to(torch.float32)


class PhytoCoreMultiTaskLoss(nn.Module):
    """Sum three standardized Huber losses and two classification losses."""

    def __init__(
        self,
        fertilizer_class_weights: torch.Tensor,
        huber_delta: float = 1.0,
        regression_weights: Sequence[float] = (1.0, 1.0, 1.0),
        fertilizer_weight: float = 1.0,
        cultivar_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if len(regression_weights) != len(REGRESSION_TARGETS):
            raise ValueError("Exactly three regression loss weights are required")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        all_weights = (*regression_weights, fertilizer_weight, cultivar_weight)
        if any(float(value) < 0 for value in all_weights):
            raise ValueError("Task loss weights cannot be negative")
        class_weights = torch.as_tensor(
            fertilizer_class_weights, dtype=torch.float32
        ).flatten()
        if class_weights.numel() < 2 or torch.any(class_weights <= 0):
            raise ValueError("Fertilizer class weights must be positive")
        if not torch.isfinite(class_weights).all():
            raise ValueError("Fertilizer class weights contain non-finite values")
        self.huber_delta = float(huber_delta)
        self.regression_weights = tuple(float(value) for value in regression_weights)
        self.fertilizer_weight = float(fertilizer_weight)
        self.cultivar_weight = float(cultivar_weight)
        self.register_buffer("fertilizer_class_weights", class_weights)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        fertilizer_class_weights: torch.Tensor,
    ) -> "PhytoCoreMultiTaskLoss":
        section = config.get("loss", config)
        if str(section["regression"]).lower() != "huber":
            raise ValueError("The frozen PhytoCore regression loss is Huber")
        if str(section["fertilizer"]).lower() != "weighted_cross_entropy":
            raise ValueError("The frozen Fertilizer loss is weighted cross entropy")
        named_weights = section.get("regression_weights", {})
        regression_weights = tuple(
            float(named_weights.get(target, 1.0)) for target in REGRESSION_TARGETS
        )
        return cls(
            fertilizer_class_weights=fertilizer_class_weights,
            huber_delta=float(section["huber_delta"]),
            regression_weights=regression_weights,
            fertilizer_weight=float(section.get("fertilizer_loss_weight", 1.0)),
            cultivar_weight=float(section.get("cultivar_loss_weight", 1.0)),
        )

    def forward(
        self,
        output: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        regression = output["regression"]
        targets = batch["regression_targets"]
        if regression.shape != targets.shape or regression.shape[-1] != 3:
            raise ValueError("Regression prediction and target shapes must be [B, 3]")
        target_losses = []
        result: dict[str, torch.Tensor] = {}
        for index, (target, weight) in enumerate(
            zip(REGRESSION_TARGETS, self.regression_weights)
        ):
            value = F.huber_loss(
                regression[:, index],
                targets[:, index],
                reduction="mean",
                delta=self.huber_delta,
            )
            result[f"regression_{target}"] = value
            target_losses.append(value * weight)
        regression_loss = torch.stack(target_losses).sum()
        fertilizer_loss = F.cross_entropy(
            output["fertilizer_logits"],
            batch["fertilizer_label"],
            weight=self.fertilizer_class_weights,
        )
        cultivar_loss = F.cross_entropy(
            output["cultivar_logits"],
            batch["cultivar_label"],
        )
        total = (
            regression_loss
            + self.fertilizer_weight * fertilizer_loss
            + self.cultivar_weight * cultivar_loss
        )
        result.update(
            {
                "regression": regression_loss,
                "fertilizer": fertilizer_loss,
                "cultivar": cultivar_loss,
                "total": total,
            }
        )
        return result


# ============================================================================
# 5b. TRAINABLE-PARAMETER EMA
# ============================================================================

class TrainableParameterEMA:
    """Track bridge and LoRA weights without copying frozen PhytoCore/Qwen."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires at least one trainable parameter")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(trainable) != set(self.shadow):
            raise ValueError("The trainable parameter set changed after EMA creation")
        self.num_updates += 1
        # A short warm-up prevents an untrained initialization from dominating.
        effective_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        for name, parameter in trainable.items():
            current = parameter.detach().float().cpu()
            self.shadow[name].mul_(effective_decay).add_(
                current, alpha=1.0 - effective_decay
            )

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily evaluate/export with EMA weights, then restore training weights."""

        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(trainable) != set(self.shadow):
            raise ValueError("The trainable parameter set differs from EMA state")
        backup = {
            name: parameter.detach().clone() for name, parameter in trainable.items()
        }
        try:
            for name, parameter in trainable.items():
                with torch.no_grad():
                    parameter.copy_(
                        self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
                    )
            yield
        finally:
            for name, parameter in trainable.items():
                with torch.no_grad():
                    parameter.copy_(backup[name])

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Replace trainable weights with their EMA values for final export."""

        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(trainable) != set(self.shadow):
            raise ValueError("The trainable parameter set differs from EMA state")
        for name, parameter in trainable.items():
            parameter.copy_(
                self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
            )

    def load_state_dict(self, state: dict[str, object]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, dict) or set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint parameter names do not match")
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.shadow = {
            str(name): tensor.detach().float().cpu().clone()
            for name, tensor in shadow.items()
            if isinstance(tensor, torch.Tensor)
        }


# ============================================================================
# 5c. ASSISTANT-ONLY VQA TRAINING / GENERATION INPUT COLLATION
# ============================================================================

class HyperLeafVQACollator:
    """Build assistant-only causal labels and stack global/local HSI tensors."""

    def __init__(
        self,
        tokenizer: Any,
        language_config: Mapping[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = str(language_config["system_prompt"])
        self.max_sequence_tokens = int(language_config["max_sequence_tokens"])
        self.max_question_tokens = int(language_config["max_question_tokens"])
        if self.max_sequence_tokens <= 0 or self.max_question_tokens <= 0:
            raise ValueError("Token limits must be positive")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty VQA batch")
        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        question_sequences: list[list[int]] = []
        for sample in samples:
            prompt_ids = self._prompt_ids(str(sample["question"]))
            answer_ids = self._answer_ids(str(sample["answer"]))
            if len(prompt_ids) + len(answer_ids) > self.max_sequence_tokens:
                raise ValueError(
                    f"Prompt and answer exceed {self.max_sequence_tokens} tokens "
                    f"for record {sample['record_id']}"
                )
            sequences.append([*prompt_ids, *answer_ids])
            labels.append([-100] * len(prompt_ids) + answer_ids)
            question_sequences.append(self._question_ids(str(sample["question"])))
        input_ids, attention_mask = self._pad(sequences)
        label_tensor, _ = self._pad(labels, padding_value=-100)
        question_ids, question_attention_mask = self._pad(question_sequences)
        return {
            **self._stack_visual(samples),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "question_input_ids": question_ids,
            "question_attention_mask": question_attention_mask,
            "record_ids": [str(sample["record_id"]) for sample in samples],
            "image_ids": [str(sample["image_id"]) for sample in samples],
            "questions": [str(sample["question"]) for sample in samples],
            "answers": [str(sample["answer"]) for sample in samples],
            "intents": [str(sample["intent"]) for sample in samples],
            "targets": [list(sample.get("targets", [])) for sample in samples],
            "answer_modes": [str(sample["answer_mode"]) for sample in samples],
        }

    def for_generation(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Build prompt-only inputs so the reference answer is never exposed."""

        if not samples:
            raise ValueError("Cannot collate an empty generation batch")
        prompts = [self._prompt_ids(str(sample["question"])) for sample in samples]
        questions = [self._question_ids(str(sample["question"])) for sample in samples]
        input_ids, attention_mask = self._pad(prompts)
        question_ids, question_attention_mask = self._pad(questions)
        return {
            **self._stack_visual(samples),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "question_input_ids": question_ids,
            "question_attention_mask": question_attention_mask,
            "record_ids": [str(sample["record_id"]) for sample in samples],
            "image_ids": [str(sample["image_id"]) for sample in samples],
            "questions": [str(sample["question"]) for sample in samples],
        }

    def _prompt_ids(self, question: str) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(
            self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )

    def _answer_ids(self, answer: str) -> list[int]:
        tokens = list(
            self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        )
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define eos_token_id")
        if not tokens or tokens[-1] != int(eos):
            tokens.append(int(eos))
        return tokens

    def _question_ids(self, question: str) -> list[int]:
        tokens = list(
            self.tokenizer(
                question,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_question_tokens,
            )["input_ids"]
        )
        if not tokens:
            raise ValueError("Question tokenization produced an empty sequence")
        return tokens

    def _pad(
        self,
        sequences: Sequence[Sequence[int]],
        padding_value: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = max(len(sequence) for sequence in sequences)
        pad = int(self.tokenizer.pad_token_id if padding_value is None else padding_value)
        values = torch.full((len(sequences), maximum), pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), maximum), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            values[index, :length] = torch.tensor(sequence, dtype=torch.long)
            mask[index, :length] = 1
        return values, mask

    @staticmethod
    def _stack_visual(samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        tensor_keys = (
            "global_hsi",
            "global_leaf_mask",
            "local_hsi",
            "local_leaf_masks",
            "local_view_mask",
            "local_target_indices",
            "primary_target_index",
            "roi_boxes_xyxy",
            "outer_fold",
        )
        return {
            key: torch.stack([sample[key] for sample in samples])
            for key in tensor_keys
        }


# ============================================================================
# 6. OPTIONAL DETERMINISTIC VISUAL OUTPUT (NOT A NEURAL IMAGE DECODER)
# ============================================================================

def robust_false_color_composite(
    cube_unit_hwc: np.ndarray,
    leaf_mask: np.ndarray,
    band_indices: Sequence[int] = (40, 100, 160),
) -> Image.Image:
    """Create a display-only HSI false-color composite from fixed band indices."""

    from PIL import Image

    cube = np.asarray(cube_unit_hwc, dtype=np.float32)
    mask = np.asarray(leaf_mask, dtype=bool)
    if cube.ndim != 3 or mask.shape != cube.shape[:2]:
        raise ValueError("Expected cube [H, W, B] and matching leaf mask")
    if len(band_indices) != 3:
        raise ValueError("Exactly three fixed false-color band indices are required")
    if any(int(index) < 0 or int(index) >= cube.shape[2] for index in band_indices):
        raise ValueError("False-color band index lies outside the HSI cube")
    channels = [_robust_scale(cube[:, :, int(index)], mask) for index in band_indices]
    image = np.stack(channels, axis=-1)
    return Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="RGB")


def grounded_evidence_text(
    target: str,
    prediction: Mapping[str, Any],
    selection: TargetSelection,
) -> str:
    """Produce a templated statement containing only model output and evidence."""

    value = prediction["value"]
    if prediction["task"] == "classification":
        value_text = f"{value} (confidence {float(prediction['confidence']):.1%})"
    elif target == "GrainWeight":
        value_text = f"{float(value):.2f}"
    else:
        value_text = f"{float(value):.4f}"
    return (
        f"PhytoCore predicts {target} = {value_text}. "
        f"The highlighted crop {list(selection.roi_box_xyxy)} contains the strongest "
        f"connected region among {len(selection.spatial_indices)} selected spatial "
        "patches for this target. This is model-internal counterfactual evidence, "
        "not a causal or biological explanation."
    )


class EvidenceRenderer:
    """Save composite, heatmap, marked ROI, zoom, spectrum, and bundle PNGs."""

    def __init__(
        self,
        output_dir: str | Path,
        false_color_band_indices: Sequence[int] = (40, 100, 160),
        zoom_factor: int = 4,
        spatial_patch_size: Sequence[int] = (8, 16),
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.false_color_band_indices = tuple(int(value) for value in false_color_band_indices)
        self.zoom_factor = int(zoom_factor)
        self.spatial_patch_size = tuple(int(value) for value in spatial_patch_size)
        if self.zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive")
        if len(self.spatial_patch_size) != 2 or min(self.spatial_patch_size) <= 0:
            raise ValueError("spatial_patch_size must contain two positive values")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        output_dir: str | Path,
    ) -> "EvidenceRenderer":
        section = config.get("evidence", config)
        return cls(
            output_dir,
            false_color_band_indices=section["false_color_band_indices"],
            zoom_factor=int(section["zoom_factor"]),
            spatial_patch_size=section["spatial_patch_size"],
        )

    def render_target(
        self,
        image_id: str,
        target: str,
        cube_unit_hwc: np.ndarray,
        leaf_mask: np.ndarray,
        evidence: EnsembleEvidence,
        prediction: Mapping[str, Any],
        disclaimer: str,
    ) -> dict[str, Any]:
        from PIL import Image

        target_index = evidence.targets.index(target)
        selection = evidence.selections[target_index]
        target_dir = self.output_dir / str(image_id) / target
        target_dir.mkdir(parents=True, exist_ok=True)
        composite = robust_false_color_composite(
            cube_unit_hwc,
            leaf_mask,
            self.false_color_band_indices,
        )
        grid_height = cube_unit_hwc.shape[0] // self.spatial_patch_size[0]
        grid_width = cube_unit_hwc.shape[1] // self.spatial_patch_size[1]
        if grid_height * grid_width != evidence.spatial_scores.shape[1]:
            raise ValueError("Evidence token grid does not match the rendered HSI shape")
        score_grid = _normalize_scores(evidence.spatial_scores[target_index]).reshape(
            grid_height, grid_width
        )
        heatmap = _heatmap_overlay(composite, score_grid, leaf_mask)
        marked = _draw_boxes(
            composite,
            selection.selected_patch_boxes,
            selection.focus_patch_boxes,
            selection.roi_box_xyxy,
        )
        focused_heatmap = _draw_boxes(
            heatmap,
            (),
            selection.focus_patch_boxes,
            selection.roi_box_xyxy,
        )
        zoom = _crop_and_zoom(
            focused_heatmap,
            selection.roi_box_xyxy,
            self.zoom_factor,
        )
        composite_path = target_dir / "false_color_composite.png"
        heatmap_path = target_dir / "counterfactual_heatmap.png"
        marked_path = target_dir / "selected_patches.png"
        zoom_path = target_dir / "focused_zoom.png"
        spectral_path = target_dir / "spectral_token_scores.png"
        bundle_path = target_dir / "evidence_bundle.png"
        composite.save(composite_path)
        heatmap.save(heatmap_path)
        marked.save(marked_path)
        zoom.save(zoom_path)
        _render_spectral_plot(
            evidence.spectral_scores[target_index],
            selection.spectral_indices,
            target,
            spectral_path,
        )
        spectral_image = Image.open(spectral_path).convert("RGB")
        statement = grounded_evidence_text(target, prediction, selection)
        text_card = _text_card(statement, disclaimer, width=700, height=300)
        _compose_bundle(
            [
                ("HSI false-color composite (fixed band indices)", composite),
                (f"{target}: token-counterfactual heatmap", heatmap),
                ("Cyan: selected; yellow: focus; white: ROI", marked),
                (f"Target-specific crop, {self.zoom_factor}x display zoom", zoom),
                ("Spectral-token sensitivity (token index, not nm)", spectral_image),
                ("Grounded model statement", text_card),
            ],
            bundle_path,
        )
        return {
            "grounded_text": statement,
            "false_color_composite": str(composite_path),
            "counterfactual_heatmap": str(heatmap_path),
            "selected_patches": str(marked_path),
            "focused_zoom": str(zoom_path),
            "spectral_token_plot": str(spectral_path),
            "evidence_bundle": str(bundle_path),
        }


def _robust_scale(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = channel[mask]
    if values.size == 0:
        return np.zeros_like(channel, dtype=np.float32)
    low, high = np.percentile(values, (1, 99)).astype(float)
    if high <= low:
        scaled = np.zeros_like(channel, dtype=np.float32)
    else:
        scaled = np.clip((channel - low) / (high - low), 0.0, 1.0)
    scaled[~mask] = 0.0
    return scaled.astype(np.float32)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(clean.max(initial=0.0))
    if maximum <= 0:
        return np.zeros_like(clean)
    return np.clip(clean / maximum, 0.0, 1.0)


def _heatmap_overlay(
    composite: Image.Image,
    score_grid: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    from matplotlib import colormaps
    from PIL import Image

    colors = (colormaps["inferno"](score_grid)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colors, mode="RGB").resize(
        composite.size, Image.Resampling.BILINEAR
    )
    alpha = Image.fromarray(np.round(score_grid * 180).astype(np.uint8)).resize(
        composite.size, Image.Resampling.BILINEAR
    )
    alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
    alpha_array *= np.asarray(mask, dtype=np.float32)
    base = np.asarray(composite, dtype=np.float32)
    heat_array = np.asarray(heat, dtype=np.float32)
    blended = base * (1 - alpha_array[..., None]) + heat_array * alpha_array[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _draw_boxes(
    image: Image.Image,
    selected_boxes: Sequence[Sequence[int]],
    focus_boxes: Sequence[Sequence[int]],
    roi_box: Sequence[int],
) -> Image.Image:
    from PIL import ImageDraw

    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for box in selected_boxes:
        x0, y0, x1, y1 = (int(value) for value in box)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 235, 255), width=1)
    for box in focus_boxes:
        x0, y0, x1, y1 = (int(value) for value in box)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 220, 0), width=2)
    x0, y0, x1, y1 = (int(value) for value in roi_box)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 255, 255), width=1)
    return marked


def _crop_and_zoom(
    image: Image.Image,
    box_xyxy: Sequence[int],
    zoom_factor: int,
) -> Image.Image:
    from PIL import Image

    crop = image.crop(tuple(int(value) for value in box_xyxy))
    return crop.resize(
        (max(1, crop.width * zoom_factor), max(1, crop.height * zoom_factor)),
        Image.Resampling.NEAREST,
    )


def _render_spectral_plot(
    scores: np.ndarray,
    selected_indices: Sequence[int],
    target: str,
    path: Path,
) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    values = np.asarray(scores, dtype=np.float32)
    figure = Figure(figsize=(7, 3), dpi=120)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    indices = np.arange(len(values))
    axis.plot(indices, values, color="#2563eb", linewidth=1.5)
    selected = np.asarray(selected_indices, dtype=int)
    if len(selected):
        axis.scatter(selected, values[selected], color="#dc2626", s=24, zorder=3)
    axis.set_title(f"{target}: counterfactual spectral-token sensitivity")
    axis.set_xlabel("Spectral token index (model space; not wavelength nm)")
    axis.set_ylabel("Absolute output change")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    figure.clear()


def _font(size: int) -> ImageFont.ImageFont:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _text_card(statement: str, disclaimer: str, width: int, height: int) -> Image.Image:
    from PIL import Image, ImageDraw

    card = Image.new("RGB", (width, height), (20, 24, 34))
    draw = ImageDraw.Draw(card)
    body_font = _font(20)
    wrapped = ""
    for size in range(20, 10, -1):
        body_font = _font(size)
        wrapped = _wrap_pixels(draw, statement, body_font, width - 40)
        bounds = draw.multiline_textbbox((0, 0), wrapped, font=body_font, spacing=6)
        if bounds[3] - bounds[1] <= 140:
            break
    draw.multiline_text(
        (20, 20),
        wrapped,
        fill="white",
        font=body_font,
        spacing=6,
    )
    draw.multiline_text(
        (20, 175),
        _wrap_pixels(draw, disclaimer, _font(15), width - 40),
        fill=(180, 190, 205),
        font=_font(15),
        spacing=6,
    )
    return card


def _wrap_pixels(draw: Any, text: str, font: Any, maximum_width: int) -> str:
    """Wrap by rendered pixel width; character counts clip variable-width text."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > maximum_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines)


def _wrap_text(text: str, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    for word in words:
        if line and len(" ".join((*line, word))) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def _compose_bundle(
    panels: Sequence[tuple[str, Image.Image]],
    path: Path,
) -> None:
    from PIL import Image, ImageDraw

    columns = 2
    cell_width, cell_height, label_height = 720, 360, 34
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row, column = divmod(index, columns)
        x0, y0 = column * cell_width, row * cell_height
        available_height = cell_height - label_height
        scale = min(
            cell_width / max(1, image.width),
            available_height / max(1, image.height),
        )
        resized = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.NEAREST,
        )
        px = x0 + (cell_width - resized.width) // 2
        py = y0 + label_height + (available_height - resized.height) // 2
        canvas.paste(resized, (px, py))
        draw.text((x0 + 10, y0 + 8), label, fill="white", font=_font(18))
    canvas.save(path)


# ============================================================================
# 7. NORMALIZATION, INPUT PREPARATION, AND TRUSTED WEIGHT LOADING
# ============================================================================

def inverse_standardize_targets(
    standardized: np.ndarray,
    statistics: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """Use training-fold target statistics, never guessed measurement units."""
    values = np.asarray(standardized, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("Last dimension must contain three regression targets")
    output = np.empty_like(values)
    for index, target in enumerate(REGRESSION_TARGETS):
        mean, std = float(statistics[target]["mean"]), float(statistics[target]["std"])
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError(f"Invalid training statistics for {target}")
        output[..., index] = values[..., index] * std + mean
    return output


def normalize_unit_hsi(
    cube_unit_hwc: np.ndarray,
    leaf_mask: np.ndarray,
    band_mean: np.ndarray,
    band_std: np.ndarray,
) -> np.ndarray:
    """Normalize foreground only; inputs must ALREADY be unit-scaled HWC.

    This is not TIFF loading or mask estimation. Use the same preprocessing
    and fold-specific band statistics as training. Local crops are resized in
    unit space BEFORE calling this function, matching the active pipeline.
    """
    cube = np.asarray(cube_unit_hwc, dtype=np.float32)
    mask = np.asarray(leaf_mask, dtype=bool)
    if cube.ndim != 3 or mask.shape != cube.shape[:2] or not mask.any():
        raise ValueError("Expected HWC cube and matching nonempty foreground mask")
    if not np.isfinite(cube).all() or cube.min() < 0 or cube.max() > 1:
        raise ValueError("HSI must be finite and unit-scaled to [0,1]")
    mean = np.asarray(band_mean, dtype=np.float32)
    std = np.asarray(band_std, dtype=np.float32)
    if mean.shape != (cube.shape[-1],) or std.shape != mean.shape:
        raise ValueError("Band mean/std must be vectors matching the band count")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Band statistics must be finite and std must be positive")
    output = np.zeros_like(cube)
    output[mask] = (cube[mask] - mean[None]) / np.maximum(std[None], 1e-6)
    return output


def prepare_visual_inputs(
    cube_unit_hwc: np.ndarray,
    leaf_mask: np.ndarray,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    roi_boxes: Mapping[str, Sequence[int]],
    primary_target: str = "GENERAL",
    bridge_config: VisualTokenBridgeConfig | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build the seven visual arguments for forward()/generate(), batch size 1.

    roi_boxes maps explicit targets to boxes selected by the grounder. Empty
    mapping means global-only input (e.g. summary). No guessed ROI or implicit
    target routing is performed. GENERAL is not a local phenotype ROI.
    """
    config = bridge_config or VisualTokenBridgeConfig()
    config.validate()
    if primary_target not in config.target_names:
        raise ValueError(f"Unknown primary target: {primary_target}")
    if len(roi_boxes) > config.max_local_views:
        raise ValueError("More ROI targets than available local slots")
    cube = np.asarray(cube_unit_hwc, dtype=np.float32).copy()
    mask = np.asarray(leaf_mask, dtype=bool)
    normalized = normalize_unit_hsi(cube, mask, band_mean, band_std)
    cube[~mask] = 0.0
    height, width, bands = cube.shape
    views = config.max_local_views
    local = np.zeros((views, bands, height, width), dtype=np.float32)
    local_masks = np.zeros((views, height, width), dtype=bool)
    active = np.zeros(views, dtype=bool)
    target_ids = np.full(views, len(config.target_names) - 1, dtype=np.int64)
    roi = np.zeros((views, 4), dtype=np.int64)
    for slot, (target, box) in enumerate(roi_boxes.items()):
        if target not in config.target_names[:-1]:
            raise ValueError(f"Unknown local phenotype target: {target}")
        crop, crop_mask = extract_local_hsi_view(cube, mask, box, (height, width))
        local[slot] = np.moveaxis(
            normalize_unit_hsi(crop, crop_mask, band_mean, band_std), -1, 0
        )
        local_masks[slot] = crop_mask
        active[slot] = True
        target_ids[slot] = config.target_names.index(target)
        roi[slot] = np.asarray(box, dtype=np.int64)
    values = {
        "global_hsi": torch.from_numpy(np.ascontiguousarray(np.moveaxis(normalized, -1, 0)))[None],
        "global_leaf_mask": torch.from_numpy(mask.copy())[None],
        "local_hsi": torch.from_numpy(local)[None],
        "local_leaf_masks": torch.from_numpy(local_masks)[None],
        "local_view_mask": torch.from_numpy(active)[None],
        "local_target_indices": torch.from_numpy(target_ids)[None],
        "primary_target_index": torch.tensor([config.target_names.index(primary_target)]),
        "roi_boxes_xyxy": torch.from_numpy(roi)[None],
    }
    return {name: tensor.to(device) for name, tensor in values.items()}


def prepare_grounded_visual_inputs(
    model: PhytoSpectraVLM,
    cube_unit_hwc: np.ndarray,
    leaf_mask: np.ndarray,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    targets: Sequence[str] = (),
    grounder: TokenCounterfactualGrounder | None = None,
) -> tuple[dict[str, torch.Tensor], EnsembleEvidence]:
    """Inference helper: global HSI -> evidence -> target crop -> VLM tensors.

    Evaluation mode is required. Computation is inference-only and is not a
    differentiable crop selector. Evidence generation can be relatively slow;
    cache it per image in an application. The current summary route uses no ROI.
    """
    if model.training:
        raise ValueError("Call model.eval() before preparing inference evidence")
    if len(targets) != len(set(targets)) or len(targets) > model.bridge.config.max_local_views:
        raise ValueError("Targets must be unique and fit the available local slots")
    if any(target not in TARGETS for target in targets):
        raise ValueError("Unsupported phenotype target")
    vision = model.bridge.vision_tower
    parameter = next(vision.parameters())
    primary = targets[0] if targets else "GENERAL"
    visual = prepare_visual_inputs(
        cube_unit_hwc, leaf_mask, band_mean, band_std, {}, primary,
        model.bridge.config, parameter.device,
    )
    # The trained vision tower is normally float32. Match its dtype explicitly.
    visual["global_hsi"] = visual["global_hsi"].to(parameter.dtype)
    evidence = (grounder or TokenCounterfactualGrounder()).ground_ensemble(
        [vision], visual["global_hsi"], visual["global_leaf_mask"]
    )
    boxes = {
        target: evidence.selections[evidence.targets.index(target)].roi_box_xyxy
        for target in targets
    }
    visual = prepare_visual_inputs(
        cube_unit_hwc, leaf_mask, band_mean, band_std, boxes, primary,
        model.bridge.config, parameter.device,
    )
    for key in ("global_hsi", "local_hsi"):
        visual[key] = visual[key].to(parameter.dtype)
    return visual, evidence


def prepare_question_inputs(
    tokenizer: Any,
    question: str,
    system_prompt: str,
    device: str | torch.device = "cpu",
    max_question_tokens: int = 96,
    max_prompt_tokens: int = 384,
) -> dict[str, torch.Tensor]:
    """Prompt-only batch of one. Pass the SAME system prompt used for alignment.

    No reference answer or symbolic predictions are inserted into the prompt.
    For training use HyperLeafVQACollator, which masks prompt/padding labels.
    """
    if not question.strip() or min(max_question_tokens, max_prompt_tokens) <= 0:
        raise ValueError("Question must be nonempty and token limits positive")
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    text = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    query = tokenizer(question, add_special_tokens=False, return_tensors="pt",
                      truncation=True, max_length=max_question_tokens)
    if text["input_ids"].shape[1] > max_prompt_tokens:
        raise ValueError("Prompt exceeds the configured alignment token limit")
    if query["input_ids"].shape[1] == 0:
        raise ValueError("Question tokenization is empty")
    return {
        "input_ids": text["input_ids"].to(device),
        "attention_mask": text["attention_mask"].to(device),
        "question_input_ids": query["input_ids"].to(device),
        "question_attention_mask": query["attention_mask"].to(device),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_idcv_classification_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[PhytoCore, dict[str, Any]]:
    """Load a TRUSTED local project checkpoint, strictly preserving all keys.

    The existing checkpoint contains Python metadata, hence weights_only=False.
    Never call this function with an untrusted downloaded pickle/checkpoint.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = PhytoCore()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = dict(checkpoint.get("metadata", {}))
    if metadata.get("fertilizer_formulation") != "WEIGHTED_CE":
        raise ValueError("Checkpoint is not the selected Weighted-CE formulation")
    if model.parameter_count() != 1_998_346:
        raise ValueError("Weighted-CE checkpoint parameter count changed")
    return model, metadata


def create_model(
    vision_checkpoint: str | Path,
    qwen_config: QwenAlignmentConfig | None = None,
    bridge_config: VisualTokenBridgeConfig | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> PhytoSpectraVLM:
    """Start NEW bridge/LoRA alignment around a trained, frozen PhytoCore.

    Does not load trained bridge/LoRA weights. For inference on the existing
    model use load_trained_model(). Qwen is local-cache-only by default.
    """
    vision, _ = load_idcv_classification_checkpoint(vision_checkpoint)
    bridge = TargetConditionedGlobalLocalBridge(vision, bridge_config)
    qwen = load_qwen_lora(qwen_config or QwenAlignmentConfig(), dtype)
    return PhytoSpectraVLM(bridge, qwen).to(device)


def load_idcv_development_bundle(
    bundle_dir: str | Path,
    vlm_config: Mapping[str, Any],
    alignment_config: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[PhytoSpectraVLM, Any, dict[str, Any]]:
    """Reload the Fold-0 Weighted-CE + EMA bridge/Qwen-LoRA bundle.

    Stage 30 deliberately references the immutable PhytoCore checkpoint rather
    than copying it. Every referenced file is hash checked before model
    construction so an evaluation cannot silently mix artifacts from two runs.
    """

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ID-CV VLM manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if str(manifest.get("status")) != "PASS":
        raise ValueError("ID-CV VLM bundle does not declare status=PASS")
    if bool(manifest.get("scientific_result", True)):
        raise ValueError("Stage-30 bundle must remain a non-scientific development run")

    _verify_file_record(manifest["visual_bridge_ema"])
    _verify_directory_record(manifest["qwen_lora_ema"])
    _verify_directory_record(manifest["tokenizer"])
    predictor_path = Path(str(manifest["predictor_checkpoint"]))
    if not predictor_path.is_file():
        raise FileNotFoundError(f"Weighted-CE PhytoCore is missing: {predictor_path}")
    if file_sha256(predictor_path) != str(manifest["predictor_checkpoint_sha256"]):
        raise ValueError("Weighted-CE PhytoCore checkpoint hash mismatch")
    vision, predictor_metadata = load_idcv_classification_checkpoint(
        predictor_path, map_location="cpu"
    )
    if predictor_metadata.get("fertilizer_formulation") != "WEIGHTED_CE":
        raise ValueError("Stage-30 VLM does not reference the Weighted-CE predictor")

    qwen_config = QwenAlignmentConfig.from_configs(vlm_config, alignment_config)
    base = AutoModelForCausalLM.from_pretrained(
        qwen_config.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=qwen_config.local_files_only,
        trust_remote_code=qwen_config.trust_remote_code,
        attn_implementation="sdpa",
    )
    base.config.use_cache = True
    language_model = PeftModel.from_pretrained(
        base,
        str(manifest["qwen_lora_ema"]["path"]),
        is_trainable=False,
    )
    bridge = TargetConditionedGlobalLocalBridge(
        vision,
        VisualTokenBridgeConfig.from_config(vlm_config),
    )
    connector_payload = torch.load(
        Path(str(manifest["visual_bridge_ema"]["path"])),
        map_location="cpu",
        weights_only=False,
    )
    if not bool(connector_payload.get("ema_applied", False)):
        raise ValueError("Primary Stage-30 bridge is not marked as EMA")
    expected_bridge_parameters = int(
        manifest["trainable_parameters"]["visual_bridge"]
    )
    if int(connector_payload.get("parameter_count", -1)) != expected_bridge_parameters:
        raise ValueError("EMA visual-bridge parameter count mismatch")
    missing, unexpected = bridge.load_state_dict(
        connector_payload["state_dict"], strict=False
    )
    allowed_missing = {
        name for name in bridge.state_dict() if name.startswith("vision_tower.")
    }
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(
            f"EMA visual bridge state mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(manifest["tokenizer"]["path"]), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = PhytoSpectraVLM(bridge, language_model).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.bridge.vision_tower.eval()
    return model, tokenizer, manifest


def _verify_file_record(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Bundle file is missing: {path}")
    if file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"Bundle file hash mismatch: {path}")


def _verify_directory_record(record: Mapping[str, Any]) -> None:
    root = Path(str(record["path"]))
    if not root.is_dir():
        raise FileNotFoundError(f"Bundle directory is missing: {root}")
    expected = {str(item["path"]): item for item in record["files"]}
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    if set(observed) != set(expected):
        raise ValueError(f"Bundle directory contents changed: {root}")
    for relative, path in observed.items():
        item = expected[relative]
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Bundle file size mismatch: {path}")
        if file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Bundle file hash mismatch: {path}")


def default_vlm_configuration() -> dict[str, Any]:
    """Current architecture values; no project YAML file is needed to import."""
    return {"model": {
        "language_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "vision_dim": 128, "language_dim": 1536,
        "text_adapter_layers": 2, "cross_modal_attention_heads": 4,
        "evidence_resampler_tokens": 16, "evidence_resampler_layers": 2,
        "feedforward_dim": 512, "max_local_views": 2, "dropout": 0.1,
        "target_order": [*TARGETS, "GENERAL"],
        "freeze_phytocore_during_language_training": True,
        "qwen": {"lora": {
            "rank": 8, "alpha": 16, "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }},
    }}


def load_trained_model(
    bundle_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    vlm_config: Mapping[str, Any] | None = None,
    alignment_config: Mapping[str, Any] | None = None,
) -> tuple[PhytoSpectraVLM, Any, dict[str, Any]]:
    """Load the existing trusted Stage-30 EMA bridge + Qwen-LoRA model.

    This explicit, potentially memory-intensive operation verifies the manifest
    hashes, loads frozen PhytoCore, Qwen base and EMA adapters, then sets eval().
    It does NOT train anything and does not substitute random weights.
    Default dtype is float32 for straightforward CPU use. With bf16 on CUDA,
    wrap forward/generate in torch.autocast('cuda', dtype=torch.bfloat16), as
    the original runtime does (vision/bridge weights remain float32).
    Project assets / pretrained Qwen cache are needed, even though the source
    code itself is self-contained. Alternate CV bundle formats are not inferred.
    """
    alignment = alignment_config or {"language": {
        "local_files_only": True, "trust_remote_code": False,
        "gradient_checkpointing": False,
    }}
    return load_idcv_development_bundle(
        bundle_dir, vlm_config or default_vlm_configuration(), alignment,
        torch.device(device), dtype,
    )


# ============================================================================
# 8. SAFE COMMAND-LINE SUMMARY AND CPU SMOKE TEST (NO QWEN DOWNLOAD)
# ============================================================================

def architecture_summary() -> dict[str, Any]:
    """Instantiate only CPU PhytoCore/bridge to verify the parameter counts."""
    vision = PhytoCore()
    bridge = TargetConditionedGlobalLocalBridge(vision)
    spatial = vision.spatial_encoder.n_tokens
    spectral = vision.spectral_encoder.n_tokens
    return {
        "model": "PhytoSpectra-VLM (single-file active architecture)",
        "weights_loaded": False,
        "scientific_result": False,
        "input_hsi": ["B", 204, 48, 352],
        "spectral_tokens": ["B", spectral, 128],
        "spatial_tokens": ["B", spatial, 128],
        "fused_memory": ["B", spatial + spectral, 128],
        "max_global_local_memory": ["B", (spatial + spectral) * 3, 128],
        "visual_prefix": ["B", 16, 1536],
        "regression_targets": list(REGRESSION_TARGETS),
        "fertilizer_classes": 3,
        "cultivar_classes": 4,
        "phytocore_parameters": vision.parameter_count(),
        "trainable_bridge_parameters": bridge.trainable_parameter_count(),
        "language_model": QwenAlignmentConfig().model_name,
        "qwen_lora": asdict(QwenAlignmentConfig()),
        "qwen_loaded": False,
        "note": "Architecture only. Instantiate/load weights explicitly for inference.",
    }


def run_cpu_smoke_test() -> dict[str, Any]:
    """Check actual default vision/bridge shapes + gradients, NOT QA accuracy."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        vision = PhytoCore()
        bridge = TargetConditionedGlobalLocalBridge(vision).train()
        cube = np.random.default_rng(2026).random((48, 352, 204), dtype=np.float32)
        mask = np.zeros((48, 352), dtype=bool)
        mask[11:36, 97:255] = True
        visual = prepare_visual_inputs(
            cube, mask, np.zeros(204, np.float32), np.ones(204, np.float32),
            {"Cultivar": (80, 8, 272, 40)}, primary_target="Cultivar",
        )
        output = bridge(
            **visual, question_embeddings=torch.randn(1, 6, 1536),
            question_attention_mask=torch.ones(1, 6, dtype=torch.long),
            return_attention=True,
        )
        assert output["prefix_embeddings"].shape == (1, 16, 1536)
        assert output["structured_predictions"]["regression"].shape == (1, 3)
        assert output["structured_predictions"]["fertilizer_logits"].shape == (1, 3)
        assert output["structured_predictions"]["cultivar_logits"].shape == (1, 4)
        assert output["visual_token_count"] == 546
        assert output["visual_padding_mask"][:, 364:].all()
        assert torch.isfinite(output["prefix_embeddings"]).all()
        output["prefix_embeddings"].square().mean().backward()
        assert not vision.training
        assert all(p.grad is None and not p.requires_grad for p in vision.parameters())
        assert bridge.language_projector[1].weight.grad.abs().sum() > 0
        assert bridge.question_adapter.project.weight.grad.abs().sum() > 0
    return {
        "status": "PASS", "device": "cpu", "scientific_result": False,
        "random_weights": True, "real_qwen_loaded": False,
        "tested": ["default PhytoCore", "fertilizer classification", "all-band crop",
                   "masked local slots", "16 x 1536 prefix", "bridge gradients",
                   "frozen vision tower"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-file PhytoSpectra-VLM model")
    parser.add_argument("--summary", action="store_true", help="Print architecture summary (default)")
    parser.add_argument("--smoke-test", action="store_true", help="CPU vision/bridge shape and gradient test")
    parser.add_argument("--print-model", action="store_true", help="Also print the vision + bridge layer tree; no Qwen load")
    args = parser.parse_args(argv)
    # Limit CPU use in this explicit CLI; importing the module changes no settings.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(json.dumps(architecture_summary(), indent=2))
    if args.print_model:
        print(TargetConditionedGlobalLocalBridge(PhytoCore()))
        print("Qwen decoder: external pretrained model; load_trained_model() loads actual layers.")
    if args.smoke_test:
        print(json.dumps(run_cpu_smoke_test(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
