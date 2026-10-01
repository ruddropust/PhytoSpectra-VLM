"""
PhytoSpectra-VLM / WheatSpectraChat
Exact port of PGC (Phenotype-Grounded Connector) feature extractor and prefix projector from Notebook 10 / 11C.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from app.config import settings
from app.core.constants import (
    MODE_CLASSES,
    INTENT_CLASSES,
    PHENOTYPES,
    REGRESSION_TARGETS,
    CULTIVAR_NAMES,
    STRENGTH_CLASSES,
    SAMPLE_STATUS_CLASSES,
    CONFOUNDER_CLASSES,
    PGC_FEATURE_DIM,
    PREFIX_TOKENS,
    LLM_HIDDEN_SIZE,
    PARAM_COUNTS,
)


# Feature scalers fitted from development data
REGRESSION_FEATURE_SCALERS = {
    "GrainWeight": {
        "value_mean": 6362.59,
        "value_std": 2115.17,
        "unc_mean": 249.42,
        "unc_std": 60.0,
    },
    "Gsw": {
        "value_mean": 0.0579,
        "value_std": 0.0306,
        "unc_mean": 0.0080,
        "unc_std": 0.0020,
    },
    "PhiPS2": {
        "value_mean": 0.5514,
        "value_std": 0.1063,
        "unc_mean": 0.0171,
        "unc_std": 0.0040,
    },
    "Fertilizer": {
        "value_mean": 0.5786,
        "value_std": 0.4676,
        "unc_mean": 0.0324,
        "unc_std": 0.0080,
    },
}


def one_hot(value: str, classes: List[str]) -> List[float]:
    return [1.0 if value == item else 0.0 for item in classes]


def zscore_val(value: float, mean: float, std: float) -> float:
    return float((float(value) - float(mean)) / max(float(std), 1e-8))


def build_pgc_feature_vector(
    route: Dict[str, Any],
    pgvrc_state: Dict[str, Any],
) -> np.ndarray:
    """
    Builds the exact 175-dimensional scientific feature vector for the PGC Prefix Projector.
    """
    vector: List[float] = []

    # 1. Answer mode one-hot (3)
    vector.extend(one_hot(pgvrc_state.get("final_answer_mode", "ABSTAIN"), MODE_CLASSES))

    # 2. Intent one-hot (5)
    vector.extend(one_hot(route.get("predicted_intent", "UNSUPPORTED"), INTENT_CLASSES))

    # 3. Selected phenotypes binary flags (5)
    selected_phenos = pgvrc_state.get("selected_phenotypes", [])
    vector.extend([1.0 if p in selected_phenos else 0.0 for p in PHENOTYPES])

    # 4. Open-set risk score (1)
    vector.append(float(route.get("open_set_score", 0.0)))

    # 5. Per-phenotype slots (5 phenotypes * (1 flag + 10 measurement + 10 spatial + 10 spectral) = 155)
    predictions = pgvrc_state.get("scientific_predictions", {})
    evidence_dict = pgvrc_state.get("evidence", {})
    verification_dict = pgvrc_state.get("verification", {})
    summary_dict = pgvrc_state.get("verification_summary", {})

    for phenotype in PHENOTYPES:
        selected = phenotype in selected_phenos
        vector.append(1.0 if selected else 0.0)

        # Measurement slot (10 dims)
        if selected and phenotype in REGRESSION_TARGETS and phenotype in predictions:
            pred = predictions[phenotype]
            scaler = REGRESSION_FEATURE_SCALERS[phenotype]
            val = float(pred.get("value", scaler["value_mean"]))
            std = float(pred.get("ensemble_std", scaler["unc_mean"]))

            vector.extend([
                1.0,  # is_regression
                zscore_val(val, scaler["value_mean"], scaler["value_std"]),
                zscore_val(std, scaler["unc_mean"], scaler["unc_std"]),
                0.0,  # interval width z (null in live mode)
            ])
            vector.extend([0.0 for _ in CULTIVAR_NAMES])  # 4
            vector.extend([0.0, 0.0])  # vote_conf, disagreement
        elif selected and phenotype == "Cultivar" and phenotype in predictions:
            pred = predictions[phenotype]
            cult_val = pred.get("value", CULTIVAR_NAMES[0])
            vote_conf = float(pred.get("vote_confidence", 1.0))
            disagree = float(pred.get("seed_disagreement", 0.0))

            vector.extend([0.0, 0.0, 0.0, 0.0])  # 4 regression dims
            vector.extend(one_hot(cult_val, CULTIVAR_NAMES))  # 4 cultivar classes
            vector.extend([vote_conf, disagree])  # 2
        else:
            vector.extend([0.0 for _ in range(10)])

        # Spatial evidence slot (10 dims)
        if selected and phenotype in evidence_dict:
            ev = evidence_dict[phenotype]["spatial"]
            ver = verification_dict.get(phenotype, {}).get("spatial", {})
            summ = summary_dict.get(phenotype, {}).get("spatial", {})

            vector.extend(one_hot(ev.get("validation_strength", "STRONG"), STRENGTH_CLASSES))  # 3
            vector.extend(one_hot(summ.get("sample_status", "VERIFIED"), SAMPLE_STATUS_CLASSES))  # 4
            vector.extend([
                float(ev.get("budget", 0.1)),
                float(np.tanh(ver.get("comprehensiveness_gain", 0.1))),
                float(np.tanh(ver.get("sufficiency_advantage", 0.1))),
            ])  # 3
        else:
            vector.extend([0.0 for _ in range(10)])

        # Spectral evidence slot (10 dims)
        if selected and phenotype in evidence_dict:
            ev = evidence_dict[phenotype]["spectral"]
            ver = verification_dict.get(phenotype, {}).get("spectral", {})
            summ = summary_dict.get(phenotype, {}).get("spectral", {})

            vector.extend(one_hot(ev.get("validation_strength", "STRONG"), STRENGTH_CLASSES))  # 3
            vector.extend(one_hot(summ.get("sample_status", "VERIFIED"), SAMPLE_STATUS_CLASSES))  # 4
            vector.extend([
                float(ev.get("budget", 0.3)),
                float(np.tanh(ver.get("comprehensiveness_gain", 0.05))),
                float(np.tanh(ver.get("sufficiency_advantage", 0.05))),
            ])  # 3
        else:
            vector.extend([0.0 for _ in range(10)])

    # 6. Confounder slot (6 dims: 4 risk classes + 2 Jaccard overlaps)
    confounder = pgvrc_state.get("confounder", {})
    risk = confounder.get("risk", "LOW")
    vector.extend(one_hot(risk, CONFOUNDER_CLASSES))  # 4
    vector.extend([
        float(confounder.get("spatial_token_jaccard", 0.0)),
        float(confounder.get("spectral_token_jaccard", 0.0)),
    ])  # 2

    array = np.asarray(vector, dtype=np.float32)
    if len(array) != PGC_FEATURE_DIM:
        raise ValueError(
            f"PGC feature vector dimension mismatch: got {len(array)}, expected {PGC_FEATURE_DIM}"
        )
    return array


class PGCPrefixProjector(nn.Module):
    """
    PGC Prefix Projector: projects 175-dim feature vector to 8 prefix tokens (1536 dim).
    Parameters: 9,584,990.
    """
    def __init__(
        self,
        input_dim: int = PGC_FEATURE_DIM,
        hidden_size: int = LLM_HIDDEN_SIZE,
        prefix_tokens: int = PREFIX_TOKENS,
    ):
        super().__init__()
        connector_hidden = min(1024, max(256, hidden_size // 2))
        self.prefix_tokens = int(prefix_tokens)
        self.hidden_size = int(hidden_size)

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, connector_hidden),
            nn.GELU(),
            nn.Linear(connector_hidden, prefix_tokens * hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.net(features.float())
        return output.reshape(features.shape[0], self.prefix_tokens, self.hidden_size)


class PGCManager:
    def __init__(
        self,
        alignment_dir: Path = settings.LANGUAGE_ALIGNMENT_DIR,
        device: torch.device = torch.device("cpu"),
    ):
        self.alignment_dir = alignment_dir
        self.device = device
        self.projector: Optional[PGCPrefixProjector] = None
        self._load()

    def _load(self) -> None:
        ckpt_path = self.alignment_dir / "pgc_prefix_projector.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing PGC prefix projector checkpoint: {ckpt_path}")

        self.projector = PGCPrefixProjector().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.projector.load_state_dict(ckpt["state_dict"])
        self.projector.eval()

        param_count = sum(p.numel() for p in self.projector.parameters())
        if param_count != PARAM_COUNTS["pgc_projector"]:
            raise ValueError(
                f"PGC parameter count mismatch: got {param_count}, expected {PARAM_COUNTS['pgc_projector']}"
            )

    @torch.inference_mode()
    def project_prefix_tokens(self, feature_vector: np.ndarray) -> torch.Tensor:
        """
        Projects [175] or [B, 175] feature vector to [B, 8, 1536] prefix embeddings.
        """
        if feature_vector.ndim == 1:
            tensor = torch.from_numpy(feature_vector).unsqueeze(0).to(self.device)
        else:
            tensor = torch.from_numpy(feature_vector).to(self.device)
        return self.projector(tensor)
