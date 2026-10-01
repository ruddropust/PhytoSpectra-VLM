"""
PhytoSpectra-VLM / WheatSpectraChat
Exact port of QPR (Question–Phenotype Router) and Binary Answerability Gate from Notebook 06C.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from app.config import settings
from app.core.constants import (
    TEXT_MODEL_NAME,
    TEXT_DIM,
    ANSWERABILITY_CLASSES,
    INTENT_CLASSES,
    PHENOTYPES,
    ANSWERABILITY_TO_INDEX,
    UNSUPPORTED_PROB_THRESHOLD,
    PHENOTYPE_PROB_THRESHOLD,
    PARAM_COUNTS,
)


class AnswerabilityQPR(nn.Module):
    """
    Binary Answerability Gate + Soft QPR + Open-Set Risk Head (168,205 parameters).
    """
    def __init__(
        self,
        text_dim: int = TEXT_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.answerability_head = nn.Linear(hidden_dim, len(ANSWERABILITY_CLASSES))
        self.intent_head = nn.Linear(hidden_dim, len(INTENT_CLASSES))
        self.phenotype_head = nn.Linear(hidden_dim, len(PHENOTYPES))
        self.unsupported_risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "answerability_logits": self.answerability_head(h),
            "intent_logits": self.intent_head(h),
            "phenotype_logits": self.phenotype_head(h),
            "unsupported_risk_logit": self.unsupported_risk_head(h).squeeze(-1),
        }


class QPRManager:
    def __init__(
        self,
        qpr_dir: Path = settings.QPR_DIR,
        device: torch.device = torch.device("cpu"),
    ):
        self.qpr_dir = qpr_dir
        self.device = device
        self.tokenizer: Optional[AutoTokenizer] = None
        self.text_encoder: Optional[AutoModel] = None
        self.qpr: Optional[AnswerabilityQPR] = None
        self._load()

    def _load(self) -> None:
        # Load frozen text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(self.device)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Load QPR head
        ckpt_path = self.qpr_dir / "answerability_qpr_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing QPR checkpoint: {ckpt_path}")

        self.qpr = AnswerabilityQPR().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.qpr.load_state_dict(state_dict)
        self.qpr.eval()

        # Check parameter count
        param_count = sum(p.numel() for p in self.qpr.parameters())
        if param_count != PARAM_COUNTS["qpr"]:
            raise ValueError(
                f"QPR parameter count mismatch: got {param_count}, expected {PARAM_COUNTS['qpr']}"
            )

    @torch.inference_mode()
    def encode_question(self, question: str) -> torch.Tensor:
        encoded = self.tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        output = self.text_encoder(**encoded)
        hidden = output.last_hidden_state
        attention_mask = encoded["attention_mask"][:, :, None].float()

        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
        pooled = F.normalize(pooled, dim=1)
        return pooled

    @torch.inference_mode()
    def route_question(self, question: str) -> Dict[str, Any]:
        embedding = self.encode_question(question)
        output = self.qpr(embedding)

        answer_prob = torch.softmax(output["answerability_logits"], dim=1)[0].cpu().numpy()
        intent_prob = torch.softmax(output["intent_logits"], dim=1)[0].cpu().numpy()
        phenotype_prob = torch.sigmoid(output["phenotype_logits"])[0].cpu().numpy()
        risk_prob = float(torch.sigmoid(output["unsupported_risk_logit"][0]).cpu())

        unsupported_answer_prob = float(answer_prob[ANSWERABILITY_TO_INDEX["UNSUPPORTED"]])
        open_set_score = float(0.5 * unsupported_answer_prob + 0.5 * risk_prob)

        intent_idx = int(intent_prob.argmax())
        predicted_intent = INTENT_CLASSES[intent_idx]
        predicted_answerability = ANSWERABILITY_CLASSES[int(answer_prob.argmax())]

        selected = [
            PHENOTYPES[i]
            for i, p in enumerate(phenotype_prob)
            if p >= PHENOTYPE_PROB_THRESHOLD
        ]

        if open_set_score >= UNSUPPORTED_PROB_THRESHOLD:
            ebsa_state = "ABSTAIN"
        elif predicted_intent in ["ESTIMATE", "EVIDENCE"] and len(selected) == 1:
            ebsa_state = "VALIDATED"
        else:
            ebsa_state = "EVIDENCE_BOUNDED"

        # If a known estimate/evidence question narrowly misses 0.5 threshold, pick top phenotype
        if len(selected) == 0 and ebsa_state == "VALIDATED":
            selected = [PHENOTYPES[int(phenotype_prob.argmax())]]

        return {
            "question": question,
            "predicted_answerability": predicted_answerability,
            "predicted_intent": predicted_intent,
            "selected_phenotypes": selected,
            "open_set_score": open_set_score,
            "ebsa_state": ebsa_state,
            "unsupported_answer_prob": unsupported_answer_prob,
            "risk_prob": risk_prob,
            "phenotype_probabilities": {
                name: float(phenotype_prob[i])
                for i, name in enumerate(PHENOTYPES)
            },
        }
