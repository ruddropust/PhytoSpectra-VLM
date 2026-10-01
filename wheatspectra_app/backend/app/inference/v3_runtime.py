"""Lazy scientific runtime backed only by newly generated v3 artifacts."""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phytospectra.config import load_config
from phytospectra.data import CULTIVAR_CLASSES, FERTILIZER_CLASSES, load_cube, load_normalization
from phytospectra.evidence import _stage42d_evidence, stage42d_direct_selections
from phytospectra.contract import expected_plan, verify_plan
from phytospectra.contract_optimization import canonical_json, score_legal_plans
from phytospectra.model import HyperLeafVQACollator, VisualTokenBridgeConfig, normalize_unit_hsi, prepare_visual_inputs, summarize_predictions
from phytospectra.training import load_phytocore_checkpoint
from phytospectra.vlm import language_config, load_vlm_bundle

from .v3_query import ConversationContext, DEFINITIONS, QueryPlan, plan_query


@dataclass
class ImageState:
    image_id: str
    cube: np.ndarray
    mask: np.ndarray
    evidence: Any
    selections: dict[str, dict[str, Any]]
    predictions: dict[str, Any]


class InvalidImageError(ValueError):
    """The uploaded TIFF cannot be used as a HyperLeaf hyperspectral cube."""


class Runtime:
    def __init__(self) -> None:
        root = Path(os.environ.get("PHYTOSPECTRA_ROOT", Path(__file__).resolve().parents[4])).resolve()
        self.run_id = os.environ.get("PHYTOSPECTRA_RUN_ID", "stage42d_exact_v1")
        self.seed = int(os.environ.get("PHYTOSPECTRA_SEED", "2026"))
        self.fold = int(os.environ.get("PHYTOSPECTRA_FOLD", "0"))
        raw_data = Path(os.environ.get("PHYTOSPECTRA_RAW_DATA", root.parent / "data" / "raw"))
        self.config = load_config(root / "configs/experiment.yaml", root=root, raw_data=raw_data, run_id=self.run_id, seed=self.seed, fold=self.fold)
        self.checkpoint = self.config.run_artifacts / "phytocore" / "phytocore_final.pt"
        self.bundle = self.config.run_artifacts / "vlm" / "contract" / "bundle"
        self.normalization_dir = root / "artifacts" / self.run_id / "normalization"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vlm = None
        self.tokenizer = None
        self.normalization = None
        self.images: dict[str, ImageState] = {}
        self.contexts: dict[str, ConversationContext] = {}

    def status(self) -> dict[str, Any]:
        return {"ready": self.checkpoint.is_file(), "vlm_ready": (self.bundle / "manifest.json").is_file(), "loaded": self.model is not None, "vlm_loaded": self.vlm is not None, "seed": self.seed, "fold": self.fold, "checkpoint": str(self.checkpoint), "architecture": "PhytoCore + global/local target/question bridge + Qwen2.5 LoRA + verifier"}

    def _load(self) -> None:
        if self.model is None:
            self.model, _ = load_phytocore_checkpoint(self.checkpoint, self.device)
            self.normalization = load_normalization(self.normalization_dir, self.fold)

    def _load_vlm(self) -> None:
        if self.vlm is None and (self.bundle / "manifest.json").is_file():
            self.vlm, self.tokenizer, _ = load_vlm_bundle(self.config, self.bundle, self.device)

    @torch.inference_mode()
    def _generate_plan(self, state: ImageState, question: str, plan: QueryPlan) -> dict[str, Any]:
        """Use the trained VLM when available; always fail closed to the legal plan."""
        self._load_vlm()
        if self.vlm is None or not plan.include_prediction:
            return {"attempted": False, "accepted": None, "fallback_used": False, "raw_answer": None}
        intent = "IMAGE_SUMMARY" if plan.intent == "IMAGE_SUMMARY" else "EVIDENCE_REQUEST" if plan.include_evidence else "TARGET_VALUE"
        chunks = [plan.targets] if intent == "IMAGE_SUMMARY" else [plan.targets[index:index + 2] for index in range(0, len(plan.targets), 2)]
        raw_answers, accepted, nbest_recovered = [], [], []
        collator = HyperLeafVQACollator(self.tokenizer, language_config(self.config))
        for index, targets in enumerate(chunks):
            boxes = {} if intent == "IMAGE_SUMMARY" else {
                target: selection["selected_box"]
                for target in targets
                if (selection := state.selections[target])["selected_box"] is not None
            }
            visual = prepare_visual_inputs(
                state.cube, state.mask, self.normalization["mean"], self.normalization["std"],
                boxes, targets[0] if targets else "GENERAL", VisualTokenBridgeConfig(), "cpu",
            )
            sample = {key: value.squeeze(0) for key, value in visual.items()}
            sample["outer_fold"] = torch.tensor(self.fold, dtype=torch.long)
            evidence_payload = {"predictions": state.predictions}
            sample.update({
                "record_id": f"live-{index}", "image_id": state.image_id,
                "question": question, "intent": intent, "targets": list(targets),
                "answer": canonical_json(expected_plan(intent, list(targets), evidence_payload)),
                "answer_mode": "REFERENCE_PLAN",
            })
            batch = collator.for_generation([sample])
            tensors = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            generated = self.vlm.generate(**{key: tensors[key] for key in (
                "global_hsi", "global_leaf_mask", "local_hsi", "local_leaf_masks", "local_view_mask",
                "local_target_indices", "primary_target_index", "roi_boxes_xyxy", "input_ids", "attention_mask",
                "question_input_ids", "question_attention_mask",
            )}, max_new_tokens=int(self.config.values["evaluation"]["max_new_tokens"]), pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            check = verify_plan(raw, intent, list(targets), evidence_payload)
            recovered = False
            if not check["accepted"]:
                ranked = score_legal_plans(self.vlm, collator, sample, self.device)
                k = int(self.config.values["training"]["contract_optimization"]["nbest_k"])
                expected = expected_plan(intent, list(targets), evidence_payload)
                recovered = any(row["plan"] == expected for row in ranked[:k])
            raw_answers.append(raw); accepted.append(bool(check["accepted"])); nbest_recovered.append(recovered)
        final_valid = all(raw or recovered for raw, recovered in zip(accepted, nbest_recovered))
        return {
            "attempted": True,
            "accepted": final_valid,
            "raw_accepted": all(accepted),
            "nbest_recovered": any(nbest_recovered),
            "fallback_used": not final_valid,
            "raw_answer": raw_answers,
        }

    def prepare_image(self, path: Path) -> ImageState:
        from phytospectra.data import create_leaf_mask
        try:
            cube = load_cube(path)
            mask = create_leaf_mask(cube)
        except (OSError, TypeError, ValueError) as exc:
            raise InvalidImageError(str(exc)) from exc
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return self.prepare_array(cube, mask, digest.hexdigest())

    def prepare_array(self, cube: np.ndarray, mask: np.ndarray, image_id: str, candidate_id: str | None = None) -> ImageState:
        """Prepare an already canonicalised upload without writing another TIFF."""
        self._load()
        normalized = normalize_unit_hsi(cube, mask, self.normalization["mean"], self.normalization["std"])
        hsi = torch.from_numpy(np.moveaxis(normalized, -1, 0).copy())[None].to(self.device)
        mask_tensor = torch.from_numpy(mask.copy())[None].to(self.device)
        candidate_seed = int(self.config.values["evidence"]["candidate_seed"]) + self.fold
        try:
            evidence, boxes = _stage42d_evidence(
                self.model, hsi, mask_tensor, mask,
                image_id=candidate_id or image_id, seed=candidate_seed,
            )
        except ValueError as exc:
            raise InvalidImageError(str(exc)) from exc
        selections = stage42d_direct_selections(boxes, evidence.spatial_scores)
        predictions = summarize_predictions(evidence, self.normalization["targets"], FERTILIZER_CLASSES, CULTIVAR_CLASSES)
        # The evidence tensor is only needed for the prediction above; keeping
        # it in the session would retain GPU memory for every uploaded image.
        state = ImageState(str(image_id), cube, mask, None, selections, predictions)
        self.images[state.image_id] = state
        return state

    def answer(self, session_id: str, image_id: str | None, question: str) -> dict[str, Any]:
        plan = plan_query(question, self.contexts.get(session_id))
        if plan.targets:
            self.contexts[session_id] = ConversationContext(plan.targets, plan.intent)
        if plan.intent == "UNSUPPORTED":
            return {"answer": "The current model does not provide validated information for that request.", "intent": plan.intent, "targets": [], "predictions": {}}
        if image_id is None and plan.include_prediction:
            return {"answer": "This question requires an uploaded wheat hyperspectral TIFF image.", "intent": plan.intent, "targets": plan.targets, "predictions": {}, "image_required": True}
        state = self.images.get(image_id) if image_id else None
        paragraphs = []
        if plan.include_definition:
            paragraphs.extend(DEFINITIONS[target] for target in plan.targets)
        selected = {target: state.predictions[target] for target in plan.targets} if state else {}
        verification = self._generate_plan(state, question, plan) if state else {"attempted": False, "accepted": None, "fallback_used": False, "raw_answer": None}
        if plan.include_prediction:
            for target, value in selected.items():
                suffix = f" with {value['confidence']:.1%} model probability" if value["task"] == "classification" else ""
                paragraphs.append(f"The model predicts {target} = {value['value']}{suffix} for this image.")
        evidence_payload = {}
        if plan.include_evidence:
            for target in plan.targets:
                selection = state.selections[target]
                evidence_payload[target] = {
                    "roi_box_xyxy": selection["selected_box"],
                    "status": selection["status"],
                    "candidate_effects": selection["candidate_effects"],
                    "interpretation": "PhytoCore grouped-token sensitivity; not biological causality.",
                }
        if "difference" in question.casefold() and {"Cultivar", "Fertilizer"}.issubset(plan.targets):
            paragraphs.insert(2, "Difference: Cultivar identifies the wheat variety, whereas Fertilizer identifies the cultivation treatment class.")
        return {"answer": "\n\n".join(paragraphs), "intent": plan.intent, "targets": plan.targets, "predictions": selected, "evidence": evidence_payload, "verification": verification}

