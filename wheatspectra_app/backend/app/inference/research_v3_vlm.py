"""Adapter that keeps the original chatbot contract on the research_v3 model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open

from app.inference.idcv_development_vlm import IDCVPreparedImageState
from app.inference.v3_query import QueryPlan
from app.inference.v3_runtime import Runtime
from phytospectra.model import normalize_unit_hsi, robust_false_color_composite

TARGETS = ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResearchV3VLMAdapter:
    """Present one evaluated seed/fold through the established app interface."""

    def __init__(self, *, device: torch.device, eager_load: bool = False) -> None:
        self.runtime = Runtime()
        self.runtime.device = device
        self.verification_error: str | None = None
        self.manifest: dict[str, Any] = {}
        self.run_report: dict[str, Any] = {}
        self.learned_parameters: dict[str, int] = {"phytocore": 1_998_346}
        try:
            config = self.runtime.config
            manifest_path = self.runtime.bundle / "manifest.json"
            report_path = config.root / "reports" / self.runtime.run_id / "runs" / f"seed_{self.runtime.seed}_fold_{self.runtime.fold}" / "report.json"
            aggregate_path = config.root / "reports" / self.runtime.run_id / "campaign" / "aggregate_report.json"
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.run_report = json.loads(report_path.read_text(encoding="utf-8"))
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if self.manifest.get("status") != "PASS" or self.run_report.get("status") != "PASS":
                raise ValueError("Selected run or VLM bundle did not pass")
            if (int(self.manifest.get("seed", -1)), int(self.manifest.get("fold", -1))) != (self.runtime.seed, self.runtime.fold):
                raise ValueError("VLM bundle seed/fold differs from selected model")
            if (int(self.run_report.get("seed", -1)), int(self.run_report.get("fold", -1))) != (self.runtime.seed, self.runtime.fold):
                raise ValueError("Evaluation report seed/fold differs from selected model")
            if aggregate.get("status") != "PASS" or int(aggregate.get("completed_runs", 0)) != 15:
                raise ValueError("The 3×5 campaign aggregate is not complete")
            checkpoint = self.manifest["predictor_checkpoint"]
            bridge = self.manifest["visual_bridge_ema"]
            if (config.root / checkpoint["path"]).resolve() != self.runtime.checkpoint:
                raise ValueError("VLM bundle is bound to another PhytoCore checkpoint")
            if _sha256(self.runtime.checkpoint) != checkpoint["sha256"]:
                raise ValueError("PhytoCore checkpoint hash changed")
            if _sha256(config.root / bridge["path"]) != bridge["sha256"]:
                raise ValueError("Visual bridge checkpoint hash changed")
            if not (config.root / self.manifest["qwen_lora_ema"]).is_dir():
                raise ValueError("Qwen LoRA adapter is missing")
            if not (config.root / self.manifest["tokenizer"]).is_dir():
                raise ValueError("VLM tokenizer is missing")
            bridge_state = torch.load(config.root / bridge["path"], map_location="cpu", weights_only=False)["state_dict"]
            self.learned_parameters["visual_bridge"] = sum(
                value.numel() for name, value in bridge_state.items()
                if not name.startswith("vision_tower.")
            )
            lora_path = config.root / self.manifest["qwen_lora_ema"] / "adapter_model.safetensors"
            with safe_open(str(lora_path), framework="pt", device="cpu") as weights:
                self.learned_parameters["qwen_lora"] = sum(
                    int(np.prod(weights.get_slice(key).get_shape())) for key in weights.keys()
                )
            if eager_load:
                self.runtime._load()
                self.runtime._load_vlm()
        except Exception as exc:
            self.verification_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        verified = self.verification_error is None
        return {
            "enabled": True,
            "verified": verified,
            "loaded": self.runtime.vlm is not None,
            "scientific_result": verified,
            "activation_allowed": verified,
            "deployment_status": "RESEARCH_V3_STAGE42D_EXACT" if verified else "ERROR",
            "bundle_dir": str(self.runtime.bundle),
            "phytocore_path": str(self.runtime.checkpoint),
            "outer_fold": self.runtime.fold,
            "model_seed": self.runtime.seed,
            "verification_error": self.verification_error,
            "ema_weights": True,
            "vision_frozen": True,
            "outer_evaluation_verified": verified,
            "campaign_runs_passed": 15 if verified else 0,
            "learned_parameters": self.learned_parameters,
            "total_learned_parameters": sum(self.learned_parameters.values()),
            "architecture": "research_v3 PhytoCore + target/question-conditioned global-local bridge + Qwen2.5 LoRA + symbolic verifier",
            "scope_warning": "Showing one evaluated seed/fold model, not a full-data model or a 15-model ensemble. Validation was internal image-wise CV only.",
        }

    @staticmethod
    def _visuals(cube: np.ndarray, mask: np.ndarray, selections: dict[str, Any], folder: Path) -> dict[str, dict[str, Any]]:
        folder.mkdir(parents=True, exist_ok=True)
        composite = robust_false_color_composite(cube, mask)
        composite.save(folder / "hsi_composite.png")
        result: dict[str, dict[str, Any]] = {}
        for target in TARGETS:
            selection = selections[target]
            box = selection["selected_box"]
            result[target] = {
                "selected_patches": None,
                "crop": None,
                "focused_zoom": None,
                "grounded_text": (
                    "No informative local ROI was selected by the frozen model."
                    if box is None else
                    "Highlighted region is PhytoCore grouped-token sensitivity, not biological or causal evidence."
                ),
            }
            if box is None:
                continue
            x0, y0, x1, y1 = map(int, box)
            marked = composite.copy()
            ImageDraw.Draw(marked).rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 220, 0), width=2)
            stem = target.lower()
            marked_path = folder / f"{stem}_marked.png"
            crop_path = folder / f"{stem}_crop.png"
            zoom_path = folder / f"{stem}_zoom.png"
            marked.save(marked_path)
            crop = composite.crop((x0, y0, x1, y1))
            crop.save(crop_path)
            crop.resize((max(1, crop.width * 4), max(1, crop.height * 4)), Image.Resampling.NEAREST).save(zoom_path)
            result[target].update(selected_patches=str(marked_path), crop=str(crop_path), focused_zoom=str(zoom_path))
        return result

    @torch.inference_mode()
    def prepare_image(self, *, cube_unit_hwc: np.ndarray, leaf_mask: np.ndarray, image_id: str, media_dir: Path) -> IDCVPreparedImageState:
        if self.verification_error:
            raise RuntimeError(self.verification_error)
        cube = np.asarray(cube_unit_hwc, dtype=np.float32)
        mask = np.asarray(leaf_mask, dtype=bool)
        if cube.shape != (48, 352, 204) or mask.shape != (48, 352) or not mask.any():
            raise ValueError("Expected a non-empty 48×352×204 HSI and matching leaf mask")
        # Upload IDs are random session keys; candidate sampling must instead
        # be stable for the same hyperspectral cube across repeated uploads.
        candidate_id = hashlib.sha256(np.ascontiguousarray(cube).tobytes()).hexdigest()
        state = self.runtime.prepare_array(cube, mask, image_id, candidate_id=candidate_id)
        folder = Path(media_dir).resolve() / image_id / "research_v3_stage42d"
        visuals = self._visuals(cube, mask, state.selections, folder / "visuals")
        exposed = {
            target: {
                "roi_box_xyxy": selection["selected_box"],
                "spatial_indices": [],
                "spectral_indices": [],
                "candidate_effects": selection["candidate_effects"],
                "acceptable_slots": selection["acceptable_slots"],
                "status": selection["status"],
                "selection_source": selection["selection_source"],
            }
            for target, selection in state.selections.items()
        }
        record = folder / "evidence.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({
            "image_id": image_id,
            "run_id": self.runtime.run_id,
            "seed": self.runtime.seed,
            "fold": self.runtime.fold,
            "predictions": state.predictions,
            "selections": state.selections,
        }, indent=2), encoding="utf-8")
        normalized = normalize_unit_hsi(cube, mask, self.runtime.normalization["mean"], self.runtime.normalization["std"])
        return IDCVPreparedImageState(
            image_id=image_id, cube_unit_hwc=cube.copy(), leaf_mask=mask.copy(),
            normalized=normalized, predictions=state.predictions,
            evidence_selections=exposed, spatial_scores=np.asarray([state.selections[t]["candidate_effects"] for t in TARGETS], dtype=np.float32),
            spectral_scores=np.empty((0,), dtype=np.float32), evidence_record_path=str(record), arrays_path="",
            visuals=visuals, media_dir=str(folder), composite_path=str(folder / "visuals" / "hsi_composite.png"),
        )

    @staticmethod
    def _value_text(target: str, prediction: dict[str, Any]) -> str:
        value = prediction["value"]
        if prediction["task"] == "classification":
            return str(value)
        return f"{float(value):.2f}" if target == "GrainWeight" else f"{float(value):.4f}"

    @torch.inference_mode()
    def answer_question(self, *, prepared_state: IDCVPreparedImageState, question: str, intent: str, targets: list[str], max_new_tokens: int = 128) -> dict[str, Any]:
        if self.verification_error:
            raise RuntimeError(self.verification_error)
        intent = str(intent).upper()
        if intent not in {"TARGET_VALUE", "EVIDENCE_REQUEST", "IMAGE_SUMMARY"}:
            raise ValueError(f"Unsupported intent: {intent}")
        requested = list(TARGETS) if intent == "IMAGE_SUMMARY" else list(targets)
        if not requested or any(target not in TARGETS for target in requested):
            raise ValueError("No supported target was routed")
        state = self.runtime.images.get(prepared_state.image_id)
        if state is None:
            raise RuntimeError("Prepared image state is not available")
        plan = QueryPlan(intent=intent, targets=requested, include_prediction=True, include_evidence=intent == "EVIDENCE_REQUEST")
        verification = self.runtime._generate_plan(state, question, plan)
        if not verification.get("attempted"):
            raise RuntimeError("The selected VLM bundle did not generate a plan")
        lines = []
        for target in requested:
            prediction = prepared_state.predictions[target]
            line = f"{target}: {self._value_text(target, prediction)}"
            if intent == "EVIDENCE_REQUEST":
                selection = state.selections[target]
                line += (
                    ". No informative local ROI was selected."
                    if selection["selected_box"] is None else
                    f". Selected model-sensitive ROI: {selection['selected_box']}."
                )
            lines.append(line)
        answer = "\n".join(lines)
        if intent == "EVIDENCE_REQUEST":
            answer += "\nThe ROI reflects grouped-token sensitivity of PhytoCore, not biological causality."
        return {
            "status": "PASS", "scientific_result": True,
            "scientific_result_reason": "Single model from the completed 15-run internal image-wise campaign; not external validation.",
            "image_id": prepared_state.image_id, "question": question, "intent": intent,
            "targets": requested, "answer": answer,
            "raw_answer_accepted": bool(verification.get("raw_accepted")),
            "fallback_used": bool(verification.get("fallback_used")),
            "verification_reasons": [] if verification.get("raw_accepted") else ["raw_plan_not_accepted"],
            "predictions": prepared_state.predictions,
            "evidence": {target: prepared_state.evidence_selections[target] for target in requested},
            "visuals": {target: prepared_state.visuals[target] for target in requested},
            "evidence_record": prepared_state.evidence_record_path,
            "ema_weights": True, "vision_frozen": True,
            "grounding_source": "DIRECT_PHYTOCORE_TOKEN_SENSITIVITY",
            "answer_source": "PHYTOCORE_STRUCTURED_PREDICTION",
            "contract_decoding_layer": "VERIFIER_RERANKED_NBEST",
        }
