"""Integrity-bound live adapter for the frozen Stage-40S Fold-0 VLM.

This is the exact post-remediation inference path used by the Stage-40U/40V
audits: a frozen weighted-CE PhytoCore, direct candidate memories, the frozen
global/local bridge, Qwen LoRA/EMA, and a symbolic-reference verifier.  The
runtime is intentionally labelled as an internal Fold-0 development model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from app.inference.idcv_development_vlm import IDCVPreparedImageState


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TARGETS = ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar")


def _ensure_project_import() -> None:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class FrozenStage40VLMAdapter:
    """Prepare an uploaded HSI once and answer many verified questions.

    Startup verifies both the Stage-40T architecture freeze and the Stage-40Y
    final-protocol freeze. Model weights remain lazy until the first upload.
    """

    def __init__(
        self,
        *,
        bundle_dir: Path,
        phytocore_path: Path,
        architecture_freeze_manifest: Path,
        protocol_freeze_manifest: Path,
        device: torch.device,
        eager_load: bool = False,
        debug_raw_qwen: bool = False,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        self.phytocore_path = Path(phytocore_path).resolve()
        self.architecture_freeze_manifest = Path(
            architecture_freeze_manifest
        ).resolve()
        self.protocol_freeze_manifest = Path(protocol_freeze_manifest).resolve()
        self.device = device
        self.debug_raw_qwen = bool(debug_raw_qwen)
        self._runtime: Optional[Any] = None
        self._bundle: Dict[str, Any] = {}
        self._architecture_freeze: Dict[str, Any] = {}
        self._protocol_freeze: Dict[str, Any] = {}
        self._verification_error: Optional[str] = None
        self._verify_frozen_runtime()
        if self._verification_error is not None:
            raise RuntimeError(self._verification_error)
        if eager_load:
            self._load_runtime()

    def _verify_frozen_runtime(self) -> None:
        try:
            _ensure_project_import()
            from research_v2.deployment.final_3x5_protocol_freeze import (
                verify_manifest,
            )
            from research_v2.deployment.vlm_architecture_freeze import (
                read_json,
                verify_freeze,
            )
            from research_v2.evaluation import direct_evidence_development as direct
            from research_v2.training.checkpointing import file_sha256

            architecture_check = verify_freeze(
                root=_PROJECT_ROOT,
                manifest_path=self.architecture_freeze_manifest,
            )
            if architecture_check.get("status") != "PASS":
                raise ValueError(
                    f"Stage-40T architecture freeze failed: "
                    f"{architecture_check.get('failures', [])}"
                )
            protocol_check = verify_manifest(
                _PROJECT_ROOT, self.protocol_freeze_manifest
            )
            if protocol_check.get("status") != "PASS":
                raise ValueError(
                    f"Stage-40Y protocol freeze failed: "
                    f"{protocol_check.get('failures', [])}"
                )

            self._architecture_freeze = read_json(
                self.architecture_freeze_manifest
            )
            self._protocol_freeze = read_json(self.protocol_freeze_manifest)
            contract = self._architecture_freeze["runtime_contract"]
            frozen_bundle = (
                _PROJECT_ROOT / str(contract["selected_bundle"])
            ).resolve()
            if frozen_bundle != self.bundle_dir:
                raise ValueError(
                    "Configured bundle is not the Stage-40T selected bundle"
                )
            if file_sha256(self.bundle_dir / "manifest.json") != str(
                contract["selected_bundle_manifest_sha256"]
            ):
                raise ValueError("Frozen Stage-40S bundle manifest hash changed")

            self._bundle = direct.check_bundle(self.bundle_dir)
            bundle_predictor = Path(
                str(self._bundle["predictor_checkpoint"])
            ).resolve()
            if bundle_predictor != self.phytocore_path:
                raise ValueError(
                    "Configured PhytoCore is not the predictor bound to Stage-40S"
                )
            if file_sha256(self.phytocore_path) != str(
                contract["predictor_checkpoint_sha256"]
            ):
                raise ValueError("Frozen PhytoCore checkpoint hash changed")
            if float(self._bundle["interpolation"]["alpha"]) != 0.6:
                raise ValueError("Frozen interpolation alpha must remain 0.6")
            if str(self._bundle.get("output_contract")) != str(
                contract["output_contract"]
            ):
                raise ValueError("Frozen symbolic output contract changed")
        except Exception as exc:
            self._verification_error = f"{type(exc).__name__}: {exc}"

    def _load_runtime(self) -> None:
        if self._runtime is not None:
            return
        if self._verification_error is not None:
            raise RuntimeError(self._verification_error)
        _ensure_project_import()
        from research_v2.evaluation.vlm_bundle import (
            load_idcv_development_bundle,
        )
        from research_v2.qa.collator import HyperLeafVQACollator
        from research_v2.training.idcv_classification import (
            IDCVClassificationConfig,
        )
        from research_v2.training.legacy_idcv import load_fold_assets

        # The Stage-40S manifest contains the frozen model settings, including
        # ROI geometry. Loading a mutable repository YAML here would silently
        # reconstruct a different bridge, so use only the bound settings.
        model, tokenizer, _ = load_idcv_development_bundle(
            self.bundle_dir,
            self._bundle["settings"]["model"],
            self._bundle["settings"]["alignment"],
            self.device,
            torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        training = IDCVClassificationConfig(root=_PROJECT_ROOT, seed=2026, fold=0)
        _frame, mean, std, target_statistics, _augmentation = load_fold_assets(
            training.legacy
        )
        runtime = SimpleNamespace(
            model=model,
            tokenizer=tokenizer,
            collator=HyperLeafVQACollator(
                tokenizer,
                self._bundle["settings"]["alignment"]["language"],
            ),
            band_mean=np.asarray(mean, dtype=np.float32),
            band_std=np.asarray(std, dtype=np.float32),
            target_statistics=target_statistics,
        )
        observed_commit = getattr(
            runtime.model.language_model.config, "_commit_hash", None
        )
        expected_commit = self._architecture_freeze["runtime_contract"][
            "qwen_base_commit"
        ]
        if observed_commit != expected_commit:
            raise RuntimeError(
                "Loaded Qwen base revision differs from the frozen Stage-40T contract"
            )
        self._runtime = runtime

    def _ensure_runtime(self) -> Any:
        self._load_runtime()
        return self._runtime

    def _checkpoint_sha256(self) -> str:
        return str(
            self._architecture_freeze["runtime_contract"]
            ["predictor_checkpoint_sha256"]
        )

    def _runtime_label(self) -> str:
        return "FROZEN_STAGE40S_FOLD0_INTERNAL"

    def _media_subdirectory(self) -> str:
        return "stage40_frozen"

    def _is_scientific_result(self) -> bool:
        return False

    def _scientific_result_reason(self) -> str:
        return (
            "Frozen single-fold internal runtime; the predeclared 3-seed "
            "x 5-fold experiment has not yet been run."
        )

    @torch.inference_mode()
    def prepare_image(
        self,
        *,
        cube_unit_hwc: np.ndarray,
        leaf_mask: np.ndarray,
        image_id: str,
        media_dir: Path,
    ) -> IDCVPreparedImageState:
        """Create the exact direct-memory representation used after Stage 40N."""

        runtime = self._ensure_runtime()
        _ensure_project_import()
        from research_v2.data.preprocessing import normalize_cube
        from research_v2.evaluation.counterfactual_evidence import (
            TokenCounterfactualGrounder,
            summarize_predictions,
        )
        from research_v2.evaluation.generalization_audit import (
            legacy_encoder_precision,
        )
        from research_v2.inference.direct_evidence import (
            DISCLAIMER,
            direct_selections,
            render_assets,
            validate_evidence,
        )
        from research_v2.inference.idcv_live_runtime import _validate_inputs
        from research_v2.models.roi_cropper import extract_local_hsi_view
        from research_v2.qa.candidate_grounding import candidate_boxes
        from research_v2.training.full_grounding import atomic_checkpoint, atomic_json

        cube, mask = _validate_inputs(cube_unit_hwc, leaf_mask)
        predictor = runtime.model.bridge.vision_tower
        mean, std = runtime.band_mean, runtime.band_std
        boxes = candidate_boxes(mask, image_id=str(image_id), seed=403026)

        def tensor_view(view_cube: np.ndarray, view_mask: np.ndarray) -> torch.Tensor:
            normalized_view = normalize_cube(
                view_cube, view_mask, mean, std
            )
            return torch.from_numpy(
                np.moveaxis(normalized_view, -1, 0).copy()
            )

        with legacy_encoder_precision():
            global_hsi = tensor_view(cube, mask)
            local_views = [
                extract_local_hsi_view(
                    cube, mask, box, output_shape_hw=mask.shape
                )
                for box in boxes
            ]
            predictor.eval()
            device = next(predictor.parameters()).device
            encoded = predictor.encode(
                global_hsi[None].to(device),
                torch.from_numpy(mask.copy())[None].to(device),
            )
            fused = predictor.fuse_encoded(encoded)
            baseline = predictor.predict_from_fused(fused)
            local_fused = predictor.fuse_encoded(
                predictor.encode(
                    torch.stack(
                        [tensor_view(c, m) for c, m in local_views]
                    ).to(device),
                    torch.stack(
                        [torch.from_numpy(m.copy()) for _, m in local_views]
                    ).to(device),
                )
            )
            features = {
                "global_memory": fused["memory"][0].float().cpu(),
                "global_padding": fused["memory_padding_mask"][0].cpu(),
                "candidate_memory": local_fused["memory"].float().cpu(),
                "candidate_padding": local_fused[
                    "memory_padding_mask"
                ].cpu(),
                "boxes_normalized": torch.from_numpy(
                    boxes.astype(np.float32)
                    / np.asarray(
                        [mask.shape[1], mask.shape[0]] * 2, dtype=np.float32
                    )
                ),
                "foreground_fraction": torch.tensor(
                    [
                        float(mask[y0:y1, x0:x1].mean())
                        for x0, y0, x1, y1 in boxes
                    ]
                ),
            }
            patch_h, patch_w = predictor.config.patch_size
            grid_w = mask.shape[1] // patch_w
            keep = torch.ones(
                len(boxes), encoded["spatial_tokens"].shape[1], device=device
            )
            for index, (x0, y0, x1, y1) in enumerate(boxes):
                ids = [
                    y * grid_w + x
                    for y in range(y0 // patch_h, y1 // patch_h)
                    for x in range(x0 // patch_w, x1 // patch_w)
                ]
                keep[index, ids] = 0
            grounder = TokenCounterfactualGrounder(variant_batch_size=4)
            variants = grounder._predict_variants(
                predictor, encoded, spatial_masks=keep
            )
            effects = grounder._phenotype_changes(baseline, variants).T

        class Values:
            regression_z = baseline["regression"][0].float().cpu().numpy()
            fertilizer_probabilities = (
                baseline["fertilizer_logits"]
                .softmax(-1)[0]
                .float()
                .cpu()
                .numpy()
            )
            cultivar_probabilities = (
                baseline["cultivar_logits"]
                .softmax(-1)[0]
                .float()
                .cpu()
                .numpy()
            )

        predictions = summarize_predictions(
            Values(),
            runtime.target_statistics,
            [0.0, 0.5, 1.0],
            ["Heerup", "Kvium", "Rembrandt", "Sheriff"],
        )
        checkpoint_hash = self._checkpoint_sha256()
        selections = direct_selections(boxes, effects)
        output_root = (
            Path(media_dir).resolve()
            / str(image_id)
            / self._media_subdirectory()
        )
        visuals_raw = render_assets(cube, mask, {
            "selections": selections
        }, output_root / "visuals")
        evidence = {
            "image_id": str(image_id),
            "checkpoint_sha256": checkpoint_hash,
            "predictions": predictions,
            "candidate_boxes": boxes.tolist(),
            "effects": effects.tolist(),
            "selections": selections,
            "shape_hwc": list(cube.shape),
            "foreground_fraction": float(mask.mean()),
            "visuals": visuals_raw,
            "composite_path": str(output_root / "visuals" / "hsi_composite.png"),
            "disclaimer": DISCLAIMER,
            "runtime": self._runtime_label(),
            "scientific_result": self._is_scientific_result(),
        }
        validate_evidence(evidence, str(image_id), checkpoint_hash)
        output_root.mkdir(parents=True, exist_ok=True)
        evidence_path = output_root / "evidence.json"
        features_path = output_root / "visual_features.pt"
        atomic_json(evidence_path, evidence)
        atomic_checkpoint(features_path, features)

        exposed_evidence: Dict[str, Any] = {}
        exposed_visuals: Dict[str, Dict[str, Any]] = {}
        for target in _TARGETS:
            selected = selections[target]
            slot = selected["selected_slot"]
            exposed_evidence[target] = {
                "roi_box_xyxy": selected["selected_box"],
                "spatial_indices": [] if slot is None else [int(slot)],
                "spectral_indices": [],
                "candidate_effects": selected["candidate_effects"],
                "acceptable_slots": selected["acceptable_slots"],
                "status": selected["status"],
                "selection_source": selected["selection_source"],
            }
            visual = visuals_raw.get(target, {})
            exposed_visuals[target] = {
                "selected_patches": visual.get("marked_region"),
                "crop": visual.get("crop"),
                "focused_zoom": visual.get("zoom_4x"),
                "grounded_text": (
                    "Frozen PhytoCore direct token-sensitivity candidate "
                    f"status: {selected['status']}."
                ),
            }

        normalized = normalize_cube(cube, mask, mean, std)
        return IDCVPreparedImageState(
            image_id=str(image_id),
            cube_unit_hwc=cube.copy(),
            leaf_mask=mask.copy(),
            normalized=normalized.copy(),
            predictions=dict(predictions),
            evidence_selections=exposed_evidence,
            spatial_scores=np.asarray(effects, dtype=np.float32),
            spectral_scores=np.empty((0,), dtype=np.float32),
            evidence_record_path=str(evidence_path),
            arrays_path=str(features_path),
            visuals=exposed_visuals,
            media_dir=str(output_root),
            composite_path=evidence["composite_path"],
        )

    @torch.inference_mode()
    def answer_question(
        self,
        *,
        prepared_state: IDCVPreparedImageState,
        question: str,
        intent: str,
        targets: List[str],
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Generate symbolic references, then resolve only trusted evidence."""

        runtime = self._ensure_runtime()
        _ensure_project_import()
        from research_v2.inference.cached_direct_bridge import generate_cached
        from research_v2.inference.direct_evidence import (
            conditioning_metadata,
            validate_evidence,
            verify_and_render,
        )

        normalized_intent = str(intent).upper()
        selected_targets = [str(target) for target in targets]
        if normalized_intent == "IMAGE_SUMMARY":
            selected_targets = list(_TARGETS)
        if normalized_intent not in {
            "TARGET_VALUE",
            "EVIDENCE_REQUEST",
            "IMAGE_SUMMARY",
        }:
            raise ValueError(f"Unsupported frozen runtime intent: {intent}")
        if normalized_intent in {"TARGET_VALUE", "EVIDENCE_REQUEST"}:
            if len(selected_targets) != 1:
                raise ValueError(
                    f"{normalized_intent} requires exactly one target"
                )
        if any(target not in _TARGETS for target in selected_targets):
            raise ValueError(f"Invalid target list: {selected_targets}")

        evidence = json.loads(
            Path(prepared_state.evidence_record_path).read_text(encoding="utf-8")
        )
        checkpoint_hash = self._checkpoint_sha256()
        validate_evidence(evidence, prepared_state.image_id, checkpoint_hash)
        features = torch.load(
            prepared_state.arrays_path, map_location="cpu", weights_only=True
        )
        query = {
            "key": f"live_{normalized_intent.lower()}_"
            + (selected_targets[0] if len(selected_targets) == 1 else "all"),
            "question": str(question),
            "intent": normalized_intent,
            "targets": selected_targets,
        }
        metadata = conditioning_metadata(
            query, evidence, runtime.model.bridge.config.max_local_views
        )
        token_cap = min(
            int(max_new_tokens),
            int(
                self._bundle["settings"]["alignment"]["language"]
                ["max_new_tokens"]
            ),
        )
        raw_answer, _prefix, _prompt, generation_info = generate_cached(
            runtime.model,
            runtime.tokenizer,
            runtime.collator,
            features,
            metadata,
            str(question),
            token_cap,
        )
        verified = verify_and_render(
            raw_answer,
            query,
            evidence,
            self._bundle["settings"]["vqa"]["vqa"],
            image_id=prepared_state.image_id,
            checkpoint_sha256=checkpoint_hash,
        )

        visible_targets = (
            [] if normalized_intent == "IMAGE_SUMMARY" else selected_targets
        )
        result: Dict[str, Any] = {
            "status": "PASS",
            "scientific_result": self._is_scientific_result(),
            "scientific_result_reason": self._scientific_result_reason(),
            "image_id": prepared_state.image_id,
            "question": str(question),
            "intent": normalized_intent,
            "targets": selected_targets,
            "answer": verified["final_answer"],
            "raw_answer_accepted": bool(
                verified["raw_reference_accepted"]
            ),
            "fallback_used": bool(verified["fallback_used"]),
            "verification_reasons": list(
                verified["verification_reasons"]
            ),
            "predictions": prepared_state.predictions,
            "evidence": {
                target: prepared_state.evidence_selections[target]
                for target in selected_targets
                if target in prepared_state.evidence_selections
            },
            "visuals": {
                target: prepared_state.visuals[target]
                for target in visible_targets
                if target in prepared_state.visuals
            },
            "evidence_record": prepared_state.evidence_record_path,
            "visual_features": prepared_state.arrays_path,
            "ema_weights": True,
            "vision_frozen": True,
            "grounding_source": "DIRECT_PHYTOCORE_TOKEN_SENSITIVITY",
            "answer_source": verified["final_answer_source"],
            "generated_tokens": generation_info["generated_tokens"],
            "hit_token_cap": generation_info["hit_token_cap"],
        }
        if self.debug_raw_qwen:
            result["raw_qwen_answer"] = raw_answer
        return result

    def status(self) -> Dict[str, Any]:
        trainable = self._bundle.get("trainable_parameters", {})
        predictor_parameters = 1_998_346
        learned = {
            "phytocore": predictor_parameters,
            "visual_bridge": int(trainable.get("visual_bridge", 0)),
            "qwen_lora": int(trainable.get("qwen_lora", 0)),
        }
        return {
            "enabled": True,
            "verified": self._verification_error is None,
            "loaded": self._runtime is not None,
            "scientific_result": False,
            "activation_allowed": self._verification_error is None,
            "deployment_status": (
                "FROZEN_STAGE40S_FOLD0_INTERNAL"
                if self._verification_error is None
                else "ERROR"
            ),
            "bundle_dir": str(self.bundle_dir),
            "phytocore_path": str(self.phytocore_path),
            "outer_fold": 0,
            "model_seed": 2026,
            "verification_error": self._verification_error,
            "ema_weights": True,
            "vision_frozen": True,
            "architecture_freeze_verified": self._verification_error is None,
            "protocol_freeze_verified": self._verification_error is None,
            "selected_interpolation_alpha": 0.6,
            "learned_parameters": learned,
            "total_learned_parameters": sum(learned.values()),
            "architecture": (
                "Frozen weighted-CE PhytoCore + direct global/local HSI memories "
                "+ target/question-conditioned bridge + Qwen2.5-1.5B LoRA/EMA "
                "+ symbolic evidence verifier"
            ),
            "scope_warning": (
                "Fold-0 internal image-wise model; not plot-disjoint, external-site, "
                "new-device, or final 3-seed x 5-fold evidence."
            ),
        }
