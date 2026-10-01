"""Stage-30 ID-CV development VLM adapter with prepare/answer separation.

This is a Fold-0 development runtime. It always reports scientific_result=false.
It is separate from the Stage-13 research_v2 deployment adapter.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _ensure_project_import() -> None:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class IDCVPreparedImageState:
    """CPU-safe serializable state from a single image preparation pass."""

    __slots__ = (
        "image_id",
        "cube_unit_hwc",
        "leaf_mask",
        "normalized",
        "predictions",
        "evidence_selections",
        "spatial_scores",
        "spectral_scores",
        "evidence_record_path",
        "arrays_path",
        "visuals",
        "media_dir",
        "composite_path",
    )

    def __init__(
        self,
        *,
        image_id: str,
        cube_unit_hwc: np.ndarray,
        leaf_mask: np.ndarray,
        normalized: np.ndarray,
        predictions: Dict[str, Any],
        evidence_selections: Dict[str, Any],
        spatial_scores: np.ndarray,
        spectral_scores: np.ndarray,
        evidence_record_path: str,
        arrays_path: str,
        visuals: Dict[str, Dict[str, Any]],
        media_dir: str,
        composite_path: str,
    ) -> None:
        self.image_id = image_id
        self.cube_unit_hwc = cube_unit_hwc
        self.leaf_mask = leaf_mask
        self.normalized = normalized
        self.predictions = predictions
        self.evidence_selections = evidence_selections
        self.spatial_scores = spatial_scores
        self.spectral_scores = spectral_scores
        self.evidence_record_path = evidence_record_path
        self.arrays_path = arrays_path
        self.visuals = visuals
        self.media_dir = media_dir
        self.composite_path = composite_path

    def to_cache_dict(self) -> Dict[str, Any]:
        """Serialize to a dict safe for session cache (no GPU tensors)."""
        return {
            "image_id": self.image_id,
            "cube_unit_hwc": self.cube_unit_hwc,
            "leaf_mask": self.leaf_mask,
            "normalized": self.normalized,
            "predictions": self.predictions,
            "evidence_selections": self.evidence_selections,
            "spatial_scores": self.spatial_scores,
            "spectral_scores": self.spectral_scores,
            "evidence_record_path": self.evidence_record_path,
            "arrays_path": self.arrays_path,
            "visuals": self.visuals,
            "media_dir": self.media_dir,
            "composite_path": self.composite_path,
        }

    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "IDCVPreparedImageState":
        return cls(
            image_id=data["image_id"],
            cube_unit_hwc=data["cube_unit_hwc"],
            leaf_mask=data["leaf_mask"],
            normalized=data["normalized"],
            predictions=data["predictions"],
            evidence_selections=data["evidence_selections"],
            spatial_scores=data["spatial_scores"],
            spectral_scores=data["spectral_scores"],
            evidence_record_path=data["evidence_record_path"],
            arrays_path=data["arrays_path"],
            visuals=data["visuals"],
            media_dir=data["media_dir"],
            composite_path=data["composite_path"],
        )


class IDCVDevelopmentVLMAdapter:
    """Two-phase adapter: prepare_image() once, answer_question() many times.

    This wraps the Stage-34 IDCVDevelopmentVLMRuntime from research_v2
    and refactors it into prepare/answer phases so the same uploaded
    image can serve multiple questions without recomputing PhytoCore
    and counterfactual evidence.
    """

    def __init__(
        self,
        *,
        bundle_dir: Path,
        phytocore_path: Path,
        device: torch.device,
        eager_load: bool = False,
        debug_raw_qwen: bool = False,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        self.phytocore_path = Path(phytocore_path).resolve()
        self.device = device
        self.debug_raw_qwen = debug_raw_qwen
        self._runtime: Optional[Any] = None
        self._manifest: Optional[Dict[str, Any]] = None
        self._verification_error: Optional[str] = None

        # Verify bundle exists
        self._verify_bundle()

        # A configured development runtime must never remain present in a
        # half-verified state.  Raising here lets the registry expose a clear
        # startup error instead of claiming that the bundle was verified and
        # failing later during an image request.
        if self._verification_error is not None:
            raise RuntimeError(self._verification_error)

        if eager_load:
            self._load_runtime()

    def _verify_bundle(self) -> None:
        """Verify that the Stage-30 bundle and PhytoCore checkpoint exist."""
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            self._verification_error = f"Bundle manifest not found: {manifest_path}"
            return
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            self._verification_error = f"Cannot read manifest: {exc}"
            return

        if not self.phytocore_path.exists():
            self._verification_error = (
                f"PhytoCore checkpoint not found: {self.phytocore_path}"
            )
            return

        manifest_predictor = Path(str(self._manifest.get("predictor_checkpoint", "")))
        if manifest_predictor.is_absolute():
            manifest_predictor = manifest_predictor.resolve()
        else:
            manifest_predictor = (_PROJECT_ROOT / manifest_predictor).resolve()
        if manifest_predictor != self.phytocore_path.resolve():
            self._verification_error = (
                "Configured PhytoCore checkpoint does not match the predictor "
                "recorded in the VLM manifest."
            )
            return

        expected_predictor_hash = str(
            self._manifest.get("predictor_checkpoint_sha256", "")
        )
        if not expected_predictor_hash:
            self._verification_error = "VLM manifest has no predictor checkpoint hash."
            return
        digest = hashlib.sha256()
        with self.phytocore_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_predictor_hash:
            self._verification_error = (
                "Configured PhytoCore checkpoint hash does not match the VLM manifest."
            )
            return

        # Verify essential bundle files
        required_files = ["visual_bridge_ema.pt"]
        required_dirs = ["qwen_lora_ema", "tokenizer"]
        for fname in required_files:
            if not (self.bundle_dir / fname).exists():
                self._verification_error = f"Missing bundle file: {fname}"
                return
        for dname in required_dirs:
            if not (self.bundle_dir / dname).is_dir():
                self._verification_error = f"Missing bundle directory: {dname}"
                return

    def _load_runtime(self) -> None:
        """Lazy-load the actual GPU runtime."""
        if self._runtime is not None:
            return
        if self._verification_error:
            raise RuntimeError(
                f"Cannot load IDCV development runtime: {self._verification_error}"
            )
        _ensure_project_import()
        from research_v2.inference.idcv_live_runtime import (
            IDCVDevelopmentVLMRuntime,
        )

        self._runtime = IDCVDevelopmentVLMRuntime.load(
            root=_PROJECT_ROOT,
            bundle_dir=self.bundle_dir,
            device=self.device,
        )

    def _ensure_runtime(self) -> Any:
        self._load_runtime()
        return self._runtime

    @torch.inference_mode()
    def prepare_image(
        self,
        *,
        cube_unit_hwc: np.ndarray,
        leaf_mask: np.ndarray,
        image_id: str,
        media_dir: Path,
    ) -> IDCVPreparedImageState:
        """Run PhytoCore + counterfactual evidence once. Return CPU-safe state."""
        runtime = self._ensure_runtime()

        _ensure_project_import()
        from research_v2.inference.idcv_live_runtime import (
            _validate_inputs,
            _normalize,
        )
        from research_v2.evaluation.counterfactual_evidence import (
            TARGETS,
            summarize_predictions,
        )
        from research_v2.evaluation.visual_evidence import (
            robust_false_color_composite,
        )
        from research_v2.training.idcv_classification import FERTILIZER_LEVELS

        cube, mask = _validate_inputs(cube_unit_hwc, leaf_mask)
        normalized = _normalize(cube, mask, runtime.band_mean, runtime.band_std)

        # Run PhytoCore + counterfactual evidence on GPU
        global_hsi = (
            torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(normalized, -1, 0))
            )
            .float()[None]
            .to(runtime.device)
        )
        global_mask = (
            torch.from_numpy(mask.copy()).bool()[None].to(runtime.device)
        )

        evidence = runtime.grounder.ground_ensemble(
            [runtime.model.bridge.vision_tower], global_hsi, global_mask
        )

        predictions = summarize_predictions(
            evidence,
            {target: runtime.target_statistics[target] for target in TARGETS[:3]},
            fertilizer_values=FERTILIZER_LEVELS,
            cultivar_names=("Heerup", "Kvium", "Rembrandt", "Sheriff"),
        )

        # Save evidence arrays to media dir
        output_root = Path(media_dir).resolve()
        image_root = output_root / str(image_id)
        image_root.mkdir(parents=True, exist_ok=True)

        arrays_path = image_root / "counterfactual_arrays.npz"
        np.savez_compressed(
            arrays_path,
            spatial_scores=evidence.spatial_scores,
            spectral_scores=evidence.spectral_scores,
            pqb_spatial_attention=evidence.pqb_spatial_attention,
            pqb_spectral_attention=evidence.pqb_spectral_attention,
        )

        # Build evidence selections dict
        evidence_selections: Dict[str, Any] = {}
        for sel in evidence.selections:
            evidence_selections[sel.target] = {
                "roi_box_xyxy": list(sel.roi_box_xyxy),
                "spatial_indices": list(sel.spatial_indices),
                "spectral_indices": list(sel.spectral_indices),
            }

        # Build evidence record for template reconstruction
        evidence_record = {
            "image_id": str(image_id),
            "outer_fold": 0,
            "scientific_result": False,
            "development_only": True,
            "arrays_path": str(arrays_path),
            "targets": [
                {
                    "target": sel.target,
                    "prediction": dict(predictions[sel.target]),
                    "roi_box_xyxy": list(sel.roi_box_xyxy),
                    "spatial_indices": list(sel.spatial_indices),
                    "spectral_indices": list(sel.spectral_indices),
                    "visuals": {},
                }
                for sel in evidence.selections
            ],
        }
        evidence_path = image_root / "evidence_record.json"
        evidence_path.write_text(
            json.dumps(evidence_record, indent=2) + "\n", encoding="utf-8"
        )

        # Render all target visuals once
        all_targets = [sel.target for sel in evidence.selections]
        visuals = runtime._render_visuals(
            cube, mask, evidence, predictions, all_targets, image_id, output_root
        )

        # Update evidence record with visual paths
        for target_entry in evidence_record["targets"]:
            target_entry["visuals"] = dict(
                visuals.get(target_entry["target"], {})
            )
        evidence_path.write_text(
            json.dumps(evidence_record, indent=2) + "\n", encoding="utf-8"
        )

        # False-color composite
        composite_path = str(image_root / "false_color_composite.png")
        if not Path(composite_path).exists():
            robust_false_color_composite(cube, mask).save(composite_path)

        # Ensure spatial/spectral scores are numpy arrays (CPU-safe)
        sp_scores = evidence.spatial_scores
        if hasattr(sp_scores, "cpu"):
            sp_scores = sp_scores.cpu().numpy()
        sp_scores = np.asarray(sp_scores, dtype=np.float32)

        spec_scores = evidence.spectral_scores
        if hasattr(spec_scores, "cpu"):
            spec_scores = spec_scores.cpu().numpy()
        spec_scores = np.asarray(spec_scores, dtype=np.float32)

        return IDCVPreparedImageState(
            image_id=image_id,
            cube_unit_hwc=cube.copy(),
            leaf_mask=mask.copy(),
            normalized=normalized.copy(),
            predictions=dict(predictions),
            evidence_selections=evidence_selections,
            spatial_scores=sp_scores,
            spectral_scores=spec_scores,
            evidence_record_path=str(evidence_path),
            arrays_path=str(arrays_path),
            visuals=visuals,
            media_dir=str(output_root),
            composite_path=composite_path,
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
        """Answer a question using cached prepared state. Runs Qwen + verification."""
        runtime = self._ensure_runtime()

        _ensure_project_import()
        from research_v2.inference.idcv_live_runtime import (
            _validate_plan,
            _select_contract,
            _normalize,
        )
        from research_v2.inference.grounded_answer import compose_verified_answer
        from research_v2.qa.templates import build_records_for_evidence
        from research_v2.models.roi_cropper import extract_local_hsi_view
        from research_v2.qa.schemas import GENERAL_TARGET_INDEX
        from research_v2.training.vlm_alignment import generation_inputs_to_device

        intent_str = str(intent).upper()
        requested_targets = _validate_plan(intent_str, targets)

        # Build sample metadata for VQA template reconstruction
        sample_metadata = {
            "protocol": "ID-CV-LIVE-DEVELOPMENT",
            "gw_group": f"IDCV_LIVE_{prepared_state.image_id}",
            "measurement_group": f"IDCV_LIVE_{prepared_state.image_id}",
            "image_path": f"memory://{prepared_state.image_id}.tiff",
            "mask_path": f"memory://{prepared_state.image_id}.npz",
            "band_stats_path": str(
                _PROJECT_ROOT
                / "artifacts/03_idcv_normalization/fold_0_band_stats.npz"
            ),
            "target_stats_path": str(
                _PROJECT_ROOT
                / "artifacts/03_idcv_normalization/fold_0_target_stats.json"
            ),
            "source_evidence_record": prepared_state.evidence_record_path,
        }

        # Load the evidence record
        with open(
            prepared_state.evidence_record_path, "r", encoding="utf-8"
        ) as f:
            evidence_record = json.load(f)

        records = build_records_for_evidence(
            evidence_record, sample_metadata, runtime.vqa_config
        )
        contract = _select_contract(records, intent_str, requested_targets)

        # Build generation sample
        cube = prepared_state.cube_unit_hwc
        mask = prepared_state.leaf_mask
        normalized = prepared_state.normalized
        height, width, bands = cube.shape

        local_hsi = np.zeros(
            (runtime.max_local_views, bands, height, width), dtype=np.float32
        )
        local_masks = np.zeros(
            (runtime.max_local_views, height, width), dtype=bool
        )
        local_view_mask = np.zeros(runtime.max_local_views, dtype=bool)
        local_target_indices = np.full(
            runtime.max_local_views, GENERAL_TARGET_INDEX, dtype=np.int64
        )
        roi_boxes = np.zeros((runtime.max_local_views, 4), dtype=np.int64)

        for index, view in enumerate(contract["local_views"]):
            local_cube, local_mask = extract_local_hsi_view(
                cube,
                mask,
                view["roi_box_xyxy"],
                output_shape_hw=(height, width),
            )
            local_hsi[index] = np.moveaxis(
                _normalize(
                    local_cube, local_mask, runtime.band_mean, runtime.band_std
                ),
                -1,
                0,
            )
            local_masks[index] = local_mask
            local_view_mask[index] = True
            local_target_indices[index] = int(view["target_index"])
            roi_boxes[index] = np.asarray(
                view["roi_box_xyxy"], dtype=np.int64
            )

        generation_sample = {
            "global_hsi": torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(normalized, -1, 0))
            ).float(),
            "global_leaf_mask": torch.from_numpy(mask.copy()).bool(),
            "local_hsi": torch.from_numpy(local_hsi).float(),
            "local_leaf_masks": torch.from_numpy(local_masks).bool(),
            "local_view_mask": torch.from_numpy(local_view_mask).bool(),
            "local_target_indices": torch.from_numpy(
                local_target_indices
            ).long(),
            "roi_boxes_xyxy": torch.from_numpy(roi_boxes).long(),
            "primary_target_index": torch.tensor(
                int(contract["primary_target_index"]), dtype=torch.long
            ),
            "question": str(question),
            "answer": str(contract["answer"]),
            "intent": str(contract["intent"]),
            "answer_mode": str(contract["answer_mode"]),
            "record_id": str(contract["record_id"]),
            "image_id": str(contract["image_id"]),
            "outer_fold": torch.tensor(0, dtype=torch.long),
        }

        # Run Qwen generation
        batch = runtime.collator.for_generation([generation_sample])
        inputs = generation_inputs_to_device(batch, runtime.device)

        with torch.autocast(
            device_type=runtime.device.type,
            dtype=(
                torch.bfloat16
                if runtime.device.type == "cuda"
                else torch.float32
            ),
            enabled=runtime.device.type == "cuda",
        ):
            generated_ids = runtime.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=int(runtime.tokenizer.pad_token_id),
                eos_token_id=int(runtime.tokenizer.eos_token_id),
                ablation_mode="full",
            )

        raw_answer = runtime.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Verify through Stage-33 grounded answer verifier
        verified = compose_verified_answer(
            contract, raw_answer, runtime.vqa_config
        )

        # Collect target-specific visuals
        target_visuals = {}
        for target in requested_targets:
            if target in prepared_state.visuals:
                target_visuals[target] = prepared_state.visuals[target]

        result: Dict[str, Any] = {
            "status": "PASS",
            "scientific_result": False,
            "scientific_result_reason": (
                "Single-fold architecture-finalization runtime; final five-fold "
                "evaluation remains deferred."
            ),
            "image_id": prepared_state.image_id,
            "question": str(question),
            "intent": intent_str,
            "targets": requested_targets,
            "answer": verified["verified_final_answer"],
            "raw_answer_accepted": bool(verified["raw_answer_accepted"]),
            "fallback_used": bool(verified["fallback_used"]),
            "verification_reasons": list(verified["verification_reasons"]),
            "predictions": prepared_state.predictions,
            "evidence": {
                t: prepared_state.evidence_selections[t]
                for t in requested_targets
                if t in prepared_state.evidence_selections
            },
            "visuals": target_visuals,
            "evidence_record": prepared_state.evidence_record_path,
            "counterfactual_arrays": prepared_state.arrays_path,
            "ema_weights": True,
            "vision_frozen": True,
            "grounding_source": "PHYTOCORE_COUNTERFACTUAL_EVIDENCE",
        }

        # Only include raw Qwen text in debug mode
        if self.debug_raw_qwen:
            result["raw_qwen_answer"] = raw_answer

        return result

    def status(self) -> Dict[str, Any]:
        """Return status information for the model registry."""
        return {
            "enabled": True,
            "verified": self._verification_error is None,
            "loaded": self._runtime is not None,
            "scientific_result": False,
            "activation_allowed": self._verification_error is None,
            "deployment_status": (
                "DEVELOPMENT_FOLD0"
                if self._verification_error is None
                else f"ERROR: {self._verification_error}"
            ),
            "bundle_dir": str(self.bundle_dir),
            "phytocore_path": str(self.phytocore_path),
            "outer_fold": 0,
            "model_seed": 2026,
            "verification_error": self._verification_error,
            "ema_weights": True,
            "vision_frozen": True,
            "architecture": (
                "Frozen PhytoCore + phenotype-conditioned visual token bridge "
                "+ Qwen-LoRA + Stage-33 grounded verifier"
            ),
        }
