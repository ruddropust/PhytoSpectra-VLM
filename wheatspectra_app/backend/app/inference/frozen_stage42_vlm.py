"""Live adapter for the verified Stage-42 contract-optimized Fold-0 model.

The adapter is deliberately fail-closed.  It verifies the frozen architecture,
protocol, Stage-42 recipe, adopted Fold-0 training report, evaluated bundle and
the completed untouched outer-fold report before accepting any weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from app.inference.frozen_stage40_vlm import (
    FrozenStage40VLMAdapter,
    _PROJECT_ROOT,
    _TARGETS,
    _ensure_project_import,
)
from app.inference.idcv_development_vlm import IDCVPreparedImageState


class FrozenStage42VLMAdapter(FrozenStage40VLMAdapter):
    """Stage-42 evidence-contract runtime evaluated on seed-2026/fold-0."""

    def __init__(
        self,
        *,
        bundle_dir: Path,
        phytocore_path: Path,
        architecture_freeze_manifest: Path,
        protocol_freeze_manifest: Path,
        recipe_freeze_manifest: Path,
        stage42c_report: Path,
        stage42d_report: Path,
        stage42d_report_sha256: str,
        device: torch.device,
        eager_load: bool = False,
        debug_raw_qwen: bool = False,
    ) -> None:
        self.recipe_freeze_manifest = Path(recipe_freeze_manifest).resolve()
        self.stage42c_report = Path(stage42c_report).resolve()
        self.stage42d_report = Path(stage42d_report).resolve()
        self.stage42d_report_sha256 = str(stage42d_report_sha256).lower()
        self._recipe_freeze: Dict[str, Any] = {}
        self._evaluation_report: Dict[str, Any] = {}
        self._evaluation_binding: Dict[str, Any] = {}
        self._verified_checkpoint_sha256 = ""
        self._nbest_k = 4
        self._expose_raw_qwen = bool(debug_raw_qwen)
        # The parent generator must return its raw plan internally so the
        # Stage-42 evidence-aware decoder can rank and audit it.  It is removed
        # from the API result unless the explicit debug flag is enabled.
        super().__init__(
            bundle_dir=bundle_dir,
            phytocore_path=phytocore_path,
            architecture_freeze_manifest=architecture_freeze_manifest,
            protocol_freeze_manifest=protocol_freeze_manifest,
            device=device,
            eager_load=eager_load,
            debug_raw_qwen=True,
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _verify_frozen_runtime(self) -> None:
        try:
            _ensure_project_import()
            from research_v2.deployment.contract_recipe_freeze import (
                verify as verify_recipe,
            )
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
                    "Stage-40T architecture freeze failed: "
                    f"{architecture_check.get('failures', [])}"
                )
            protocol_check = verify_manifest(
                _PROJECT_ROOT, self.protocol_freeze_manifest
            )
            if protocol_check.get("status") != "PASS":
                raise ValueError(
                    "Stage-40Y protocol freeze failed: "
                    f"{protocol_check.get('failures', [])}"
                )
            recipe_check = verify_recipe(
                _PROJECT_ROOT, self.recipe_freeze_manifest
            )
            if recipe_check.get("status") != "PASS":
                raise ValueError(
                    "Stage-42B recipe freeze failed: "
                    f"{recipe_check.get('failures', [])}"
                )

            self._architecture_freeze = read_json(
                self.architecture_freeze_manifest
            )
            self._protocol_freeze = read_json(self.protocol_freeze_manifest)
            self._recipe_freeze = self._read_json(self.recipe_freeze_manifest)
            stage42c = self._read_json(self.stage42c_report)
            self._evaluation_report = self._read_json(self.stage42d_report)
            binding_path = self.stage42d_report.parent / "run_binding.json"
            status_path = self.stage42d_report.parent / "status.json"
            self._evaluation_binding = self._read_json(binding_path)
            evaluation_status = self._read_json(status_path)

            if file_sha256(self.stage42d_report) != self.stage42d_report_sha256:
                raise ValueError("Configured Stage-42D evaluation report hash changed")
            if (
                self._evaluation_report.get("status") != "PASS"
                or self._evaluation_report.get("stage") != "42D"
                or int(self._evaluation_report.get("model_seed", -1)) != 2026
                or int(self._evaluation_report.get("outer_fold", -1)) != 0
                or self._evaluation_report.get("scientific_result") is not True
                or int(
                    self._evaluation_report.get("population", {}).get("images", 0)
                ) != 318
                or int(
                    self._evaluation_report.get("population", {}).get("questions", 0)
                ) != 3816
            ):
                raise ValueError("Stage-42D is not the completed Fold-0 evaluation")
            if (
                evaluation_status.get("status") != "COMPLETE"
                or evaluation_status.get("scientific_result") is not True
            ):
                raise ValueError("Stage-42D completion status is invalid")
            if (
                self._evaluation_binding.get("stage") != "42D"
                or int(self._evaluation_binding.get("model_seed", -1)) != 2026
                or int(self._evaluation_binding.get("outer_fold", -1)) != 0
                or self._evaluation_binding.get("no_training") is not True
                or self._evaluation_binding.get(
                    "outer_validation_used_for_model_selection"
                )
                is not False
            ):
                raise ValueError("Stage-42D run binding is not the frozen Fold-0 job")

            configured_bundle = Path(
                str(self._evaluation_binding["contract_bundle_dir"])
            ).resolve()
            if configured_bundle != self.bundle_dir:
                raise ValueError("Configured bundle is not the Stage-42D contract bundle")
            configured_c = [
                (Path(path).resolve(), digest)
                for path, digest in self._evaluation_binding["inputs"].items()
                if Path(path).name == "stage42c_report.json"
            ]
            if len(configured_c) != 1 or configured_c[0][0] != self.stage42c_report:
                raise ValueError("Configured Stage-42C report is not bound by Stage-42D")
            if file_sha256(self.stage42c_report) != configured_c[0][1]:
                raise ValueError("Stage-42C report hash changed")
            if (
                stage42c.get("status") != "PASS"
                or stage42c.get("stage") != "42C"
                or stage42c.get("mode") != "adopt"
                or int(stage42c.get("model_seed", -1)) != 2026
                or int(stage42c.get("outer_fold", -1)) != 0
                or stage42c.get("architecture_changed") is not False
                or stage42c.get("outer_validation_used") is not False
                or Path(str(stage42c.get("recipe_manifest"))).resolve()
                != self.recipe_freeze_manifest
            ):
                raise ValueError("Stage-42C Fold-0 adoption identity is invalid")

            bundle_manifest = self.bundle_dir / "manifest.json"
            bound_bundle_hash = self._evaluation_binding["inputs"].get(
                str(bundle_manifest.resolve())
            )
            if not bound_bundle_hash or file_sha256(bundle_manifest) != bound_bundle_hash:
                raise ValueError("Stage-42 contract bundle manifest hash changed")
            self._bundle = direct.check_bundle(self.bundle_dir)
            predictor = Path(str(self._bundle["predictor_checkpoint"])).resolve()
            if predictor != self.phytocore_path:
                raise ValueError("Configured PhytoCore is not bound to Stage-42")
            checkpoint_hash = str(
                self._bundle["predictor_checkpoint_sha256"]
            ).lower()
            if (
                file_sha256(self.phytocore_path) != checkpoint_hash
                or self._evaluation_binding.get("checkpoint_sha256")
                != checkpoint_hash
            ):
                raise ValueError("Stage-42 PhytoCore checkpoint hash changed")
            if (
                self._bundle.get("status") != "PASS"
                or self._bundle.get("output_contract")
                != "evidence_references_v1"
                or self._bundle.get("use_roi_geometry") is not True
                or self._bundle.get("training", {}).get("stage") != "42A"
                or self._bundle.get("training", {}).get("architecture_changed")
                is not False
                or self._bundle.get("training", {}).get("outer_validation_used")
                is not False
            ):
                raise ValueError("Stage-42 contract bundle identity is invalid")

            verified = self._evaluation_report.get("verified_final_system", {})
            if (
                verified.get("trusted_evidence_field_consistency") != 1.0
                or verified.get("unsupported_exact_abstention") != 1.0
                or verified.get("raw_to_final_source_contract") != 1.0
            ):
                raise ValueError("Stage-42 verified-system gates are incomplete")
            self._nbest_k = int(
                self._recipe_freeze["frozen_recipe"]["config"]["nbest_k"]
            )
            if self._nbest_k != 4:
                raise ValueError("Stage-42 evidence-aware n-best setting changed")
            self._verified_checkpoint_sha256 = checkpoint_hash
        except Exception as exc:
            self._verification_error = f"{type(exc).__name__}: {exc}"

    def _checkpoint_sha256(self) -> str:
        return self._verified_checkpoint_sha256

    def _runtime_label(self) -> str:
        return "FROZEN_STAGE42_CONTRACT_SEED2026_FOLD0"

    def _media_subdirectory(self) -> str:
        return "stage42_contract_frozen"

    def _is_scientific_result(self) -> bool:
        return True

    def _scientific_result_reason(self) -> str:
        return (
            "Completed untouched seed-2026/fold-0 outer evaluation. This is a "
            "single internal image-wise fold; final claims still require all 15 runs."
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
        """Run raw generation, then the frozen Stage-42 top-k contract layer."""

        result = super().answer_question(
            prepared_state=prepared_state,
            question=question,
            intent=intent,
            targets=targets,
            max_new_tokens=max_new_tokens,
        )
        raw_answer = str(result.get("raw_qwen_answer", ""))
        runtime = self._ensure_runtime()
        _ensure_project_import()
        from research_v2.inference.direct_evidence import (
            expected_reference,
            validate_evidence,
            verify_and_render,
        )
        from research_v2.models.contract_optimization import score_reference_plans

        normalized_intent = str(intent).upper()
        selected_targets = [str(target) for target in targets]
        if normalized_intent == "IMAGE_SUMMARY":
            selected_targets = list(_TARGETS)
        evidence = self._read_json(Path(prepared_state.evidence_record_path))
        validate_evidence(
            evidence, prepared_state.image_id, self._checkpoint_sha256()
        )
        features = torch.load(
            prepared_state.arrays_path, map_location="cpu", weights_only=True
        )
        query = {
            "key": "live_"
            + normalized_intent.lower()
            + "_"
            + (selected_targets[0] if len(selected_targets) == 1 else "all"),
            "question": str(question),
            "intent": normalized_intent,
            "targets": selected_targets,
        }
        ranked = score_reference_plans(
            runtime.model, runtime.collator, features, evidence, query
        )
        expected = expected_reference(query, evidence)
        eligible = ranked[: min(self._nbest_k, len(ranked))]
        selected = next(
            (candidate for candidate in eligible if candidate["plan"] == expected),
            ranked[0],
        )
        verified = verify_and_render(
            selected["text"],
            query,
            evidence,
            self._bundle["settings"]["vqa"]["vqa"],
            image_id=prepared_state.image_id,
            checkpoint_sha256=self._checkpoint_sha256(),
        )
        result.update(
            {
                "answer": verified["final_answer"],
                "fallback_used": bool(verified["fallback_used"]),
                "verification_reasons": list(verified["verification_reasons"]),
                "answer_source": verified["final_answer_source"],
                "scientific_result": True,
                "scientific_result_reason": self._scientific_result_reason(),
                "contract_decoding_layer": "VERIFIER_RERANKED_NBEST",
                "contract_nbest_k": self._nbest_k,
                "contract_candidate_rank": int(selected["rank"]),
                "verifier_recovered_within_nbest": bool(
                    selected["plan"] == expected and int(selected["rank"]) > 1
                ),
                "raw_greedy_accepted": bool(result["raw_answer_accepted"]),
            }
        )
        if self._expose_raw_qwen:
            result["raw_qwen_answer"] = raw_answer
        else:
            result.pop("raw_qwen_answer", None)
        return result

    def status(self) -> Dict[str, Any]:
        trainable = self._bundle.get("trainable_parameters", {})
        learned = {
            "phytocore": 1_998_346,
            "visual_bridge": int(trainable.get("visual_bridge", 0)),
            "qwen_lora": int(trainable.get("qwen_lora", 0)),
        }
        layers = self._evaluation_report.get("generation_layers", {})
        raw = layers.get("raw_greedy", {})
        nbest = layers.get("verifier_reranked_nbest", {})
        return {
            "enabled": True,
            "verified": self._verification_error is None,
            "loaded": self._runtime is not None,
            "scientific_result": True,
            "activation_allowed": self._verification_error is None,
            "deployment_status": (
                self._runtime_label()
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
            "recipe_freeze_verified": self._verification_error is None,
            "outer_evaluation_verified": self._verification_error is None,
            "contract_decoding": "verifier_reranked_nbest_k4",
            "learned_parameters": learned,
            "total_learned_parameters": sum(learned.values()),
            "architecture": (
                "Frozen weighted-CE PhytoCore + direct global/local HSI memories "
                "+ target/question-conditioned bridge + Stage-42 contract-trained "
                "Qwen2.5-1.5B LoRA/EMA + evidence-aware top-k verifier"
            ),
            "evaluated_images": self._evaluation_report.get("population", {}).get(
                "images"
            ),
            "evaluated_questions": self._evaluation_report.get("population", {}).get(
                "questions"
            ),
            "raw_reference_acceptance": raw.get("raw_reference_acceptance"),
            "raw_class_agreement_with_phytocore": raw.get(
                "raw_class_agreement_with_phytocore"
            ),
            "verified_nbest_acceptance": nbest.get("raw_reference_acceptance"),
            "scope_warning": (
                "Scientifically evaluated seed-2026/fold-0 internal image-wise "
                "model; not yet the final 3-seed x 5-fold estimate, plot-disjoint "
                "validation, or external-site validation."
            ),
        }
