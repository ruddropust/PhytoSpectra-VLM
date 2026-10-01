"""Structured orchestration for knowledge, image state, and PhytoCore routes."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent.intent_schema import QueryPlan, ScientificIntent
from app.agent.query_understanding import QueryUnderstandingAgent
from app.config import settings
from app.core.session_cache import session_cache
from app.inference.pgvrc import run_pgvrc
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    PredictionItem,
    ScopeDetails,
    VerificationDetails,
    VisualsPayload,
)
from app.services.image_state import ImageStateBuilder
from app.services.response_composer import (
    compose_curated_knowledge_answer,
    deduplicate_answer,
    compose_image_required_answer,
    compose_image_summary,
    compose_prediction_answer,
    compose_target_difference,
    compose_unsupported_answer,
    load_target_knowledge,
)
from app.inference.idcv_development_vlm import (
    IDCVDevelopmentVLMAdapter,
    IDCVPreparedImageState,
)


class IDCVRuntimeFailure(RuntimeError):
    """Raised when an enabled ID-CV runtime cannot safely answer."""


class ChatPipeline:
    def __init__(self, registry):
        self.registry = registry
        self.router = QueryUnderstandingAgent(llm_manager=self.registry.llm_manager)

    def _has_idcv_development(self) -> bool:
        """Check if the IDCV development runtime is available and verified."""
        adapter = getattr(self.registry, "idcv_development_vlm", None)
        if adapter is None:
            return False
        status = adapter.status()
        return bool(status.get("verified", False))

    @staticmethod
    def _development_warning(result: Optional[Dict[str, Any]]) -> str:
        if result and result.get("scientific_result_reason", "").startswith("Single model from the completed 15-run"):
            return (
                "This answer uses research_v3 stage42d_exact_v1, seed "
                "2026/fold 0. The 15-run campaign passed internal image-wise "
                "evaluation; this live answer is from one fold model, not a "
                "full-data checkpoint or external validation."
            )
        if result and result.get("grounding_source") == "DIRECT_PHYTOCORE_TOKEN_SENSITIVITY":
            if result.get("contract_decoding_layer") == "VERIFIER_RERANKED_NBEST":
                return (
                    "This answer uses the verified Stage-42 seed-2026/fold-0 "
                    "contract model. Its untouched outer-fold evaluation is "
                    "complete, but it is not yet the final 3-seed x 5-fold or "
                    "external-validation result."
                )
            return (
                "This answer uses the frozen Stage-40S Fold-0 internal model. "
                "It is not a final 3-seed x 5-fold or external-validation result."
            )
        return (
            "This result is from Fold-0 development evaluation. "
            "It is not a validated scientific result."
        )

    def _get_idcv_prepared(
        self, image_data: Dict[str, Any]
    ) -> Optional[IDCVPreparedImageState]:
        """Retrieve cached IDCV prepared image state."""
        raw = image_data.get("idcv_prepared")
        if raw is None:
            return None
        try:
            return IDCVPreparedImageState.from_cache_dict(raw)
        except (KeyError, TypeError):
            return None

    def _idcv_answer(
        self,
        req: ChatRequest,
        image_data: Dict[str, Any],
        plan: QueryPlan,
    ) -> Optional[Dict[str, Any]]:
        """Run the IDCV development VLM for an image-derived question."""
        adapter = getattr(self.registry, "idcv_development_vlm", None)
        if adapter is None:
            if (
                settings.RESEARCH_V3_VLM_ENABLED
                or settings.FROZEN_STAGE42_VLM_ENABLED
                or settings.FROZEN_STAGE40_VLM_ENABLED
                or settings.IDCV_DEVELOPMENT_VLM_ENABLED
            ):
                raise IDCVRuntimeFailure("configured_runtime_unavailable")
            return None
        if not self._has_idcv_development():
            raise IDCVRuntimeFailure("bundle_verification_failed")
        prepared = self._get_idcv_prepared(image_data)
        if prepared is None:
            raise IDCVRuntimeFailure("prepared_image_state_unavailable")

        # Map intents to runtime intents. The frozen runtime's trained contract
        # is single-target for value/evidence questions, so multi-field user
        # requests are decomposed and each field is independently verified.
        intent_map = {
            "TARGET_VALUE": "TARGET_VALUE",
            "TARGET_COMPARISON": "TARGET_VALUE",
            "EVIDENCE_REQUEST": "EVIDENCE_REQUEST",
            "IMAGE_REGION_REQUEST": "EVIDENCE_REQUEST",
            "IMAGE_SUMMARY": "IMAGE_SUMMARY",
        }
        runtime_intent = intent_map.get(plan.intent.value, plan.intent.value)

        # For IMAGE_SUMMARY, no targets
        if runtime_intent == "IMAGE_SUMMARY":
            targets: list[str] = []
        else:
            targets = plan.targets

        try:
            if len(targets) > 1:
                return self._idcv_multi_target_answer(
                    adapter=adapter,
                    prepared=prepared,
                    question=req.question,
                    runtime_intent=runtime_intent,
                    targets=targets,
                    comparison=plan.intent == ScientificIntent.TARGET_COMPARISON,
                )
            return adapter.answer_question(
                prepared_state=prepared,
                question=req.question,
                intent=runtime_intent,
                targets=targets,
            )
        except Exception as exc:
            print(
                f"IDCV development answer failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise IDCVRuntimeFailure("answer_generation_failed") from exc

    @staticmethod
    def _restrict_result_to_targets(
        result: Dict[str, Any], targets: List[str]
    ) -> Dict[str, Any]:
        """Prevent cached non-requested fields from leaking into a task reply."""
        if not targets:
            return result
        restricted = dict(result)
        for field in ("predictions", "evidence", "visuals"):
            values = result.get(field, {})
            restricted[field] = {
                target: values[target]
                for target in targets
                if isinstance(values, dict) and target in values
            }
        restricted["targets"] = list(targets)
        return restricted

    @staticmethod
    def _idcv_multi_target_answer(
        *,
        adapter: IDCVDevelopmentVLMAdapter,
        prepared: IDCVPreparedImageState,
        question: str,
        runtime_intent: str,
        targets: List[str],
        comparison: bool,
    ) -> Dict[str, Any]:
        """Compose a multi-field answer from verified single-field results.

        This deliberately does not create a new multi-target generation
        contract.  Every displayed field passes through the already frozen
        Stage-42 single-target verifier, and aggregation only formats those
        accepted outputs.
        """
        results: List[Dict[str, Any]] = []
        for target in targets:
            if runtime_intent == "EVIDENCE_REQUEST":
                target_question = (
                    f"Show the verified evidence for {target} for this "
                    "hyperspectral wheat image."
                )
            else:
                target_question = (
                    f"What is the predicted {target} value for this "
                    "hyperspectral wheat image?"
                )
            results.append(
                adapter.answer_question(
                    prepared_state=prepared,
                    question=target_question,
                    intent=runtime_intent,
                    targets=[target],
                )
            )

        lines = []
        for target, result in zip(targets, results):
            answer_text = str(result.get("answer", "")).strip()
            if answer_text.startswith(f"{target}:"):
                answer_text = answer_text[len(target) + 1:].strip()
            lines.append(f"- {target}: {answer_text}")
        heading = (
            "For the current image, the independently verified outputs are:"
            if not comparison
            else "For the current image, the independently verified outputs for comparison are:"
        )
        answer = heading + "\n" + "\n".join(lines)
        if comparison:
            answer += (
                "\nThese outputs are shown side by side and do not establish "
                "a causal relationship."
            )

        combined: Dict[str, Any] = {
            "status": "PASS",
            "scientific_result": all(
                bool(result.get("scientific_result", False)) for result in results
            ),
            "scientific_result_reason": results[0].get(
                "scientific_result_reason"
            ),
            "image_id": results[0].get("image_id"),
            "question": str(question),
            "intent": (
                "TARGET_COMPARISON" if comparison else runtime_intent
            ),
            "targets": list(targets),
            "answer": answer,
            "raw_answer_accepted": all(
                bool(result.get("raw_answer_accepted", False))
                for result in results
            ),
            "fallback_used": any(
                bool(result.get("fallback_used", False)) for result in results
            ),
            "verification_reasons": [
                f"{target}:{reason}"
                for target, result in zip(targets, results)
                for reason in result.get("verification_reasons", [])
            ],
            "predictions": {},
            "evidence": {},
            "visuals": {},
            "ema_weights": all(
                bool(result.get("ema_weights", False)) for result in results
            ),
            "vision_frozen": all(
                bool(result.get("vision_frozen", False)) for result in results
            ),
            "grounding_source": "DIRECT_PHYTOCORE_TOKEN_SENSITIVITY",
            "answer_source": "COMPOSED_VERIFIED_SINGLE_TARGET_ANSWERS",
            "contract_decoding_layer": "DECOMPOSED_VERIFIER_RERANKED_NBEST",
            "component_results": results,
        }
        for target, result in zip(targets, results):
            predictions = result.get("predictions", {})
            if target in predictions:
                combined["predictions"][target] = predictions[target]
            if target in result.get("evidence", {}):
                combined["evidence"][target] = result["evidence"][target]
            if target in result.get("visuals", {}):
                combined["visuals"][target] = result["visuals"][target]
        return combined

    @staticmethod
    def _plan_dict(plan: QueryPlan) -> Dict[str, Any]:
        return plan.model_dump(mode="json")

    def _record(self, req: ChatRequest, plan: QueryPlan, response: ChatResponse) -> ChatResponse:
        if req.session_id:
            session_cache.append_chat_turn(
                req.session_id,
                req.question,
                self._plan_dict(plan),
                response.answer_type,
            )
        return response

    @staticmethod
    def _empty_response(
        *,
        plan: QueryPlan,
        answer: str,
        answer_type: str,
        answer_mode: str,
        reason: str,
        evidence_status: str,
    ) -> ChatResponse:
        return ChatResponse(
            answer_mode=answer_mode,
            answer_type=answer_type,
            answer=answer,
            numeric_permission=False,
            intent=plan.intent.value,
            selected_phenotypes=plan.targets,
            predictions={},
            scope=ScopeDetails(
                hcsg=plan.route.value,
                reason=reason,
                open_set_score=1.0 - plan.confidence,
                predicted_intent=plan.intent.value,
            ),
            verification=VerificationDetails(
                response_policy=answer_mode,
                evidence_status=evidence_status,
                all_dual_verified=False,
                downgrade_reasons=[],
            ),
            evidence={},
            warnings=[],
            visuals=VisualsPayload(),
        )

    @staticmethod
    def _idcv_runtime_failure_response(
        plan: QueryPlan,
        failure: IDCVRuntimeFailure,
    ) -> ChatResponse:
        """Return a structured, non-numeric failure without model substitution."""
        reason_code = str(failure) or "unknown_runtime_failure"
        return ChatResponse(
            answer_mode="ABSTAIN",
            answer_type="RUNTIME_ERROR",
            answer=(
                "The current model does not provide validated information. "
                "The configured vision-language runtime could not complete this "
                "request, and no legacy model response was substituted."
            ),
            numeric_permission=False,
            intent=plan.intent.value,
            selected_phenotypes=plan.targets,
            predictions={},
            scope=ScopeDetails(
                hcsg="SYSTEM_FAILURE",
                reason="The configured ID-CV runtime failed closed.",
                open_set_score=1.0 - plan.confidence,
                predicted_intent=plan.intent.value,
            ),
            verification=VerificationDetails(
                response_policy="ABSTAIN",
                evidence_status="RUNTIME_FAILURE",
                all_dual_verified=False,
                downgrade_reasons=[reason_code],
                verification_reasons=[reason_code],
                scientific_result=False,
            ),
            evidence={},
            warnings=[
                "IDCV_RUNTIME_FAILED: no prediction or numerical value was returned."
            ],
            visuals=VisualsPayload(),
            development_warning=(
                "This Fold-0 development runtime is unavailable for the current request."
            ),
        )

    def _knowledge_response(self, plan: QueryPlan, req: ChatRequest) -> ChatResponse:
        # Exact target documents use deterministic rendering. This is safer and
        # faster than asking an LLM to repeat a short curated definition.
        context = load_target_knowledge(plan.targets)
        if context:
            answer = compose_curated_knowledge_answer(context)
            q = plan.normalized_question
            if "categorical" in q:
                categorical = [
                    target for target in plan.targets
                    if target in {"Fertilizer", "Cultivar"}
                ]
                continuous = [
                    target for target in plan.targets
                    if target in {"GrainWeight", "Gsw", "PhiPS2"}
                ]
                if categorical and not continuous:
                    prefix = (
                        f"{', '.join(categorical)} "
                        f"{'is' if len(categorical) == 1 else 'are'} categorical."
                    )
                elif categorical:
                    prefix = f"The categorical target(s) are {', '.join(categorical)}."
                else:
                    prefix = "None of the referenced targets is categorical."
                answer = prefix + "\n\n" + answer
            elif "continuous" in q or "regression" in q:
                continuous = [
                    target for target in plan.targets
                    if target in {"GrainWeight", "Gsw", "PhiPS2"}
                ]
                prefix = (
                    f"The continuous target(s) are {', '.join(continuous)}."
                    if continuous
                    else "None of the referenced targets is continuous."
                )
                answer = prefix + "\n\n" + answer
        else:
            retrieved = ""
            if self.registry.knowledge_retriever:
                try:
                    retrieved = self.registry.knowledge_retriever.retrieve(req.question)
                except Exception:
                    retrieved = ""
            if retrieved and self.registry.llm_manager:
                answer = self.registry.llm_manager.generate_knowledge_response(
                    req.question,
                    retrieved,
                )
            else:
                answer = "The current model does not provide validated information."

        return self._empty_response(
            plan=plan,
            answer=deduplicate_answer(answer),
            answer_type="CONCEPT_EXPLANATION",
            answer_mode="EVIDENCE_BOUNDED",
            reason="Response grounded in the curated scientific knowledge base.",
            evidence_status="CURATED_KNOWLEDGE",
        )

    @staticmethod
    def _prediction_items(raw_predictions: Dict[str, Any]) -> Dict[str, PredictionItem]:
        predictions: Dict[str, PredictionItem] = {}
        for name, value in raw_predictions.items():
            if not isinstance(value, dict) or value.get("value") is None:
                continue
            predictions[name] = PredictionItem(
                value=value.get("value"),
                unit=value.get("unit"),
                ensemble_std=value.get("ensemble_std"),
                interval_low=value.get("interval_low"),
                interval_high=value.get("interval_high"),
                vote_confidence=value.get("vote_confidence", value.get("confidence")),
                cultivar_probabilities=value.get("cultivar_probabilities"),
                seed_disagreement=value.get("seed_disagreement"),
                calibration_status=value.get(
                    "calibration_status",
                    "NOT_AVAILABLE_FOR_LIVE_MODE",
                ),
            )
        return predictions

    def _image_summary_response(
        self,
        plan: QueryPlan,
        image_state: Dict[str, Any],
        req: ChatRequest,
        image_data: Dict[str, Any],
    ) -> ChatResponse:
        raw_predictions = image_state.get("predicted_targets", {})
        evidence = image_state.get("evidence", {})
        answer = compose_image_summary(image_state)
        # Try IDCV development VLM first (Stage-30 Fold-0)
        idcv_result = self._idcv_answer(req, image_data, plan)
        v2_result = None
        if idcv_result is not None:
            raw_predictions = self._research_v2_predictions(idcv_result["predictions"])
            evidence = idcv_result.get("evidence", {})
            answer = idcv_result.get("answer", answer)
        else:
            v2_result = self._run_research_v2(req, image_data, targets=[])
            if v2_result is not None:
                raw_predictions = self._research_v2_predictions(v2_result["predictions"])
                evidence = v2_result.get("evidence", {})
                answer = v2_result.get("answer") or answer
        answer = deduplicate_answer(answer)
        predictions = self._prediction_items(raw_predictions)
        return ChatResponse(
            answer_mode="EVIDENCE_BOUNDED",
            answer_type="IMAGE_SUMMARY",
            answer=answer,
            numeric_permission=True,
            intent=plan.intent.value,
            selected_phenotypes=list(predictions.keys()),
            predictions=predictions,
            scope=ScopeDetails(
                hcsg="IMAGE_STATE",
                reason="Description rendered only from typed image state and cached model outputs.",
                open_set_score=0.0,
                predicted_intent=plan.intent.value,
            ),
            verification=VerificationDetails(
                response_policy="EVIDENCE_BOUNDED",
                evidence_status="PREDICTION_ONLY",
                all_dual_verified=False,
                downgrade_reasons=["IMAGE_SUMMARY_DOES_NOT_RUN_DUAL_EVIDENCE_VERIFICATION"],
                raw_answer_accepted=idcv_result.get("raw_answer_accepted") if idcv_result else None,
                fallback_used=idcv_result.get("fallback_used") if idcv_result else None,
                verification_reasons=idcv_result.get("verification_reasons", []) if idcv_result else [],
                grounding_source=idcv_result.get("grounding_source") if idcv_result else None,
                ema_weights=idcv_result.get("ema_weights") if idcv_result else None,
                vision_frozen=idcv_result.get("vision_frozen") if idcv_result else None,
                scientific_result=idcv_result.get("scientific_result") if idcv_result else None,
            ),
            evidence=evidence,
            warnings=[
                "Model outputs are based on internal image-level validation and are not calibrated confidence estimates."
            ],
            visuals=(
                self._research_v2_visuals(idcv_result, image_state.get("preview_url"))
                if idcv_result is not None
                else self._research_v2_visuals(v2_result, image_state.get("preview_url"))
                if v2_result is not None
                else VisualsPayload(composite_url=image_state.get("preview_url"))
            ),
            development_warning=(
                self._development_warning(idcv_result)
                if idcv_result is not None
                else None
            ),
            debug_raw_qwen=idcv_result.get("raw_qwen_answer") if idcv_result else None,
        )

    def _run_research_v2(
        self,
        req: ChatRequest,
        image_data: Dict[str, Any],
        targets: list[str],
    ) -> Optional[Dict[str, Any]]:
        manager = getattr(self.registry, "research_v2_vlm", None)
        if manager is None or not bool(
            manager.manifest.get("activation_allowed", False)
        ):
            return None
        return manager.infer(
            cube_unit_hwc=image_data["cube_unit"],
            leaf_mask=image_data["mask"],
            question=req.question,
            targets=targets,
            image_id=req.image_id or "live",
            visual_output_dir=settings.MEDIA_DIR,
        )

    @staticmethod
    def _research_v2_predictions(
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        cultivar_names = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]
        output: Dict[str, Any] = {}
        for target, prediction in values.items():
            item: Dict[str, Any] = {"value": prediction.get("value")}
            if prediction.get("task") == "classification":
                item["vote_confidence"] = prediction.get("confidence")
                probabilities = prediction.get("probabilities")
                if target == "Cultivar" and isinstance(probabilities, list):
                    item["cultivar_probabilities"] = {
                        name: float(value)
                        for name, value in zip(cultivar_names, probabilities)
                    }
            output[target] = item
        return output

    @staticmethod
    def _media_url(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            relative = Path(path).resolve().relative_to(settings.MEDIA_DIR.resolve())
        except (OSError, ValueError):
            return None
        return "/media/" + relative.as_posix()

    def _research_v2_visuals(
        self,
        result: Dict[str, Any],
        composite_url: Optional[str],
    ) -> VisualsPayload:
        targets: Dict[str, Any] = {}
        first: Dict[str, Optional[str]] = {}
        for target, visual in result.get("visuals", {}).items():
            evidence = result.get("evidence", {}).get(target, {})
            prediction = result.get("predictions", {}).get(target, {})
            marked = self._media_url(visual.get("selected_patches"))
            crop = self._media_url(visual.get("crop"))
            zoom = self._media_url(visual.get("focused_zoom"))
            spectral = self._media_url(visual.get("spectral_token_plot"))
            bundle = self._media_url(visual.get("evidence_bundle"))
            heatmap = self._media_url(visual.get("counterfactual_heatmap"))
            targets[target] = {
                "target": target,
                "prediction_text": str(prediction.get("value", "")),
                "visual_method": (
                    "DIRECT_PHYTOCORE_TOKEN_SENSITIVITY"
                    if result.get("grounding_source")
                    == "DIRECT_PHYTOCORE_TOKEN_SENSITIVITY"
                    else "RESEARCH_V2_PCTG_COUNTERFACTUAL_ATTRIBUTION"
                ),
                "evidence_available": bool(evidence.get("roi_box_xyxy")),
                "focus_box": evidence.get("roi_box_xyxy"),
                "selected_token_indices": evidence.get("spatial_indices", []),
                "selected_token_count": len(evidence.get("spatial_indices", [])),
                "spatial_summary": visual.get("grounded_text", ""),
                "heatmap_url": heatmap,
                "marked_url": marked,
                "crop_url": crop,
                "zoom_url": zoom,
                "spectral_plot_url": spectral,
                "bundle_url": bundle,
            }
            if not first:
                first = {
                    "marked_url": marked,
                    "zoom_url": zoom,
                    "spectral_plot_url": spectral,
                    "bundle_url": bundle,
                }
        return VisualsPayload(
            composite_url=composite_url,
            target_visuals=targets,
            **first,
        )

    def _render_visuals(
        self,
        image_data: Dict[str, Any],
        plan: QueryPlan,
        session_id: str,
        composite_url: Optional[str],
    ) -> VisualsPayload:
        payload: Dict[str, Any] = {
            "composite_url": composite_url,
            "target_visuals": {},
        }
        if not plan.targets or not self.registry.ver_renderer:
            return VisualsPayload(**payload)
        raw_predictions = image_data.get("ensemble_predictions", {})
        cached_visuals = image_data.setdefault("target_visuals", {})
        for target in plan.targets:
            try:
                rendered = cached_visuals.get(target)
                if not isinstance(rendered, dict):
                    rendered = self.registry.ver_renderer.render_phenotype_visuals(
                        cube_unit=image_data["cube_unit"],
                        mask=image_data["mask"],
                        pctg_result=image_data.get("pctg_result", {}),
                        phenotype=target,
                        session_id=session_id,
                        prediction=raw_predictions.get(target),
                    )
                    cached_visuals[target] = rendered
                payload["target_visuals"][target] = rendered
                if not payload.get("marked_url"):
                    for key in ("marked_url", "zoom_url", "spectral_plot_url", "bundle_url"):
                        payload[key] = rendered.get(key)
            except (KeyError, IndexError, OSError, TypeError, ValueError):
                # The scientific answer remains usable when optional rendering fails.
                continue
        return VisualsPayload(**payload)

    def _prediction_response(
        self,
        plan: QueryPlan,
        req: ChatRequest,
        image_data: Dict[str, Any],
        image_state: Dict[str, Any],
    ) -> ChatResponse:
        initial_mode = (
            "EVIDENCE_BOUNDED"
            if plan.intent in {
                ScientificIntent.TARGET_COMPARISON,
                ScientificIntent.EVIDENCE_REQUEST,
                ScientificIntent.IMAGE_REGION_REQUEST,
            }
            else "VALIDATED"
        )
        route = {
            "question": req.question,
            "predicted_answerability": "SUPPORTED",
            "predicted_intent": plan.intent.value,
            "selected_phenotypes": plan.targets,
            "numeric_permission": True,
            "ebsa_state": initial_mode,
            "open_set_score": 1.0 - plan.confidence,
            "hcsg_scope": "SUPPORTED_PHENOTYPE",
            "hcsg_reason": "VALIDATED_STRUCTURED_QUERY_PLAN",
            "routing_override": False,
        }
        state = run_pgvrc(
            route,
            image_data.get("pctg_result", {}),
            image_data.get("ensemble_predictions", {}),
        )
        raw_predictions = state.get("scientific_predictions", {})
        is_comparison = plan.intent == ScientificIntent.TARGET_COMPARISON
        answer = compose_prediction_answer(raw_predictions, plan.targets, is_comparison)
        v2_result = None
        idcv_result = self._idcv_answer(req, image_data, plan)
        if idcv_result is not None:
            idcv_result = self._restrict_result_to_targets(
                idcv_result, plan.targets
            )
            raw_predictions = self._research_v2_predictions(idcv_result["predictions"])
            answer = idcv_result.get("answer") or answer
        else:
            v2_result = self._run_research_v2(req, image_data, targets=plan.targets)
            if v2_result is not None:
                v2_result = self._restrict_result_to_targets(
                    v2_result, plan.targets
                )
                raw_predictions = self._research_v2_predictions(v2_result["predictions"])
                answer = v2_result.get("answer") or answer
        if plan.include_definition:
            knowledge = compose_curated_knowledge_answer(
                load_target_knowledge(plan.targets)
            )
            if knowledge and not knowledge.startswith("The current model does not"):
                distinction = compose_target_difference(plan.targets)
                if plan.intent != ScientificIntent.TARGET_COMPARISON:
                    distinction = ""
                scientific_context = (
                    distinction + "\n\n" + knowledge
                    if distinction
                    else knowledge
                )
                answer = (
                    "Scientific meaning:\n"
                    + scientific_context
                    + "\n\nVerified image prediction:\n"
                    + answer
                )
        answer = deduplicate_answer(answer)
        predictions = self._prediction_items(raw_predictions)
        if idcv_result is None and plan.intent in {
            ScientificIntent.EVIDENCE_REQUEST,
            ScientificIntent.IMAGE_REGION_REQUEST,
        }:
            answer += (
                "\nThe accompanying panels show model-internal counterfactual spatial "
                "and spectral-token attribution. They are not causal biological evidence, "
                "and physical wavelength values are not asserted."
            )

        all_verified = bool(state.get("all_dual_verified", False))
        evidence_status = "DUAL_VERIFIED" if all_verified else "PARTIAL_OR_MISSING"
        if idcv_result is not None:
            evidence_status = (
                "VERIFIED_GROUNDED_FALLBACK"
                if idcv_result.get("fallback_used")
                else "VERIFIED_QWEN_OUTPUT"
            )
        answer_type = (
            "EVIDENCE_EXPLANATION"
            if plan.intent in {
                ScientificIntent.EVIDENCE_REQUEST,
                ScientificIntent.IMAGE_REGION_REQUEST,
            }
            else "MODEL_PREDICTION"
        )
        visuals = (
            self._research_v2_visuals(idcv_result, image_state.get("preview_url"))
            if idcv_result is not None
            else self._research_v2_visuals(v2_result, image_state.get("preview_url"))
            if v2_result is not None
            else self._render_visuals(
                image_data,
                plan,
                req.session_id or "session",
                image_state.get("preview_url"),
            )
        )
        visual_summaries = [
            visual.spatial_summary
            for visual in visuals.target_visuals.values()
            if visual.spatial_summary
        ]
        if idcv_result is None and visual_summaries:
            answer += "\n\nVisual explanation:\n" + "\n".join(
                f"- {summary}" for summary in visual_summaries
            )

        return ChatResponse(
            answer_mode="EVIDENCE_BOUNDED" if idcv_result is not None else state.get("final_answer_mode", "EVIDENCE_BOUNDED"),
            answer_type=answer_type,
            answer=answer,
            numeric_permission=True if idcv_result is not None and (idcv_result.get("raw_answer_accepted") or idcv_result.get("fallback_used")) else bool(state.get("numeric_permission", False)),
            intent=plan.intent.value,
            selected_phenotypes=(
                list(idcv_result.get("targets", plan.targets))
                if idcv_result is not None
                else state.get("selected_phenotypes", [])
            ),
            predictions=predictions,
            scope=ScopeDetails(
                hcsg="SUPPORTED_PHENOTYPE",
                reason="Request executed from a validated structured query plan.",
                open_set_score=1.0 - plan.confidence,
                predicted_intent=plan.intent.value,
            ),
            verification=VerificationDetails(
                response_policy="EVIDENCE_BOUNDED" if idcv_result is not None else state.get("final_answer_mode", "EVIDENCE_BOUNDED"),
                evidence_status=evidence_status,
                all_dual_verified=all_verified,
                downgrade_reasons=state.get("downgrade_reasons", []),
                raw_answer_accepted=idcv_result.get("raw_answer_accepted") if idcv_result else None,
                fallback_used=idcv_result.get("fallback_used") if idcv_result else None,
                verification_reasons=idcv_result.get("verification_reasons", []) if idcv_result else [],
                grounding_source=idcv_result.get("grounding_source") if idcv_result else None,
                ema_weights=idcv_result.get("ema_weights") if idcv_result else None,
                vision_frozen=idcv_result.get("vision_frozen") if idcv_result else None,
                scientific_result=idcv_result.get("scientific_result") if idcv_result else None,
            ),
            evidence=(
                idcv_result.get("evidence", {})
                if idcv_result is not None
                else v2_result.get("evidence", {})
                if v2_result is not None
                else state.get("evidence", {})
            ),
            warnings=state.get("warnings", []),
            visuals=visuals,
            development_warning=(
                self._development_warning(idcv_result)
                if idcv_result is not None
                else None
            ),
            debug_raw_qwen=idcv_result.get("raw_qwen_answer") if idcv_result else None,
        )

    def process_chat(self, req: ChatRequest) -> ChatResponse:
        session = session_cache.get_session(req.session_id) if req.session_id else None
        history = (
            session_cache.get_chat_history(req.session_id)
            if req.session_id and session
            else []
        )
        plan = self.router.analyze(req.question, history=history)

        if plan.intent == ScientificIntent.CLARIFICATION_REQUIRED:
            response = self._empty_response(
                plan=plan,
                answer=plan.clarification_message or "Could you clarify what information you need?",
                answer_type="CLARIFICATION",
                answer_mode="ABSTAIN",
                reason="The request has more than one plausible scientific meaning.",
                evidence_status="NOT_REQUESTED",
            )
            return self._record(req, plan, response)

        if plan.intent == ScientificIntent.CONCEPT_EXPLANATION:
            return self._record(req, plan, self._knowledge_response(plan, req))

        if plan.intent == ScientificIntent.UNSUPPORTED:
            response = self._empty_response(
                plan=plan,
                answer=compose_unsupported_answer(),
                answer_type="UNSUPPORTED",
                answer_mode="ABSTAIN",
                reason="The request is outside the supported scientific ontology.",
                evidence_status="UNAVAILABLE",
            )
            return self._record(req, plan, response)

        if plan.requires_image and (not session or not req.image_id):
            response = self._empty_response(
                plan=plan,
                answer=compose_image_required_answer(),
                answer_type="IMAGE_REQUIRED",
                answer_mode="ABSTAIN",
                reason="The selected route requires an active uploaded image.",
                evidence_status="IMAGE_REQUIRED",
            )
            return self._record(req, plan, response)

        image_data = session_cache.get_image_data(req.image_id or "")
        if not image_data:
            raise ValueError(f"Image '{req.image_id}' was not found or has expired.")
        if image_data.get("session_id") != req.session_id:
            raise ValueError("The requested image does not belong to this session.")

        image_state = ImageStateBuilder.build(image_data)
        try:
            if plan.intent == ScientificIntent.IMAGE_SUMMARY:
                response = self._image_summary_response(plan, image_state, req, image_data)
            else:
                response = self._prediction_response(plan, req, image_data, image_state)
        except IDCVRuntimeFailure as exc:
            response = self._idcv_runtime_failure_response(plan, exc)
        return self._record(req, plan, response)
