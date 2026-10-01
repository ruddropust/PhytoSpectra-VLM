"""
PhytoSpectra-VLM / WheatSpectraChat
ScientificModelRegistry: Loads all models once at startup and manages scientific components.
"""

from typing import Any, Dict, Optional
import torch

from app.config import settings
from app.core.constants import (
    PARAM_COUNTS,
    PHENOTYPES,
)
from app.inference.normalization import NormalizationManager
from app.inference.phytocore_ensemble import PhytoCoreEnsembleManager
from app.inference.qpr import QPRManager
from app.inference.pctg import PCTGManager
from app.inference.pgc import PGCManager
from app.inference.llm import LLMManager
from app.inference.research_v2_vlm import ResearchV2VLMManager
from app.inference.idcv_development_vlm import IDCVDevelopmentVLMAdapter
from app.inference.frozen_stage40_vlm import FrozenStage40VLMAdapter
from app.inference.frozen_stage42_vlm import FrozenStage42VLMAdapter
from app.inference.research_v3_vlm import ResearchV3VLMAdapter
from app.services.ver_renderer import VERRenderer


class ScientificModelRegistry:
    _instance: Optional["ScientificModelRegistry"] = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

        self.norm_manager: Optional[NormalizationManager] = None
        self.phytocore_ensemble: Optional[PhytoCoreEnsembleManager] = None
        self.qpr_manager: Optional[QPRManager] = None
        self.pctg_manager: Optional[PCTGManager] = None
        self.pgc_manager: Optional[PGCManager] = None
        self.llm_manager: Optional[LLMManager] = None
        self.ver_renderer: Optional[VERRenderer] = None
        self.knowledge_retriever: Optional[Any] = None
        self.research_v2_vlm: Optional[ResearchV2VLMManager] = None
        self.idcv_development_vlm: Optional[IDCVDevelopmentVLMAdapter] = None
        self.idcv_development_error: Optional[str] = None
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ScientificModelRegistry":
        if cls._instance is None:
            cls._instance = ScientificModelRegistry()
        return cls._instance

    def load_all_models(self) -> None:
        """
        Loads all scientific models and checkpoints once during startup.
        Fails clearly if any artifact is missing or incompatible.
        """
        if self.is_loaded:
            return

        print("=" * 70)
        print("LOADING PHYTOSPECTRA-VLM SCIENTIFIC MODEL REGISTRY")
        print(f"Device: {self.device} (GPU: {self.gpu_name})")
        print("=" * 70)

        frozen_exclusive = bool(
            (
                settings.RESEARCH_V3_VLM_ENABLED
                and settings.RESEARCH_V3_EXCLUSIVE_RUNTIME
            )
            or (
                settings.FROZEN_STAGE42_VLM_ENABLED
                and settings.FROZEN_STAGE42_EXCLUSIVE_RUNTIME
            )
            or (
                settings.FROZEN_STAGE40_VLM_ENABLED
                and settings.FROZEN_STAGE40_EXCLUSIVE_RUNTIME
            )
        )
        if frozen_exclusive:
            print("Frozen image-VLM exclusive runtime: skipping obsolete vision stack.")
        else:
            # 1. Normalization
            print("Loading normalization artifacts...")
            self.norm_manager = NormalizationManager()

            # 2. PhytoCore Ensemble (15 checkpoints)
            print("Loading 15-checkpoint PhytoCore ensemble...")
            self.phytocore_ensemble = PhytoCoreEnsembleManager(
                device=self.device,
                norm_manager=self.norm_manager,
                mode=settings.SCIENTIFIC_ENSEMBLE_MODE,
            )

            # 3. QPR & Answerability Gate
            print("Loading QPR & MiniLM text encoder...")
            self.qpr_manager = QPRManager(device=self.device)

            # 4. PCTG Grounder
            print("Loading PCTG budget artifacts...")
            self.pctg_manager = PCTGManager(
                norm_manager=self.norm_manager,
                device=self.device,
            )

            # 5. PGC Prefix Projector
            print("Loading PGC Prefix Projector...")
            self.pgc_manager = PGCManager(device=self.device)

            # 6. LLM & LoRA Adapter
            print("Loading Qwen2.5-1.5B-Instruct + LoRA adapter...")
            self.llm_manager = LLMManager(
                device=self.device,
                pgc_manager=self.pgc_manager,
            )

        # 7. VER Renderer
        print("Initializing Vision Evidence Renderer...")
        self.ver_renderer = VERRenderer()

        # 8. Knowledge Retriever (Optional RAG)
        try:
            from app.retrieval.retriever import KnowledgeRetriever
            print("Loading Knowledge Retriever...")
            self.knowledge_retriever = KnowledgeRetriever()
        except Exception as exc:
            self.knowledge_retriever = None
            print(f"Knowledge Retriever unavailable: {type(exc).__name__}")

        # 9. Locked research_v2 VLM (opt-in). Verification happens before any
        # weights are accepted; model weights may remain lazy until first chat.
        if settings.RESEARCH_V2_VLM_ENABLED:
            print("Verifying locked research_v2 Stage-13 VLM bundle...")
            self.research_v2_vlm = ResearchV2VLMManager(
                bundle_dir=settings.RESEARCH_V2_VLM_BUNDLE_DIR,
                device=self.device,
                require_scientific=settings.RESEARCH_V2_VLM_REQUIRE_SCIENTIFIC,
                eager_load=settings.RESEARCH_V2_VLM_EAGER_LOAD,
            )

        # 10. Completed research_v3 campaign, deployed as one selected fold.
        if settings.RESEARCH_V3_VLM_ENABLED:
            print("Verifying research_v3 Stage-42D campaign model and artifacts...")
            try:
                self.idcv_development_vlm = ResearchV3VLMAdapter(
                    device=self.device,
                    eager_load=settings.RESEARCH_V3_VLM_EAGER_LOAD,
                )
                self.idcv_development_error = self.idcv_development_vlm.verification_error
                if self.idcv_development_error:
                    print(f"research_v3 model verification failed: {self.idcv_development_error}")
                else:
                    from app.core.session_cache import session_cache
                    session_cache.image_expiry_callback = (
                        lambda image_id: self.idcv_development_vlm.runtime.images.pop(image_id, None)
                    )
                    print("research_v3 seed-2026/fold-0 bundle verified successfully.")
            except Exception as exc:
                self.idcv_development_error = f"{type(exc).__name__}: {exc}"
                self.idcv_development_vlm = None
                print(f"research_v3 runtime failed: {self.idcv_development_error}")

        # 11. Verified legacy Stage-42 contract-optimized Fold-0 model. This occupies
        # the established image-VLM adapter slot so upload/chat stay fail-closed.
        elif settings.FROZEN_STAGE42_VLM_ENABLED:
            print("Verifying Stage-42 contract model and outer-fold result...")
            try:
                self.idcv_development_vlm = FrozenStage42VLMAdapter(
                    bundle_dir=settings.FROZEN_STAGE42_VLM_BUNDLE_DIR,
                    phytocore_path=settings.FROZEN_STAGE42_PHYTOCORE_PATH,
                    architecture_freeze_manifest=(
                        settings.FROZEN_STAGE42_ARCHITECTURE_MANIFEST
                    ),
                    protocol_freeze_manifest=(
                        settings.FROZEN_STAGE42_PROTOCOL_MANIFEST
                    ),
                    recipe_freeze_manifest=(
                        settings.FROZEN_STAGE42_RECIPE_MANIFEST
                    ),
                    stage42c_report=settings.FROZEN_STAGE42_TRAINING_REPORT,
                    stage42d_report=settings.FROZEN_STAGE42_EVALUATION_REPORT,
                    stage42d_report_sha256=(
                        settings.FROZEN_STAGE42_EVALUATION_REPORT_SHA256
                    ),
                    device=self.device,
                    eager_load=settings.FROZEN_STAGE42_VLM_EAGER_LOAD,
                    debug_raw_qwen=settings.FROZEN_STAGE42_DEBUG_RAW_QWEN,
                )
                self.idcv_development_error = None
                print("Stage-42 seed-2026/fold-0 contract VLM verified successfully.")
            except Exception as exc:
                self.idcv_development_error = f"{type(exc).__name__}: {exc}"
                print(
                    "Stage-42 Fold-0 VLM failed: "
                    f"{self.idcv_development_error}"
                )
                self.idcv_development_vlm = None

        # 11. Frozen post-remediation Stage-40S model. This occupies the same
        # chat-pipeline adapter slot as the older Stage-30 runtime so all image
        # questions remain fail-closed and use one authoritative VLM.
        elif settings.FROZEN_STAGE40_VLM_ENABLED:
            print("Verifying frozen Stage-40S Fold-0 VLM and protocol binding...")
            try:
                self.idcv_development_vlm = FrozenStage40VLMAdapter(
                    bundle_dir=settings.FROZEN_STAGE40_VLM_BUNDLE_DIR,
                    phytocore_path=settings.FROZEN_STAGE40_PHYTOCORE_PATH,
                    architecture_freeze_manifest=(
                        settings.FROZEN_STAGE40_ARCHITECTURE_MANIFEST
                    ),
                    protocol_freeze_manifest=(
                        settings.FROZEN_STAGE40_PROTOCOL_MANIFEST
                    ),
                    device=self.device,
                    eager_load=settings.FROZEN_STAGE40_VLM_EAGER_LOAD,
                    debug_raw_qwen=settings.FROZEN_STAGE40_DEBUG_RAW_QWEN,
                )
                self.idcv_development_error = None
                print("Frozen Stage-40S Fold-0 VLM verified successfully.")
            except Exception as exc:
                self.idcv_development_error = f"{type(exc).__name__}: {exc}"
                print(
                    "Frozen Stage-40S Fold-0 VLM failed: "
                    f"{self.idcv_development_error}"
                )
                self.idcv_development_vlm = None

        # 12. Legacy Stage-30 ID-CV development VLM (opt-in, Fold-0 only)
        elif settings.IDCV_DEVELOPMENT_VLM_ENABLED:
            print("Verifying Stage-30 ID-CV development VLM bundle...")
            try:
                self.idcv_development_vlm = IDCVDevelopmentVLMAdapter(
                    bundle_dir=settings.IDCV_DEVELOPMENT_VLM_BUNDLE_DIR,
                    phytocore_path=settings.IDCV_DEVELOPMENT_PHYTOCORE_PATH,
                    device=self.device,
                    eager_load=settings.IDCV_DEVELOPMENT_VLM_EAGER_LOAD,
                    debug_raw_qwen=settings.IDCV_DEVELOPMENT_DEBUG_RAW_QWEN,
                )
                self.idcv_development_error = None
                print("Stage-30 ID-CV development VLM verified successfully.")
            except Exception as exc:
                self.idcv_development_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"Stage-30 ID-CV development VLM failed: "
                    f"{self.idcv_development_error}"
                )
                self.idcv_development_vlm = None

        self.is_loaded = True
        print("=" * 70)
        if self.idcv_development_error is None:
            print("ALL CONFIGURED SCIENTIFIC MODELS LOADED SUCCESSFULLY")
        else:
            print("CORE MODELS LOADED; CONFIGURED ID-CV VLM IS UNAVAILABLE")
        runtime_parameter_count = PARAM_COUNTS["total_project_specific"]
        if self.idcv_development_vlm is not None:
            runtime_parameter_count = int(
                self.idcv_development_vlm.status().get(
                    "total_learned_parameters", runtime_parameter_count
                )
            )
        print(
            "Total project-specific learned parameters: "
            f"{runtime_parameter_count:,}"
        )
        print("=" * 70)

    def get_status(self) -> Dict[str, Any]:
        research_v2 = (
            self.research_v2_vlm.status()
            if self.research_v2_vlm is not None
            else {
                "enabled": bool(settings.RESEARCH_V2_VLM_ENABLED),
                "verified": False,
                "loaded": False,
                "scientific_result": False,
                "activation_allowed": False,
                "deployment_status": "DISABLED",
                "bundle_dir": str(settings.RESEARCH_V2_VLM_BUNDLE_DIR),
                "outer_fold": None,
                "model_seed": None,
            }
        )
        image_vlm_status = (
            self.idcv_development_vlm.status()
            if self.idcv_development_vlm is not None
            else {}
        )
        learned_parameters = image_vlm_status.get(
            "learned_parameters", PARAM_COUNTS
        )
        total_learned_parameters = image_vlm_status.get(
            "total_learned_parameters",
            PARAM_COUNTS.get("total_project_specific", 13_930_483),
        )
        configured_image_vlm = bool(
            settings.RESEARCH_V3_VLM_ENABLED
            or settings.FROZEN_STAGE42_VLM_ENABLED
            or settings.FROZEN_STAGE40_VLM_ENABLED
            or settings.IDCV_DEVELOPMENT_VLM_ENABLED
        )
        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(self.device),
            "gpu_name": self.gpu_name,
            "phytocore_loaded": self.phytocore_ensemble is not None,
            "phytocore_checkpoints_loaded": (
                self.phytocore_ensemble.total_checkpoints_loaded
                if self.phytocore_ensemble
                else 0
            ),
            "qpr_loaded": self.qpr_manager is not None,
            "hcsg_active": True,
            "pctg_available": self.pctg_manager is not None,
            "pgc_projector_loaded": self.pgc_manager is not None,
            "qwen_loaded": (
                self.llm_manager is not None
                or bool(image_vlm_status.get("loaded", False))
            ),
            "lora_loaded": (
                self.llm_manager is not None
                or bool(image_vlm_status.get("loaded", False))
            ),
            "ver_available": self.ver_renderer is not None,
            "scientific_ensemble_mode": settings.SCIENTIFIC_ENSEMBLE_MODE,
            "learned_parameters": learned_parameters,
            "total_learned_parameters": total_learned_parameters,
            "supported_targets": PHENOTYPES,
            "research_v2_vlm": research_v2,
            "idcv_development_vlm": (
                self.idcv_development_vlm.status()
                if self.idcv_development_vlm is not None
                else {
                    "enabled": configured_image_vlm,
                    "verified": False,
                    "loaded": False,
                    "scientific_result": False,
                    "activation_allowed": False,
                    "deployment_status": (
                        "ERROR"
                        if configured_image_vlm
                        else "DISABLED"
                    ),
                    "bundle_dir": str(
                        (settings.PROJECT_ROOT / "artifacts" / "stage42d_exact_v1" / "runs" / "seed_2026_fold_0" / "vlm" / "contract" / "bundle")
                        if settings.RESEARCH_V3_VLM_ENABLED
                        else settings.FROZEN_STAGE42_VLM_BUNDLE_DIR
                        if settings.FROZEN_STAGE42_VLM_ENABLED
                        else settings.FROZEN_STAGE40_VLM_BUNDLE_DIR
                        if settings.FROZEN_STAGE40_VLM_ENABLED
                        else settings.IDCV_DEVELOPMENT_VLM_BUNDLE_DIR
                    ),
                    "phytocore_path": str(
                        (settings.PROJECT_ROOT / "artifacts" / "stage42d_exact_v1" / "runs" / "seed_2026_fold_0" / "phytocore" / "phytocore_final.pt")
                        if settings.RESEARCH_V3_VLM_ENABLED
                        else settings.FROZEN_STAGE42_PHYTOCORE_PATH
                        if settings.FROZEN_STAGE42_VLM_ENABLED
                        else settings.FROZEN_STAGE40_PHYTOCORE_PATH
                        if settings.FROZEN_STAGE40_VLM_ENABLED
                        else settings.IDCV_DEVELOPMENT_PHYTOCORE_PATH
                    ),
                    "outer_fold": 0,
                    "model_seed": 2026,
                    "verification_error": self.idcv_development_error,
                    "ema_weights": True,
                    "vision_frozen": True,
                    "architecture": (
                        "Frozen PhytoCore + phenotype-conditioned visual token bridge "
                        "+ Qwen-LoRA + Stage-33 grounded verifier"
                    ),
                }
            ),
        }


def get_model_registry() -> ScientificModelRegistry:
    return ScientificModelRegistry.get_instance()
