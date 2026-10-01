"""
PhytoSpectra-VLM / WheatSpectraChat
Application configuration and path management.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_APP_ROOT = _DEFAULT_BACKEND_ROOT.parent
_DEFAULT_PROJECT_ROOT = _DEFAULT_APP_ROOT.parent
_DEFAULT_RESEARCH_ROOT = _DEFAULT_PROJECT_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "WheatSpectraChat API"
    APP_VERSION: str = "1.0.0"
    PROJECT_NAME: str = "PhytoSpectra-VLM"
    DESCRIPTION: str = "Phenotype-Grounded Hyperspectral Reasoning for Wheat"

    # Host & Port
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Root paths
    PROJECT_ROOT: Path = _DEFAULT_PROJECT_ROOT
    APP_ROOT: Path = _DEFAULT_APP_ROOT
    BACKEND_ROOT: Path = _DEFAULT_BACKEND_ROOT
    RESEARCH_ROOT: Path = _DEFAULT_RESEARCH_ROOT
    MEDIA_DIR: Path = BACKEND_ROOT / "media"

    # Artifact paths
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    NORMALIZATION_DIR: Path = ARTIFACTS_DIR / "03_idcv_normalization"
    BACKBONE_DIR: Path = ARTIFACTS_DIR / "05_phytocore_3seed_5fold"
    QPR_DIR: Path = ARTIFACTS_DIR / "06C_binary_answerability_qpr"
    PCTG_DIR: Path = ARTIFACTS_DIR / "07D_token_counterfactual_grounder"
    LANGUAGE_ALIGNMENT_DIR: Path = ARTIFACTS_DIR / "10_full_hyperleafqa_pgc_alignment" / "language_alignment"

    # Completed research_v3 3x5 campaign; the live app uses one evaluated
    # seed/fold checkpoint and does not claim to deploy the aggregate.
    RESEARCH_V3_VLM_ENABLED: bool = True
    RESEARCH_V3_EXCLUSIVE_RUNTIME: bool = True
    RESEARCH_V3_VLM_EAGER_LOAD: bool = False

    # Stage-13 research VLM. It remains opt-in until a scientific bundle has
    # passed the locked evaluation gates. A smoke bundle is always rejected
    # when REQUIRE_SCIENTIFIC is true.
    RESEARCH_V2_VLM_ENABLED: bool = False
    RESEARCH_V2_VLM_BUNDLE_DIR: Path = (
        RESEARCH_ROOT
        / "artifacts"
        / "18_phytospectra_deployment"
        / "fold_0"
        / "seed_2026"
    )
    RESEARCH_V2_VLM_REQUIRE_SCIENTIFIC: bool = True
    RESEARCH_V2_VLM_EAGER_LOAD: bool = False

    # Stage-30 ID-CV development VLM (Fold-0 only). Separate from the
    # Stage-13 research adapter above. Uses the classification PhytoCore
    # and Stage-30 EMA Qwen-LoRA bundle.
    IDCV_DEVELOPMENT_VLM_ENABLED: bool = False
    IDCV_DEVELOPMENT_VLM_BUNDLE_DIR: Path = PROJECT_ROOT / "artifacts" / "30_idcv_phytospectra_vlm"
    IDCV_DEVELOPMENT_PHYTOCORE_PATH: Path = (
        PROJECT_ROOT / "artifacts" / "28_idcv_classification_phytocore"
        / "phytocore_weighted_ce_seed2026_fold0_best.pt"
    )
    IDCV_DEVELOPMENT_VLM_EAGER_LOAD: bool = False
    IDCV_DEVELOPMENT_DEBUG_RAW_QWEN: bool = False

    # Frozen post-remediation Fold-0 runtime used by the chat application.
    # It is integrity-bound to both the Stage-40T architecture freeze and the
    # Stage-40Y final-evaluation protocol freeze. This remains an internal
    # development runtime until the frozen 3-seed x 5-fold protocol completes.
    FROZEN_STAGE40_VLM_ENABLED: bool = False
    FROZEN_STAGE40_VLM_BUNDLE_DIR: Path = (
        PROJECT_ROOT / "results" / "40s_soup_select_soup_v2" / "bundle_alpha_060"
    )
    FROZEN_STAGE40_PHYTOCORE_PATH: Path = (
        PROJECT_ROOT / "artifacts" / "39_idcv_locked_5x5_ep50" / "phytocore"
        / "phytocore_weighted_ce_seed2026_fold0_final.pt"
    )
    FROZEN_STAGE40_ARCHITECTURE_MANIFEST: Path = (
        PROJECT_ROOT / "artifacts" / "40t_architecture_freeze_v1"
        / "architecture_freeze_manifest.json"
    )
    FROZEN_STAGE40_PROTOCOL_MANIFEST: Path = (
        PROJECT_ROOT / "artifacts" / "40y_final_3x5_protocol_user_v1"
        / "final_3x5_protocol_manifest.json"
    )
    FROZEN_STAGE40_VLM_EAGER_LOAD: bool = False
    FROZEN_STAGE40_DEBUG_RAW_QWEN: bool = False
    # Avoid loading/running the obsolete chatbot vision stack beside the
    # frozen model. Curated knowledge routing remains available.
    FROZEN_STAGE40_EXCLUSIVE_RUNTIME: bool = True

    # Current verified chatbot runtime: the Stage-42 contract-optimized
    # seed-2026/fold-0 model.  The exact outer-evaluation report is hash-bound,
    # so a changed or incomplete run fails closed instead of silently loading.
    FROZEN_STAGE42_VLM_ENABLED: bool = False
    FROZEN_STAGE42_VLM_BUNDLE_DIR: Path = (
        PROJECT_ROOT / "results" / "42a_contract_optimization_train_contract_all_v4"
        / "bundle"
    )
    FROZEN_STAGE42_PHYTOCORE_PATH: Path = (
        PROJECT_ROOT / "artifacts" / "41_idcv_final_3x5_v1" / "runs"
        / "seed_2026_fold_0" / "phytocore"
        / "phytocore_weighted_ce_seed2026_fold0_final.pt"
    )
    FROZEN_STAGE42_ARCHITECTURE_MANIFEST: Path = (
        PROJECT_ROOT / "artifacts" / "40t_architecture_freeze_v1"
        / "architecture_freeze_manifest.json"
    )
    FROZEN_STAGE42_PROTOCOL_MANIFEST: Path = (
        PROJECT_ROOT / "artifacts" / "40y_final_3x5_protocol_user_v1"
        / "final_3x5_protocol_manifest.json"
    )
    FROZEN_STAGE42_RECIPE_MANIFEST: Path = (
        PROJECT_ROOT / "artifacts" / "42b_contract_recipe_freeze_recipe_user_v1"
        / "contract_recipe_freeze_manifest.json"
    )
    FROZEN_STAGE42_TRAINING_REPORT: Path = (
        PROJECT_ROOT / "results" / "42c_final_contract_train_contract_cv_pilot_v2"
        / "runs" / "seed_2026_fold_0" / "stage42c_report.json"
    )
    FROZEN_STAGE42_EVALUATION_REPORT: Path = (
        PROJECT_ROOT / "results"
        / "42d_final_contract_outer_evaluate_contract_outer_pilot_v3"
        / "runs" / "seed_2026_fold_0" / "stage42d_report.json"
    )
    FROZEN_STAGE42_EVALUATION_REPORT_SHA256: str = (
        "62c86adf0c28e17022206f1aa9525ab974ddad6098faf8335397a2daca9bb8be"
    )
    FROZEN_STAGE42_VLM_EAGER_LOAD: bool = False
    FROZEN_STAGE42_DEBUG_RAW_QWEN: bool = False
    FROZEN_STAGE42_EXCLUSIVE_RUNTIME: bool = True

    # Scientific inference mode: 'cv15' or 'deployment3'
    SCIENTIFIC_ENSEMBLE_MODE: str = "cv15"

    # Session cache TTL in seconds (default 1 hour)
    CACHE_TTL_SECONDS: int = 3600

    # Max upload size in megabytes
    MAX_UPLOAD_SIZE_MB: int = 100

    # Allowed CORS origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

settings = Settings()

_configured_image_runtimes = sum(
    bool(value)
    for value in (
        settings.RESEARCH_V3_VLM_ENABLED,
        settings.FROZEN_STAGE42_VLM_ENABLED,
        settings.FROZEN_STAGE40_VLM_ENABLED,
        settings.IDCV_DEVELOPMENT_VLM_ENABLED,
    )
)
if _configured_image_runtimes > 1:
    raise ValueError(
        "Enable only one image VLM runtime: research_v3, Stage-42, Stage-40, or legacy Stage-30."
    )

# Ensure media directory exists
settings.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
