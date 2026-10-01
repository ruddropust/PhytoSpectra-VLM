"""
PhytoSpectra-VLM / WheatSpectraChat
FastAPI request and response schemas.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    project: str = "PhytoSpectra-VLM"


class ModelStatusResponse(BaseModel):
    cuda_available: bool = False
    device: str = "cpu"
    gpu_name: Optional[str] = None
    phytocore_loaded: bool = False
    phytocore_checkpoints_loaded: int = 0
    qpr_loaded: bool = False
    hcsg_active: bool = True
    pctg_available: bool = False
    pgc_projector_loaded: bool = False
    qwen_loaded: bool = False
    lora_loaded: bool = False
    ver_available: bool = False
    scientific_ensemble_mode: str = "cv15"
    learned_parameters: Dict[str, int] = Field(default_factory=dict)
    total_learned_parameters: int = 0
    supported_targets: List[str] = Field(default_factory=list)
    research_v2_vlm: Dict[str, Any] = Field(default_factory=dict)
    idcv_development_vlm: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    created_at: str


class MaskStatistics(BaseModel):
    raw_foreground_pixels: int
    final_foreground_pixels: int
    raw_components: int
    kept_components: int
    retained_foreground_ratio: Optional[float] = None
    foreground_fraction: float
    sat_fraction: Optional[float] = None
    near_sat_fraction: Optional[float] = None
    bbox: Optional[List[int]] = None


class UploadResponse(BaseModel):
    session_id: str
    image_id: str
    original_shape: List[int]
    canonical_shape: List[int]
    mask_statistics: MaskStatistics
    preview_url: str
    supported_targets: List[str]


class PredictionItem(BaseModel):
    value: Optional[Union[float, str]] = None
    unit: Optional[str] = None
    ensemble_std: Optional[float] = None
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    calibration_status: str = "NOT_AVAILABLE_FOR_LIVE_MODE"
    cultivar_probabilities: Optional[Dict[str, float]] = None
    vote_confidence: Optional[float] = None
    seed_disagreement: Optional[float] = None


class ScopeDetails(BaseModel):
    hcsg: str
    reason: str
    open_set_score: float = 0.0
    predicted_intent: str


class VerificationDetails(BaseModel):
    response_policy: str
    evidence_status: str
    spatial_status: Optional[str] = None
    spectral_status: Optional[str] = None
    global_evidence_strength: Optional[str] = None
    all_dual_verified: bool = False
    downgrade_reasons: List[str] = Field(default_factory=list)
    raw_answer_accepted: Optional[bool] = None
    fallback_used: Optional[bool] = None
    verification_reasons: List[str] = Field(default_factory=list)
    grounding_source: Optional[str] = None
    ema_weights: Optional[bool] = None
    vision_frozen: Optional[bool] = None
    scientific_result: Optional[bool] = None


class SpatialEvidence(BaseModel):
    selected_tokens: List[Dict[str, Any]] = Field(default_factory=list)
    budget: float = 0.1
    validation_strength: str = "STRONG"
    comprehensiveness_gain: Optional[float] = None
    sufficiency_advantage: Optional[float] = None


class SpectralEvidence(BaseModel):
    selected_token_indices: List[int] = Field(default_factory=list)
    budget: float = 0.3
    validation_strength: str = "STRONG"
    comprehensiveness_gain: Optional[float] = None
    sufficiency_advantage: Optional[float] = None
    disclaimer: str = (
        "Spectral evidence is reported in model token/band-index space; "
        "physical wavelength mapping is not asserted."
    )


class TargetVisualExplanation(BaseModel):
    target: str
    prediction_text: str
    visual_method: str = "PCTG_COUNTERFACTUAL_PATCH_ATTRIBUTION"
    evidence_available: bool = False
    focus_box: Optional[List[int]] = None
    selected_token_indices: List[int] = Field(default_factory=list)
    selected_token_count: int = 0
    spatial_summary: str = ""
    heatmap_url: Optional[str] = None
    marked_url: Optional[str] = None
    crop_url: Optional[str] = None
    zoom_url: Optional[str] = None
    spectral_plot_url: Optional[str] = None
    bundle_url: Optional[str] = None


class VisualsPayload(BaseModel):
    composite_url: Optional[str] = None
    marked_url: Optional[str] = None
    zoom_url: Optional[str] = None
    spectral_plot_url: Optional[str] = None
    bundle_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    target_visuals: Dict[str, TargetVisualExplanation] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    image_id: Optional[str] = None
    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer_mode: Literal["VALIDATED", "EVIDENCE_BOUNDED", "ABSTAIN"]
    answer_type: Literal[
        "CONCEPT_EXPLANATION",
        "IMAGE_SUMMARY",
        "MODEL_PREDICTION",
        "EVIDENCE_EXPLANATION",
        "CLARIFICATION",
        "IMAGE_REQUIRED",
        "UNSUPPORTED",
        "RUNTIME_ERROR",
    ]
    answer: str
    numeric_permission: bool
    intent: str
    selected_phenotypes: List[str]
    predictions: Dict[str, PredictionItem] = Field(default_factory=dict)
    scope: ScopeDetails
    verification: VerificationDetails
    evidence: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    visuals: VisualsPayload = Field(default_factory=VisualsPayload)
    development_warning: Optional[str] = None
    debug_raw_qwen: Optional[str] = None
