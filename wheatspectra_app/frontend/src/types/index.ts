export interface ScientificRuntimeStatus {
  enabled: boolean;
  verified: boolean;
  loaded: boolean;
  scientific_result: boolean;
  activation_allowed: boolean;
  deployment_status: string;
  bundle_dir?: string;
  phytocore_path?: string;
  outer_fold?: number | null;
  model_seed?: number | null;
  verification_error?: string | null;
  ema_weights?: boolean;
  vision_frozen?: boolean;
  architecture?: string;
  architecture_freeze_verified?: boolean;
  protocol_freeze_verified?: boolean;
  recipe_freeze_verified?: boolean;
  outer_evaluation_verified?: boolean;
  selected_interpolation_alpha?: number;
  contract_decoding?: string;
  evaluated_images?: number;
  evaluated_questions?: number;
  raw_reference_acceptance?: number;
  raw_class_agreement_with_phytocore?: number;
  verified_nbest_acceptance?: number;
  scope_warning?: string;
}

export interface ModelStatus {
  cuda_available: boolean;
  device: string;
  gpu_name: string | null;
  phytocore_loaded: boolean;
  phytocore_checkpoints_loaded: number;
  qpr_loaded: boolean;
  hcsg_active: boolean;
  pctg_available: boolean;
  pgc_projector_loaded: boolean;
  qwen_loaded: boolean;
  lora_loaded: boolean;
  ver_available: boolean;
  scientific_ensemble_mode: string;
  learned_parameters: Record<string, number>;
  total_learned_parameters: number;
  supported_targets: string[];
  research_v2_vlm: ScientificRuntimeStatus;
  idcv_development_vlm: ScientificRuntimeStatus;
}

export interface MaskStatistics {
  raw_foreground_pixels: number;
  final_foreground_pixels: number;
  raw_components: number;
  kept_components: number;
  retained_foreground_ratio: number | null;
  foreground_fraction: number;
  sat_fraction: number | null;
  near_sat_fraction: number | null;
  bbox: number[] | null;
}

export interface UploadResult {
  session_id: string;
  image_id: string;
  original_shape: number[];
  canonical_shape: number[];
  mask_statistics: MaskStatistics;
  preview_url: string;
  supported_targets: string[];
}

export interface PredictionItem {
  value: number | string | null;
  unit: string | null;
  ensemble_std: number | null;
  interval_low: number | null;
  interval_high: number | null;
  calibration_status: string;
  cultivar_probabilities?: Record<string, number>;
  vote_confidence?: number;
  seed_disagreement?: number;
}

export interface ScopeDetails {
  hcsg: string;
  reason: string;
  open_set_score: number;
  predicted_intent: string;
}

export interface VerificationDetails {
  response_policy: string;
  evidence_status: string;
  spatial_status?: string;
  spectral_status?: string;
  global_evidence_strength?: string;
  all_dual_verified: boolean;
  downgrade_reasons: string[];
  raw_answer_accepted?: boolean | null;
  fallback_used?: boolean | null;
  verification_reasons?: string[];
  grounding_source?: string | null;
  ema_weights?: boolean | null;
  vision_frozen?: boolean | null;
  scientific_result?: boolean | null;
}

export interface VisualsPayload {
  composite_url?: string;
  marked_url?: string;
  zoom_url?: string;
  spectral_plot_url?: string;
  bundle_url?: string;
  heatmap_url?: string;
  target_visuals?: Record<string, TargetVisualExplanation>;
}

export interface TargetVisualExplanation {
  target: string;
  prediction_text: string;
  visual_method: "PCTG_COUNTERFACTUAL_PATCH_ATTRIBUTION" | "RESEARCH_V2_PCTG_COUNTERFACTUAL_ATTRIBUTION" | "DIRECT_PHYTOCORE_TOKEN_SENSITIVITY";
  evidence_available: boolean;
  focus_box: number[] | null;
  selected_token_indices: number[];
  selected_token_count: number;
  spatial_summary: string;
  heatmap_url?: string;
  marked_url?: string;
  crop_url?: string;
  zoom_url?: string;
  spectral_plot_url?: string;
  bundle_url?: string;
}

export type AnswerMode = "VALIDATED" | "EVIDENCE_BOUNDED" | "ABSTAIN";

export type AnswerType =
  | "CONCEPT_EXPLANATION"
  | "IMAGE_SUMMARY"
  | "MODEL_PREDICTION"
  | "EVIDENCE_EXPLANATION"
  | "CLARIFICATION"
  | "IMAGE_REQUIRED"
  | "UNSUPPORTED"
  | "RUNTIME_ERROR";

export interface ChatResponse {
  answer_mode: AnswerMode;
  answer_type: AnswerType;
  answer: string;
  numeric_permission: boolean;
  intent: string;
  selected_phenotypes: string[];
  predictions: Record<string, PredictionItem>;
  scope: ScopeDetails;
  verification: VerificationDetails;
  evidence: Record<string, any>;
  warnings: string[];
  visuals: VisualsPayload;
  development_warning?: string | null;
  debug_raw_qwen?: string | null;
}

export interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  text?: string;
  response?: ChatResponse;
  timestamp: string;
}

export interface ConversationSession {
  id: string;
  sessionId?: string | null;
  title: string;
  createdAt: number;
  updatedAt: number;
  uploadData: UploadResult | null;
  activeFileName: string | null;
  messages: ChatMessage[];
}
