"""
PhytoSpectra-VLM / WheatSpectraChat
Exact port of PGVRC (Phenotype-Grounded Verification Reasoning Chain) from Notebook 08B / 11C.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from app.core.constants import (
    PHENOTYPES,
    REGRESSION_TARGETS,
    DISCLAIMER_CONFOUNDER,
    DISCLAIMER_GROUNDING,
    DISCLAIMER_SPECTRAL,
)


def compute_token_jaccard(tokens_a: List[int], tokens_b: List[int]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return float(len(set_a & set_b) / len(set_a | set_b))


def evaluate_confounder_risk(
    selected_phenotypes: List[str],
    pctg_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates confounder risk specifically between GrainWeight and Fertilizer.
    """
    gw_spatial = []
    gw_spectral = []
    fert_spatial = []
    fert_spectral = []

    if "GrainWeight" in pctg_evidence:
        gw_spatial = pctg_evidence["GrainWeight"]["spatial"]["selected_token_indices"]
        gw_spectral = pctg_evidence["GrainWeight"]["spectral"]["selected_token_indices"]

    if "Fertilizer" in pctg_evidence:
        fert_spatial = pctg_evidence["Fertilizer"]["spatial"]["selected_token_indices"]
        fert_spectral = pctg_evidence["Fertilizer"]["spectral"]["selected_token_indices"]

    spatial_jaccard = compute_token_jaccard(gw_spatial, fert_spatial)
    spectral_jaccard = compute_token_jaccard(gw_spectral, fert_spectral)

    has_gw = "GrainWeight" in selected_phenotypes
    has_fert = "Fertilizer" in selected_phenotypes

    if has_gw and has_fert:
        risk = "HIGH"
    elif has_gw or has_fert:
        risk = "HIGH_CONTEXTUAL" if (spatial_jaccard > 0.3 or spectral_jaccard > 0.3) else "MODERATE"
    else:
        risk = "LOW"

    return {
        "risk": risk,
        "spatial_token_jaccard": spatial_jaccard,
        "spectral_token_jaccard": spectral_jaccard,
        "warning_needed": has_gw or has_fert,
    }


def run_pgvrc(
    route: Dict[str, Any],
    pctg_result: Dict[str, Any],
    predictions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes the 5-stage Phenotype-Grounded Verification Reasoning Chain.
    Stage 1: Query Support Assessment
    Stage 2: Phenotype Hypothesis Routing
    Stage 3: Spatial-Spectral Evidence Verification
    Stage 4: Counterfactual + Confounder Verification
    Stage 5: Evidence-Bounded Selective Answering
    """
    initial_mode = route.get("ebsa_state", "EVIDENCE_BOUNDED")
    selected_phenotypes = route.get("selected_phenotypes", [])
    numeric_permission = route.get("numeric_permission", False)

    downgrade_reasons: List[str] = []
    warnings: List[str] = []

    # If already abstaining, return clean abstention
    if initial_mode == "ABSTAIN":
        return {
            "final_answer_mode": "ABSTAIN",
            "initial_mode": "ABSTAIN",
            "numeric_permission": False,
            "selected_phenotypes": [],
            "scientific_predictions": {},
            "evidence": {},
            "verification": {},
            "verification_summary": {},
            "confounder": {"risk": "LOW"},
            "downgrade_reasons": ["UNSUPPORTED_QUERY"],
            "warnings": [],
            "all_dual_verified": False,
        }

    evidence_dict = pctg_result.get("evidence", {})
    verification_dict: Dict[str, Dict[str, Any]] = {}
    verification_summary: Dict[str, Dict[str, Any]] = {}
    all_dual_verified = bool(selected_phenotypes)

    for p in selected_phenotypes:
        if p in evidence_dict:
            p_ev = evidence_dict[p]
            s_comp = p_ev["spatial"]["comprehensiveness_gain"]
            s_suff = p_ev["spatial"]["sufficiency_advantage"]
            l_comp = p_ev["spectral"]["comprehensiveness_gain"]
            l_suff = p_ev["spectral"]["sufficiency_advantage"]

            s_verified = s_comp > 0 and s_suff > 0
            l_verified = l_comp > 0 and l_suff > 0

            verification_dict[p] = {
                "spatial": {
                    "comprehensiveness_gain": s_comp,
                    "sufficiency_advantage": s_suff,
                    "verified": s_verified,
                },
                "spectral": {
                    "comprehensiveness_gain": l_comp,
                    "sufficiency_advantage": l_suff,
                    "verified": l_verified,
                },
            }

            verification_summary[p] = {
                "spatial": {
                    "sample_status": "VERIFIED" if s_verified else "PARTIAL",
                },
                "spectral": {
                    "sample_status": "VERIFIED" if l_verified else "PARTIAL",
                },
            }

            if not (s_verified and l_verified):
                all_dual_verified = False
                if not s_verified:
                    downgrade_reasons.append(f"{p}_spatial_evidence_weak")
                if not l_verified:
                    downgrade_reasons.append(f"{p}_spectral_evidence_weak")
        else:
            all_dual_verified = False
            downgrade_reasons.append(f"{p}_evidence_missing")

    # Evaluate confounder
    confounder_info = evaluate_confounder_risk(selected_phenotypes, evidence_dict)
    if confounder_info["warning_needed"]:
        warnings.append(DISCLAIMER_CONFOUNDER)

    # Stage 4 controls Stage 5 final response state
    final_mode = initial_mode
    if initial_mode == "VALIDATED" and not all_dual_verified:
        final_mode = "EVIDENCE_BOUNDED"
        downgrade_reasons.append("STAGE4_EVIDENCE_VERIFICATION_PARTIAL")

    # Filter predictions for selected phenotypes
    filtered_predictions: Dict[str, Any] = {}
    if numeric_permission:
        for p in selected_phenotypes:
            if p in predictions:
                filtered_predictions[p] = predictions[p]

    return {
        "final_answer_mode": final_mode,
        "initial_mode": initial_mode,
        "numeric_permission": numeric_permission,
        "selected_phenotypes": selected_phenotypes,
        "scientific_predictions": filtered_predictions,
        "evidence": {p: evidence_dict[p] for p in selected_phenotypes if p in evidence_dict},
        "verification": verification_dict,
        "verification_summary": verification_summary,
        "confounder": confounder_info,
        "downgrade_reasons": downgrade_reasons,
        "warnings": warnings,
        "all_dual_verified": all_dual_verified,
    }
