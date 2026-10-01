"""
PhytoSpectra-VLM / WheatSpectraChat
Exact port of the Head-Constrained Scientific Scope Gate (HCSG) from Notebook 11C.
HCSG has ZERO learned parameters.
"""

import re
from typing import Any, Dict, List, Optional
from app.core.constants import PHENOTYPES


VALIDATED_TARGET_PATTERNS = {
    "GrainWeight": [
        r"\bgrain[\s\-]*weight\b",
        r"\bgrain[\s\-]*mass\b",
        r"\bkernel[\s\-]*weight\b",
    ],
    "Gsw": [
        r"\bgsw\b",
        r"\bstomatal\s+conductance\b",
    ],
    "PhiPS2": [
        r"\bphips?2\b",
        r"\bphi\s*ps\s*2\b",
        r"\bphi\s*psii\b",
        r"\bφ\s*psii\b",
    ],
    "Fertilizer": [
        r"\bfertili[sz]er\b",
    ],
    "Cultivar": [
        r"\bcultivar\b",
        r"\bwheat\s+variet(?:y|ies)\b",
    ],
}

EVIDENCE_OPERATOR_PATTERNS = [
    r"\bevidence\b",
    r"\bsupport(?:s|ed|ing)?\b",
    r"\bspatial\b",
    r"\bspectral\b",
    r"\bregion(?:s)?\b",
    r"\btoken(?:s)?\b",
    r"\bband(?:s)?\b",
    r"\battribution\b",
    r"\bexplain\b",
]

COMPARE_OPERATOR_PATTERNS = [
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\bdiffer(?:ence|ent)?\b",
    r"\bjointly\b",
    r"\bboth\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\band\b",
]

SAFE_NEGATION_PATTERNS = [
    r"\bdo\s+not\s+diagnos",
    r"\bdon['’]?t\s+diagnos",
    r"\bwithout\s+(?:a\s+)?diagnos",
    r"\bavoid(?:ing)?\s+(?:a\s+)?(?:disease\s+)?diagnos",
    r"\bdo\s+not\s+(?:claim|infer|classify|assign|determine|predict|estimate)\b",
    r"\bwithout\s+(?:claiming|inferring|classifying|assigning|determining|predicting|estimating)\b",
    r"\bavoid(?:ing)?\s+(?:claiming|inferring|classifying|assigning|determining|predicting|estimating)\b",
    r"\bnot\s+claim\b",
    r"\bnot\s+infer\b",
    r"\bnot\s+predict\b",
    r"\bnot\s+estimate\b",
]

DIAGNOSTIC_REQUEST_PATTERNS = [
    r"\bdiagnos(?:e|is|ing)\b",
    r"\bdisease\s+status\b",
    r"\binfection\s+status\b",
    r"\bdetect\s+(?:a\s+)?disease\b",
    r"\bclassify\s+(?:a\s+)?disease\b",
    r"\bidentify\s+(?:the\s+)?(?:disease|pathogen)\b",
    r"\bdetermine\s+(?:whether|if).*(?:disease|infection|pathogen)\b",
    r"\bdoes\s+.*\s+have\s+.*(?:disease|infection)\b",
    r"\bis\s+.*(?:infected|diseased)\b",
    r"\brust\b",
    r"\bmildew\b",
    r"\bblight\b",
    r"\bpathogen\b",
    r"\bseptoria\b",
    r"\bfusarium\b",
    r"\bsmut\b",
]

UNKNOWN_QUANTITY_PATTERNS = [
    r"\bconcentration\b",
    r"\bcontent\b",
    r"\bpercentage\b",
    r"\bpercent\b",
    r"\brate\b",
    r"\bdensity\b",
    r"\bbiomass\b",
    r"\btemperature\b",
    r"\bthickness\b",
    r"\bpotential\b",
    r"\bindex\b",
    r"\bseverity\b",
    r"\baperture\b",
    r"\barea\s+fraction\b",
    r"\bmass\s+fraction\b",
    r"\brespiration\b",
    r"\btranspiration\b",
    r"\bconductivity\b",
    r"\bmoisture\b",
    r"\bprotein\b",
    r"\bpigment\b",
    r"\bmineral\b",
    r"\bchlorophyll\b",
    r"\bnitrogen\b",
    r"\bcalcium\b",
    r"\bmagnesium\b",
    r"\bpotassium\b",
    r"\bphosphorus\b",
    r"\bspad\b",
    r"\bndvi\b",
    r"\blai\b",
    r"\bleaf\s+area\b",
    r"\banthocyanin\b",
    r"\bwater\b",
]

QUANTITATIVE_OPERATOR_PATTERNS = [
    r"\bestimat(?:e|ed|ing)\b",
    r"\bpredict(?:ed|ion|ing)?\b",
    r"\bmeasure(?:d|ment|ing)?\b",
    r"\bquantif(?:y|ied|ication)\b",
    r"\bnumerical\b",
    r"\bnumeric\b",
    r"\bvalue\b",
    r"\breturn\s+the\b",
    r"\breport\s+the\b",
    r"\bstate\s+the\b",
    r"\bgive\s+the\b",
]

GENERAL_HSI_PATTERNS = [
    r"\bhsi\b",
    r"\bhyperspectral\b",
    r"\bspectral\s+structure\b",
    r"\bspatial\s+structure\b",
    r"\bmodel[\-\s]+supported\b",
    r"\bevidence\b",
    r"\bwhich\s+part\s+of\s+the\s+leaf\b",
    r"\bwhich\s+region\b",
]


def regex_any(patterns: List[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) is not None for pattern in patterns)


def recognized_validated_targets(question: str) -> List[str]:
    targets = []
    for phenotype, patterns in VALIDATED_TARGET_PATTERNS.items():
        if regex_any(patterns, question):
            targets.append(phenotype)
    return targets


def remove_validated_target_mentions(question: str) -> str:
    residual = question
    for patterns in VALIDATED_TARGET_PATTERNS.values():
        for pattern in patterns:
            residual = re.sub(pattern, " ", residual, flags=re.IGNORECASE)
    residual = re.sub(r"\s+", " ", residual).strip()
    return residual


def hcsg_scope(question: str) -> Dict[str, Any]:
    """
    Exact HCSG scope determination logic from Notebook 11C.
    """
    targets = recognized_validated_targets(question)
    safe_negation = regex_any(SAFE_NEGATION_PATTERNS, question)
    diagnostic_request = regex_any(DIAGNOSTIC_REQUEST_PATTERNS, question)
    residual = remove_validated_target_mentions(question)

    compare_signal = regex_any(COMPARE_OPERATOR_PATTERNS, question) or len(targets) > 1
    residual_unknown_quantity = regex_any(UNKNOWN_QUANTITY_PATTERNS, residual)

    unknown_quantity_signal = residual_unknown_quantity and (
        regex_any(QUANTITATIVE_OPERATOR_PATTERNS, question)
        or compare_signal
        or len(targets) > 0
    )

    evidence_signal = regex_any(EVIDENCE_OPERATOR_PATTERNS, question)
    general_hsi_signal = regex_any(GENERAL_HSI_PATTERNS, question)

    # 1. Explicit safe negation dominates diagnostic vocabulary when 0 targets
    if safe_negation and len(targets) == 0:
        return {
            "scope": "SAFE_GENERAL_HSI",
            "targets": [],
            "intent": "GENERAL_HSI",
            "reason": "EXPLICIT_SAFE_NEGATION",
        }

    # 2. Any non-negated diagnosis is outside validated heads
    if diagnostic_request and not safe_negation:
        return {
            "scope": "UNSUPPORTED",
            "targets": targets,
            "intent": "UNSUPPORTED",
            "reason": "UNSUPPORTED_DIAGNOSTIC_REQUEST",
        }

    # 3. Unknown scientific quantity remains unsupported even when a known phenotype is also mentioned
    if unknown_quantity_signal and not safe_negation:
        return {
            "scope": "UNSUPPORTED",
            "targets": targets,
            "intent": "UNSUPPORTED",
            "reason": "UNKNOWN_QUANTITATIVE_VARIABLE",
        }

    # 4. Explicit mapping to one or more dedicated heads
    if len(targets) > 0:
        if compare_signal and len(targets) > 1:
            intent = "COMPARE"
        elif evidence_signal:
            intent = "EVIDENCE"
        else:
            intent = "ESTIMATE"

        return {
            "scope": "SUPPORTED_PHENOTYPE",
            "targets": targets,
            "intent": intent,
            "reason": "VALIDATED_HEAD_MAPPING",
        }

    # 5. No validated target and no unsupported measurement/diagnosis: descriptive HSI allowed
    if general_hsi_signal or safe_negation:
        return {
            "scope": "SAFE_GENERAL_HSI",
            "targets": [],
            "intent": "GENERAL_HSI",
            "reason": "DESCRIPTIVE_HSI_ONLY",
        }

    # 6. Conservative default for unresolved requests
    return {
        "scope": "UNSUPPORTED",
        "targets": [],
        "intent": "UNSUPPORTED",
        "reason": "UNRESOLVED_OUTSIDE_VALIDATED_HEADS",
    }


def hcsg_route_question(qpr_route: Dict[str, Any], question: str) -> Dict[str, Any]:
    gate = hcsg_scope(question)

    if gate["scope"] == "UNSUPPORTED":
        return {
            **qpr_route,
            "predicted_answerability": "UNSUPPORTED",
            "predicted_intent": "UNSUPPORTED",
            "selected_phenotypes": [],
            "numeric_permission": False,
            "ebsa_state": "ABSTAIN",
            "open_set_score": 1.0,
            "hcsg_scope": "UNSUPPORTED",
            "hcsg_reason": gate["reason"],
            "routing_override": True,
        }

    if gate["scope"] == "SAFE_GENERAL_HSI":
        return {
            **qpr_route,
            "predicted_answerability": "SUPPORTED",
            "predicted_intent": "GENERAL_HSI",
            "selected_phenotypes": [],
            "numeric_permission": False,
            "ebsa_state": "EVIDENCE_BOUNDED",
            "open_set_score": 0.0,
            "hcsg_scope": "SAFE_GENERAL_HSI",
            "hcsg_reason": gate["reason"],
            "routing_override": True,
        }

    targets = gate["targets"]
    intent = gate["intent"]

    if intent in ["ESTIMATE", "EVIDENCE"] and len(targets) == 1:
        initial_mode = "VALIDATED"
    else:
        initial_mode = "EVIDENCE_BOUNDED"

    numeric_permission = intent in ["ESTIMATE", "EVIDENCE", "COMPARE"] and len(targets) > 0

    return {
        **qpr_route,
        "predicted_answerability": "SUPPORTED",
        "predicted_intent": intent,
        "selected_phenotypes": targets,
        "numeric_permission": bool(numeric_permission),
        "ebsa_state": initial_mode,
        "open_set_score": 0.0,
        "hcsg_scope": "SUPPORTED_PHENOTYPE",
        "hcsg_reason": gate["reason"],
        "routing_override": (
            qpr_route.get("predicted_intent") != intent
            or set(qpr_route.get("selected_phenotypes", [])) != set(targets)
            or qpr_route.get("ebsa_state") != initial_mode
        ),
    }
