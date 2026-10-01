"""
PhytoSpectra-VLM / WheatSpectraChat
Core scientific constants, shapes, classes, and parameter definitions.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# HSI Cube Specifications
# ============================================================
N_BANDS = 204
HEIGHT = 48
WIDTH = 352

EXPECTED_RAW_SHAPE: Tuple[int, int, int] = (204, 48, 352)
EXPECTED_HWB_SHAPE: Tuple[int, int, int] = (48, 352, 204)

UINT16_MAX = 65535.0
MASK_SUPPORT_THRESHOLD = 0.75
MIN_COMPONENT_SIZE = 20
FILL_HOLES = True

# ============================================================
# Tokenizer / Architecture Configurations
# ============================================================
D_MODEL = 128
N_HEADS = 4
SPECTRAL_DEPTH = 2
SPATIAL_DEPTH = 2
DROPOUT = 0.1
SPATIAL_PATCH: Tuple[int, int] = (8, 16)

N_SPATIAL_TOKENS = (HEIGHT // SPATIAL_PATCH[0]) * (WIDTH // SPATIAL_PATCH[1])  # 6 * 22 = 132
N_SPECTRAL_TOKENS = ((N_BANDS - 8) // 4) + 1  # 50 tokens

# ============================================================
# Validated Targets & Classes
# ============================================================
PHENOTYPES: List[str] = [
    "GrainWeight",
    "Gsw",
    "PhiPS2",
    "Fertilizer",
    "Cultivar",
]

REGRESSION_TARGETS: List[str] = [
    "GrainWeight",
    "Gsw",
    "PhiPS2",
    "Fertilizer",
]

CULTIVAR_NAMES: List[str] = [
    "Heerup",
    "Kvium",
    "Rembrandt",
    "Sheriff",
]

# Scientific units - null because physical mapping / metadata is not asserted
TARGET_UNITS: Dict[str, None] = {
    "GrainWeight": None,
    "Gsw": None,
    "PhiPS2": None,
    "Fertilizer": None,
    "Cultivar": None,
}

# ============================================================
# Routing / Gate Definitions
# ============================================================
ANSWERABILITY_CLASSES: List[str] = [
    "SUPPORTED",
    "UNSUPPORTED",
]

INTENT_CLASSES: List[str] = [
    "ESTIMATE",
    "EVIDENCE",
    "COMPARE",
    "GENERAL_HSI",
    "UNSUPPORTED",
]

EBSA_STATES: List[str] = [
    "VALIDATED",
    "EVIDENCE_BOUNDED",
    "ABSTAIN",
]

MODE_CLASSES: List[str] = [
    "VALIDATED",
    "EVIDENCE_BOUNDED",
    "ABSTAIN",
]

STRENGTH_CLASSES: List[str] = [
    "STRONG",
    "MODERATE",
    "WEAK",
]

SAMPLE_STATUS_CLASSES: List[str] = [
    "VERIFIED",
    "PARTIAL",
    "UNVERIFIED",
    "WEAK_GLOBAL_EVIDENCE",
]

CONFOUNDER_CLASSES: List[str] = [
    "LOW",
    "MODERATE",
    "HIGH",
    "HIGH_CONTEXTUAL",
]

INTENT_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(INTENT_CLASSES)}
PHENOTYPE_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(PHENOTYPES)}
ANSWERABILITY_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ANSWERABILITY_CLASSES)}
EBSA_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(EBSA_STATES)}

# Thresholds from 06C calibration
UNSUPPORTED_PROB_THRESHOLD = 0.44
PHENOTYPE_PROB_THRESHOLD = 0.50

# ============================================================
# Language & Connector Specifications
# ============================================================
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384

PREFIX_TOKENS = 8
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_HIDDEN_SIZE = 1536
PGC_FEATURE_DIM = 175
MAX_SEQ_LEN = 512

# ============================================================
# Exact Parameter Counts
# ============================================================
PARAM_COUNTS = {
    "phytocore_single": 1_998_216,
    "qpr": 168_205,
    "hcsg": 0,
    "pgc_projector": 9_584_990,
    "lora": 2_179_072,
    "total_project_specific": 13_930_483,
}

# ============================================================
# Required Scientific Disclaimers
# ============================================================
DISCLAIMER_GROUNDING = (
    "Evidence reflects model-internal counterfactual attribution and should not "
    "be interpreted as causal biological evidence."
)

DISCLAIMER_SPECTRAL = (
    "Spectral evidence is reported in model token/band-index space; "
    "physical wavelength mapping is not asserted."
)

DISCLAIMER_CONFOUNDER = (
    "GrainWeight and fertilizer are strongly associated in this dataset; "
    "this relationship should not be interpreted causally."
)
