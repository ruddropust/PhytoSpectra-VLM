"""Typed contract for scientific query understanding."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ScientificIntent(str, Enum):
    CONCEPT_EXPLANATION = "CONCEPT_EXPLANATION"
    IMAGE_SUMMARY = "IMAGE_SUMMARY"
    TARGET_VALUE = "TARGET_VALUE"
    TARGET_COMPARISON = "TARGET_COMPARISON"
    EVIDENCE_REQUEST = "EVIDENCE_REQUEST"
    IMAGE_REGION_REQUEST = "IMAGE_REGION_REQUEST"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"


class ExecutionRoute(str, Enum):
    KNOWLEDGE_RAG = "KNOWLEDGE_RAG"
    IMAGE_STATE = "IMAGE_STATE"
    PHYTOCORE = "PHYTOCORE"
    VER = "VER"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"


VALID_TARGETS = [
    "GrainWeight",
    "Gsw",
    "PhiPS2",
    "Fertilizer",
    "Cultivar",
]


IMAGE_REQUIRED_INTENTS = {
    ScientificIntent.IMAGE_SUMMARY,
    ScientificIntent.TARGET_VALUE,
    ScientificIntent.TARGET_COMPARISON,
    ScientificIntent.EVIDENCE_REQUEST,
    ScientificIntent.IMAGE_REGION_REQUEST,
}


ROUTE_BY_INTENT = {
    ScientificIntent.CONCEPT_EXPLANATION: ExecutionRoute.KNOWLEDGE_RAG,
    ScientificIntent.IMAGE_SUMMARY: ExecutionRoute.IMAGE_STATE,
    ScientificIntent.TARGET_VALUE: ExecutionRoute.PHYTOCORE,
    ScientificIntent.TARGET_COMPARISON: ExecutionRoute.PHYTOCORE,
    ScientificIntent.EVIDENCE_REQUEST: ExecutionRoute.VER,
    ScientificIntent.IMAGE_REGION_REQUEST: ExecutionRoute.VER,
    ScientificIntent.CLARIFICATION_REQUIRED: ExecutionRoute.CLARIFY,
    ScientificIntent.UNSUPPORTED: ExecutionRoute.ABSTAIN,
}


class QueryPlan(BaseModel):
    """A validated plan produced before any scientific executor is called."""

    intent: ScientificIntent
    targets: List[str] = Field(default_factory=list)
    requires_image: bool = False
    route: ExecutionRoute
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    normalized_question: str = ""
    source: str = "RULE"
    # True when the user explicitly asks for both scientific meaning and an
    # image-derived prediction.  The executor remains unchanged; the pipeline
    # composes curated knowledge with independently verified target calls.
    include_definition: bool = False
    # Explicit, machine-checkable semantics used by the response layer and by
    # the linguistic generalization benchmark.  ``targets`` remains the
    # executable allow-list; these fields preserve how the user requested it.
    requested_order: List[str] = Field(default_factory=list)
    excluded_targets: List[str] = Field(default_factory=list)
    include_evidence: bool = False
    operations: List[str] = Field(default_factory=list)

    @property
    def target(self) -> Optional[str]:
        return self.targets[0] if self.targets else None
