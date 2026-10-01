"""Generalized ontology-aware query planning with follow-up context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches

TARGETS = ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar")
ALIASES = {
    "GrainWeight": ("grainweight", "grain weight", "grain wieght", "grainwieght", "yield", "yield weight", "yield related weight"),
    "Gsw": ("gsw", "gsw valu", "stomatal conductance", "stomtal conductance"),
    "PhiPS2": ("phips2", "phipsii", "phips ii", "photosystem ii", "photosystem 2", "photosynthetic efficiency"),
    "Fertilizer": ("fertilizer", "fertiliser", "fertillizer", "fertilizer treatment"),
    "Cultivar": ("cultivar", "culitvar", "variety"),
}
DEFINITIONS = {
    "GrainWeight": "GrainWeight is the grain-weight phenotype associated with the wheat sample.",
    "Gsw": "Gsw represents stomatal conductance, describing regulation of gas exchange between leaf tissue and the atmosphere.",
    "PhiPS2": "PhiPS2 represents the effective quantum yield of photosystem II and relates to photosynthetic efficiency.",
    "Fertilizer": "Fertilizer is the categorical fertilizer-treatment condition used during wheat cultivation.",
    "Cultivar": "Cultivar means a cultivated wheat variety; it is a categorical prediction task.",
}


@dataclass
class ConversationContext:
    targets: list[str]
    intent: str | None = None


@dataclass
class QueryPlan:
    intent: str
    targets: list[str]
    include_definition: bool = False
    include_prediction: bool = False
    include_evidence: bool = False


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _correct_words(text: str) -> str:
    vocabulary = {word for aliases in ALIASES.values() for alias in aliases for word in alias.split()}
    vocabulary |= {"value", "prediction", "predictions", "predicted", "give", "please", "explain", "define", "evidence", "summary", "all", "except", "categorical", "continuous", "measurement", "measurements", "tell", "list", "identify", "estimate"}
    corrected = []
    for token in _tokens(text):
        match = get_close_matches(token, vocabulary, n=1, cutoff=0.74)
        corrected.append(match[0] if match else token)
    return " ".join(corrected)


def _explicit_targets(text: str) -> list[str]:
    normalized = _correct_words(text)
    found = []
    positions = []
    for target, aliases in ALIASES.items():
        hits = [normalized.find(alias) for alias in aliases if normalized.find(alias) >= 0]
        if hits:
            positions.append((min(hits), target))
    for _, target in sorted(positions):
        if target not in found:
            found.append(target)
    return found


def _targets(text: str) -> list[str]:
    normalized = _correct_words(text)
    found = _explicit_targets(text)
    if "categorical" in normalized:
        found = ["Fertilizer", "Cultivar"]
    elif "continuous" in normalized or "measurements" in normalized:
        found = ["GrainWeight", "Gsw", "PhiPS2"]
    elif any(phrase in normalized for phrase in ("all available", "all phenotype", "everything", "report all")):
        found = list(TARGETS)
    negative = re.search(r"\b(?:except|do not|dont)\b(.+)$", normalized)
    if negative:
        excluded = set(_explicit_targets(negative.group(1)))
        positive_text = normalized[: negative.start()]
        positive = _explicit_targets(positive_text)
        found = [target for target in (positive or TARGETS) if target not in excluded]
    return found


def plan_query(question: str, context: ConversationContext | None = None) -> QueryPlan:
    normalized = _correct_words(question)
    targets = _targets(question)
    pronoun_followup = any(token in normalized.split() for token in ("it", "its", "their", "them", "that", "one"))
    if not targets and context and pronoun_followup:
        targets = list(context.targets)
    if not targets and context and normalized in {"yes", "prediction", "value", "evidence", "explain", "explanation"}:
        targets = list(context.targets)
    if any(word in normalized for word in ("disease", "rust", "consumption", "wavelength nm")):
        return QueryPlan("UNSUPPORTED", [])
    if "summary" in normalized or "describe this" in normalized or "tell me more" in normalized:
        return QueryPlan("IMAGE_SUMMARY", list(TARGETS), include_prediction=True)
    definition = any(word in normalized for word in ("what is", "define", "meaning", "explain", "difference"))
    evidence = any(word in normalized for word in ("evidence", "region", "patch", "supports"))
    prediction = any(word in normalized for word in ("value", "prediction", "predicted", "give", "report", "read", "how much", "model think", "number", "measurement", "measurements", "list", "identify", "estimate"))
    if "tell" in normalized and ("its" in normalized or "this" in normalized):
        prediction = True
    if context and any(normalized.startswith(prefix) for prefix in ("what about", "and ")):
        prediction = context.intent in {"TARGET_VALUE", "COMPOUND", "EVIDENCE_REQUEST"}
    if normalized in {"yes", "value", "prediction"}:
        prediction = True
    if normalized in {"explain", "explanation"}:
        definition = True
    if evidence:
        return QueryPlan("EVIDENCE_REQUEST", targets, include_prediction=True, include_evidence=True)
    if not targets:
        return QueryPlan("UNSUPPORTED", [])
    if definition and prediction:
        return QueryPlan("COMPOUND", targets, include_definition=True, include_prediction=True)
    if prediction:
        return QueryPlan("TARGET_VALUE", targets, include_prediction=True)
    return QueryPlan("KNOWLEDGE", targets, include_definition=True)
