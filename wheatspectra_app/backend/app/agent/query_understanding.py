"""Hybrid scientific query planner.

High-confidence scientific patterns are deterministic. Ambiguous paraphrases can
fall back to Qwen, but every result is validated against :class:`QueryPlan`.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

from app.agent.intent_schema import (
    IMAGE_REQUIRED_INTENTS,
    ROUTE_BY_INTENT,
    VALID_TARGETS,
    QueryPlan,
    ScientificIntent,
)


TARGET_PATTERNS = {
    "GrainWeight": (
        r"\bgrain[\s-]*weight\b",
        r"\bgrain[\s-]*mass\b",
        r"\byield[\s-]+related\s+weight\b",
        r"\b(?:leaf'?s|grain)\s+yield\b",
        r"\bkernel\s+weight\b",
    ),
    "Gsw": (r"\bgsw\b", r"\bstomatal\s+conductance\b"),
    "PhiPS2": (
        r"\bphips?2\b",
        r"\bphipsii\b",
        r"\bphi\s*ps\s*2\b",
        r"\bphi\s*psii\b",
        r"\bphotosystem[\s-]*(?:ii|2)\b",
        r"\bphotosynthetic\s+efficiency\b",
    ),
    "Fertilizer": (r"\bfertili[sz]ers?\b",),
    "Cultivar": (r"\bcultivar\b", r"\bwheat\s+variet(?:y|ies)\b"),
}

# Canonical ontology aliases used by the general typo matcher. Matching stays
# inside this closed vocabulary; arbitrary question words are never corrected.
TARGET_ALIASES = {
    "GrainWeight": (
        "grainweight", "grain weight", "grain mass", "kernel weight",
        "yield related weight",
    ),
    "Gsw": ("gsw", "stomatal conductance"),
    "PhiPS2": (
        "phips2", "phipsii", "phi ps2", "phi psii", "photosystem ii",
        "photosystem 2", "photosynthetic efficiency",
    ),
    "Fertilizer": ("fertilizer", "fertiliser", "fertilizer treatment"),
    "Cultivar": ("cultivar", "wheat variety"),
}

CONCEPT_PATTERNS = (
    r"\bwhat\s+(?:is|are)\b",
    r"\bwhat\s+does\b.+\bmean\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bmeaning\s+of\b",
    r"\bexplain\b",
    r"\bexplanation\b",
    r"\btell\s+me\s+about\b",
    r"\bdescribe\b",
)
EXPLICIT_DEFINITION_PATTERNS = (
    r"\bwhat\s+does\b.+\bmean\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bmeaning\s+of\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\bdifference\s+between\b",
)
VALUE_PATTERNS = (
    r"\bvalue\b",
    r"\bpredictions?\b",
    r"\bpredict(?:ed|s|ing)?\b",
    r"\bestimat(?:e|ed|ion)\b",
    r"\bmeasure(?:ments?|d)?\b",
    r"\bresult\b",
    r"\bclassif(?:y|ication)\b",
    r"\bidentify\b",
    r"\b(?:give|provide|report|list|read|show)\b",
    r"\btell\s+me\b",
    r"\bhow\s+much\b",
    r"\bmodel\s+(?:think|thinks|say|says)\b",
    r"\boutput(?:s)?\b",
)
IMAGE_REFERENCE_PATTERNS = (
    r"\bthis\s+(?:image|leaf|sample)\b",
    r"\bthe\s+(?:image|leaf|sample)\b",
    r"\buploaded\s+(?:image|sample)\b",
    r"\bfrom\s+(?:this|the)\b",
    r"\bin\s+(?:this|the)\b",
    r"\bfor\s+(?:this|the|current)\s+(?:image|leaf|sample)\b",
    r"\bits\s+(?:grain|yield|gsw|stomatal|phips|photosystem|fertili|cultivar)\w*\b",
)
IMAGE_SUMMARY_PATTERNS = (
    r"\bdescribe\s+(?:this\s+|the\s+)?(?:current\s+|uploaded\s+)?(?:hyperspectral\s+)?(?:wheat\s+)?(?:image|leaf|sample)\b",
    r"\bsummari[sz]e\s+(?:this\s+|the\s+)?(?:current\s+|uploaded\s+)?(?:hyperspectral\s+)?(?:wheat\s+)?(?:image|leaf|sample)\b",
    r"\btells?\s+me\s+(?:more\s+)?about\s+(?:(?:this|the|uploaded)\s+)?(?:image|leaf|sample)\b",
    r"\bgive\s+me\s+(?:more\s+)?details?\s+about\s+(?:this|the|uploaded)?\s*(?:image|leaf|sample)\b",
    r"\banaly[sz]e\s+(?:this|the|uploaded)?\s*(?:image|leaf|sample)\b",
    r"\bwhat\s+do\s+you\s+see\b",
    r"\b(?:short\s+)?scientific\s+summary\s+of\s+(?:this|the|(?:the\s+)?current|uploaded)\s+(?:hyperspectral\s+)?(?:image|leaf|sample)\b",
)
EVIDENCE_PATTERNS = (
    r"\bevidence\b",
    r"\battribution\b",
    r"\bspectral\s+(?:support|evidence|tokens?)\b",
)
REGION_PATTERNS = (
    r"\bwhich\s+(?:part|region)\b",
    r"\bwhere\s+(?:in|on)\s+(?:the|this)\s+(?:image|leaf)\b",
    r"\bspatial\s+(?:region|evidence|support)\b",
    r"\bbounding\s+box\b",
    r"\bcrop(?:ped|ping)?\b",
    r"\bzoom(?:ed|ing)?\b",
    r"\bfocus(?:ed)?\s+(?:area|region)\b",
    r"\bvisual\s+(?:explanation|representation|evidence)\b",
)
COMPARISON_PATTERNS = (
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\bdifference\s+between\b",
    r"\bversus\b",
    r"\bvs\.?\b",
)
UNSUPPORTED_PATTERNS = (
    r"\bdiagnos(?:e|is|ing)\b",
    r"\bdisease\b",
    r"\brust\b",
    r"\bmildew\b",
    r"\bblight\b",
    r"\bpathogen\b",
    r"\bchlorophyll\b",
    r"\bndvi\b",
    r"\bprotein\b",
    r"\bmoisture\b",
)


def _matches(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _edit_distance(left: str, right: str) -> int:
    """Small deterministic Levenshtein distance for ontology tokens."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _ontology_token_matches(observed: str, canonical: str) -> bool:
    if observed == canonical:
        return True
    # Short scientific symbols such as Gsw are exact-only. Requiring nearly
    # equal token lengths also prevents the valid word "cultivator" from being
    # silently changed into "cultivar".
    if min(len(observed), len(canonical)) < 5:
        return False
    if abs(len(observed) - len(canonical)) > 1:
        return False
    # Adjacent transpositions count as two Levenshtein edits (wieght/weight).
    # Two edits are accepted only for target words of at least six characters.
    allowed = 2 if max(len(observed), len(canonical)) >= 6 else 1
    return _edit_distance(observed, canonical) <= allowed


def _fuzzy_alias_present(text: str, alias: str) -> bool:
    observed_tokens = re.findall(r"[a-z0-9]+", text.casefold())
    alias_tokens = re.findall(r"[a-z0-9]+", alias.casefold())
    width = len(alias_tokens)
    if not width or len(observed_tokens) < width:
        return False
    return any(
        all(
            _ontology_token_matches(observed, canonical)
            for observed, canonical in zip(
                observed_tokens[start : start + width], alias_tokens
            )
        )
        for start in range(len(observed_tokens) - width + 1)
    )


def _fuzzy_alias_position(text: str, alias: str) -> Optional[int]:
    """Return the token position of a closed-ontology fuzzy alias."""
    observed = list(re.finditer(r"[a-z0-9]+", text.casefold()))
    alias_tokens = re.findall(r"[a-z0-9]+", alias.casefold())
    width = len(alias_tokens)
    if not width or len(observed) < width:
        return None
    for start in range(len(observed) - width + 1):
        if all(
            _ontology_token_matches(item.group(0), canonical)
            for item, canonical in zip(observed[start : start + width], alias_tokens)
        ):
            return observed[start].start()
    return None


ALL_TARGET_PATTERNS = (
    r"\ball\s+(?:available\s+)?(?:phenotype\s+)?predictions?\b",
    r"\ball\s+(?:available\s+)?phenotypes?\b",
    r"\beverything\b",
)
CATEGORICAL_GROUP_PATTERNS = (
    r"\b(?:two|2|both|the)\s+categorical\b",
    r"\bcategorical\s+(?:phenotypes?|predictions?|outputs?|classes?)\b",
)
CONTINUOUS_GROUP_PATTERNS = (
    r"\b(?:three|3|the)\s+(?:continuous|regression)\b",
    r"\b(?:continuous|regression)\s+(?:measurements?|phenotypes?|predictions?|outputs?)\b",
)
REFERENCE_PATTERNS = (
    r"\b(?:its|their|that|those|these)\b",
    r"\b(?:what|how)\s+about\b",
    r"^and\b",
)
EXCLUSION_PATTERNS = (
    r"\bexcept\s+{alias}",
    r"\bexcluding\s+{alias}",
    r"\bwithout\s+{alias}",
    r"\bbut\s+not\s+{alias}",
    r"\bdo\s+not\s+(?:report|show|include|give|provide|list)\s+{alias}",
)


class QueryUnderstandingAgent:
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager

    @staticmethod
    def normalize_question(question: str) -> str:
        normalized = question.casefold().replace("’", "'")
        normalized = re.sub(r"\bwhat(?:'s|s)\b", "what is", normalized)
        # Only routing vocabulary is repaired here.  This is intentionally not
        # a general spell-corrector, so arbitrary scientific terms are untouched.
        repairs = {
            r"\bwht\b": "what", r"\bteh\b": "the", r"\bplz\b": "please",
            r"\bgiv\b": "give", r"\bvalu\b": "value", r"\bval\b": "value",
            r"\bpredcted\b": "predicted", r"\bpredction\b": "prediction",
            r"\bimge\b": "image", r"\bstomtal\b": "stomatal",
            r"\bphipsii\b": "phips2", r"\bn\b": "and", r"\br\b": "are",
        }
        for pattern, replacement in repairs.items():
            normalized = re.sub(pattern, replacement, normalized)
        normalized = re.sub(r"[^\w\s?'-]", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def extract_targets(question: str) -> List[str]:
        positions: Dict[str, int] = {}
        for target in VALID_TARGETS:
            found: List[int] = []
            for pattern in TARGET_PATTERNS[target]:
                match = re.search(pattern, question, flags=re.IGNORECASE)
                if match:
                    found.append(match.start())
            for alias in TARGET_ALIASES[target]:
                position = _fuzzy_alias_position(question, alias)
                if position is not None:
                    found.append(position)
            if found:
                positions[target] = min(found)
        return sorted(positions, key=lambda target: positions[target])

    @staticmethod
    def _excluded_targets(question: str) -> List[str]:
        excluded: List[str] = []
        for target, aliases in TARGET_ALIASES.items():
            alias_pattern = "(?:" + "|".join(
                re.escape(alias).replace(r"\ ", r"[\s-]*") for alias in aliases
            ) + ")"
            if any(
                re.search(template.format(alias=alias_pattern), question)
                for template in EXCLUSION_PATTERNS
            ):
                excluded.append(target)
        return excluded

    @staticmethod
    def _expand_target_groups(question: str, targets: List[str]) -> List[str]:
        expanded = list(targets)
        group: List[str] = []
        if _matches(ALL_TARGET_PATTERNS, question):
            group = list(VALID_TARGETS)
        elif _matches(CATEGORICAL_GROUP_PATTERNS, question):
            group = ["Fertilizer", "Cultivar"]
        elif _matches(CONTINUOUS_GROUP_PATTERNS, question):
            group = ["GrainWeight", "Gsw", "PhiPS2"]
        for target in group:
            if target not in expanded:
                expanded.append(target)
        excluded = QueryUnderstandingAgent._excluded_targets(question)
        return [target for target in expanded if target not in excluded]

    @staticmethod
    def _last_targets(history: Optional[List[Dict[str, Any]]]) -> List[str]:
        for turn in reversed(history or []):
            plan = turn.get("plan", {}) if isinstance(turn, dict) else {}
            targets = plan.get("targets", []) if isinstance(plan, dict) else []
            valid = [target for target in targets if target in VALID_TARGETS]
            if valid:
                return valid
        return []

    @staticmethod
    def _last_turn(history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        for turn in reversed(history or []):
            if isinstance(turn, dict):
                return turn
        return {}

    def _build_plan(
        self,
        intent: ScientificIntent,
        question: str,
        targets: Optional[List[str]] = None,
        confidence: float = 0.95,
        source: str = "RULE",
        clarification_message: Optional[str] = None,
        include_definition: bool = False,
        requested_order: Optional[List[str]] = None,
        excluded_targets: Optional[List[str]] = None,
        include_evidence: Optional[bool] = None,
        operations: Optional[List[str]] = None,
    ) -> QueryPlan:
        targets = [target for target in (targets or []) if target in VALID_TARGETS]
        requested_order = [
            target for target in (requested_order or targets)
            if target in targets
        ]
        excluded_targets = [
            target for target in (excluded_targets or self._excluded_targets(question))
            if target in VALID_TARGETS and target not in targets
        ]
        if include_evidence is None:
            include_evidence = intent in {
                ScientificIntent.EVIDENCE_REQUEST,
                ScientificIntent.IMAGE_REGION_REQUEST,
            }
        if operations is None:
            operations_by_intent = {
                ScientificIntent.CONCEPT_EXPLANATION: ["DEFINE"],
                ScientificIntent.IMAGE_SUMMARY: ["SUMMARIZE"],
                ScientificIntent.TARGET_VALUE: ["PREDICT"],
                ScientificIntent.TARGET_COMPARISON: ["COMPARE", "PREDICT"],
                ScientificIntent.EVIDENCE_REQUEST: ["SHOW_EVIDENCE"],
                ScientificIntent.IMAGE_REGION_REQUEST: ["SHOW_REGION"],
                ScientificIntent.CLARIFICATION_REQUIRED: ["CLARIFY"],
                ScientificIntent.UNSUPPORTED: ["ABSTAIN"],
            }
            operations = list(operations_by_intent[intent])
            if include_definition and "DEFINE" not in operations:
                operations.insert(0, "DEFINE")
        needs_clarification = intent == ScientificIntent.CLARIFICATION_REQUIRED
        return QueryPlan(
            intent=intent,
            targets=targets,
            requires_image=intent in IMAGE_REQUIRED_INTENTS,
            route=ROUTE_BY_INTENT[intent],
            confidence=confidence,
            needs_clarification=needs_clarification,
            clarification_message=clarification_message,
            normalized_question=question,
            source=source,
            include_definition=include_definition,
            requested_order=requested_order,
            excluded_targets=excluded_targets,
            include_evidence=bool(include_evidence),
            operations=operations,
        )

    def _from_llm(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]],
    ) -> Optional[QueryPlan]:
        if not self.llm_manager or not hasattr(self.llm_manager, "classify_query"):
            return None

        try:
            raw = self.llm_manager.classify_query(question, history=history or [])
            intent = ScientificIntent(str(raw.get("intent", "UNSUPPORTED")))
            targets = raw.get("targets") or []
            if isinstance(targets, str):
                targets = [targets]
            targets = [target for target in targets if target in VALID_TARGETS]
            excluded_targets = raw.get("excluded_targets") or []
            if isinstance(excluded_targets, str):
                excluded_targets = [excluded_targets]
            excluded_targets = [
                target for target in excluded_targets
                if target in VALID_TARGETS and target not in targets
            ]
            requested_order = raw.get("requested_order") or targets
            if isinstance(requested_order, str):
                requested_order = [requested_order]
            requested_order = [
                target for target in requested_order if target in targets
            ]
            for target in targets:
                if target not in requested_order:
                    requested_order.append(target)
            confidence = float(raw.get("confidence", 0.6))
            confidence = max(0.0, min(confidence, 1.0))
            include_definition = raw.get("include_definition", False) is True
            include_evidence = bool(
                raw.get("include_evidence", False) is True
                or intent in {
                    ScientificIntent.EVIDENCE_REQUEST,
                    ScientificIntent.IMAGE_REGION_REQUEST,
                }
            )

            if intent == ScientificIntent.TARGET_COMPARISON and len(targets) < 2:
                return self._build_plan(
                    ScientificIntent.CLARIFICATION_REQUIRED,
                    question,
                    targets,
                    confidence,
                    "QWEN_VALIDATED",
                    "Which two supported phenotypes would you like to compare?",
                )
            if intent in {
                ScientificIntent.TARGET_VALUE,
                ScientificIntent.EVIDENCE_REQUEST,
                ScientificIntent.IMAGE_REGION_REQUEST,
            } and not targets:
                return self._build_plan(
                    ScientificIntent.CLARIFICATION_REQUIRED,
                    question,
                    [],
                    confidence,
                    "QWEN_VALIDATED",
                    "Which supported phenotype do you mean: GrainWeight, Gsw, PhiPS2, Fertilizer, or Cultivar?",
                )
            return self._build_plan(
                intent,
                question,
                targets,
                confidence,
                "QWEN_VALIDATED",
                include_definition=include_definition,
                requested_order=requested_order,
                excluded_targets=excluded_targets,
                include_evidence=include_evidence,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def analyze(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryPlan:
        q = self.normalize_question(question)
        targets = self._expand_target_groups(q, self.extract_targets(q))
        previous = self._last_turn(history)
        previous_plan = previous.get("plan", {}) if isinstance(previous, dict) else {}
        previous_intent = previous_plan.get("intent") if isinstance(previous_plan, dict) else None

        # A bare affirmative is meaningful only after our explicit one-target
        # clarification.  The clarification wording defines "yes" as the
        # image-value branch, so this is deterministic conversation state, not
        # an LLM guess.
        if q in {"yes", "yes please", "please", "sure", "ok", "okay"}:
            previous_targets = [
                target
                for target in previous_plan.get("targets", [])
                if target in VALID_TARGETS
            ] if isinstance(previous_plan, dict) else []
            if (
                previous.get("answer_type") == "CLARIFICATION"
                and previous_plan.get("intent") == ScientificIntent.CLARIFICATION_REQUIRED.value
                and len(previous_targets) == 1
            ):
                return self._build_plan(
                    ScientificIntent.TARGET_VALUE,
                    q,
                    previous_targets,
                    0.99,
                    "CONVERSATION_CONTEXT",
                )

        # Domain-aware typo handling: never silently rewrite a valid English word.
        if re.search(r"\bcultivator\b", q) and "Cultivar" not in targets:
            return self._build_plan(
                ScientificIntent.CLARIFICATION_REQUIRED,
                q,
                clarification_message=(
                    "Did you mean 'cultivar'? A cultivar is a cultivated plant variety; "
                    "a cultivator usually means a person or an agricultural tool used for cultivation."
                ),
            )

        # Explicit unsupported scientific requests take precedence over a known target.
        if _matches(UNSUPPORTED_PATTERNS, q):
            return self._build_plan(ScientificIntent.UNSUPPORTED, q, targets, 0.99)

        # Resolve pronouns and short follow-ups before evidence/region routing.
        # This makes "show its evidence" and "give their predictions" retain
        # the same target set as the preceding turn.
        if not targets and (
            re.search(r"\b(?:its|that)\s+(?:value|prediction|result)\b", q)
            or _matches(REFERENCE_PATTERNS, q)
            or q in {"value", "prediction", "current value", "image value", "definition", "explanation"}
        ):
            targets = self._last_targets(history)

        # A bare target can complete the immediately preceding evidence/region
        # clarification without asking an unrelated value-vs-definition question.
        if targets and previous.get("answer_type") == "CLARIFICATION":
            previous_question = self.normalize_question(str(previous.get("question", "")))
            if _matches(REGION_PATTERNS, previous_question):
                return self._build_plan(
                    ScientificIntent.IMAGE_REGION_REQUEST, q, targets, 0.99,
                    "CONVERSATION_CONTEXT",
                )
            if _matches(EVIDENCE_PATTERNS, previous_question):
                return self._build_plan(
                    ScientificIntent.EVIDENCE_REQUEST, q, targets, 0.99,
                    "CONVERSATION_CONTEXT",
                )

        if _matches(REGION_PATTERNS, q):
            if targets:
                return self._build_plan(ScientificIntent.IMAGE_REGION_REQUEST, q, targets)
            return self._build_plan(
                ScientificIntent.CLARIFICATION_REQUIRED,
                q,
                clarification_message="Which phenotype's spatial region would you like to inspect?",
            )

        if _matches(EVIDENCE_PATTERNS, q):
            if targets:
                return self._build_plan(ScientificIntent.EVIDENCE_REQUEST, q, targets)
            return self._build_plan(
                ScientificIntent.CLARIFICATION_REQUIRED,
                q,
                clarification_message="Which phenotype's evidence would you like to inspect?",
            )

        # Explicit comparison language is distinct from asking for several
        # values.  Multi-field value requests are handled below as TARGET_VALUE
        # and executed as independently verified single-target calls.
        if _matches(COMPARISON_PATTERNS, q):
            if len(targets) >= 2:
                return self._build_plan(
                    ScientificIntent.TARGET_COMPARISON, q, targets, 0.98,
                    include_definition=_matches(EXPLICIT_DEFINITION_PATTERNS, q),
                )
            return self._build_plan(
                ScientificIntent.CLARIFICATION_REQUIRED,
                q,
                targets,
                clarification_message="Which two supported phenotypes would you like to compare?",
            )

        if _matches(IMAGE_SUMMARY_PATTERNS, q):
            return self._build_plan(ScientificIntent.IMAGE_SUMMARY, q, confidence=0.99)

        has_value_signal = _matches(VALUE_PATTERNS, q)
        has_image_reference = _matches(IMAGE_REFERENCE_PATTERNS, q)
        asks_which_cultivar = bool(
            re.search(r"\b(?:what|which)\s+(?:wheat\s+)?cultivar\b", q)
            and re.search(r"\b(?:this|image|leaf|sample|belong)\b", q)
        )

        inherits_value_intent = bool(
            targets
            and previous_intent in {
                ScientificIntent.TARGET_VALUE.value,
                ScientificIntent.TARGET_COMPARISON.value,
            }
            and (_matches(REFERENCE_PATTERNS, q) or re.match(r"^and\b", q))
            and not _matches(EXPLICIT_DEFINITION_PATTERNS, q)
        )
        if targets and (has_value_signal or has_image_reference or asks_which_cultivar or inherits_value_intent):
            # "What is Gsw, and what value ..." contains two operations;
            # "what are the predicted Gsw and PhiPS2" merely coordinates two
            # targets.  Inspect the clauses rather than treating every "and"
            # after "what is/are" as a definition request.
            first_clause, separator, remaining_clause = q.partition(" and ")
            mixed_definition_clause = bool(
                separator
                and _matches(CONCEPT_PATTERNS, first_clause)
                and self.extract_targets(first_clause)
                and not _matches(VALUE_PATTERNS, first_clause)
                and _matches(VALUE_PATTERNS, remaining_clause)
            )
            include_definition = bool(
                (has_value_signal or asks_which_cultivar)
                and (
                    _matches(EXPLICIT_DEFINITION_PATTERNS, q)
                    or mixed_definition_clause
                )
            )
            return self._build_plan(
                ScientificIntent.TARGET_VALUE, q, targets, 0.98,
                include_definition=include_definition,
            )

        # Contextual property questions such as "Which one is categorical?"
        # are knowledge requests over the prior target set.
        if not targets and re.search(r"\bwhich\s+one\s+is\s+(?:categorical|continuous)\b", q):
            targets = self._last_targets(history)
            if targets:
                return self._build_plan(
                    ScientificIntent.CONCEPT_EXPLANATION, q, targets, 0.99,
                    "CONVERSATION_CONTEXT",
                )

        # A bare entity is genuinely ambiguous when an image may be active.
        residual = q
        for target in targets:
            for pattern in TARGET_PATTERNS[target]:
                residual = re.sub(pattern, " ", residual, flags=re.IGNORECASE)
        residual = re.sub(r"[\s?]+", " ", residual).strip()
        if targets and not residual:
            return self._build_plan(
                ScientificIntent.CLARIFICATION_REQUIRED,
                q,
                targets,
                clarification_message=(
                    f"Do you want the predicted {targets[0]} value for the current "
                    "image? Reply 'yes' for the value, or reply 'explanation' for "
                    "the scientific explanation and definition."
                ),
            )

        if _matches(CONCEPT_PATTERNS, q):
            return self._build_plan(ScientificIntent.CONCEPT_EXPLANATION, q, targets, 0.98)

        llm_plan = self._from_llm(q, history)
        if llm_plan and llm_plan.confidence >= 0.65:
            return llm_plan

        return self._build_plan(
            ScientificIntent.UNSUPPORTED,
            q,
            targets,
            0.5,
            "CONSERVATIVE_FALLBACK",
        )
