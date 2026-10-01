"""
PhytoSpectra-VLM Router Agent

Responsible for:
- Intent detection (IMAGE_DESCRIPTION, KNOWLEDGE, PREDICTION, EVIDENCE, UNSUPPORTED)
- Target extraction (GrainWeight, Gsw, PhiPS2, Fertilizer, Cultivar)
- Context routing
"""

import re
from typing import Dict, List, Optional


class RouterAgent:
    def __init__(self):
        self.targets = [
            "GrainWeight",
            "Gsw",
            "PhiPS2",
            "Fertilizer",
            "Cultivar",
        ]

    def extract_targets(self, question: str) -> List[str]:
        q = question.lower()
        found = []

        mapping = {
            "GrainWeight": [
                "grain weight",
                "grain mass",
                "kernel weight",
                "grainweight",
            ],
            "Gsw": [
                "gsw",
                "stomatal conductance",
            ],
            "PhiPS2": [
                "phips2",
                "phi ps2",
                "photosystem ii",
                "phipsii",
                "φpsii",
            ],
            "Fertilizer": [
                "fertilizer",
                "nitrogen treatment",
                "fertilizer treatment",
            ],
            "Cultivar": [
                "cultivar",
                "variety",
                "wheat variety",
            ],
        }

        for key, words in mapping.items():
            for w in words:
                if re.search(r"\b" + re.escape(w) + r"\b", q):
                    found.append(key)
                    break

        return found

    def classify_intent(self, question: str, image_available: bool = True) -> str:
        q = question.lower().strip()
        targets = self.extract_targets(question)

        # -------------------------------------------------------------
        # 1. Scientific Knowledge Detection (Priority 1)
        # -------------------------------------------------------------
        knowledge_starters = [
            "what is",
            "what are",
            "explain",
            "define",
            "meaning of",
            "tell me about",
            "how does",
            "how do",
            "describe",
        ]

        # Explicit image reference phrases
        image_ref_phrases = [
            "this image",
            "this hyperspectral image",
            "uploaded image",
            "the image",
            "analyze this image",
            "tell me details about this image",
            "describe this image",
            "what is this image",
            "what is shown",
            "about this image",
        ]

        has_image_ref = any(phrase in q for phrase in image_ref_phrases)

        # Prediction verbs
        prediction_verbs = [
            "predict",
            "estimate",
            "calculate",
            "measure",
            "quantify",
            "value of",
            "how much",
            "level of",
        ]
        has_prediction_verb = any(re.search(r"\b" + re.escape(v) + r"\b", q) for v in prediction_verbs)

        # If question has knowledge phrasing and does NOT have explicit image references or prediction verbs
        has_knowledge_starter = any(q.startswith(phrase) or re.search(r"\b" + re.escape(phrase) + r"\b", q) for phrase in knowledge_starters)

        if has_knowledge_starter and not has_image_ref and not has_prediction_verb:
            return "KNOWLEDGE"

        if has_knowledge_starter and targets and not has_image_ref and not has_prediction_verb:
            return "KNOWLEDGE"

        # -------------------------------------------------------------
        # 2. Prediction Detection (Priority 2)
        # -------------------------------------------------------------
        if has_prediction_verb and targets:
            return "PREDICTION"

        # -------------------------------------------------------------
        # 3. Evidence Detection (Priority 3)
        # -------------------------------------------------------------
        evidence_words = [
            "evidence",
            "spatial",
            "spectral",
            "attribution",
            "tokens",
            "bands",
            "support",
        ]
        has_evidence_word = any(re.search(r"\b" + re.escape(ew) + r"\b", q) for ew in evidence_words)

        if has_evidence_word and targets:
            return "EVIDENCE"

        # -------------------------------------------------------------
        # 4. Image Description Detection (Priority 4)
        # -------------------------------------------------------------
        # Only when referring to actual uploaded image
        if has_image_ref:
            # If asking to predict a specific target on this image (e.g. "predict grain weight for this image")
            if has_prediction_verb and targets:
                return "PREDICTION"
            return "IMAGE_DESCRIPTION"

        # -------------------------------------------------------------
        # 5. Fallback Rules
        # -------------------------------------------------------------
        if targets and not has_image_ref:
            # Bare target name or implicit target prediction (e.g. "grain weight")
            return "PREDICTION"

        if has_knowledge_starter:
            return "KNOWLEDGE"

        return "UNSUPPORTED"

    def route(self, question: str, image_available: bool = False) -> Dict:
        intent = self.classify_intent(question, image_available=image_available)
        targets = self.extract_targets(question)

        # Fallback to KNOWLEDGE if image description is requested without an uploaded image
        if intent == "IMAGE_DESCRIPTION" and not image_available:
            intent = "KNOWLEDGE"

        return {
            "intent": intent,
            "target": targets[0] if targets else None,
            "targets": targets,
            "action": intent,
            "image_available": image_available,
        }