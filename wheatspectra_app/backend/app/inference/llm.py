"""
PhytoSpectra-VLM
Qwen2.5 + LoRA Language Module

Responsibilities:

- Grounded scientific generation
- RAG knowledge response
- HSI image explanation
- Prediction explanation
- PGC prefix injection
- Hallucination prevention
"""


from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import json
import re
import numpy as np
import torch


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


from app.config import settings

from app.core.constants import (
    LLM_NAME
)

from app.inference.pgc import (
    PGCManager
)



NUMBER_PATTERN = re.compile(
    r"-?\d+(?:\.\d+)?"
)



# =========================================================
# Numeric Safety
# =========================================================


def extract_allowed_numbers(
    predictions: Dict[str,Any]
)->Set[str]:

    allowed=set()


    for _,item in predictions.items():

        if not isinstance(item,dict):
            continue


        for field in ("value", "ensemble_std", "interval_low", "interval_high", "vote_confidence"):
            value = item.get(field)
            if isinstance(value, (int, float)) and np.isfinite(value):
                allowed.update(
                    {
                        str(value),
                        f"{value:.2f}",
                        f"{value:.3f}",
                        f"{value:.4f}",
                    }
                )


    return allowed


# Backwards-compatible public name used by the regression suite.
allowed_numeric_strings = extract_allowed_numbers




# =========================================================
# Prompt Builder
# =========================================================


def build_grounded_prompt(
        question:str,
        image_context:str="",
        knowledge_context:str="",
        predictions:Dict=None
):


    prompt="""

You are PhytoSpectra-VLM,
a scientific assistant for wheat hyperspectral image analysis.

Rules:

1. Answer only from provided scientific context.
2. Never invent spectral bands.
3. Never create fake measurements.
4. Never diagnose disease.
5. If information is unavailable say:
   "The current model does not provide validated information."


"""


    if image_context:

        prompt += """

IMAGE INFORMATION:

""" + image_context + "\n"



    if knowledge_context:

        prompt += """

SCIENTIFIC KNOWLEDGE:

""" + knowledge_context + "\n"



    if predictions:


        prompt += """

MODEL PREDICTIONS:

"""


        for k,v in predictions.items():

            prompt += (
                f"{k}: {v}\n"
            )



    prompt += f"""

USER QUESTION:

{question}


ANSWER:

"""


    return prompt





# =========================================================
# Deterministic fallback
# =========================================================


def safe_fallback(
    state:Dict[str,Any]
):


    mode = state.get(
        "final_answer_mode",
        "ABSTAIN"
    )


    if mode=="ABSTAIN":

        return (
            "The requested information is outside "
            "the validated scope of PhytoSpectra-VLM."
        )


    predictions = state.get(
        "scientific_predictions",
        {}
    )


    lines=[]


    lines.append(
        "Validated scientific information:"
    )


    for k,v in predictions.items():

        if isinstance(v,dict):

            lines.append(
                f"{k}: {v.get('value')}"
            )


    return "\n".join(lines)


def _format_scientific_number(value: float) -> str:
    absolute = abs(float(value))
    if absolute >= 100:
        return f"{value:.2f}"
    return f"{value:.4f}"


def generate_deterministic_safe_answer(
    question: str,
    state: Dict[str, Any],
) -> str:
    """Render only values already present in the verified scientific state."""
    del question  # The renderer intentionally does not generate from user text.
    if state.get("final_answer_mode", "ABSTAIN") == "ABSTAIN":
        return (
            "The requested information is outside the validated quantitative scope "
            "of PhytoSpectra-VLM."
        )

    predictions = state.get("scientific_predictions", {})
    if not predictions:
        return "The current model does not provide validated information."

    lines: List[str] = []
    for target, item in predictions.items():
        if not isinstance(item, dict) or item.get("value") is None:
            continue
        value = item["value"]
        if isinstance(value, (int, float)):
            rendered_value = _format_scientific_number(float(value))
        else:
            rendered_value = str(value)
        line = f"{target}: {rendered_value}"
        std = item.get("ensemble_std")
        if isinstance(std, (int, float)):
            line += f" (ensemble dispersion: {_format_scientific_number(float(std))})"
        lines.append(line)

    if not lines:
        return "The current model does not provide validated information."
    return "\n".join(lines)




# =========================================================
# LLM Manager
# =========================================================


class LLMManager:



    def __init__(
        self,
        alignment_dir:Path=settings.LANGUAGE_ALIGNMENT_DIR,
        device=torch.device("cpu"),
        pgc_manager:Optional[PGCManager]=None,
        knowledge_retriever=None

    ):


        self.alignment_dir=alignment_dir

        self.device=device

        self.pgc_manager=pgc_manager

        self.knowledge_retriever=knowledge_retriever


        self.tokenizer=None

        self.model=None


        self.load_model()




    # -----------------------------------------------------

    def load_model(self):


        tokenizer_path = (
            self.alignment_dir /
            "tokenizer"
        )


        lora_path = (
            self.alignment_dir /
            "lora_adapter"
        )



        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(tokenizer_path)
            )
        )


        if self.tokenizer.pad_token is None:

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )



        base_model = (
            AutoModelForCausalLM
            .from_pretrained(

                LLM_NAME,

                dtype=torch.float32,

                low_cpu_mem_usage=True

            )
        )



        self.model = (
            PeftModel
            .from_pretrained(

                base_model,

                str(lora_path)

            )
            .to(
                self.device
            )
        )



        self.model.eval()


    @torch.inference_mode()
    def _generate_continuation(self, prompt: str, max_new_tokens: int) -> str:
        """Generate and decode only tokens created after the input prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = int(inputs["input_ids"].shape[1])
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


    def classify_query(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Qwen fallback router with a small, machine-validated JSON contract."""
        history_summary = []
        for turn in (history or [])[-3:]:
            plan = turn.get("plan", {}) if isinstance(turn, dict) else {}
            if isinstance(plan, dict):
                history_summary.append(
                    {
                        "intent": plan.get("intent"),
                        "targets": plan.get("targets", []),
                        "requested_order": plan.get("requested_order", []),
                        "excluded_targets": plan.get("excluded_targets", []),
                        "operations": plan.get("operations", []),
                    }
                )

        prompt = f"""You classify questions for a wheat hyperspectral assistant.
Return exactly one JSON object and no prose.
Allowed intents: CONCEPT_EXPLANATION, IMAGE_SUMMARY, TARGET_VALUE,
TARGET_COMPARISON, EVIDENCE_REQUEST, IMAGE_REGION_REQUEST,
CLARIFICATION_REQUIRED, UNSUPPORTED.
Allowed targets: GrainWeight, Gsw, PhiPS2, Fertilizer, Cultivar.
Definitions ask what a term means. Predictions ask for an image-derived value.
Image summaries describe the uploaded image without diagnosing disease.
If the meaning is ambiguous, use CLARIFICATION_REQUIRED.
Preserve the user's target order in requested_order. Record explicitly excluded
targets in excluded_targets. Set include_definition=true only when both meaning
and prediction/comparison are requested. Set include_evidence=true only for an
evidence or image-region request. Never invent a target outside the allowed list.

Recent context: {json.dumps(history_summary, ensure_ascii=True)}
User question: {json.dumps(question, ensure_ascii=True)}

JSON schema:
{{"intent":"...","targets":[],"requested_order":[],"excluded_targets":[],"include_definition":false,"include_evidence":false,"confidence":0.0}}
"""
        text = self._generate_continuation(prompt, max_new_tokens=160)
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Qwen router did not return a JSON object")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Qwen router returned an invalid plan")
        return parsed




    # =====================================================
    # Main Generation
    # =====================================================


    @torch.inference_mode()
    def generate_response(

        self,

        question:str,

        pgvrc_state:Dict[str,Any],

        pgc_feature_vector=None

    )->Tuple[str,bool]:



        image_context = (
            pgvrc_state
            .get(
                "image_context",
                ""
            )
        )


        knowledge_context = (
            pgvrc_state
            .get(
                "knowledge_context",
                ""
            )
        )



        predictions = (
            pgvrc_state
            .get(
                "scientific_predictions",
                {}
            )
        )



        prompt = build_grounded_prompt(

            question,

            image_context,

            knowledge_context,

            predictions

        )



        inputs = (
            self.tokenizer(

                prompt,

                return_tensors="pt"

            )
        )


        input_ids = (
            inputs["input_ids"]
            .to(
                self.device
            )
        )



        embeds = (
            self.model
            .get_input_embeddings()
            (
                input_ids
            )
        )



        # -------------------------------
        # PGC Prefix Injection
        # -------------------------------

        if (
            pgc_feature_vector is not None
            and self.pgc_manager is not None
        ):


            prefix = (
                self.pgc_manager
                .project_prefix_tokens(
                    pgc_feature_vector
                )
            )


            embeds = torch.cat(

                [
                    prefix,
                    embeds
                ],

                dim=1

            )



        attention_mask = torch.ones(

            embeds.shape[:2],

            dtype=torch.long,

            device=self.device

        )



        output = self.model.generate(

            inputs_embeds=embeds,

            attention_mask=attention_mask,

            max_new_tokens=220,

            do_sample=False,

            pad_token_id=
            self.tokenizer.pad_token_id,

            eos_token_id=
            self.tokenizer.eos_token_id

        )



        text = (
            self.tokenizer
            .decode(

                output[0],

                skip_special_tokens=True

            )
            .strip()
        )

        if any(
            marker in text
            for marker in ("IMAGE INFORMATION:", "SCIENTIFIC KNOWLEDGE:", "USER QUESTION:")
        ):
            return safe_fallback(pgvrc_state), True



        # ==================================================
        # Safety Check
        # ==================================================


        if (
            pgvrc_state
            .get(
                "numeric_permission",
                False
            )
        ):


            allowed = extract_allowed_numbers(

                predictions

            )


            generated_numbers=set(
                NUMBER_PATTERN.findall(
                    text
                )
            )


            unexpected = (
                generated_numbers -
                allowed
            )


            if unexpected:


                return (
                    safe_fallback(
                        pgvrc_state
                    ),

                    True
                )

        elif NUMBER_PATTERN.search(text):
            return safe_fallback(pgvrc_state), True

        return text,False




    # =====================================================
    # RAG Knowledge Response
    # =====================================================


    def generate_knowledge_response(

        self,

        question:str,

        context:str

    ):


        if not context or not context.strip():
            return "The current model does not provide validated information."

        prompt = build_grounded_prompt(

            question,

            knowledge_context=context

        )


        text = self._generate_continuation(prompt, max_new_tokens=200)
        if not text or any(
            marker in text
            for marker in ("IMAGE INFORMATION:", "SCIENTIFIC KNOWLEDGE:", "USER QUESTION:")
        ):
            return "The current model does not provide validated information."
        return text



    # =====================================================
    # Image Description
    # =====================================================


    def generate_image_description(

        self,

        question:str,

        image_context:str

    ):


        prompt = build_grounded_prompt(

            question,

            image_context=image_context

        )


        text = self._generate_continuation(prompt, max_new_tokens=220)
        if not text or any(
            marker in text
            for marker in ("IMAGE INFORMATION:", "SCIENTIFIC KNOWLEDGE:", "USER QUESTION:")
        ):
            return "The current model does not provide validated information."
        return text
