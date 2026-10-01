"""Deterministic rendering for scientific facts and model predictions.

The composer never derives new measurements. It only formats values already
present in the typed image/prediction state.
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


KNOWLEDGE_FILES = {
    "GrainWeight": "grainweight.md",
    "Gsw": "gsw.md",
    "PhiPS2": "phips2.md",
    "Fertilizer": "fertilizer.md",
    "Cultivar": "cultivar.md",
}

TARGET_ROLE_DESCRIPTIONS = {
    "GrainWeight": "a continuous yield-related grain-weight output",
    "Gsw": "a continuous stomatal-conductance output",
    "PhiPS2": "a continuous photosystem-II efficiency output",
    "Fertilizer": "a categorical fertilizer-treatment condition",
    "Cultivar": "a categorical cultivated wheat-variety identity",
}


def _format_number(value: float) -> str:
    absolute = abs(float(value))
    if absolute >= 100:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _prediction_sentence(target: str, item: Dict[str, Any]) -> Optional[str]:
    value = item.get("value")
    if value is None:
        return None

    if target == "Cultivar":
        sentence = f"The model predicts the cultivar as {value}"
        vote = item.get("vote_confidence", item.get("confidence"))
        if isinstance(vote, (int, float)):
            sentence += f" (ensemble vote proportion: {vote:.0%})"
        return sentence + "."

    if isinstance(value, (int, float)):
        rendered = _format_number(float(value))
    else:
        rendered = str(value)
    sentence = f"The model prediction for {target} is {rendered}"
    unit = item.get("unit")
    if unit:
        sentence += f" {unit}"
    sentence += "."

    dispersion = item.get("ensemble_std")
    if isinstance(dispersion, (int, float)):
        sentence += (
            f" The dispersion across ensemble checkpoints is "
            f"{_format_number(float(dispersion))}; this is not a calibrated confidence interval."
        )
    return sentence


def compose_prediction_answer(
    predictions: Dict[str, Dict[str, Any]],
    selected_targets: Iterable[str],
    comparison: bool = False,
) -> str:
    sentences = [
        sentence
        for target in selected_targets
        if target in predictions
        for sentence in [_prediction_sentence(target, predictions[target])]
        if sentence
    ]
    if not sentences:
        return "The current model does not provide validated information for the requested phenotype."

    if comparison:
        body = "\n".join(f"- {sentence}" for sentence in sentences)
        return (
            "For the current image, the requested model outputs are:\n"
            + body
            + "\nThese outputs are reported side by side; they do not establish a causal relationship."
        )
    body = " ".join(sentences)
    return "For the current image, " + body[:1].lower() + body[1:]


def compose_image_summary(image_state: Dict[str, Any]) -> str:
    height = image_state.get("height")
    width = image_state.get("width")
    bands = image_state.get("bands")
    lines = [
        f"This is a {height} × {width} hyperspectral wheat image containing {bands} spectral channels."
    ]

    mask = image_state.get("mask_statistics", {})
    foreground = mask.get("foreground_fraction")
    if isinstance(foreground, (int, float)):
        lines.append(f"The segmented leaf foreground covers approximately {foreground:.2%} of the image.")
    bbox = mask.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        lines.append(f"The detected leaf region has pixel bounding box {list(bbox)}.")

    predictions = image_state.get("predicted_targets", {})
    prediction_lines = [
        sentence
        for target, item in predictions.items()
        if isinstance(item, dict)
        for sentence in [_prediction_sentence(target, item)]
        if sentence
    ]
    if prediction_lines:
        lines.append("The current in-distribution model outputs are:")
        lines.extend(f"- {sentence}" for sentence in prediction_lines)

    lines.append(
        "These are model outputs from internal image-level validation, not an external field validation or a disease diagnosis."
    )
    return "\n".join(lines)


def load_target_knowledge(targets: Iterable[str]) -> str:
    knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge"
    documents: List[str] = []
    for target in targets:
        filename = KNOWLEDGE_FILES.get(target)
        if not filename:
            continue
        path = knowledge_dir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(text)
    return "\n\n".join(documents)


def compose_curated_knowledge_answer(context: str) -> str:
    """Turn curated Markdown into readable text without asking an LLM to invent facts."""
    if not context.strip():
        return "The current model does not provide validated information."
    paragraphs: List[str] = []
    current: List[str] = []
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            line = re.sub(r"^#+\s*", "", line)
            if current:
                paragraphs.append(" ".join(current))
                current = []
            if line:
                paragraphs.append(line)
        elif not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def compose_target_difference(targets: Iterable[str]) -> str:
    """State the ontology-level distinction without deriving new evidence."""
    selected = [
        target for target in targets if target in TARGET_ROLE_DESCRIPTIONS
    ]
    if len(selected) == 2:
        first, second = selected
        return (
            f"{first} represents {TARGET_ROLE_DESCRIPTIONS[first]}, whereas "
            f"{second} represents {TARGET_ROLE_DESCRIPTIONS[second]}."
        )
    if selected:
        return "The requested targets differ as follows:\n" + "\n".join(
            f"- {target}: {TARGET_ROLE_DESCRIPTIONS[target]}."
            for target in selected
        )
    return ""


def deduplicate_answer(answer: str) -> str:
    """Remove an accidental exact double emission without rewriting content."""
    text = str(answer or "").strip()
    if len(text) < 40:
        return text
    midpoint = len(text) // 2
    if len(text) % 2 == 0 and text[:midpoint] == text[midpoint:]:
        return text[:midpoint].strip()
    return text


def compose_image_required_answer() -> str:
    return (
        "This question requires an uploaded wheat hyperspectral image. "
        "Attach a TIFF image, then ask the question again."
    )


def compose_unsupported_answer() -> str:
    return (
        "The current model does not provide validated information for that request. "
        "The request is outside the validated quantitative scope of PhytoSpectra-VLM. "
        "It supports scientific explanations and image-based outputs for GrainWeight, "
        "Gsw, PhiPS2, Fertilizer, and Cultivar, but it does not diagnose disease."
    )
