"""Generate the fold-specific 32-question-per-image HyperLeaf-VQA corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import TARGETS
from .io import file_record, read_json, write_json


VALUE_TEMPLATES = (
    "What is the predicted {target} for this hyperspectral wheat image?",
    "Give the model's {target} result for this sample.",
    "According to the image model, what is this sample's {target}?",
)
EVIDENCE_TEMPLATES = (
    "Show the visual evidence used for the {target} output.",
    "Which image region supports the model's {target} prediction?",
)
SUMMARY_TEMPLATES = (
    "Describe this hyperspectral wheat image using the available model outputs.",
    "Tell me more about this image using only validated outputs.",
    "Summarize the model results for this wheat image.",
)
UNSUPPORTED = (
    "What disease does this wheat leaf have?",
    "What is the chlorophyll content of this leaf?",
    "Which physical wavelength in nm proves this prediction?",
    "Is this wheat safe for consumption?",
)


def _plan(intent: str, targets: list[str], evidence: dict[str, Any]) -> str:
    from .contract import expected_plan
    return json.dumps(expected_plan(intent, targets, evidence), separators=(",", ":"))


def build_vqa(config: ExperimentConfig, *, smoke: bool = False) -> dict[str, Any]:
    cache = config.run_artifacts / "evidence/alignment"
    evidence_files = sorted(cache.glob("*.json"))
    if not evidence_files:
        raise FileNotFoundError("Alignment evidence cache is empty")
    records = []
    for path in evidence_files:
        evidence = read_json(path)
        image_id = evidence["image_id"]
        alignment_partition = evidence.get("alignment_partition", "train")
        counter = 0
        for target in TARGETS:
            for template in VALUE_TEMPLATES:
                records.append({"record_id": f"{image_id}-{counter:02d}", "image_id": image_id, "template_index": VALUE_TEMPLATES.index(template), "alignment_partition": alignment_partition, "intent": "TARGET_VALUE", "targets": [target], "question": template.format(target=target), "answer": _plan("TARGET_VALUE", [target], evidence), "answer_mode": "REFERENCE_PLAN", "evidence_path": str(path.resolve())})
                counter += 1
            for template in EVIDENCE_TEMPLATES:
                records.append({"record_id": f"{image_id}-{counter:02d}", "image_id": image_id, "template_index": EVIDENCE_TEMPLATES.index(template), "alignment_partition": alignment_partition, "intent": "EVIDENCE_REQUEST", "targets": [target], "question": template.format(target=target), "answer": _plan("EVIDENCE_REQUEST", [target], evidence), "answer_mode": "REFERENCE_PLAN", "evidence_path": str(path.resolve())})
                counter += 1
        for template in SUMMARY_TEMPLATES:
            records.append({"record_id": f"{image_id}-{counter:02d}", "image_id": image_id, "template_index": SUMMARY_TEMPLATES.index(template), "alignment_partition": alignment_partition, "intent": "IMAGE_SUMMARY", "targets": list(TARGETS), "question": template, "answer": _plan("IMAGE_SUMMARY", list(TARGETS), evidence), "answer_mode": "REFERENCE_PLAN", "evidence_path": str(path.resolve())})
            counter += 1
        for question in UNSUPPORTED:
            records.append({"record_id": f"{image_id}-{counter:02d}", "image_id": image_id, "template_index": UNSUPPORTED.index(question), "alignment_partition": alignment_partition, "intent": "UNSUPPORTED", "targets": [], "question": question, "answer": _plan("UNSUPPORTED", [], evidence), "answer_mode": "EXACT_ABSTENTION", "evidence_path": str(path.resolve())})
            counter += 1
        if counter != 32:
            raise RuntimeError(f"Expected 32 records for {image_id}, observed {counter}")
    train_records = [row for row in records if row["alignment_partition"] == "train"]
    dev_records = [row for row in records if row["alignment_partition"] == "dev" and row["template_index"] == 0]
    if smoke and (len(train_records), len(dev_records)) != (3 * 32, 2 * 12):
        raise RuntimeError(f"Stage-42D smoke VQA contract changed: train={len(train_records)} dev={len(dev_records)}")
    if not smoke and (len(train_records) != 128 * 32 or len(dev_records) != 16 * 12):
        raise RuntimeError(f"Stage-42D VQA contract changed: train={len(train_records)} dev={len(dev_records)}")
    records = sorted(train_records + dev_records, key=lambda row: row["record_id"])
    output = config.run_artifacts / "vqa/questions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    manifest = {"status": "PASS", "stage": "vqa", "scientific_result": False, "recipe": "historical_stage42d_exact", "images": len(evidence_files), "records": len(records), "train_records": len(train_records), "dev_records": len(dev_records), "train_records_per_image": 32, "dev_records_per_image": 12, "questions": file_record(output, config.root)}
    write_json(config.run_artifacts / "vqa/manifest.json", manifest)
    return manifest
