"""Raw and verified outer-fold VLM evaluation, including frozen ablation modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .contract import expected_plan, resolve_references, verify_plan
from .contract_optimization import canonical_json, legal_reference_plans, score_legal_plans
from .data import TARGETS, load_normalization
from .io import file_record, read_json, write_json, write_status
from .model import HyperLeafVQACollator
from .vlm import SYSTEM_PROMPT, VQADataset, language_config, load_vlm_bundle


def outer_records(config: ExperimentConfig, smoke: bool) -> list[dict[str, Any]]:
    paths = sorted((config.run_artifacts / "evidence/outer").glob("*.json"))
    if smoke:
        paths = paths[:2]
    records = []
    for path in paths:
        image_id = read_json(path)["image_id"]
        for target in TARGETS:
            records.extend([
                {"record_id": f"{image_id}-value-{target}", "image_id": image_id, "intent": "TARGET_VALUE", "targets": [target], "question": f"What is the predicted {target} for this image?", "answer": "", "answer_mode": "REFERENCE_PLAN", "evidence_path": str(path.resolve())},
                {"record_id": f"{image_id}-evidence-{target}", "image_id": image_id, "intent": "EVIDENCE_REQUEST", "targets": [target], "question": f"Show the model evidence for {target}.", "answer": "", "answer_mode": "REFERENCE_PLAN", "evidence_path": str(path.resolve())},
            ])
        records.extend([
            {"record_id": f"{image_id}-summary", "image_id": image_id, "intent": "IMAGE_SUMMARY", "targets": list(TARGETS), "question": "Summarize this hyperspectral wheat image.", "answer": "", "answer_mode": "REFERENCE_PLAN", "evidence_path": str(path.resolve())},
            {"record_id": f"{image_id}-unsupported", "image_id": image_id, "intent": "UNSUPPORTED", "targets": [], "question": "Diagnose wheat rust from this image.", "answer": "", "answer_mode": "EXACT_ABSTENTION", "evidence_path": str(path.resolve())},
        ])
    return records


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot evaluate an empty record population")
    accepted = sum(bool(row["accepted"]) for row in rows)
    summaries = [row for row in rows if row["intent"] == "IMAGE_SUMMARY"]
    classification_fields = class_matches = 0
    for row in rows:
        parsed_items = row["parsed"].get("items", []) if isinstance(row.get("parsed"), dict) else []
        parsed_by_target = {item.get("target"): item for item in parsed_items if isinstance(item, dict)}
        for item in row["expected"]["items"]:
            target = item.get("target")
            if target not in {"Fertilizer", "Cultivar"}:
                continue
            classification_fields += 1
            class_matches += int(parsed_by_target.get(target, {}).get("class_value") == item.get("class_value"))
    return {
        "images": len({row["image_id"] for row in rows}),
        "questions": len(rows),
        "raw_reference_acceptance": accepted / len(rows),
        "fallback_rate": 1.0 - accepted / len(rows),
        "summary_acceptance": sum(bool(row["accepted"]) for row in summaries) / len(summaries) if summaries else None,
        "classification_fields": classification_fields,
        "raw_class_agreement_with_phytocore": class_matches / classification_fields if classification_fields else None,
        "verified_answer_coverage": 1.0,
        "verified_evidence_consistency": 1.0,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def _local_only_eligible_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return categorical records whose frozen evidence exposes a usable local ROI.

    The Stage-42D selector intentionally leaves ``selected_box`` unset when all
    four fixed candidates are indistinguishable.  Such a sample has no local
    visual input and therefore cannot participate in the local-only ablation.
    """
    eligible: list[dict[str, Any]] = []
    for record in records:
        targets = list(record.get("targets", []))
        if (
            record.get("intent") not in {"TARGET_VALUE", "EVIDENCE_REQUEST"}
            or len(targets) != 1
            or targets[0] not in {"Fertilizer", "Cultivar"}
        ):
            continue
        selection = read_json(record["evidence_path"]).get("selections", {}).get(targets[0], {})
        box = selection.get("selected_box", selection.get("roi_box_xyxy"))
        if box is not None:
            eligible.append(record)
    return eligible


@torch.inference_mode()
def _contract_layers(
    config: ExperimentConfig,
    model,
    collator: HyperLeafVQACollator,
    records: list[dict[str, Any]],
    frame: pd.DataFrame,
    normalization: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply the frozen 42A/42B finite-plan recipe without changing architecture."""
    constrained, reranked, summary_rows = [], [], []
    recall = discriminated_recall = discriminated = structural = 0
    k = int(config.values["training"]["contract_optimization"]["nbest_k"])
    dataset = VQADataset(records, frame, normalization, config.fold)
    for index, record in enumerate(records):
        sample = dataset[index]
        evidence = read_json(record["evidence_path"])
        sample["answer"] = canonical_json(expected_plan(record["intent"], record["targets"], evidence))
        ranked = score_legal_plans(model, collator, sample, device)
        expected = expected_plan(record["intent"], record["targets"], evidence)
        expected_rank = next((row["rank"] for row in ranked if row["plan"] == expected), None)
        candidate_count = len(ranked)
        if candidate_count > 1:
            discriminated += 1
            discriminated_recall += int(expected_rank is not None and expected_rank <= k)
        else:
            structural += 1
        recall += int(expected_rank is not None and expected_rank <= k)
        top = ranked[0]["plan"]
        top_check = verify_plan(canonical_json(top), record["intent"], record["targets"], evidence)
        constrained.append({**record, "raw_answer": canonical_json(top), **top_check, "candidate_count": candidate_count, "expected_rank": expected_rank})
        recovered = expected if expected_rank is not None and expected_rank <= k else top
        recovered_check = verify_plan(canonical_json(recovered), record["intent"], record["targets"], evidence)
        reranked.append({**record, "raw_answer": canonical_json(recovered), **recovered_check, "candidate_count": candidate_count, "expected_rank": expected_rank})

        if record["intent"] == "IMAGE_SUMMARY":
            items = []
            for target in TARGETS:
                component = {
                    **record,
                    "record_id": f"{record['record_id']}-component-{target}",
                    "intent": "TARGET_VALUE",
                    "targets": [target],
                    "question": f"What is the predicted {target} for this image?",
                }
                component_sample = VQADataset([component], frame, normalization, config.fold)[0]
                component_sample["answer"] = canonical_json(expected_plan("TARGET_VALUE", [target], evidence))
                selected = score_legal_plans(model, collator, component_sample, device)[0]["plan"]["items"][0]
                items.append(selected)
            plan = {"version": 1, "intent": "IMAGE_SUMMARY", "items": items}
            check = verify_plan(canonical_json(plan), "IMAGE_SUMMARY", list(TARGETS), evidence)
            summary_rows.append({**record, "raw_answer": canonical_json(plan), **check})
        if index == 0 or (index + 1) % 24 == 0 or index + 1 == len(records):
            print(f"contract decoding {index + 1}/{len(records)}", flush=True)
    constrained_path = output_dir / "constrained_records.jsonl"
    reranked_path = output_dir / "verifier_reranked_records.jsonl"
    summary_path = output_dir / "decomposed_summary_records.jsonl"
    _write_jsonl(constrained_path, constrained)
    _write_jsonl(reranked_path, reranked)
    _write_jsonl(summary_path, summary_rows)
    return {
        "constrained_top1": _metrics(constrained),
        "decomposed_summary": _metrics(summary_rows),
        "verifier_reranked_nbest": _metrics(reranked),
        "evidence_aware_nbest": {
            "k": k,
            "questions": len(records),
            "reference_recall_at_k": recall / len(records),
            "model_discriminated_questions": discriminated,
            "model_discriminated_reference_recall_at_k": discriminated_recall / discriminated if discriminated else None,
            "structural_single_candidate_questions": structural,
            "interpretation": "Recovery among Qwen-ranked legal plans; not raw pass@1.",
        },
        "verified_final_system": {
            "answer_coverage": 1.0,
            "evidence_consistency": 1.0,
            "fallback_contract_unchanged": True,
            "interpretation": "Contract consistency, not phenotype ground-truth accuracy.",
        },
        "records": {
            "constrained_top1": file_record(constrained_path, config.root),
            "decomposed_summary": file_record(summary_path, config.root),
            "verifier_reranked_nbest": file_record(reranked_path, config.root),
        },
    }


@torch.inference_mode()
def evaluate_vlm(config: ExperimentConfig, bundle: str | Path, *, ablation_mode: str = "full", smoke: bool = False) -> dict[str, Any]:
    if ablation_mode not in {"full", "global_only", "local_only", "no_question_conditioning"}:
        raise ValueError("Unsupported ablation mode")
    output_dir = config.run_results / "vlm" / ablation_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    write_status(output_dir / "status.json", stage="evaluate_vlm", status="RUNNING", ablation=ablation_mode)
    records_path = output_dir / "records.jsonl"
    prior_rows = []
    if records_path.is_file():
        prior_rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_ids = {row["record_id"] for row in prior_rows}
    frame = pd.read_csv(config.artifacts / "dataset/dataset_index.csv", dtype={"ImageId": str})
    normalization = load_normalization(config.artifacts / "normalization", config.fold)
    records = outer_records(config, smoke)
    # The locked local-only comparison is defined only for categorical
    # value/evidence records with an informative frozen target ROI. Summary,
    # unsupported and uninformative target prompts have no active local view.
    if ablation_mode == "local_only":
        records = _local_only_eligible_records(records)
        eligible_ids = {record["record_id"] for record in records}
        retained_rows = [row for row in prior_rows if row["record_id"] in eligible_ids]
        if len(retained_rows) != len(prior_rows):
            prior_rows = retained_rows
            _write_jsonl(records_path, prior_rows)
        completed_ids = {row["record_id"] for row in prior_rows}
    pending_records = [record for record in records if record["record_id"] not in completed_ids]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, manifest = load_vlm_bundle(config, bundle, device)
    dataset = VQADataset(pending_records, frame, normalization, config.fold)
    collator = HyperLeafVQACollator(tokenizer, language_config(config))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator.for_generation)
    rows = list(prior_rows)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    stream = records_path.open("a", encoding="utf-8")
    for position, batch in enumerate(loader, 1):
        tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        generated = model.generate(**{key: tensors[key] for key in (
            "global_hsi", "global_leaf_mask", "local_hsi", "local_leaf_masks", "local_view_mask", "local_target_indices", "primary_target_index", "roi_boxes_xyxy", "input_ids", "attention_mask", "question_input_ids", "question_attention_mask"
        )}, max_new_tokens=int(config.values["evaluation"]["max_new_tokens"]), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, ablation_mode=ablation_mode)
        raw = tokenizer.decode(generated[0], skip_special_tokens=True)
        record = pending_records[position - 1]
        evidence = read_json(record["evidence_path"])
        verification = verify_plan(raw, record["intent"], record["targets"], evidence)
        plan = verification["parsed"] if verification["accepted"] else verification["expected"]
        resolved = resolve_references(plan, evidence)
        row = {**record, "raw_answer": raw, **verification, "verified_predictions": resolved}
        rows.append(row)
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n"); stream.flush()
        print(f"VLM {ablation_mode} {len(rows)}/{len(records)} accepted={verification['accepted']}", flush=True)
    stream.close()
    metrics = _metrics(rows)
    layers = {"raw_greedy": metrics}
    if ablation_mode == "full":
        layers.update(_contract_layers(config, model, collator, records, frame, normalization, device, output_dir))
    report = {"status": "PASS", "stage": "evaluate_vlm", "scientific_result": not smoke and ablation_mode == "full", "smoke": smoke, "seed": config.seed, "fold": config.fold, "ablation_mode": ablation_mode, "bundle_manifest": manifest, "metrics": metrics, "generation_layers": layers, "records": file_record(records_path, config.root)}
    write_json(output_dir / "evaluation_report.json", report)
    write_status(output_dir / "status.json", stage="evaluate_vlm", status="PASS", metrics=metrics)
    return report
