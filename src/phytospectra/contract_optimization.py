"""Frozen Stage-42 contract optimization and finite-schema decoding.

This module changes no model layer.  It refines only the existing bridge and
LoRA tensors using the development-selected recipe: balanced categorical
plans, semantic/class token weighting, one single-target pass, two joint
summary passes, and inner-development-only checkpoint selection.
"""

from __future__ import annotations

from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .contract import expected_plan, verify_plan
from .data import REGRESSION_TARGETS, TARGETS, load_normalization
from .io import read_json, write_json, write_status
from .model import HyperLeafVQACollator, TrainableParameterEMA
from .refinement import CLASS_VALUES, MODEL_KEYS, _enable_training
from .training import seed_everything
from .vlm import (
    VQADataset,
    _records,
    export_vlm_bundle,
    language_config,
    load_vlm_bundle,
    restore_trainable_state,
    trainable_state,
)


SEMANTIC_FIELD = re.compile(
    r'"(?:version|intent|target|value_ref|class_value|roi_ref)"\s*:\s*'
    r'(?:(?:"(?:[^"\\]|\\.)*")|-?\d+(?:\.\d+)?)'
)
CLASS_FIELD = re.compile(r'"class_value"\s*:\s*"(?:[^"\\]|\\.)*"')


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), separators=(",", ":"), ensure_ascii=False)


def legal_reference_plans(intent: str, targets: Sequence[str]) -> list[dict[str, Any]]:
    targets = list(targets)
    if intent not in {"TARGET_VALUE", "EVIDENCE_REQUEST", "IMAGE_SUMMARY", "UNSUPPORTED"}:
        raise ValueError(f"Unsupported intent: {intent}")
    if intent == "UNSUPPORTED":
        if targets:
            raise ValueError("UNSUPPORTED plans cannot name targets")
        return [{"version": 1, "intent": intent, "items": []}]
    if not targets or any(target not in TARGETS for target in targets):
        raise ValueError("Legal plan targets are missing or unsupported")
    if intent == "IMAGE_SUMMARY" and targets != list(TARGETS):
        raise ValueError("IMAGE_SUMMARY must retain the frozen target order")
    categorical = [target for target in targets if target in CLASS_VALUES]
    combinations = itertools.product(*(CLASS_VALUES[target] for target in categorical))
    plans = []
    for values in combinations:
        classes = dict(zip(categorical, values))
        items = []
        for target in targets:
            item: dict[str, Any] = {"target": target, "value_ref": f"{target}.value"}
            if target in classes:
                item["class_value"] = classes[target]
            if intent == "EVIDENCE_REQUEST":
                item["roi_ref"] = f"{target}.roi"
            items.append(item)
        plans.append({"version": 1, "intent": intent, "items": items})
    return plans


def _token_positions(collator: HyperLeafVQACollator, answer: str) -> tuple[list[int], list[int]]:
    tokenized = collator.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(tokenized["offset_mapping"])
    semantic_spans = [match.span() for match in SEMANTIC_FIELD.finditer(answer)]
    class_spans = [match.span() for match in CLASS_FIELD.finditer(answer)]

    def overlap(spans: list[tuple[int, int]]) -> list[int]:
        result = []
        for index, pair in enumerate(offsets):
            start, stop = map(int, pair)
            if stop > start and any(start < right and stop > left for left, right in spans):
                result.append(index)
        return result

    semantic, categorical = overlap(semantic_spans), overlap(class_spans)
    if not semantic or not set(categorical).issubset(semantic):
        raise ValueError("Canonical contract token spans are invalid")
    return semantic, categorical


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _weighted_contract_loss(
    model,
    batch: dict[str, Any],
    collator: HyperLeafVQACollator,
    answer: str,
    *,
    reference_weight: float,
    class_weight: float,
) -> torch.Tensor:
    output = model(**{key: batch[key] for key in MODEL_KEYS})
    prefix = int(output["prefix_embeddings"].shape[1])
    labels = batch["labels"]
    combined = torch.cat([
        torch.full((labels.shape[0], prefix), -100, device=labels.device, dtype=labels.dtype),
        labels,
    ], dim=1)
    logits, shifted = output["logits"][:, :-1].float(), combined[:, 1:]
    losses = F.cross_entropy(logits.transpose(1, 2), shifted, ignore_index=-100, reduction="none")
    semantic, categorical = _token_positions(collator, answer)
    start = int(labels[0].ne(-100).nonzero(as_tuple=False)[0].item())
    semantic_positions = [prefix + start + value - 1 for value in semantic]
    class_positions = [prefix + start + value - 1 for value in categorical]
    weights = torch.ones_like(losses)
    weights[:, semantic_positions] = float(reference_weight)
    weights[:, class_positions] = float(class_weight)
    valid = shifted.ne(-100)
    return (losses * weights * valid).sum() / (weights * valid).sum()


def _categorical_ranking_loss(
    model,
    sample: dict[str, Any],
    collator: HyperLeafVQACollator,
    device: torch.device,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    target = sample["targets"][0]
    plans = legal_reference_plans(sample["intent"], [target])
    expected = json.loads(sample["answer"])
    matches = [index for index, plan in enumerate(plans) if plan == expected]
    if len(matches) != 1:
        raise ValueError("Categorical reference is outside the frozen legal plan set")
    positive = matches[0]
    answers = [canonical_json(plan) for plan in plans]
    variants = [{**sample, "answer": answer} for answer in answers]
    batch = _batch_to_device(collator(variants), device)
    output = model(**{key: batch[key] for key in MODEL_KEYS})
    prefix = int(output["prefix_embeddings"].shape[1])
    labels = batch["labels"]
    combined = torch.cat([
        torch.full((labels.shape[0], prefix), -100, device=device, dtype=labels.dtype), labels
    ], dim=1)
    logits, shifted = output["logits"][:, :-1].float(), combined[:, 1:]
    token_losses = F.cross_entropy(logits.transpose(1, 2), shifted, ignore_index=-100, reduction="none")
    positive_valid = shifted[positive].ne(-100)
    semantic, categorical = _token_positions(collator, answers[positive])
    start = int(labels[positive].ne(-100).nonzero(as_tuple=False)[0].item())
    weights = torch.ones_like(token_losses[positive])
    weights[[prefix + start + value - 1 for value in semantic]] = float(settings["reference_token_weight"])
    hard = float(sample.get("_hard_multiplier", 1.0))
    weights[[prefix + start + value - 1 for value in categorical]] = float(settings["class_token_weight"]) * hard
    causal = (token_losses[positive] * weights * positive_valid).sum() / (weights * positive_valid).sum()
    scores = []
    for row, answer in enumerate(answers):
        _, class_tokens = _token_positions(collator, answer)
        row_start = int(labels[row].ne(-100).nonzero(as_tuple=False)[0].item())
        positions = [prefix + row_start + value - 1 for value in class_tokens]
        ids = labels[row, [row_start + value for value in class_tokens]]
        scores.append(logits[row, positions].log_softmax(-1).gather(1, ids[:, None]).mean())
    ranking = F.cross_entropy(
        torch.stack(scores)[None] / float(settings["ranking_temperature"]),
        torch.tensor([positive], device=device),
    )
    return causal + float(settings["ranking_weight"]) * ranking


def score_legal_plans(
    model,
    collator: HyperLeafVQACollator,
    sample: dict[str, Any],
    device: torch.device,
    *,
    ablation_mode: str = "full",
) -> list[dict[str, Any]]:
    """Rank finite schema-valid plans by mean assistant-token log probability."""
    plans = legal_reference_plans(sample["intent"], sample["targets"])
    variants = [{**sample, "answer": canonical_json(plan)} for plan in plans]
    batch = _batch_to_device(collator(variants), device)
    with torch.inference_mode():
        output = model(**{key: batch[key] for key in MODEL_KEYS}, ablation_mode=ablation_mode)
    prefix = int(output["prefix_embeddings"].shape[1])
    labels = batch["labels"]
    combined = torch.cat([
        torch.full((labels.shape[0], prefix), -100, device=device, dtype=labels.dtype), labels
    ], dim=1)
    logits, shifted = output["logits"][:, :-1].float(), combined[:, 1:]
    valid = shifted.ne(-100)
    targets = shifted.masked_fill(~valid, 0)
    token_logp = logits.log_softmax(-1).gather(2, targets[..., None]).squeeze(-1)
    scores = (token_logp * valid).sum(1) / valid.sum(1).clamp_min(1)
    ranked = [
        {"rank": 0, "score": float(scores[index]), "plan": plan}
        for index, plan in enumerate(plans)
    ]
    ranked.sort(key=lambda row: (-row["score"], canonical_json(row["plan"])))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def _class_value(record: Mapping[str, Any], target: str) -> str:
    plan = json.loads(record["answer"])
    return str(next(item for item in plan["items"] if item["target"] == target)["class_value"])


def _evaluation_records(records: list[dict[str, Any]], image_ids: Sequence[str]) -> list[dict[str, Any]]:
    selected = []
    for image_id in image_ids:
        rows = [row for row in records if row["image_id"] == image_id]
        for target in TARGETS:
            for intent in ("TARGET_VALUE", "EVIDENCE_REQUEST"):
                selected.append(next(row for row in rows if row["targets"] == [target] and row["intent"] == intent))
        selected.append(next(row for row in rows if row["intent"] == "IMAGE_SUMMARY"))
        selected.append(next(row for row in rows if row["intent"] == "UNSUPPORTED"))
    return selected


def build_curriculum(
    records: list[dict[str, Any]],
    train_ids: Sequence[str],
    hard_labels: Mapping[str, Sequence[str]],
    summary_repeat: int,
    *,
    require_all_classes: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = [copy.deepcopy(row) for row in records if row["image_id"] in set(train_ids)]
    single = [
        row for row in train
        if (len(row["targets"]) == 1 and row["targets"][0] in REGRESSION_TARGETS)
        or row["intent"] == "UNSUPPORTED"
    ]
    regression_records = sum(
        len(row["targets"]) == 1 and row["targets"][0] in REGRESSION_TARGETS
        for row in single
    )
    # Keep one unsupported control per image, as in the frozen recipe.
    kept_unsupported, filtered = set(), []
    for row in single:
        if row["intent"] != "UNSUPPORTED" or row["image_id"] not in kept_unsupported:
            filtered.append(row)
            if row["intent"] == "UNSUPPORTED":
                kept_unsupported.add(row["image_id"])
    single = filtered
    balanced_counts: dict[str, dict[str, int]] = {}
    for target in CLASS_VALUES:
        candidates = [row for row in train if row["targets"] == [target]]
        groups = {label: [row for row in candidates if _class_value(row, target) == label] for label in CLASS_VALUES[target]}
        if require_all_classes and any(not values for values in groups.values()):
            raise ValueError(f"Contract curriculum lacks predicted class support for {target}")
        populated = [values for values in groups.values() if values]
        if not populated:
            raise ValueError(f"Contract curriculum has no records for {target}")
        target_count = max(map(len, populated))
        balanced_counts[target] = {}
        for label, values in groups.items():
            if not values:
                balanced_counts[target][label] = 0
                continue
            for index in range(target_count):
                row = copy.deepcopy(values[index % len(values)])
                row["record_id"] = f"contract-{target}-{label}-{index:04d}-{row['record_id']}"
                row["_hard_multiplier"] = (
                    float(1.0) if label not in set(hard_labels.get(target, [])) else None
                )
                single.append(row)
            balanced_counts[target][label] = target_count
    # Replace sentinel with the configured multiplier at training time.
    summaries = [row for row in train if row["intent"] == "IMAGE_SUMMARY"]
    joint = list(single)
    for repeat in range(summary_repeat):
        for row in summaries:
            value = copy.deepcopy(row)
            value["record_id"] = f"contract-summary-r{repeat}-{row['record_id']}"
            joint.append(value)
    for index, row in enumerate(single):
        row.setdefault("_curriculum_order", index)
    for index, row in enumerate(joint):
        row.setdefault("_curriculum_order", index)
    summary = {
        "single_target_records": len(single),
        "joint_records": len(joint),
        "balanced_class_counts": balanced_counts,
        "hard_labels": {key: list(value) for key, value in hard_labels.items()},
        "summary_source_records": len(summaries),
        "summary_repeat": summary_repeat,
        "regression_records": regression_records,
        "unsupported_retained_one_per_image": len(kept_unsupported),
        "outer_fold_used": False,
    }
    return single, joint, summary


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = sum(row["accepted"] for row in rows)
    summaries = [row for row in rows if row["intent"] == "IMAGE_SUMMARY"]
    fields = correct = 0
    for row in rows:
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        by_target = {item.get("target"): item for item in parsed.get("items", []) if isinstance(item, dict)}
        for item in row["expected"]["items"]:
            if item.get("target") in CLASS_VALUES:
                fields += 1
                correct += int(by_target.get(item["target"], {}).get("class_value") == item.get("class_value"))
    return {
        "questions": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "raw_reference_acceptance": accepted / len(rows),
        "fallback_rate": 1.0 - accepted / len(rows),
        "classification_fields": fields,
        "raw_class_agreement_with_phytocore": correct / fields if fields else None,
        "summary_acceptance": sum(row["accepted"] for row in summaries) / len(summaries) if summaries else None,
    }


@torch.inference_mode()
def _raw_evaluate(
    model,
    tokenizer,
    collator: HyperLeafVQACollator,
    records: list[dict[str, Any]],
    frame: pd.DataFrame,
    normalization: dict[str, Any],
    fold: int,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows = []
    for record in records:
        sample = VQADataset([record], frame, normalization, fold)[0]
        batch = _batch_to_device(collator.for_generation([sample]), device)
        generated = model.generate(**{key: batch[key] for key in MODEL_KEYS if key != "labels"}, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(generated[0], skip_special_tokens=True)
        evidence = read_json(record["evidence_path"])
        checked = verify_plan(raw, record["intent"], record["targets"], evidence)
        rows.append({**record, "raw_answer": raw, **checked})
    return _metrics(rows), rows


def train_contract_optimization(
    config: ExperimentConfig,
    input_bundle: Path,
    *,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    settings = config.values["training"]["contract_optimization"]
    contract_seed = 420026
    seed_everything(contract_seed)
    output = config.run_artifacts / "vlm/contract"
    output.mkdir(parents=True, exist_ok=True)
    write_status(output / "status.json", stage="contract_optimization", status="RUNNING")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, source = load_vlm_bundle(config, input_bundle, device)
    bridge, lora = _enable_training(model)
    if hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([
        {"params": bridge, "lr": float(settings["bridge_learning_rate"])},
        {"params": lora, "lr": float(settings["lora_learning_rate"])},
    ], weight_decay=float(settings["weight_decay"]))
    ema = TrainableParameterEMA(model, float(settings["ema_decay"]))
    collator = HyperLeafVQACollator(tokenizer, language_config(config))
    records = _records(config.run_artifacts / "vqa/questions.jsonl")
    train_ids = sorted({row["image_id"] for row in records if row.get("alignment_partition") == "train"})
    dev_ids = sorted({row["image_id"] for row in records if row.get("alignment_partition") == "dev"})
    if not smoke and (len(train_ids), len(dev_ids)) != (128, 16):
        raise ValueError("Contract optimization requires the frozen 128/16 inner split")
    calibration_count = 1 if smoke else int(settings["inner_calibration_images"])
    selection_count = 1 if smoke else int(settings["inner_selection_images"])
    calibration_ids = dev_ids[::2][:calibration_count]
    selection_ids = dev_ids[1::2][:selection_count]
    if set(calibration_ids) & set(selection_ids) and not smoke:
        raise ValueError("Inner calibration and selection images overlap")
    frame = pd.read_csv(config.artifacts / "dataset/dataset_index.csv", dtype={"ImageId": str})
    normalization = load_normalization(config.artifacts / "normalization", config.fold)
    calibration_rows = [
        row for row in _evaluation_records(records, calibration_ids)
        if len(row["targets"]) == 1 and row["targets"][0] in CLASS_VALUES
    ]
    if smoke:
        calibration_rows = calibration_rows[:2]
    _, calibration_outputs = _raw_evaluate(model, tokenizer, collator, calibration_rows, frame, normalization, config.fold, device, int(config.values["evaluation"]["max_new_tokens"]))
    hard_labels: dict[str, list[str]] = {target: [] for target in CLASS_VALUES}
    for row in calibration_outputs:
        if row["accepted"]:
            continue
        target = row["targets"][0]
        label = _class_value(row, target)
        if label not in hard_labels[target]:
            hard_labels[target].append(label)
    single, joint, curriculum = build_curriculum(
        records,
        train_ids,
        hard_labels,
        int(settings["summary_repeat"]),
        require_all_classes=not smoke,
    )
    if not smoke:
        expected_images = 128
        expected_regression = expected_images * len(REGRESSION_TARGETS) * 5
        expected_summaries = expected_images * 3
        class_balance_valid = all(
            len(set(counts.values())) == 1 and min(counts.values()) > 0
            for counts in curriculum["balanced_class_counts"].values()
        )
        frozen_invariants = {
            "train_images": len(train_ids) == expected_images,
            "regression_records": curriculum["regression_records"] == expected_regression,
            "unsupported_one_per_image": curriculum["unsupported_retained_one_per_image"] == expected_images,
            "summary_sources": curriculum["summary_source_records"] == expected_summaries,
            "summary_oversampling": len(joint) - len(single) == expected_summaries * int(settings["summary_repeat"]),
            "categorical_classes_balanced": class_balance_valid,
        }
        if not all(frozen_invariants.values()):
            raise ValueError(f"Frozen contract curriculum invariants failed: {frozen_invariants}")
        curriculum["frozen_invariants"] = frozen_invariants
    for rows in (single, joint):
        for row in rows:
            if row.get("_hard_multiplier") is None:
                row["_hard_multiplier"] = float(settings["hard_label_multiplier"])
    selection_rows = _evaluation_records(records, selection_ids)
    if smoke:
        selection_rows = selection_rows[:2]
        single, joint = single[:4], joint[:4]
    base_metrics, _ = _raw_evaluate(model, tokenizer, collator, selection_rows, frame, normalization, config.fold, device, int(config.values["evaluation"]["max_new_tokens"]))
    best_state = trainable_state(model)
    best_score = (1, float(base_metrics.get("raw_class_agreement_with_phytocore") or 0), float(base_metrics["raw_reference_acceptance"]), float(base_metrics.get("summary_acceptance") or 0))
    history = [{"label": "base", "raw_noninferior_to_base": True, "metrics": base_metrics, "score": list(best_score)}]
    phases = [("single_target", single)] + [(f"joint_summary_{index + 1}", joint) for index in range(1 if smoke else int(settings["joint_epochs"]))]
    start_phase = 0
    latest = output / "latest.pt"
    if resume and latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        restore_trainable_state(model, state["trainable_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        ema.load_state_dict(state["ema"])
        best_state, best_score = state["best_state"], tuple(state["best_score"])
        history, start_phase = list(state["history"]), int(state["phase"])
    optimizer_steps = sum(int(item.get("optimizer_steps", 0)) for item in history)
    accumulation = 1 if smoke else int(settings["gradient_accumulation_steps"])
    for phase_index in range(start_phase, len(phases)):
        label, phase_records = phases[phase_index]
        dataset = VQADataset(phase_records, frame, normalization, config.fold)
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(contract_seed + phase_index)).tolist()
        losses = []
        for block_start in range(0, len(order), accumulation):
            block = order[block_start:block_start + accumulation]
            model.train(); model.bridge.vision_tower.eval(); optimizer.zero_grad(set_to_none=True)
            block_losses = []
            for index in block:
                sample = dataset[index]
                if len(sample["targets"]) == 1 and sample["targets"][0] in CLASS_VALUES:
                    loss = _categorical_ranking_loss(model, sample, collator, device, settings)
                else:
                    batch = _batch_to_device(collator([sample]), device)
                    loss = _weighted_contract_loss(model, batch, collator, sample["answer"], reference_weight=float(settings["reference_token_weight"]), class_weight=float(settings["class_token_weight"]))
                (loss / len(block)).backward()
                block_losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_([*bridge, *lora], float(settings["gradient_clip_norm"]))
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite contract optimization gradient")
            optimizer.step(); ema.update(model); optimizer_steps += 1
            losses.extend(block_losses)
            if optimizer_steps == 1 or optimizer_steps % int(settings["log_every"]) == 0 or block_start + len(block) == len(order):
                print(f"contract {label} step={optimizer_steps} records={min(block_start + len(block), len(order))}/{len(order)} loss={np.mean(block_losses):.5f}", flush=True)
        with ema.average_parameters(model):
            candidate_metrics, _ = _raw_evaluate(model, tokenizer, collator, selection_rows, frame, normalization, config.fold, device, int(config.values["evaluation"]["max_new_tokens"]))
            candidate_state = trainable_state(model)
        noninferior = (
            candidate_metrics["raw_reference_acceptance"] >= base_metrics["raw_reference_acceptance"]
            and float(candidate_metrics.get("raw_class_agreement_with_phytocore") or 0) >= float(base_metrics.get("raw_class_agreement_with_phytocore") or 0)
        )
        score = (int(noninferior), float(candidate_metrics.get("raw_class_agreement_with_phytocore") or 0), float(candidate_metrics["raw_reference_acceptance"]), float(candidate_metrics.get("summary_acceptance") or 0))
        history.append({"label": label, "raw_noninferior_to_base": noninferior, "metrics": candidate_metrics, "score": list(score), "optimizer_steps": len(range(0, len(order), accumulation)), "mean_loss": float(np.mean(losses))})
        if score > best_score:
            best_score, best_state = score, candidate_state
        torch.save({"format": "phytospectra_v3_contract_resume_v1", "phase": phase_index + 1, "trainable_state_dict": trainable_state(model), "optimizer_state_dict": optimizer.state_dict(), "ema": ema.state_dict(), "best_state": best_state, "best_score": list(best_score), "history": history}, latest)
    restore_trainable_state(model, best_state)
    checkpoint = config.root / source["predictor_checkpoint"]["path"]
    manifest = export_vlm_bundle(config, model, tokenizer, checkpoint, output / "bundle", stage="contract_optimization", metadata={
        "scientific_result": False,
        "smoke": smoke,
        "source_bundle": str(input_bundle.relative_to(config.root).as_posix()),
        "curriculum": curriculum,
        "recipe": "historical_stage42d_exact",
        "training_seed": contract_seed,
        "inner_calibration_images": calibration_ids,
        "inner_selection_images": selection_ids,
        "outer_fold_used_for_selection": False,
        "selection_history": history,
        "selected_score": list(best_score),
        "nbest_k": int(settings["nbest_k"]),
        "constrained_decoding_available": True,
        "decomposed_summary_available": True,
    })
    write_json(output / "training_report.json", manifest)
    write_status(output / "status.json", stage="contract_optimization", status="PASS")
    return manifest
