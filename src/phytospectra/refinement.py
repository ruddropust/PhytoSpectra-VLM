"""Fixed class-aware remediation, Cultivar sequence ranking, and alpha soup."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import load_normalization
from .io import read_json, write_json, write_status
from .model import HyperLeafVQACollator, TrainableParameterEMA
from .training import seed_everything
from .vlm import VQADataset, _records, export_vlm_bundle, language_config, load_vlm_bundle, restore_trainable_state, trainable_state
from .contract import expected_plan


CLASS_VALUES = {"Fertilizer": ["0", "0.5", "1"], "Cultivar": ["Heerup", "Kvium", "Rembrandt", "Sheriff"]}
MODEL_KEYS = ("global_hsi", "global_leaf_mask", "local_hsi", "local_leaf_masks", "local_view_mask", "local_target_indices", "primary_target_index", "roi_boxes_xyxy", "input_ids", "attention_mask", "labels", "question_input_ids", "question_attention_mask")


def _enable_training(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    bridge, lora = [], []
    for name, parameter in model.named_parameters():
        trainable = (name.startswith("bridge.") and not name.startswith("bridge.vision_tower.")) or "lora_" in name
        parameter.requires_grad_(trainable)
        if trainable:
            (bridge if name.startswith("bridge.") else lora).append(parameter)
    if not bridge or not lora:
        raise RuntimeError("Bridge/LoRA trainable parameter groups are incomplete")
    return bridge, lora


def _common_prefix(sequences: list[list[int]]) -> int:
    for index in range(min(map(len, sequences))):
        if len({sequence[index] for sequence in sequences}) != 1:
            return index
    return min(map(len, sequences))


def _common_suffix(sequences: list[list[int]], prefix: int) -> int:
    limit = min(len(sequence) - prefix for sequence in sequences)
    for width in range(1, limit + 1):
        if len({sequence[-width] for sequence in sequences}) != 1:
            return width - 1
    return limit


def categorical_variants(collator, answer: str, target: str) -> tuple[list[str], int, list[list[int]]]:
    parsed = json.loads(answer)
    item = next(value for value in parsed["items"] if value["target"] == target)
    correct = str(item["class_value"])
    answers = []
    for value in CLASS_VALUES[target]:
        candidate = copy.deepcopy(parsed)
        next(entry for entry in candidate["items"] if entry["target"] == target)["class_value"] = value
        answers.append(json.dumps(candidate, separators=(",", ":")))
    encoded = [collator._answer_ids(value) for value in answers]
    return answers, CLASS_VALUES[target].index(correct), encoded


def _class_positions(collator, answer: str, target: str) -> list[int]:
    _, correct, sequences = categorical_variants(collator, answer, target)
    prefix = _common_prefix(sequences); suffix = _common_suffix(sequences, prefix)
    stop = len(sequences[correct]) - suffix
    positions = list(range(prefix, stop))
    if not positions:
        raise RuntimeError("No class-discriminating answer tokens were found")
    return positions


def _weighted_loss(model, batch: dict[str, Any], collator, answer: str, targets: list[str], weight: float) -> tuple[torch.Tensor, int]:
    output = model(**{key: batch[key] for key in MODEL_KEYS})
    prefix_tokens = int(output["prefix_embeddings"].shape[1])
    labels = batch["labels"]
    combined = torch.cat([torch.full((labels.shape[0], prefix_tokens), -100, device=labels.device, dtype=labels.dtype), labels], dim=1)
    logits, shifted = output["logits"][:, :-1].float(), combined[:, 1:]
    losses = F.cross_entropy(logits.transpose(1, 2), shifted, ignore_index=-100, reduction="none")
    weights = torch.ones_like(losses); count = 0
    if len(targets) == 1 and targets[0] in CLASS_VALUES:
        positions = _class_positions(collator, answer, targets[0])
        answer_start = int(labels[0].ne(-100).nonzero()[0])
        shifted_positions = [prefix_tokens + answer_start + position - 1 for position in positions]
        weights[:, shifted_positions] = float(weight); count = len(shifted_positions)
    valid = shifted.ne(-100)
    return (losses * weights * valid).sum() / (weights * valid).sum(), count


def _donor_map(image_ids: list[str], rows_by_image: dict[str, list[dict[str, Any]]], target: str) -> dict[str, str]:
    labels = {}
    for image_id in image_ids:
        row = next(row for row in rows_by_image[image_id] if row["targets"] == [target])
        evidence = read_json(row["evidence_path"])
        labels[image_id] = str(evidence["predictions"][target]["value"])
    result = {}
    for image_id in image_ids:
        candidates = [candidate for candidate in image_ids if labels[candidate] != labels[image_id]]
        if not candidates:
            raise ValueError(f"No different-label donor for {target}")
        start = image_ids.index(image_id) % len(candidates)
        result[image_id] = candidates[start]
    return result


def _rotation_donor_map(image_ids: list[str]) -> dict[str, str]:
    if len(image_ids) < 2 or len(set(image_ids)) != len(image_ids):
        raise ValueError("Need distinct recipient/donor images")
    return {image_id: image_ids[(index + 1) % len(image_ids)] for index, image_id in enumerate(image_ids)}


EXPECTED_CULTIVARS = ("Heerup", "Kvium", "Rembrandt", "Sheriff")


def _balanced_cultivar_donor_map(
    image_ids: list[str], rows_by_image: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Exact Stage-41F deterministic forbidden-diagonal max-flow allocation."""
    ids = sorted(image_ids)
    labels = {
        image_id: str(read_json(rows_by_image[image_id][0]["evidence_path"])["predictions"]["Cultivar"]["value"])
        for image_id in ids
    }
    recipient_counts = Counter(labels.values())
    if set(recipient_counts) != set(EXPECTED_CULTIVARS) or len(ids) % 2:
        raise ValueError(f"Cultivar ranking cannot be balanced: {dict(recipient_counts)}")
    target_assignments = len(ids) // 2
    quotas = {label: target_assignments - recipient_counts[label] for label in EXPECTED_CULTIVARS}
    if any(value < 0 for value in quotas.values()) or sum(quotas.values()) != len(ids):
        raise ValueError(f"Infeasible Cultivar donor quotas: {quotas}")

    source, sink = "source", "sink"
    residual: dict[str, dict[str, int]] = {}
    adjacency: dict[str, list[str]] = {}

    def add_edge(left: str, right: str, capacity: int) -> None:
        residual.setdefault(left, {})[right] = capacity
        residual.setdefault(right, {}).setdefault(left, 0)
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)

    for label in EXPECTED_CULTIVARS:
        add_edge(source, f"recipient:{label}", recipient_counts[label])
    for recipient_label in EXPECTED_CULTIVARS:
        for donor_label in EXPECTED_CULTIVARS:
            if donor_label != recipient_label:
                add_edge(f"recipient:{recipient_label}", f"donor:{donor_label}", len(ids))
    for label in EXPECTED_CULTIVARS:
        add_edge(f"donor:{label}", sink, quotas[label])

    total_flow = 0
    while True:
        parent: dict[str, str | None] = {source: None}
        queue = [source]
        for node in queue:
            for neighbour in adjacency.get(node, []):
                if neighbour not in parent and residual[node].get(neighbour, 0) > 0:
                    parent[neighbour] = node
                    queue.append(neighbour)
                    if neighbour == sink:
                        break
            if sink in parent:
                break
        if sink not in parent:
            break
        amount = len(ids)
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            amount = min(amount, residual[previous][node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= amount
            residual[node][previous] += amount
            node = previous
        total_flow += amount
    if total_flow != len(ids):
        raise ValueError(f"Cultivar donor flow incomplete: {total_flow}/{len(ids)}")

    allocation = {
        recipient: {
            donor: residual[f"donor:{donor}"].get(f"recipient:{recipient}", 0)
            for donor in EXPECTED_CULTIVARS if donor != recipient
        }
        for recipient in EXPECTED_CULTIVARS
    }
    groups = {label: sorted(image_id for image_id in ids if labels[image_id] == label) for label in EXPECTED_CULTIVARS}
    cursors: Counter[str] = Counter()
    mapping: dict[str, str] = {}
    for recipient_label in EXPECTED_CULTIVARS:
        donor_labels: list[str] = []
        for donor_label in EXPECTED_CULTIVARS:
            if donor_label != recipient_label:
                donor_labels.extend([donor_label] * allocation[recipient_label][donor_label])
        if len(donor_labels) != len(groups[recipient_label]):
            raise ValueError("Cultivar donor-flow row does not match recipient count")
        for image_id, donor_label in zip(groups[recipient_label], donor_labels, strict=True):
            candidates = groups[donor_label]
            mapping[image_id] = candidates[cursors[donor_label] % len(candidates)]
            cursors[donor_label] += 1
    donor_counts = Counter(labels[donor] for donor in mapping.values())
    if donor_counts != Counter(quotas) or any(labels[left] == labels[right] for left, right in mapping.items()):
        raise ValueError("Cultivar donor balance audit failed")
    return mapping, {
        "method": "deterministic_forbidden_diagonal_max_flow",
        "recipient_cultivar_counts": dict(recipient_counts),
        "target_image_assignments_per_class": target_assignments,
        "donor_cultivar_quotas": quotas,
        "donor_cultivar_counts": dict(donor_counts),
        "class_flow": allocation,
    }


def _donor_record(row: dict[str, Any], donor_id: str, donor_rows: list[dict[str, Any]], suffix: str) -> dict[str, Any]:
    donor_evidence_path = donor_rows[0]["evidence_path"]
    evidence = read_json(donor_evidence_path)
    return {
        **row,
        "record_id": f"{row['record_id']}__{suffix}__{donor_id}",
        "image_id": donor_id,
        "evidence_path": donor_evidence_path,
        "answer": json.dumps(expected_plan(row["intent"], list(row["targets"]), evidence), separators=(",", ":")),
    }


def _first(rows: list[dict[str, Any]], *, target: str | None = None, intent: str) -> dict[str, Any]:
    return next(row for row in rows if row["intent"] == intent and (target is None or row["targets"] == [target]))


def _training_records(config: ExperimentConfig, *, ranking: bool, smoke: bool) -> list[dict[str, Any]]:
    records = _records(config.run_artifacts / "vqa/questions.jsonl")
    records = [row for row in records if row.get("alignment_partition") == "train"]
    image_ids = sorted({row["image_id"] for row in records})
    rows_by_image = {image_id: [row for row in records if row["image_id"] == image_id] for image_id in image_ids}
    # A one-epoch smoke PhytoCore may collapse all categorical predictions;
    # smoke therefore exercises donor plumbing with deterministic rotation.
    # Full runs retain the strict different-label Stage-42D donor contract.
    fertilizer_donors = _rotation_donor_map(image_ids) if smoke else _donor_map(image_ids, rows_by_image, "Fertilizer")
    cultivar_donor_audit = None
    if ranking and not smoke:
        cultivar_donors, cultivar_donor_audit = _balanced_cultivar_donor_map(image_ids, rows_by_image)
    else:
        cultivar_donors = _rotation_donor_map(image_ids) if smoke else _donor_map(image_ids, rows_by_image, "Cultivar")
    summary_donors = _rotation_donor_map(image_ids)
    selected: list[dict[str, Any]] = []
    for image_id in image_ids:
        rows = rows_by_image[image_id]
        if ranking:
            donor = cultivar_donors[image_id]
            for intent in ("TARGET_VALUE", "EVIDENCE_REQUEST"):
                row = _first(rows, target="Cultivar", intent=intent)
                selected.append({**row, "record_id": f"{row['record_id']}__ranked_matched"})
                selected.append(_donor_record(row, donor, rows_by_image[donor], "ranked_hard_donor"))
            for intent in ("TARGET_VALUE", "EVIDENCE_REQUEST"):
                row = _first(rows, target="Fertilizer", intent=intent)
                selected.append({**row, "record_id": f"{row['record_id']}__fertilizer_retention"})
            row = _first(rows, intent="IMAGE_SUMMARY")
            selected.append({**row, "record_id": f"{row['record_id']}__summary_retention"})
            row = _first(rows, intent="UNSUPPORTED")
            selected.append({**row, "record_id": f"{row['record_id']}__unsupported_retention"})
            for target in ("GrainWeight", "Gsw", "PhiPS2"):
                row = _first(rows, target=target, intent="TARGET_VALUE")
                selected.append({**row, "record_id": f"{row['record_id']}__regression_retention"})
        else:
            for target, donors in (("Fertilizer", fertilizer_donors), ("Cultivar", cultivar_donors)):
                donor = donors[image_id]
                for row in [value for value in rows if value["targets"] == [target]]:
                    selected.append({**row, "record_id": f"{row['record_id']}__matched"})
                    selected.append(_donor_record(row, donor, rows_by_image[donor], "hard_donor"))
            summary_donor = summary_donors[image_id]
            for row in [value for value in rows if value["intent"] == "IMAGE_SUMMARY"]:
                selected.append({**row, "record_id": f"{row['record_id']}__matched_summary"})
                selected.append(_donor_record(row, summary_donor, rows_by_image[summary_donor], "coherent_donor_summary"))
            row = _first(rows, intent="UNSUPPORTED")
            selected.append({**row, "record_id": f"{row['record_id']}__unsupported_retention"})
            for target in ("GrainWeight", "Gsw", "PhiPS2"):
                for intent in ("TARGET_VALUE", "EVIDENCE_REQUEST"):
                    row = _first(rows, target=target, intent=intent)
                    selected.append({**row, "record_id": f"{row['record_id']}__regression_retention"})
    expected = len(image_ids) * (11 if ranking else 33)
    if len(selected) != expected or len({row["record_id"] for row in selected}) != expected:
        raise RuntimeError(f"Stage-42D refinement curriculum changed: expected={expected} observed={len(selected)}")
    if ranking and not smoke:
        counts = Counter(
            str(read_json(row["evidence_path"])["predictions"]["Cultivar"]["value"])
            for row in selected if "ranked_" in row["record_id"]
        )
        expected_counts = Counter({label: len(image_ids) for label in EXPECTED_CULTIVARS})
        if counts != expected_counts:
            raise RuntimeError(f"Stage-42D Cultivar ranking balance changed: {dict(counts)}")
    return selected[:4] if smoke else selected


def _train_phase(config: ExperimentConfig, input_bundle: Path, output_dir: Path, *, phase: str, ranking: bool, smoke: bool, resume: bool) -> dict[str, Any]:
    settings = config.values["training"]["cultivar_ranking" if ranking else "remediation"]
    training_seed = (418026 if ranking else 417026) + 10 * (config.seed - 2026) + config.fold
    seed_everything(training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, source = load_vlm_bundle(config, input_bundle, device)
    bridge, lora = _enable_training(model)
    model.train(); model.bridge.vision_tower.eval()
    if hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    optimizer = torch.optim.AdamW([{"params": bridge, "lr": float(settings["bridge_learning_rate"])}, {"params": lora, "lr": float(settings["lora_learning_rate"])}], weight_decay=1e-4)
    ema = TrainableParameterEMA(model, float(settings["ema_decay"]))
    collator = HyperLeafVQACollator(tokenizer, language_config(config))
    frame = pd.read_csv(config.artifacts / "dataset/dataset_index.csv", dtype={"ImageId": str})
    normalization = load_normalization(config.artifacts / "normalization", config.fold)
    records = _training_records(config, ranking=ranking, smoke=smoke)
    dataset = VQADataset(records, frame, normalization, config.fold)
    epochs = 1 if smoke else int(settings["epochs"]); accumulation = 1 if smoke else int(settings["gradient_accumulation_steps"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.pt"
    write_status(output_dir / "status.json", stage=phase, status="RUNNING")
    history, step_count, start_epoch = [], 0, 0
    if resume and latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        restore_trainable_state(model, state["trainable_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        ema.load_state_dict(state["ema"])
        history = list(state["history"])
        step_count = int(state["optimizer_steps"])
        start_epoch = int(state["epoch"])
    for epoch in range(start_epoch, epochs):
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(training_seed + epoch + 1)).tolist()
        optimizer.zero_grad(set_to_none=True); losses, ranked = [], 0
        for cursor, index in enumerate(order, 1):
            sample = dataset[index]; target = sample["targets"][0] if len(sample["targets"]) == 1 else None
            if ranking and target == "Cultivar":
                answers, correct, _ = categorical_variants(collator, sample["answer"], target)
                variants = [{**sample, "answer": answer} for answer in answers]
                batch = collator(variants)
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(**{key: batch[key] for key in MODEL_KEYS})
                prefix = int(output["prefix_embeddings"].shape[1]); labels = batch["labels"]
                combined = torch.cat([torch.full((labels.shape[0], prefix), -100, device=device, dtype=labels.dtype), labels], 1)
                logits = output["logits"][:, :-1].float(); shifted = combined[:, 1:]
                token_losses = F.cross_entropy(logits.transpose(1, 2), shifted, ignore_index=-100, reduction="none")
                scores = []
                for row_index, answer in enumerate(answers):
                    positions = _class_positions(collator, answer, target); start = int(labels[row_index].ne(-100).nonzero()[0])
                    shifted_pos = [prefix + start + value - 1 for value in positions]
                    token_ids = labels[row_index, [start + value for value in positions]]
                    scores.append(logits[row_index, shifted_pos].log_softmax(-1).gather(1, token_ids[:, None]).mean())
                rank_loss = F.cross_entropy(torch.stack(scores)[None] / float(settings["ranking_temperature"]), torch.tensor([correct], device=device))
                valid = shifted[correct].ne(-100)
                causal = (token_losses[correct] * valid).sum() / valid.sum()
                loss = causal + float(settings["ranking_weight"]) * rank_loss; ranked += 1
            else:
                batch = collator([sample]); batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                loss, _ = _weighted_loss(model, batch, collator, sample["answer"], sample["targets"], float(settings["class_token_weight"]))
            divisor = min(accumulation, len(order) - ((cursor - 1) // accumulation) * accumulation)
            (loss / divisor).backward(); losses.append(float(loss.detach()))
            if cursor % accumulation == 0 or cursor == len(order):
                torch.nn.utils.clip_grad_norm_([*bridge, *lora], 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); ema.update(model); step_count += 1
                print(f"{phase} epoch={epoch+1}/{epochs} step={step_count} records={cursor}/{len(order)} loss={np.mean(losses[-accumulation:]):.5f}", flush=True)
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "optimizer_steps": step_count, "ranking_records": ranked})
        torch.save({
            "epoch": epoch + 1,
            "optimizer_steps": step_count,
            "format": "phytospectra_v3_trainable_resume_v1",
            "trainable_state_dict": trainable_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "history": history,
        }, latest)
    ema.copy_to(model)
    checkpoint = config.root / source["predictor_checkpoint"]["path"]
    manifest = export_vlm_bundle(config, model, tokenizer, checkpoint, output_dir / "bundle", stage=phase, metadata={"scientific_result": False, "smoke": smoke, "recipe": "historical_stage42d_exact", "training_seed": training_seed, "history": history, "records": len(records), "source_bundle": str(input_bundle.relative_to(config.root).as_posix())})
    write_json(output_dir / "training_report.json", manifest)
    write_status(output_dir / "status.json", stage=phase, status="PASS")
    return manifest


def train_remediation(config: ExperimentConfig, input_bundle: Path, *, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    return _train_phase(config, input_bundle, config.run_artifacts / "vlm/remediation", phase="class_aware_remediation", ranking=False, smoke=smoke, resume=resume)


def train_cultivar_ranking(config: ExperimentConfig, input_bundle: Path, *, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    return _train_phase(config, input_bundle, config.run_artifacts / "vlm/ranking", phase="cultivar_sequence_ranking", ranking=True, smoke=smoke, resume=resume)


def fixed_interpolation(config: ExperimentConfig, remediation_bundle: Path, ranking_bundle: Path, *, smoke: bool = False) -> dict[str, Any]:
    alpha = float(config.values["model"]["interpolation_alpha"])
    if alpha != 0.60:
        raise ValueError("Frozen interpolation alpha changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_model, _, q_manifest = load_vlm_bundle(config, remediation_bundle, device)
    names = [name for name, _ in q_model.named_parameters() if (name.startswith("bridge.") and not name.startswith("bridge.vision_tower.")) or "lora_" in name]
    q_state = {name: dict(q_model.named_parameters())[name].detach().cpu().clone() for name in names}
    del q_model
    if device.type == "cuda": torch.cuda.empty_cache()
    r_model, tokenizer, r_manifest = load_vlm_bundle(config, ranking_bundle, device)
    parameters = dict(r_model.named_parameters())
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(((1.0 - alpha) * q_state[name] + alpha * parameters[name].detach().cpu()).to(parameters[name].device, parameters[name].dtype))
    checkpoint = config.root / r_manifest["predictor_checkpoint"]["path"]
    output = config.run_artifacts / "vlm/final"
    manifest = export_vlm_bundle(config, r_model, tokenizer, checkpoint, output / "bundle", stage="fixed_interpolation", metadata={"scientific_result": False, "smoke": smoke, "alpha": alpha, "selection_repeated_per_fold": False, "remediation_bundle": str(remediation_bundle.relative_to(config.root).as_posix()), "ranking_bundle": str(ranking_bundle.relative_to(config.root).as_posix())})
    write_json(output / "interpolation_report.json", manifest)
    return manifest
