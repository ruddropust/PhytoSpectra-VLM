"""Question-conditioned global/local VLM alignment and bundle export."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data import TARGETS, load_cube, load_normalization
from .io import file_record, read_json, utc_now, write_json, write_status
from .model import (
    HyperLeafVQACollator,
    QwenAlignmentConfig,
    TrainableParameterEMA,
    VisualTokenBridgeConfig,
    create_model,
    load_qwen_tokenizer,
    prepare_visual_inputs,
)
from .training import seed_everything


SYSTEM_PROMPT = """You are PhytoSpectra-VLM. Use the attached continuous
hyperspectral visual tokens and question. Return one JSON object only, no prose.
Schema: {"version":1,"intent":"TARGET_VALUE|EVIDENCE_REQUEST|IMAGE_SUMMARY|UNSUPPORTED",
"items":[{"target":"TARGET","value_ref":"TARGET.value"}]}.
TARGET is GrainWeight, Gsw, PhiPS2, Fertilizer, or Cultivar. For Fertilizer and
Cultivar include "class_value": a string predicted from the visual input.
Fertilizer labels: "0", "0.5", "1". Cultivar labels: "Heerup", "Kvium",
"Rembrandt", "Sheriff". EVIDENCE_REQUEST additionally requires
"roi_ref":"TARGET.roi" in its item. IMAGE_SUMMARY uses all five items in the
target order above. UNSUPPORTED uses an empty items list. Never output confidence,
measurements, coordinates, file paths, image IDs, disease diagnoses or units.
The trusted evidence renderer supplies exact fields after validating references."""

MODEL_KEYS = (
    "global_hsi", "global_leaf_mask", "local_hsi", "local_leaf_masks",
    "local_view_mask", "local_target_indices", "primary_target_index",
    "roi_boxes_xyxy",
    "input_ids", "attention_mask", "labels", "question_input_ids",
    "question_attention_mask",
)


def language_config(config: ExperimentConfig) -> dict[str, Any]:
    values = config.values["training"]["vlm"]
    return {
        "system_prompt": SYSTEM_PROMPT,
        "max_sequence_tokens": int(values["max_sequence_tokens"]),
        "max_question_tokens": int(values["max_question_tokens"]),
    }


class VQADataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], frame: pd.DataFrame, normalization: dict[str, Any], fold: int):
        self.records = records
        self.rows = frame.set_index("ImageId")
        self.normalization = normalization
        self.fold = int(fold)
        self.bridge_config = VisualTokenBridgeConfig()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        row = self.rows.loc[str(record["image_id"])]
        cube = load_cube(row["ImagePath"])
        mask = np.load(row["MaskPath"], allow_pickle=False).astype(bool)
        evidence = read_json(record["evidence_path"])
        targets = list(record["targets"])
        local_targets = targets[: self.bridge_config.max_local_views] if record["intent"] != "IMAGE_SUMMARY" else []
        boxes = {}
        for target in local_targets:
            selection = evidence["selections"][target]
            box = selection.get("selected_box", selection.get("roi_box_xyxy"))
            if box is not None:
                boxes[target] = box
        primary = local_targets[0] if local_targets else "GENERAL"
        visual = prepare_visual_inputs(cube, mask, self.normalization["mean"], self.normalization["std"], boxes, primary, self.bridge_config, "cpu")
        item = {key: value.squeeze(0) for key, value in visual.items()}
        item.update({
            "outer_fold": torch.tensor(self.fold, dtype=torch.long),
            **record,
        })
        return item


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trainable_state(model) -> dict[str, torch.Tensor]:
    """Serialize only bridge/LoRA tensors; frozen base weights are reproducible inputs."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if set(state) != expected:
        raise ValueError("Resume checkpoint trainable parameter names changed")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


def export_vlm_bundle(config: ExperimentConfig, model, tokenizer, checkpoint: str | Path, bundle: Path, *, stage: str, metadata: dict[str, Any]) -> dict[str, Any]:
    bridge_path = bundle / "visual_bridge_ema.pt"
    lora_path = bundle / "qwen_lora_ema"
    tokenizer_path = bundle / "tokenizer"
    bundle.mkdir(parents=True, exist_ok=True)
    bridge_state = {name: value.detach().cpu() for name, value in model.bridge.state_dict().items() if not name.startswith("vision_tower.")}
    torch.save({"state_dict": bridge_state, "ema_applied": True, "parameter_count": sum(value.numel() for value in bridge_state.values())}, bridge_path)
    model.language_model.save_pretrained(lora_path, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_path)
    manifest = {
        "status": "PASS", "format": "phytospectra_v3_vlm_bundle_v1", "created_at_utc": utc_now(),
        "stage": stage, "seed": config.seed, "fold": config.fold,
        "architecture": "Frozen PhytoCore + global/local target/question bridge + Qwen2.5-1.5B LoRA + EMA",
        "predictor_checkpoint": file_record(checkpoint, config.root),
        "visual_bridge_ema": file_record(bridge_path, config.root),
        "qwen_lora_ema": str(lora_path.relative_to(config.root).as_posix()),
        "tokenizer": str(tokenizer_path.relative_to(config.root).as_posix()),
        **metadata,
    }
    write_json(bundle / "manifest.json", manifest)
    return manifest


def train_vlm(config: ExperimentConfig, checkpoint: str | Path, *, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    """Train bridge + Qwen LoRA while keeping PhytoCore and base Qwen frozen."""
    matched_seed = 415026 + 10 * (config.seed - 2026) + config.fold
    seed_everything(matched_seed)
    status_path = config.run_artifacts / "vlm/matched/status.json"
    write_status(status_path, stage="vlm", status="RUNNING")
    question_path = config.run_artifacts / "vqa/questions.jsonl"
    if not question_path.is_file():
        raise FileNotFoundError("Run VQA generation before VLM training")
    records = _records(question_path)
    train_records = [record for record in records if record.get("alignment_partition") == "train"]
    dev_records = [record for record in records if record.get("alignment_partition") == "dev"]
    train_ids = {record["image_id"] for record in train_records}
    dev_ids = {record["image_id"] for record in dev_records}
    if set(train_ids) & set(dev_ids):
        raise RuntimeError("Stage-42D alignment train/dev overlap")
    if not smoke and (len(train_ids), len(dev_ids), len(train_records), len(dev_records)) != (128, 16, 4096, 192):
        raise ValueError("Stage-42D alignment population changed")
    if smoke:
        train_records = [next(row for row in train_records if row["image_id"] == image_id) for image_id in sorted(train_ids)[:2]]
        dev_records = [next(row for row in dev_records if row["image_id"] == image_id) for image_id in sorted(dev_ids)[:2]]
    frame = pd.read_csv(config.artifacts / "dataset/dataset_index.csv", dtype={"ImageId": str})
    normalization = load_normalization(config.artifacts / "normalization", config.fold)
    settings = config.values["training"]["vlm"]
    qwen_config = QwenAlignmentConfig(
        local_files_only=bool(settings["local_files_only"]),
        gradient_checkpointing=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = load_qwen_tokenizer(qwen_config)
    model = create_model(checkpoint, qwen_config=qwen_config, dtype=dtype, device=device)
    collator = HyperLeafVQACollator(tokenizer, language_config(config))
    dataset = VQADataset(train_records, frame, normalization, config.fold)
    bridge_params = [parameter for name, parameter in model.bridge.named_parameters() if parameter.requires_grad and not name.startswith("vision_tower.")]
    lora_params = [parameter for parameter in model.language_model.parameters() if parameter.requires_grad]
    run_dir = config.run_artifacts / "vlm" / "matched"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Historical Stage-41D phase 1: fixed reference-contract warm-up from the
    # fresh bridge/LoRA. Its final EMA weights initialize matched alignment.
    warmup_seed = 402026 + 10 * (config.seed - 2026) + config.fold
    warmup_state = run_dir / "warmup_final.pt"
    warmup_report = run_dir / "warmup_report.json"
    if resume and warmup_state.is_file() and warmup_report.is_file():
        restore_trainable_state(model, torch.load(warmup_state, map_location="cpu", weights_only=True)["trainable_state_dict"])
    else:
        seed_everything(warmup_seed)
        warmup_optimizer = torch.optim.AdamW([
            {"params": bridge_params, "lr": float(settings["bridge_learning_rate"])},
            {"params": lora_params, "lr": float(settings["lora_learning_rate"])},
        ], weight_decay=float(settings["weight_decay"]))
        warmup_epochs = 1 if smoke else int(settings["warmup_epochs"])
        warmup_steps_per_epoch = 1 if smoke else int(settings["warmup_steps_per_epoch"])
        warmup_accumulation = 1 if smoke else int(settings["warmup_accumulation"])
        warmup_steps = warmup_epochs * warmup_steps_per_epoch
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            warmup_optimizer,
            lambda step: min(1.0, (step + 1) / max(1, round(warmup_steps * 0.05)))
            * 0.5 * (1 + math.cos(math.pi * min(step, warmup_steps) / warmup_steps)),
        )
        warmup_ema = TrainableParameterEMA(model, decay=float(settings["ema_decay"]))
        warmup_losses = []
        model.train(); model.bridge.vision_tower.eval()
        for warmup_step in range(warmup_steps):
            epoch = warmup_step // warmup_steps_per_epoch + 1
            within = warmup_step % warmup_steps_per_epoch
            order = list(range(len(dataset)))
            random.Random(warmup_seed + epoch).shuffle(order)
            start = within * warmup_accumulation * 2
            chosen = [order[(start + offset) % len(order)] for offset in range(warmup_accumulation * 2)]
            warmup_optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for micro in range(warmup_accumulation):
                samples = [dataset[index] for index in chosen[2 * micro:2 * micro + 2]]
                batch = collator(samples)
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(**{key: batch[key] for key in MODEL_KEYS})["loss"]
                (loss / warmup_accumulation).backward()
                step_losses.append(float(loss.detach().cpu()))
            torch.nn.utils.clip_grad_norm_([*bridge_params, *lora_params], float(settings["gradient_clip_norm"]))
            warmup_optimizer.step(); warmup_ema.update(model); warmup_scheduler.step()
            warmup_losses.append(float(np.mean(step_losses)))
            print(f"VLM warm-up step={warmup_step + 1}/{warmup_steps} loss={warmup_losses[-1]:.5f}", flush=True)
        warmup_ema.copy_to(model)
        torch.save({"trainable_state_dict": trainable_state(model)}, warmup_state)
        write_json(warmup_report, {"status": "PASS", "phase": "reference_warmup", "seed": warmup_seed, "optimizer_steps": warmup_steps, "mean_loss": float(np.mean(warmup_losses)), "ema_applied": True})

    seed_everything(matched_seed)
    optimizer = torch.optim.AdamW([
        {"params": bridge_params, "lr": float(settings["bridge_learning_rate"])},
        {"params": lora_params, "lr": float(settings["lora_learning_rate"])},
    ], weight_decay=float(settings["weight_decay"]))
    ema = TrainableParameterEMA(model, decay=float(settings["ema_decay"]))
    epochs = 1 if smoke else int(settings["epochs"])
    accumulation = 1 if smoke else int(settings["gradient_accumulation_steps"])
    max_optimizer_steps = 2 if smoke else int(settings["max_optimizer_steps"])
    latest = run_dir / "latest.pt"
    start_epoch, global_step, history, evaluation_history = 0, 0, [], []
    best_score = (-1.0, -1.0)
    best_epoch: int | None = None
    best_path = run_dir / "best_ema.pt"
    if resume and latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        restore_trainable_state(model, state["trainable_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        ema.load_state_dict(state["ema"])
        start_epoch, global_step, history = int(state["epoch"]), int(state["global_step"]), list(state["history"])
        evaluation_history = list(state.get("evaluation_history", []))
        best_score = tuple(float(value) for value in state.get("best_score", [-1.0, -1.0]))
        best_epoch = state.get("best_epoch")

    def evaluate_candidate(epoch: int, step: int) -> None:
        nonlocal best_score, best_epoch
        # Local import avoids a module-level cycle: contract optimization reuses
        # the same strict raw generator and verifier.
        from .contract_optimization import _raw_evaluate

        model.eval()
        raw_metrics, _ = _raw_evaluate(
            model, tokenizer, collator, dev_records, frame, normalization,
            config.fold, device, int(config.values["evaluation"]["max_new_tokens"]),
        )
        with ema.average_parameters(model):
            ema_metrics, _ = _raw_evaluate(
                model, tokenizer, collator, dev_records, frame, normalization,
                config.fold, device, int(config.values["evaluation"]["max_new_tokens"]),
            )
            score = (
                float(ema_metrics.get("raw_class_agreement_with_phytocore") or 0.0),
                float(ema_metrics.get("raw_reference_acceptance") or 0.0),
            )
            if score > best_score:
                torch.save({"format": "phytospectra_v3_best_ema_v1", "ema_applied": True, "trainable_state_dict": trainable_state(model)}, best_path)
                best_score, best_epoch = score, epoch
        evaluation_history.append({
            "step": step,
            "epoch": epoch,
            "evaluation_population": "bounded_two_images" if smoke else "alignment_dev_16_images_x_12_queries",
            "raw": raw_metrics,
            "ema": ema_metrics,
            "selection_score": list(score),
            "scientific_result": False,
        })
        print(f"VLM evaluation step={step} epoch={epoch} EMA score={score}", flush=True)

    expected_steps = math.ceil(len(dataset) / accumulation) * epochs
    if not smoke and max_optimizer_steps != expected_steps:
        raise ValueError(f"Stage-42D matched schedule changed: expected {expected_steps} steps, configured {max_optimizer_steps}")
    if not smoke and not evaluation_history:
        evaluate_candidate(0, 0)
    for epoch in range(start_epoch, epochs):
        model.train()
        model.bridge.vision_tower.eval()
        losses = []
        order = list(range(len(dataset)))
        random.Random(matched_seed + epoch + 1).shuffle(order)
        for start in range(0, len(order), accumulation):
            selected = order[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            block_losses = []
            for index in selected:
                batch = collator([dataset[index]])
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    output = model(**{key: batch[key] for key in MODEL_KEYS})
                (output["loss"] / len(selected)).backward()
                block_losses.append(float(output["loss"].detach().cpu()))
            torch.nn.utils.clip_grad_norm_([*bridge_params, *lora_params], float(settings["gradient_clip_norm"]))
            optimizer.step(); ema.update(model); global_step += 1
            losses.extend(block_losses)
            print(f"VLM epoch {epoch + 1}/{epochs} optimizer_step={global_step}/{max_optimizer_steps} loss={np.mean(block_losses):.5f}", flush=True)
            if global_step >= max_optimizer_steps:
                break
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "optimizer_steps": global_step})
        evaluate_candidate(epoch + 1, global_step)
        torch.save({"format": "phytospectra_v3_trainable_resume_v1", "epoch": epoch + 1, "global_step": global_step, "trainable_state_dict": trainable_state(model), "optimizer_state_dict": optimizer.state_dict(), "ema": ema.state_dict(), "history": history, "evaluation_history": evaluation_history, "best_score": list(best_score), "best_epoch": best_epoch}, latest)
        if global_step >= max_optimizer_steps:
            break
    if not best_path.is_file():
        raise RuntimeError("Matched alignment did not produce a selectable EMA checkpoint")
    restore_trainable_state(model, torch.load(best_path, map_location="cpu", weights_only=True)["trainable_state_dict"])
    bundle = run_dir / "bundle"
    manifest = export_vlm_bundle(config, model, tokenizer, checkpoint, bundle, stage="matched_alignment", metadata={
        "scientific_result": False, "smoke": smoke,
        "recipe": "historical_stage42d_exact", "warmup_seed": warmup_seed,
        "warmup_optimizer_steps": 1 if smoke else int(settings["warmup_epochs"]) * int(settings["warmup_steps_per_epoch"]),
        "training_seed": matched_seed,
        "train_images": len(train_ids), "dev_images": len(dev_ids), "train_records": len(train_records), "dev_records": len(dev_records),
        "history": history, "evaluation_history": evaluation_history,
        "best_score": list(best_score), "selected_epoch": best_epoch,
        "selection_rule": "EMA dev class agreement, then reference acceptance; ties retain the earlier checkpoint",
    })
    write_json(run_dir / "training_report.json", manifest)
    write_status(status_path, stage="vlm", status="PASS", bundle=str(bundle))
    return manifest


def load_vlm_bundle(config: ExperimentConfig, bundle: str | Path, device: torch.device):
    """Load a newly generated v3 bundle; all stored paths are repository-relative."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .model import PhytoSpectraVLM, TargetConditionedGlobalLocalBridge
    from .training import load_phytocore_checkpoint

    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "manifest.json")
    checkpoint = config.root / manifest["predictor_checkpoint"]["path"]
    vision, _ = load_phytocore_checkpoint(checkpoint, torch.device("cpu"))
    settings = config.values["training"]["vlm"]
    qwen_config = QwenAlignmentConfig(local_files_only=bool(settings["local_files_only"]), gradient_checkpointing=False)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(qwen_config.model_name, dtype=dtype, low_cpu_mem_usage=True, local_files_only=qwen_config.local_files_only, trust_remote_code=False, attn_implementation="sdpa")
    language_model = PeftModel.from_pretrained(base, str(config.root / manifest["qwen_lora_ema"]), is_trainable=False)
    bridge = TargetConditionedGlobalLocalBridge(vision, VisualTokenBridgeConfig())
    payload = torch.load(config.root / manifest["visual_bridge_ema"]["path"], map_location="cpu", weights_only=False)
    missing, unexpected = bridge.load_state_dict(payload["state_dict"], strict=False)
    allowed = {name for name in bridge.state_dict() if name.startswith("vision_tower.")}
    if set(missing) != allowed or unexpected:
        raise ValueError(f"Bridge bundle mismatch: missing={missing}, unexpected={unexpected}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.root / manifest["tokenizer"]), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PhytoSpectraVLM(bridge, language_model).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, manifest
