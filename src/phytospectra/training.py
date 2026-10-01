"""Training loops for the unchanged PhytoCore and VLM architecture."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import FERTILIZER_CLASSES, HyperLeafDataset, load_normalization
from .io import file_record, read_json, utc_now, write_json, write_status
from .model import PhytoCore, PhytoCoreMultiTaskLoss, inverse_sqrt_class_weights
from .splits import split_fold


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def train_phytocore(config: ExperimentConfig, *, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    """Train one fold for a fixed budget; the outer fold is never used to select epochs."""
    seed_everything(config.seed)
    stage = "phytocore"
    status_path = config.run_artifacts / stage / "status.json"
    write_status(status_path, stage=stage, status="RUNNING", seed=config.seed, fold=config.fold)
    index_path = config.artifacts / "dataset" / "dataset_index.csv"
    fold_path = config.artifacts / "splits" / "folds.csv"
    if not index_path.is_file() or not fold_path.is_file():
        raise FileNotFoundError("Run the prepare stage before training")
    index = pd.read_csv(index_path, dtype={"ImageId": str})
    folds = pd.read_csv(fold_path, dtype={"ImageId": str})[["ImageId", "Fold"]]
    frame = index.merge(folds, on="ImageId", validate="one_to_one")
    train_frame, _ = split_fold(frame, config.fold)
    normalization = load_normalization(config.artifacts / "normalization", config.fold)
    dataset = HyperLeafDataset(train_frame, normalization, augment=True, base_seed=config.seed)
    settings = config.values["training"]["phytocore"]
    epochs = 1 if smoke else int(settings["epochs"])
    batch_size = min(4, int(settings["batch_size"])) if smoke else int(settings["batch_size"])
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(settings["workers"]),
        generator=generator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = PhytoCore().to(device)
    labels = torch.tensor([FERTILIZER_CLASSES.index(float(value)) for value in train_frame["Fertilizer"]])
    criterion = PhytoCoreMultiTaskLoss(inverse_sqrt_class_weights(labels, len(FERTILIZER_CLASSES))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(settings.get("minimum_learning_rate", 1.0e-6))
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch, history = 0, []
    checkpoint_dir = config.run_artifacts / stage
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    if resume and latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"])
        history = list(state["history"])
    max_steps = 2 if smoke else None
    for epoch in range(start_epoch, epochs):
        model.train()
        dataset.set_epoch(epoch + 1)
        losses = []
        for step, batch in enumerate(loader, 1):
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(batch["hsi"], batch["leaf_mask"])
                loss = criterion(output, batch)["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if step % int(settings["log_every"]) == 0 or (max_steps and step == max_steps):
                print(f"PhytoCore epoch {epoch + 1}/{epochs} step {step}/{len(loader)} loss={np.mean(losses[-10:]):.5f}", flush=True)
            if max_steps and step >= max_steps:
                break
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "steps": len(losses), "learning_rate": float(optimizer.param_groups[0]["lr"])})
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
        }, latest)
    final_path = checkpoint_dir / ("smoke.pt" if smoke else "phytocore_final.pt")
    metadata = {
        "format": "phytospectra_v3_phytocore_v1",
        "created_at_utc": utc_now(),
        "seed": config.seed,
        "fold": config.fold,
        "epochs": epochs,
        "fixed_epoch_selection": True,
        "outer_fold_used_for_selection": False,
        "fertilizer_formulation": "WEIGHTED_CE",
        "model_parameters": model.parameter_count(),
        "smoke": smoke,
        "recipe": "historical_stage42d_exact",
        "train_only_augmentation": True,
        "scheduler": "CosineAnnealingLR",
        "minimum_learning_rate": float(settings.get("minimum_learning_rate", 1.0e-6)),
        "automatic_mixed_precision": amp,
    }
    torch.save({"model_state_dict": model.cpu().state_dict(), "metadata": metadata}, final_path)
    report = {"status": "PASS", "stage": stage, "scientific_result": False, "metadata": metadata, "history": history, "checkpoint": file_record(final_path, config.root)}
    write_json(checkpoint_dir / "training_report.json", report)
    write_status(status_path, stage=stage, status="PASS", report=str(checkpoint_dir / "training_report.json"))
    return report


def load_phytocore_checkpoint(path: str | Path, device: torch.device) -> tuple[PhytoCore, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = PhytoCore()
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval(), dict(state.get("metadata", {}))
