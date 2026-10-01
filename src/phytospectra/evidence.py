"""Direct PhytoCore token-sensitivity evidence and ROI cache generation."""

from __future__ import annotations

from dataclasses import asdict
from collections import defaultdict
import hashlib
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import CULTIVAR_CLASSES, FERTILIZER_CLASSES, load_cube, load_normalization
from .io import file_record, read_json, write_json, write_status
from .model import (
    EnsembleEvidence,
    TargetSelection,
    TokenCounterfactualGrounder,
    normalize_unit_hsi,
    summarize_predictions,
)
from .training import load_phytocore_checkpoint


def split_stage42d_alignment_images(
    frame: pd.DataFrame,
    *,
    fold: int,
    train_images: int = 128,
    dev_images: int = 16,
) -> tuple[list[str], list[str]]:
    """Historical Stage-41C split used by the frozen Stage-42D recipe."""
    eligible = frame.loc[frame["Fold"] != int(fold)].copy()
    strata: dict[str, list[str]] = defaultdict(list)
    for row in eligible.itertuples(index=False):
        label = str(getattr(row, "Stratum", f"{row.Cultivar}__F{float(row.Fertilizer)}"))
        strata[label].append(str(row.ImageId))
    rng = random.Random(402026 + int(fold))
    for label in sorted(strata):
        rng.shuffle(strata[label])
    ordered: list[str] = []
    while any(strata.values()):
        for label in sorted(strata):
            if strata[label]:
                ordered.append(strata[label].pop())
    dev = sorted(ordered[:dev_images])
    train = sorted(ordered[dev_images:dev_images + train_images])
    if len(train) != train_images or len(dev) != dev_images or set(train) & set(dev):
        raise RuntimeError("Invalid Stage-42D alignment split")
    return train, dev


def stage42d_candidate_boxes(mask: np.ndarray, *, image_id: str, seed: int, count: int = 4) -> np.ndarray:
    """Four equal, unordered candidates; no target label influences sampling."""
    height, width = mask.shape
    boxes = [
        (x, y, x + 64, y + 16)
        for y in range(0, height - 16 + 1, 8)
        for x in range(0, width - 64 + 1, 16)
        if float(mask[y:y + 16, x:x + 64].mean()) >= 0.05
    ]
    if len(boxes) < count:
        raise ValueError(f"Image {image_id} has fewer than {count} candidate windows")
    digest = hashlib.sha256(f"{seed}:{image_id}:candidates".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    chosen = rng.sample(boxes, count)
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def stage42d_direct_selections(boxes: np.ndarray, effects: np.ndarray) -> dict[str, dict[str, Any]]:
    """Historical tie/uninformative policy for the four candidate effects."""
    targets = ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar")
    result = {}
    for target, values in zip(targets, np.asarray(effects, dtype=np.float64)):
        spread = float(np.ptp(values))
        tolerance = 1e-6 + 0.01 * spread
        acceptable = values.max() - values <= tolerance
        informative = bool(spread > tolerance and not acceptable.all())
        slots = np.flatnonzero(acceptable).tolist()
        selected = int(slots[0]) if informative else None
        box = None if selected is None else [int(value) for value in boxes[selected]]
        result[target] = {
            "status": "uninformative" if not informative else "tied" if len(slots) > 1 else "informative",
            "candidate_effects": values.tolist(),
            "acceptable_slots": slots,
            "selected_slot": selected,
            "selected_box": box,
            "roi_box_xyxy": box,
            "selection_source": "direct_phytocore_token_sensitivity",
            "unique_winner": bool(informative and len(slots) == 1),
        }
    return result


@torch.inference_mode()
def _stage42d_evidence(model, hsi: torch.Tensor, mask_tensor: torch.Tensor, mask: np.ndarray, *, image_id: str, seed: int) -> tuple[EnsembleEvidence, np.ndarray]:
    encoded = model.encode(hsi, mask_tensor)
    baseline = model.predict_from_encoded(encoded, return_attention=True)
    boxes = stage42d_candidate_boxes(mask, image_id=image_id, seed=seed)
    patch_h, patch_w = model.config.patch_size
    grid_w = model.config.image_width // patch_w
    keep = torch.ones(len(boxes), encoded["spatial_tokens"].shape[1], device=hsi.device, dtype=hsi.dtype)
    token_ids: list[tuple[int, ...]] = []
    for candidate, (x0, y0, x1, y1) in enumerate(boxes):
        ids = tuple(y * grid_w + x for y in range(y0 // patch_h, y1 // patch_h) for x in range(x0 // patch_w, x1 // patch_w))
        keep[candidate, list(ids)] = 0
        token_ids.append(ids)
    helper = TokenCounterfactualGrounder(spatial_top_k=1, spectral_top_k=1, variant_batch_size=4, roi_margin_patches=0)
    variants = helper._predict_variants(model, encoded, spatial_masks=keep)
    effects = helper._phenotype_changes(baseline, variants).T.astype(np.float32)
    selections = []
    for target_index, target in enumerate(("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar")):
        best = int(np.argmax(effects[target_index]))
        box = tuple(int(value) for value in boxes[best])
        selections.append(TargetSelection(
            target=target,
            target_index=target_index,
            spatial_indices=token_ids[best],
            spectral_indices=(),
            focus_spatial_indices=token_ids[best],
            selected_patch_boxes=(box,),
            focus_patch_boxes=(box,),
            roi_box_xyxy=box,
            comprehensiveness_spatial=float(effects[target_index, best]),
            sufficiency_error_spatial=0.0,
            comprehensiveness_spectral=0.0,
            sufficiency_error_spectral=0.0,
        ))
    attention = baseline["pqb_attention"][0].mean(dim=0).detach().float().cpu().numpy()
    n_spatial = int(encoded["spatial_tokens"].shape[1])
    result = EnsembleEvidence(
        targets=("GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar"),
        regression_z=baseline["regression"][0].detach().float().cpu().numpy(),
        fertilizer_probabilities=torch.softmax(baseline["fertilizer_logits"][0], 0).detach().float().cpu().numpy(),
        cultivar_probabilities=torch.softmax(baseline["cultivar_logits"][0], 0).detach().float().cpu().numpy(),
        spatial_scores=effects,
        spectral_scores=np.zeros((5, 0), dtype=np.float32),
        pqb_spatial_attention=attention[:, :n_spatial].astype(np.float32),
        pqb_spectral_attention=attention[:, n_spatial:].astype(np.float32),
        selections=tuple(selections),
        model_count=1,
    )
    return result, boxes


@torch.inference_mode()
def build_evidence_cache(
    config: ExperimentConfig,
    checkpoint: str | Path,
    *,
    partition: str = "alignment",
    smoke: bool = False,
) -> dict[str, Any]:
    """Cache evidence once per image; evidence is sensitivity, not causality."""
    if partition not in {"alignment", "outer"}:
        raise ValueError("partition must be alignment or outer")
    status_path = config.run_artifacts / "evidence" / f"{partition}_status.json"
    write_status(status_path, stage="evidence", status="RUNNING", partition=partition)
    index = pd.read_csv(config.artifacts / "dataset/dataset_index.csv", dtype={"ImageId": str})
    folds = pd.read_csv(config.artifacts / "splits/folds.csv", dtype={"ImageId": str})[["ImageId", "Fold"]]
    frame = index.merge(folds, on="ImageId", validate="one_to_one")
    if partition == "outer":
        selected = frame.loc[frame["Fold"] == config.fold].sort_values("ImageId")
    else:
        # Historical Stage-41C smoke fixture: three train and two dev images.
        train_count = 3 if smoke else int(config.values["training"]["vlm"]["alignment_images"]) - int(config.values["training"]["vlm"]["dev_images"])
        dev_count = 2 if smoke else int(config.values["training"]["vlm"]["dev_images"])
        train_ids, dev_ids = split_stage42d_alignment_images(
            frame,
            fold=config.fold,
            train_images=train_count,
            dev_images=dev_count,
        )
        selected_ids = train_ids + dev_ids
        selected = frame.loc[frame["ImageId"].isin(selected_ids)].sort_values("ImageId")
    if smoke and partition == "outer":
        selected = selected.head(2)
    normalization = load_normalization(config.artifacts / "normalization", config.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_phytocore_checkpoint(checkpoint, device)
    checkpoint_record = file_record(checkpoint, config.root)
    candidate_seed = int(config.values["evidence"]["candidate_seed"]) + config.fold
    cache_dir = config.run_artifacts / "evidence" / partition
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for position, row in enumerate(selected.itertuples(index=False), 1):
        arrays_path = cache_dir / f"{row.ImageId}.npz"
        record_path = cache_dir / f"{row.ImageId}.json"
        if arrays_path.is_file() and record_path.is_file():
            prior = read_json(record_path)
            if (
                prior.get("status") == "PASS" and prior.get("seed") == config.seed
                and prior.get("fold") == config.fold
                and prior.get("predictor_checkpoint", {}).get("sha256") == checkpoint_record["sha256"]
            ):
                records.append(file_record(record_path, config.root))
                print(f"evidence {partition} {position}/{len(selected)} image={row.ImageId} SKIPPED_VERIFIED", flush=True)
                continue
        cube = load_cube(row.ImagePath)
        mask = np.load(row.MaskPath, allow_pickle=False).astype(bool)
        normalized = normalize_unit_hsi(cube, mask, normalization["mean"], normalization["std"])
        hsi = torch.from_numpy(np.moveaxis(normalized, -1, 0).copy())[None].to(device)
        mask_tensor = torch.from_numpy(mask.copy())[None].to(device)
        evidence, candidate_boxes = _stage42d_evidence(
            model, hsi, mask_tensor, mask, image_id=row.ImageId, seed=candidate_seed
        )
        predictions = summarize_predictions(evidence, normalization["targets"], FERTILIZER_CLASSES, CULTIVAR_CLASSES)
        np.savez_compressed(
            arrays_path,
            spatial_scores=evidence.spatial_scores,
            spectral_scores=evidence.spectral_scores,
            pqb_spatial_attention=evidence.pqb_spatial_attention,
            pqb_spectral_attention=evidence.pqb_spectral_attention,
            candidate_boxes=candidate_boxes,
            candidate_effects=evidence.spatial_scores,
        )
        payload = {
            "status": "PASS",
            "image_id": row.ImageId,
            "seed": config.seed,
            "fold": config.fold,
            "partition": partition,
            "alignment_partition": (
                "dev" if partition == "alignment" and row.ImageId in set(dev_ids) else
                "train" if partition == "alignment" else None
            ),
            "predictor_checkpoint": checkpoint_record,
            "predictions": predictions,
            "selections": stage42d_direct_selections(candidate_boxes, evidence.spatial_scores),
            "arrays": file_record(arrays_path, config.root),
            "candidate_seed": candidate_seed,
            "candidate_count": 4,
            "interpretation": "Frozen PhytoCore grouped spatial-token zeroing among four fixed equal candidates; not biological localization.",
        }
        write_json(record_path, payload)
        records.append(file_record(record_path, config.root))
        print(f"evidence {partition} {position}/{len(selected)} image={row.ImageId}", flush=True)
    report = {
        "status": "PASS", "stage": "evidence", "scientific_result": False,
        "recipe": "historical_stage42d_exact", "partition": partition,
        "images": len(selected), "checkpoint_metadata": metadata, "records": records,
        "alignment_split": ({"split_seed": 402026 + config.fold, "train_image_ids": train_ids, "dev_image_ids": dev_ids} if partition == "alignment" else None),
    }
    write_json(config.run_artifacts / "evidence" / f"{partition}_report.json", report)
    write_status(status_path, stage="evidence", status="PASS", images=len(selected))
    return report
