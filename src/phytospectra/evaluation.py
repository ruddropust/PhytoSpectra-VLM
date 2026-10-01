"""Image-level PhytoCore evaluation with bootstrap confidence intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import CULTIVAR_CLASSES, FERTILIZER_CLASSES, REGRESSION_TARGETS, HyperLeafDataset, load_normalization
from .io import file_record, write_json, write_status
from .splits import split_fold
from .training import load_phytocore_checkpoint


def _ci(values: np.ndarray, statistic: Callable[[np.ndarray], float], seed: int, resamples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.asarray([statistic(values[rng.integers(0, n, n)]) for _ in range(resamples)])
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


@torch.inference_mode()
def evaluate_phytocore(config: ExperimentConfig, checkpoint: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    status_path = config.run_results / "phytocore" / "status.json"
    write_status(status_path, stage="evaluate_phytocore", status="RUNNING")
    index = pd.read_csv(config.artifacts / "dataset/dataset_index.csv", dtype={"ImageId": str})
    folds = pd.read_csv(config.artifacts / "splits/folds.csv", dtype={"ImageId": str})[["ImageId", "Fold"]]
    frame = index.merge(folds, on="ImageId", validate="one_to_one")
    _, validation = split_fold(frame, config.fold)
    if smoke:
        validation = validation.head(8)
    normalization = load_normalization(config.artifacts / "normalization", config.fold)
    dataset = HyperLeafDataset(validation, normalization)
    loader = DataLoader(dataset, batch_size=int(config.values["evaluation"]["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_phytocore_checkpoint(checkpoint, device)
    rows = []
    for batch in loader:
        output = model(batch["hsi"].to(device), batch["leaf_mask"].to(device))
        regression_z = output["regression"].float().cpu().numpy()
        regression = np.column_stack([
            regression_z[:, i] * normalization["targets"][target]["std"] + normalization["targets"][target]["mean"]
            for i, target in enumerate(REGRESSION_TARGETS)
        ])
        fertilizer = output["fertilizer_logits"].argmax(dim=1).cpu().numpy()
        cultivar = output["cultivar_logits"].argmax(dim=1).cpu().numpy()
        for i, image_id in enumerate(batch["image_id"]):
            row: dict[str, Any] = {"ImageId": image_id}
            for j, target in enumerate(REGRESSION_TARGETS):
                row[f"true_{target}"] = float(batch["regression_targets_raw"][i, j])
                row[f"pred_{target}"] = float(regression[i, j])
            row.update({
                "true_Fertilizer": str(FERTILIZER_CLASSES[int(batch["fertilizer_label"][i])]),
                "pred_Fertilizer": str(FERTILIZER_CLASSES[int(fertilizer[i])]),
                "true_Cultivar": CULTIVAR_CLASSES[int(batch["cultivar_label"][i])],
                "pred_Cultivar": CULTIVAR_CLASSES[int(cultivar[i])],
            })
            rows.append(row)
    predictions = pd.DataFrame(rows)
    output_dir = config.run_results / "phytocore"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics: dict[str, Any] = {"images": len(predictions), "independent_unit": "image", "regression": {}, "classification": {}}
    for target in REGRESSION_TARGETS:
        true = predictions[f"true_{target}"].to_numpy()
        pred = predictions[f"pred_{target}"].to_numpy()
        absolute = np.abs(true - pred)
        metrics["regression"][target] = {
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(mean_squared_error(true, pred) ** 0.5),
            "r2": float(r2_score(true, pred)) if len(true) > 1 else None,
            "mae_bootstrap_95ci": _ci(absolute, np.mean, config.seed, int(config.values["evaluation"]["bootstrap_resamples"])) if not smoke else None,
        }
    for target, labels in (("Fertilizer", [str(value) for value in FERTILIZER_CLASSES]), ("Cultivar", list(CULTIVAR_CLASSES))):
        true, pred = predictions[f"true_{target}"], predictions[f"pred_{target}"]
        metrics["classification"][target] = {
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
            "class_order": labels,
        }
    report = {"status": "PASS", "stage": "evaluate_phytocore", "scientific_result": not smoke, "seed": config.seed, "fold": config.fold, "training_metadata": metadata, "metrics": metrics, "predictions": file_record(predictions_path, config.root)}
    write_json(output_dir / "evaluation_report.json", report)
    write_status(status_path, stage="evaluate_phytocore", status="PASS", report=str(output_dir / "evaluation_report.json"))
    return report
