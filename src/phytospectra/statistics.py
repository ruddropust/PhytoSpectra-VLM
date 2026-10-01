"""Paired image-cluster bootstrap, permutation, effect size, and Holm control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation import ABLATIONS
from .config import ExperimentConfig
from .io import write_json


def _load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = []
    for row in rows:
        parsed_items = row.get("parsed", {}).get("items", []) if isinstance(row.get("parsed"), dict) else []
        parsed = {item.get("target"): item for item in parsed_items if isinstance(item, dict)}
        class_fields = 0
        class_correct = 0
        for expected in row.get("expected", {}).get("items", []):
            target = expected.get("target")
            if target not in {"Fertilizer", "Cultivar"}:
                continue
            class_fields += 1
            class_correct += int(parsed.get(target, {}).get("class_value") == expected.get("class_value"))
        normalized.append({
            "record_id": row["record_id"],
            "image_id": row["image_id"],
            "accepted": float(row["accepted"]),
            "class_fields": class_fields,
            "class_correct": class_correct,
        })
    return pd.DataFrame(normalized)


def _resampled_statistics(per_image: np.ndarray, rng: np.random.Generator, resamples: int, permutations: int) -> dict[str, Any]:
    observed = float(per_image.mean())
    boot = np.asarray([
        per_image[rng.integers(0, len(per_image), len(per_image))].mean()
        for _ in range(resamples)
    ])
    signs = rng.choice((-1.0, 1.0), size=(permutations, len(per_image)))
    null = np.mean(signs * per_image[None, :], axis=1)
    p = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (permutations + 1))
    std = float(per_image.std(ddof=1)) if len(per_image) > 1 else 0.0
    return {
        "delta": observed,
        "image_cluster_bootstrap_95ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "paired_permutation_two_sided_p": p,
        "cohen_dz": observed / std if std > 0 else None,
    }


def _paired(full: pd.DataFrame, ablated: pd.DataFrame, seed: int, resamples: int, permutations: int) -> dict[str, Any]:
    merged = full.merge(ablated, on=["record_id", "image_id"], suffixes=("_full", "_ablated"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    acceptance_by_image = merged.assign(
        delta=merged["accepted_full"] - merged["accepted_ablated"]
    ).groupby("image_id")["delta"].mean().to_numpy()
    acceptance = _resampled_statistics(acceptance_by_image, rng, resamples, permutations)
    class_rows = merged[(merged["class_fields_full"] > 0) & (merged["class_fields_ablated"] > 0)].copy()
    if len(class_rows):
        class_rows["full_rate"] = class_rows["class_correct_full"] / class_rows["class_fields_full"]
        class_rows["ablated_rate"] = class_rows["class_correct_ablated"] / class_rows["class_fields_ablated"]
        class_by_image = class_rows.assign(
            delta=class_rows["full_rate"] - class_rows["ablated_rate"]
        ).groupby("image_id")["delta"].mean().to_numpy()
        class_stats = _resampled_statistics(class_by_image, rng, resamples, permutations)
        full_class = float(class_rows["class_correct_full"].sum() / class_rows["class_fields_full"].sum())
        ablated_class = float(class_rows["class_correct_ablated"].sum() / class_rows["class_fields_ablated"].sum())
    else:
        class_stats = None
        full_class = ablated_class = None
    return {
        "images": len(acceptance_by_image), "paired_questions": len(merged),
        "paired_classification_fields": int(class_rows["class_fields_full"].sum()) if len(class_rows) else 0,
        "point_estimates": {
            "full_acceptance": float(merged["accepted_full"].mean()),
            "ablated_acceptance": float(merged["accepted_ablated"].mean()),
            "full_minus_ablated_acceptance": acceptance["delta"],
            "full_class_agreement": full_class,
            "ablated_class_agreement": ablated_class,
            "full_minus_ablated_class_agreement": None if class_stats is None else class_stats["delta"],
        },
        "acceptance_statistics": acceptance,
        "class_agreement_statistics": class_stats,
        "independent_unit": "image",
    }


def _holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted, running = {}, 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * raw[key]))
        adjusted[key] = running
    return adjusted


def paired_statistics(config: ExperimentConfig, *, smoke: bool = False) -> dict[str, Any]:
    base = config.run_results / "vlm"
    full = _load(base / "full/records.jsonl")
    settings = config.values["evaluation"]
    resamples = 100 if smoke else int(settings["bootstrap_resamples"])
    permutations = 200 if smoke else int(settings["permutations"])
    results = {mode: _paired(full, _load(base / mode / "records.jsonl"), config.seed + index, resamples, permutations) for index, mode in enumerate(ABLATIONS)}
    raw_p = {
        mode: (
            value["class_agreement_statistics"] or value["acceptance_statistics"]
        )["paired_permutation_two_sided_p"]
        for mode, value in results.items()
    }
    report = {
        "status": "PASS", "stage": "statistics", "scientific_result": not smoke,
        "seed": config.seed, "fold": config.fold, "results": results,
        "multiple_comparison_control": {"method": "Holm-Bonferroni", "raw_p": raw_p, "adjusted_p": _holm(raw_p), "alpha": 0.05},
    }
    write_json(config.run_results / "statistics.json", report)
    return report
