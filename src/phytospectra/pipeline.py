"""End-to-end orchestration. Every stage creates new artifacts and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .ablation import run_ablations
from .aggregate import aggregate_campaign
from .binding import bind_or_verify
from .config import ExperimentConfig
from .contract_optimization import train_contract_optimization
from .data import fit_normalization, load_metadata, prepare_images
from .evaluation import evaluate_phytocore
from .evidence import build_evidence_cache
from .io import read_json, write_json
from .reporting import build_report
from .splits import create_folds
from .statistics import paired_statistics
from .provenance import freeze_source
from .refinement import fixed_interpolation, train_cultivar_ranking, train_remediation
from .training import train_phytocore
from .vlm import train_vlm
from .vlm_evaluation import evaluate_vlm
from .vqa import build_vqa
from .visualization import run_figures
from .parity import audit_stage42d_parity


def prepare(config: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    dataset_dir = config.artifacts / "dataset"
    index_path = dataset_dir / "dataset_index.csv"
    if force or not index_path.is_file():
        frame = prepare_images(load_metadata(config.raw_data), dataset_dir)
    else:
        frame = pd.read_csv(index_path, dtype={"ImageId": str})
    split_path = config.artifacts / "splits/folds.csv"
    if force or not split_path.is_file():
        folds = create_folds(frame, split_path, seed=int(config.values["protocol"]["split_seed"]))
    else:
        folds = pd.read_csv(split_path, dtype={"ImageId": str})
    merged = frame.merge(folds[["ImageId", "Fold"]], on="ImageId", validate="one_to_one")
    normalization_path = config.artifacts / "normalization" / f"fold_{config.fold}_band_stats.npz"
    if force or not normalization_path.is_file():
        normalization = fit_normalization(merged, config.fold, config.artifacts / "normalization")
    else:
        from .data import load_normalization
        normalization = load_normalization(config.artifacts / "normalization", config.fold)
    report = {
        "status": "PASS", "stage": "prepare", "scientific_result": False,
        "images": len(frame), "fold": config.fold,
        "train_images": int((merged["Fold"] != config.fold).sum()),
        "outer_images": int((merged["Fold"] == config.fold).sum()),
        "normalization_bands": int(len(normalization["mean"])),
    }
    write_json(config.run_artifacts / "prepare_report.json", report)
    return report


def checkpoint_path(config: ExperimentConfig, smoke: bool) -> Path:
    return config.run_artifacts / "phytocore" / ("smoke.pt" if smoke else "phytocore_final.pt")


def bundle_path(config: ExperimentConfig) -> Path:
    """Publication/runtime bundle after the frozen Stage-42 contract recipe."""
    return config.run_artifacts / "vlm" / "contract" / "bundle"


def fixed_bundle_path(config: ExperimentConfig) -> Path:
    return config.run_artifacts / "vlm" / "final" / "bundle"


def matched_bundle_path(config: ExperimentConfig) -> Path:
    return config.run_artifacts / "vlm" / "matched" / "bundle"


def remediation_bundle_path(config: ExperimentConfig) -> Path:
    return config.run_artifacts / "vlm" / "remediation" / "bundle"


def ranking_bundle_path(config: ExperimentConfig) -> Path:
    return config.run_artifacts / "vlm" / "ranking" / "bundle"


def _completed(config: ExperimentConfig, stage: str, smoke: bool) -> dict[str, Any] | None:
    paths = {
        "prepare": config.run_artifacts / "prepare_report.json",
        "phytocore": config.run_artifacts / "phytocore/training_report.json",
        "phytocore-evaluate": config.run_results / "phytocore/evaluation_report.json",
        "evidence-alignment": config.run_artifacts / "evidence/alignment_report.json",
        "evidence-outer": config.run_artifacts / "evidence/outer_report.json",
        "vqa": config.run_artifacts / "vqa/manifest.json",
        "vlm": config.run_artifacts / "vlm/matched/training_report.json",
        "remediation": config.run_artifacts / "vlm/remediation/training_report.json",
        "cultivar-ranking": config.run_artifacts / "vlm/ranking/training_report.json",
        "fixed-interpolation": config.run_artifacts / "vlm/final/interpolation_report.json",
        "contract-optimization": config.run_artifacts / "vlm/contract/training_report.json",
        "vlm-evaluate": config.run_results / "vlm/full/evaluation_report.json",
        "ablation": config.run_results / "ablations.json",
        "statistics": config.run_results / "statistics.json",
        "report": config.run_reports / "report.json",
        "figures": config.run_reports / "figures/figure_manifest.json",
        "stage42d-parity-check": config.run_artifacts / "stage42d_parity_report.json",
    }
    path = paths.get(stage)
    if path is None or not path.is_file():
        return None
    report = read_json(path)
    if report.get("status") not in {"PASS", "BOUND"}:
        return None
    if stage == "phytocore" and bool(report.get("metadata", {}).get("smoke")) != smoke:
        return None
    if stage in {"phytocore-evaluate", "vlm-evaluate", "ablation", "statistics"} and bool(report.get("scientific_result")) != (not smoke):
        return None
    if stage in {"vlm", "remediation", "cultivar-ranking", "fixed-interpolation", "contract-optimization"} and bool(report.get("smoke")) != smoke:
        return None
    expected_images = (
        (2 if stage == "evidence-outer" else 5)
        if smoke else
        (318 if stage == "evidence-outer" else int(config.values["training"]["vlm"]["alignment_images"]))
    )
    if stage in {"evidence-alignment", "evidence-outer", "vqa"} and int(report.get("images", -1)) != expected_images:
        return None
    return report


def run_stage(config: ExperimentConfig, stage: str, *, smoke: bool = False, resume: bool = False, force: bool = False) -> dict[str, Any]:
    if stage not in {"aggregate", "campaign-figures", "source-freeze"}:
        bind_or_verify(config, create=(stage in {"prepare", "all"}), force=force)
    if resume and stage != "all":
        completed = _completed(config, stage, smoke)
        if completed is not None:
            return {**completed, "resume_action": "SKIPPED_VERIFIED"}
    checkpoint = checkpoint_path(config, smoke)
    bundle = bundle_path(config)
    if stage == "prepare":
        return prepare(config, force=force)
    if stage == "stage42d-parity-check":
        return audit_stage42d_parity(config)
    if stage == "phytocore":
        return train_phytocore(config, smoke=smoke, resume=resume)
    if stage == "phytocore-evaluate":
        return evaluate_phytocore(config, checkpoint, smoke=smoke)
    if stage == "evidence-alignment":
        return build_evidence_cache(config, checkpoint, partition="alignment", smoke=smoke)
    if stage == "evidence-outer":
        return build_evidence_cache(config, checkpoint, partition="outer", smoke=smoke)
    if stage == "vqa":
        return build_vqa(config, smoke=smoke)
    if stage == "vlm":
        return train_vlm(config, checkpoint, smoke=smoke, resume=resume)
    if stage == "remediation":
        return train_remediation(config, matched_bundle_path(config), smoke=smoke, resume=resume)
    if stage == "cultivar-ranking":
        return train_cultivar_ranking(config, remediation_bundle_path(config), smoke=smoke, resume=resume)
    if stage == "fixed-interpolation":
        return fixed_interpolation(config, remediation_bundle_path(config), ranking_bundle_path(config), smoke=smoke)
    if stage == "contract-optimization":
        return train_contract_optimization(config, fixed_bundle_path(config), smoke=smoke, resume=resume)
    if stage == "vlm-evaluate":
        return evaluate_vlm(config, bundle, smoke=smoke)
    if stage == "ablation":
        return run_ablations(config, bundle, smoke=smoke)
    if stage == "statistics":
        return paired_statistics(config, smoke=smoke)
    if stage == "report":
        return build_report(config)
    if stage == "aggregate":
        return aggregate_campaign(config, allow_incomplete=smoke)
    if stage == "figures":
        return run_figures(config, aggregate=False)
    if stage == "campaign-figures":
        return run_figures(config, aggregate=True)
    if stage == "source-freeze":
        return freeze_source(config)
    if stage == "all":
        stages = ("prepare", "stage42d-parity-check", "phytocore", "phytocore-evaluate", "evidence-alignment", "evidence-outer", "vqa", "vlm", "remediation", "cultivar-ranking", "fixed-interpolation", "contract-optimization", "vlm-evaluate", "ablation", "statistics", "report", "figures")
        return {name: run_stage(config, name, smoke=smoke, resume=resume, force=force) for name in stages}
    raise ValueError(f"Unknown stage: {stage}")
