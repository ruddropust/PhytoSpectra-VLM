"""Self-contained audit of the immutable historical Stage-42D recipe."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .config import ExperimentConfig
from .data import STAGE42D_AUGMENTATION
from .evidence import split_stage42d_alignment_images
from .io import write_json
from .model import PhytoCore, TargetConditionedGlobalLocalBridge, VisualTokenBridgeConfig


def audit_stage42d_parity(config: ExperimentConfig) -> dict:
    values = config.values
    phyto = values["training"]["phytocore"]
    vlm = values["training"]["vlm"]
    evidence = values["evidence"]
    bridge = TargetConditionedGlobalLocalBridge(PhytoCore(), VisualTokenBridgeConfig(use_roi_geometry=True))
    gates = {
        "recipe_name": values["protocol"].get("frozen_recipe") == "historical_stage42d_exact",
        "phytocore_parameters": PhytoCore().parameter_count() == 1_998_346,
        "bridge_parameters_with_roi_geometry": sum(parameter.numel() for parameter in bridge.parameters()) == 3_476_618,
        "phytocore_schedule": (
            int(phyto["epochs"]) == 50 and int(phyto["batch_size"]) == 16
            and float(phyto["learning_rate"]) == 3e-4
            and float(phyto["minimum_learning_rate"]) == 1e-6
            and float(phyto["weight_decay"]) == 1e-4
        ),
        "augmentation_contract": STAGE42D_AUGMENTATION == {
            "translation_probability": 0.5, "max_dx": 8, "max_dy": 2,
            "gain_probability": 0.5, "gain_low": 0.98, "gain_high": 1.02,
            "high_saturation_threshold": 0.05, "noise_probability": 0.5,
            "noise_sigma_max": 0.002, "smooth_spectral_jitter_probability": 0.35,
            "smooth_spectral_jitter_amp_max": 0.01,
            "smooth_spectral_jitter_sigma_bands": 8.0,
        },
        "alignment_population": int(vlm["alignment_images"]) == 144 and int(vlm["dev_images"]) == 16,
        "warmup_schedule": int(vlm["warmup_epochs"]) == 3 and int(vlm["warmup_steps_per_epoch"]) == 20 and int(vlm["warmup_accumulation"]) == 8,
        "matched_schedule": int(vlm["epochs"]) == 3 and int(vlm["gradient_accumulation_steps"]) == 16 and int(vlm["max_optimizer_steps"]) == 768,
        "language_budget": int(vlm["max_sequence_tokens"]) == 1024 and int(values["evaluation"]["max_new_tokens"]) == 384,
        "candidate_grounding": (
            evidence["method"] == "four_fixed_candidate_grouped_spatial_token_zeroing"
            and int(evidence["candidate_seed"]) == 403026
            and int(evidence["candidate_count"]) == 4
            and int(evidence["variant_batch_size"]) == 4
        ),
        "fixed_interpolation": float(values["model"]["interpolation_alpha"]) == 0.60,
    }
    split = None
    fold_path = config.artifacts / "splits/folds.csv"
    if fold_path.is_file():
        frame = pd.read_csv(fold_path, dtype={"ImageId": str})
        train, dev = split_stage42d_alignment_images(frame, fold=config.fold)
        split = {"split_seed": 402026 + config.fold, "train_image_ids": train, "dev_image_ids": dev}
        gates["alignment_split"] = len(train) == 128 and len(dev) == 16 and not (set(train) & set(dev))
    report = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "stage": "stage42d-parity-check",
        "scientific_result": False,
        "seed": config.seed,
        "fold": config.fold,
        "gates": gates,
        "alignment_split": split,
        "frozen_seeds": {
            "alignment_split": 402026 + config.fold,
            "candidate_layout": 403026 + config.fold,
            "reference_warmup": 402026 + 10 * (config.seed - 2026) + config.fold,
            "matched_alignment": 415026 + 10 * (config.seed - 2026) + config.fold,
            "remediation": 417026 + 10 * (config.seed - 2026) + config.fold,
            "cultivar_ranking": 418026 + 10 * (config.seed - 2026) + config.fold,
            "contract_optimization": 420026,
        },
        "bridge_config": asdict(VisualTokenBridgeConfig(use_roi_geometry=True)),
        "note": "This audit does not read or modify research_v2 or historical Stage-42D outputs.",
    }
    output = config.run_artifacts / "stage42d_parity_report.json"
    write_json(output, report)
    if report["status"] != "PASS":
        raise ValueError(f"Stage-42D parity gates failed: {[name for name, passed in gates.items() if not passed]}")
    return report
