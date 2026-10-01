"""Frozen component ablations; no model parameters are retrained."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .io import read_json, write_json
from .vlm_evaluation import evaluate_vlm


ABLATIONS = ("global_only", "local_only", "no_question_conditioning")


def run_ablations(config: ExperimentConfig, bundle: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    results = {}
    for mode in ABLATIONS:
        completed = config.run_results / "vlm" / mode / "evaluation_report.json"
        if completed.is_file() and read_json(completed).get("status") == "PASS":
            results[mode] = read_json(completed)
        else:
            results[mode] = evaluate_vlm(config, bundle, ablation_mode=mode, smoke=smoke)
    report = {
        "status": "PASS",
        "stage": "ablation",
        "scientific_result": not smoke,
        "seed": config.seed,
        "fold": config.fold,
        "modes": list(ABLATIONS),
        "parameter_updates": 0,
        "results": {mode: value["metrics"] for mode, value in results.items()},
    }
    output = config.run_results / "ablations.json"
    write_json(output, report)
    return report
