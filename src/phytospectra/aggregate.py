"""Aggregate the locked 3-seed x 5-fold experiment without model selection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import REGRESSION_TARGETS
from .io import read_json, utc_now, write_json


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "runs": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_campaign(config: ExperimentConfig, *, allow_incomplete: bool = False) -> dict[str, Any]:
    seeds = [int(value) for value in config.values["protocol"]["seeds"]]
    folds = list(range(int(config.values["protocol"]["folds"])))
    expected = [(seed, fold) for seed in seeds for fold in folds]
    found: list[tuple[int, int, dict[str, Any]]] = []
    missing: list[str] = []
    for seed, fold in expected:
        path = config.reports / "runs" / f"seed_{seed}_fold_{fold}" / "report.json"
        if path.is_file():
            found.append((seed, fold, read_json(path)))
        else:
            missing.append(str(path))
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Campaign is incomplete ({len(missing)}/15 missing); first={missing[0]}")
    if not found:
        raise FileNotFoundError("No completed seed/fold reports were found")

    collected: dict[str, list[float]] = defaultdict(list)
    run_rows = []
    for seed, fold, report in found:
        phytocore = report["results"]["phytocore"]["metrics"]
        vlm = report["results"]["vlm"]["metrics"]
        row: dict[str, Any] = {"seed": seed, "fold": fold, "images": phytocore["images"]}
        for target in REGRESSION_TARGETS:
            for metric in ("mae", "rmse", "r2"):
                value = phytocore["regression"][target][metric]
                if value is not None:
                    key = f"phytocore.{target}.{metric}"
                    collected[key].append(float(value))
                    row[key] = float(value)
        for target in ("Fertilizer", "Cultivar"):
            for metric in ("accuracy", "macro_f1"):
                value = float(phytocore["classification"][target][metric])
                key = f"phytocore.{target}.{metric}"
                collected[key].append(value)
                row[key] = value
        for metric in ("raw_reference_acceptance", "raw_class_agreement_with_phytocore", "fallback_rate", "summary_acceptance", "verified_evidence_consistency"):
            if vlm[metric] is None:
                continue
            value = float(vlm[metric])
            key = f"vlm.{metric}"
            collected[key].append(value)
            row[key] = value
        layers = report["results"]["vlm"].get("generation_layers", {})
        for layer in ("constrained_top1", "decomposed_summary", "verifier_reranked_nbest"):
            layer_values = layers.get(layer, {})
            for metric in ("raw_reference_acceptance", "raw_class_agreement_with_phytocore", "summary_acceptance"):
                if layer_values.get(metric) is None:
                    continue
                value = float(layer_values[metric])
                key = f"vlm.{layer}.{metric}"
                collected[key].append(value)
                row[key] = value
        run_rows.append(row)

    report = {
        "status": "PASS" if not missing else "INCOMPLETE",
        "stage": "aggregate",
        "created_at_utc": utc_now(),
        "run_id": config.run_id,
        "protocol": "3-seed x 5-fold image-wise internal cross-validation",
        "expected_runs": len(expected),
        "completed_runs": len(found),
        "missing_runs": len(missing),
        "scientific_result": not missing,
        "scope_warning": "Internal image-wise CV; not external-site, plot-disjoint, or new-device validation.",
        "metrics_across_runs": {key: _summary(values) for key, values in sorted(collected.items())},
        "runs": run_rows,
    }
    output = config.reports / "campaign"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate_report.json", report)
    return report
