"""Generate publication/professor-ready figures from newly produced reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .io import read_json, write_json


def _save(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def architecture_figure(output: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.axis("off")
    boxes = [
        (0.2, 2.1, 2.0, 1.6, "HSI cube\n48×352×204", "#DCEEFF"),
        (2.8, 3.4, 2.2, 1.4, "Spectral encoder\n50 tokens", "#C8E6C9"),
        (2.8, 1.1, 2.2, 1.4, "Spatial encoder\n132 tokens", "#C8E6C9"),
        (5.6, 2.1, 2.2, 1.6, "PhytoCore fusion\n182×128 memory", "#FFF3BF"),
        (8.4, 3.4, 2.2, 1.4, "5 phenotype heads\n3 reg. + 2 class.", "#FFD8A8"),
        (8.4, 1.1, 2.2, 1.4, "Token-zeroing ROI\nglobal + local", "#FFD8A8"),
        (11.2, 2.1, 1.8, 1.6, "Question/target\nconditioned bridge", "#E5DBFF"),
        (13.5, 2.1, 1.3, 1.6, "Qwen +\nLoRA", "#D0EBFF"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#263238", linewidth=1.4))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10, weight="bold")
    arrows = [((2.2,2.9),(2.8,4.1)),((2.2,2.9),(2.8,1.8)),((5.0,4.1),(5.6,3.1)),((5.0,1.8),(5.6,2.7)),((7.8,3.1),(8.4,4.1)),((7.8,2.7),(8.4,1.8)),((10.6,4.1),(11.2,3.2)),((10.6,1.8),(11.2,2.6)),((13.0,2.9),(13.5,2.9))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#37474F"))
    ax.text(7.5, 5.55, "PhytoSpectra-VLM frozen architecture", ha="center", fontsize=16, weight="bold")
    ax.text(7.5, .25, "Matched alignment → remediation → Cultivar ranking → α=0.60 → contract optimization → verified references", ha="center", fontsize=9.5, color="#455A64")
    return _save(fig, output / "architecture")


def run_figures(config: ExperimentConfig, *, aggregate: bool = False) -> dict[str, Any]:
    output = config.reports / ("campaign/figures" if aggregate else f"runs/{config.run_key}/figures")
    files = architecture_figure(output)
    if aggregate:
        report = read_json(config.reports / "campaign/aggregate_report.json")
        metrics = report["metrics_across_runs"]
        keys = [key for key in metrics if key.endswith("accuracy") or key.startswith("vlm.")]
        labels = [key.replace("phytocore.", "").replace("vlm.", "") for key in keys]
        means = [metrics[key]["mean"] for key in keys]
        errors = [metrics[key]["standard_deviation"] for key in keys]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(keys)); ax.bar(x, means, yerr=errors, capsize=4, color="#4C78A8")
        ax.set_xticks(x, labels, rotation=30, ha="right"); ax.set_ylim(0, 1.05); ax.set_ylabel("Mean across 15 runs")
        ax.set_title("Classification and VLM contract metrics (mean ± SD)"); ax.grid(axis="y", alpha=.25)
        files.extend(_save(fig, output / "campaign_metrics"))
    else:
        report = read_json(config.run_reports / "report.json")
        phytocore = report["results"]["phytocore"]
        predictions_path = config.root / phytocore["predictions"]["path"]
        predictions = pd.read_csv(predictions_path)
        regression = list(phytocore["metrics"]["regression"])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, target in zip(axes, regression):
            true = predictions[f"true_{target}"]; pred = predictions[f"pred_{target}"]
            ax.scatter(true, pred, s=18, alpha=.7, color="#2F6B9A")
            low, high = min(true.min(), pred.min()), max(true.max(), pred.max())
            ax.plot([low, high], [low, high], "--", color="#D1495B", lw=1)
            ax.set(title=target, xlabel="Observed", ylabel="Predicted")
            ax.grid(alpha=.2)
        fig.suptitle("PhytoCore outer-fold regression")
        files.extend(_save(fig, output / "regression_observed_vs_predicted"))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
        for ax, target in zip(axes, ("Fertilizer", "Cultivar")):
            values = phytocore["metrics"]["classification"][target]
            matrix = np.asarray(values["confusion_matrix"])
            image = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks(range(len(values["class_order"])), values["class_order"], rotation=30, ha="right")
            ax.set_yticks(range(len(values["class_order"])), values["class_order"])
            ax.set(title=target, xlabel="Predicted", ylabel="Observed")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]): ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
            fig.colorbar(image, ax=ax, fraction=.046)
        fig.suptitle("Outer-fold classification confusion matrices")
        files.extend(_save(fig, output / "classification_confusion_matrices"))

        vlm = report["results"]["vlm"]["metrics"]
        labels = ["Raw acceptance", "Raw class agreement", "Summary acceptance", "Verified consistency"]
        values = [vlm["raw_reference_acceptance"], vlm["raw_class_agreement_with_phytocore"], vlm["summary_acceptance"], vlm["verified_evidence_consistency"]]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        bars = ax.bar(labels, values, color=["#4C78A8", "#F2A541", "#59A14F"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Rate"); ax.set_title("VLM raw and verified contract outcomes")
        for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value+.02, f"{value:.3f}", ha="center")
        files.extend(_save(fig, output / "vlm_contract_metrics"))
    manifest = {"status": "PASS", "stage": "figures", "files": files, "source": "newly generated v3 reports"}
    write_json(output / "figure_manifest.json", manifest)
    return manifest
