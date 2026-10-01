"""Assemble machine-readable and professor/paper-readable run reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .io import read_json, utc_now, write_json


def build_report(config: ExperimentConfig) -> dict[str, Any]:
    run = config.run_key
    sources = {
        "phytocore": config.run_results / "phytocore/evaluation_report.json",
        "vlm": config.run_results / "vlm/full/evaluation_report.json",
        "ablations": config.run_results / "ablations.json",
        "statistics": config.run_results / "statistics.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot report; missing outputs: {missing}")
    report = {
        "status": "PASS", "stage": "report", "created_at_utc": utc_now(),
        "run_id": config.run_id, "seed": config.seed, "fold": config.fold,
        "scientific_scope": "internal image-wise outer fold",
        "results": {name: read_json(path) for name, path in sources.items()},
    }
    output_dir = config.run_reports
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    p = report["results"]["phytocore"]["metrics"]
    v = report["results"]["vlm"]["metrics"]
    lines = [
        f"# PhytoSpectra-VLM — seed {config.seed}, fold {config.fold}", "",
        "## PhytoCore", "",
        "| Target | Metric | Value |", "|---|---:|---:|",
    ]
    for target, values in p["regression"].items():
        lines.append(f"| {target} | MAE | {values['mae']:.6f} |")
    for target, values in p["classification"].items():
        lines.append(f"| {target} | Accuracy | {values['accuracy']:.4f} |")
    summary = "—" if v["summary_acceptance"] is None else f"{v['summary_acceptance']:.4f}"
    class_agreement = "—" if v["raw_class_agreement_with_phytocore"] is None else f"{v['raw_class_agreement_with_phytocore']:.4f}"
    lines.extend(["", "## VLM contract", "", "| Metric | Value |", "|---|---:|", f"| Raw reference acceptance | {v['raw_reference_acceptance']:.4f} |", f"| Raw VLM–PhytoCore class agreement | {class_agreement} |", f"| Summary acceptance | {summary} |", f"| Verified evidence consistency | {v['verified_evidence_consistency']:.4f} |", "", "> Evidence consistency is not phenotype ground-truth accuracy."])
    layers = report["results"]["vlm"].get("generation_layers", {})
    lines.extend(["", "## Frozen decoding layers", "", "| Layer | Acceptance | Class agreement | Summary acceptance |", "|---|---:|---:|---:|"])
    for name in ("raw_greedy", "constrained_top1", "decomposed_summary", "verifier_reranked_nbest"):
        values = layers.get(name)
        if not isinstance(values, dict):
            continue
        acceptance = values.get("raw_reference_acceptance")
        agreement = values.get("raw_class_agreement_with_phytocore")
        summary_value = values.get("summary_acceptance")
        lines.append(
            f"| {name} | {'—' if acceptance is None else f'{acceptance:.4f}'} | "
            f"{'—' if agreement is None else f'{agreement:.4f}'} | "
            f"{'—' if summary_value is None else f'{summary_value:.4f}'} |"
        )
    lines.extend(["", "## Frozen component ablations", "", "| Ablation | Raw acceptance | Raw class agreement |", "|---|---:|---:|"])
    for mode, metrics in report["results"]["ablations"]["results"].items():
        class_value = metrics.get("raw_class_agreement_with_phytocore")
        class_text = "—" if class_value is None else f"{class_value:.4f}"
        lines.append(f"| {mode} | {metrics['raw_reference_acceptance']:.4f} | {class_text} |")
    lines.extend(["", "## Paired image-cluster tests", "", "| Ablation | Primary raw p | Holm-adjusted p |", "|---|---:|---:|"])
    stats = report["results"]["statistics"]["multiple_comparison_control"]
    for mode in stats["raw_p"]:
        lines.append(f"| {mode} | {stats['raw_p'][mode]:.6g} | {stats['adjusted_p'][mode]:.6g} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
