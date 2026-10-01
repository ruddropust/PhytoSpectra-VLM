"""Typed experiment configuration and directory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    raw_data: Path
    run_id: str
    seed: int
    fold: int
    values: dict[str, Any]

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / self.run_id

    @property
    def results(self) -> Path:
        return self.root / "results" / self.run_id

    @property
    def reports(self) -> Path:
        return self.root / "reports" / self.run_id

    @property
    def run_key(self) -> str:
        """Stable identity for one independent seed/fold execution."""
        return f"seed_{self.seed}_fold_{self.fold}"

    @property
    def run_artifacts(self) -> Path:
        return self.artifacts / "runs" / self.run_key

    @property
    def run_results(self) -> Path:
        return self.results / "runs" / self.run_key

    @property
    def run_reports(self) -> Path:
        return self.reports / "runs" / self.run_key

    def ensure_output_dirs(self) -> None:
        for path in (
            self.artifacts, self.results, self.reports,
            self.run_artifacts, self.run_results, self.run_reports,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(
    path: str | Path,
    *,
    root: str | Path,
    raw_data: str | Path,
    run_id: str,
    seed: int,
    fold: int,
) -> ExperimentConfig:
    path = Path(path).resolve()
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values["protocol"]["seeds"] != [2026, 2027, 2028]:
        raise ValueError("Frozen seed contract changed")
    if int(values["protocol"]["folds"]) != 5:
        raise ValueError("Frozen five-fold contract changed")
    if seed not in values["protocol"]["seeds"] or not 0 <= fold < 5:
        raise ValueError("Seed/fold is outside the frozen protocol")
    config = ExperimentConfig(
        root=Path(root).resolve(),
        raw_data=Path(raw_data).resolve(),
        run_id=str(run_id),
        seed=int(seed),
        fold=int(fold),
        values=values,
    )
    config.ensure_output_dirs()
    return config
