"""Create a tamper-evident source/configuration manifest for a public release."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .io import utc_now, write_json


INCLUDED_ROOTS = ("src", "configs", "scripts", "notebooks", "tests", "chatbot_app/backend", "chatbot_app/frontend/src")
INCLUDED_FILES = (
    "README.md", "pyproject.toml", ".gitignore", "LICENSE", "CITATION.cff", "DATASET.md",
    "chatbot_app/backend/requirements.txt", "chatbot_app/frontend/package.json",
    "chatbot_app/frontend/package-lock.json", "chatbot_app/frontend/vite.config.ts",
    "chatbot_app/frontend/vitest.config.ts", "chatbot_app/frontend/tsconfig.json",
    "chatbot_app/frontend/tsconfig.node.json", "chatbot_app/frontend/index.html",
    "chatbot_app/frontend/.env.example",
    "chatbot_app/frontend/tailwind.config.js", "chatbot_app/frontend/postcss.config.js",
    "chatbot_app/run_backend.ps1", "chatbot_app/run_frontend.ps1",
)


def _record(root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def freeze_source(config: ExperimentConfig) -> dict[str, Any]:
    files: list[Path] = []
    for relative in INCLUDED_ROOTS:
        directory = config.root / relative
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.extend(config.root / name for name in INCLUDED_FILES if (config.root / name).is_file())
    records = [_record(config.root, path) for path in sorted(set(files))]
    dependencies = sorted(f"{distribution.metadata['Name']}=={distribution.version}" for distribution in importlib.metadata.distributions() if distribution.metadata.get("Name"))
    output = config.root / "release" / config.run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment-lock.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")
    manifest = {
        "status": "FROZEN", "stage": "source-freeze", "created_at_utc": utc_now(),
        "scientific_result": False, "run_id": config.run_id,
        "architecture_contract": {"phytocore_parameters": 1_998_346, "spectral_tokens": 50, "spatial_tokens": 132, "visual_prefix": [16, 1536]},
        "protocol_contract": {"seeds": [2026, 2027, 2028], "folds": 5, "split_unit": "image"},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()},
        "files": records,
        "environment_lock": _record(config.root, output / "environment-lock.txt"),
        "exclusions": ["raw dataset", "generated artifacts", "generated results", "generated reports", "node_modules", "frontend dist"],
    }
    write_json(output / "source_manifest.json", manifest)
    return manifest
