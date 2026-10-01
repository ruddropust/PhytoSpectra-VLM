"""Bind every run to exact source and configuration bytes before training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .io import read_json, utc_now, write_json


def _source_files(root: Path) -> list[Path]:
    files = list((root / "src").rglob("*.py")) + list((root / "configs").rglob("*.yaml")) + list((root / "scripts").glob("*.ps1"))
    return sorted(path for path in files if "__pycache__" not in path.parts)


def source_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    records, combined = [], hashlib.sha256()
    for path in _source_files(root):
        data = path.read_bytes(); relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        records.append({"path": relative, "sha256": digest, "bytes": len(data)})
        combined.update(relative.encode()); combined.update(bytes.fromhex(digest))
    return combined.hexdigest(), records


def bind_or_verify(config: ExperimentConfig, *, create: bool, force: bool = False) -> dict[str, Any]:
    path = config.artifacts / "run_binding.json"
    digest, records = source_digest(config.root)
    identity = {"run_id": config.run_id, "source_digest": digest, "protocol_seeds": [2026, 2027, 2028], "protocol_folds": 5}
    if path.is_file():
        existing = read_json(path)
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Run binding mismatch: source/config changed. Start a new RunId; do not mix outputs.")
        return existing
    if not create:
        raise FileNotFoundError("Run binding is missing. Execute the prepare stage first.")
    binding = {"status": "BOUND", "created_at_utc": utc_now(), **identity, "files": records}
    write_json(path, binding)
    return binding
