"""Guarded adapter for the locked research_v2 Stage-13 deployment bundle."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch


RESEARCH_ROOT = Path(__file__).resolve().parents[4]


class ResearchV2VLMManager:
    """Verify eagerly, load model weights lazily, and reject smoke activation."""

    def __init__(
        self,
        *,
        bundle_dir: str | Path,
        device: torch.device,
        require_scientific: bool = True,
        eager_load: bool = False,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        self.device = device
        self.require_scientific = bool(require_scientific)
        self.runtime: Optional[Any] = None
        self.manifest = self._verify()
        if eager_load:
            self.load()

    @staticmethod
    def _ensure_research_import() -> None:
        root = str(RESEARCH_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _verify(self) -> dict[str, Any]:
        self._ensure_research_import()
        from research_v2.deployment.verification import verify_deployment_bundle

        return verify_deployment_bundle(
            self.bundle_dir,
            require_scientific=self.require_scientific,
        )

    def load(self) -> None:
        if self.runtime is not None:
            return
        self._ensure_research_import()
        from research_v2.deployment.runtime import DeployedPhytoSpectraRuntime

        self.runtime = DeployedPhytoSpectraRuntime.load(
            self.bundle_dir,
            device=self.device,
            require_scientific=self.require_scientific,
        )

    def infer(
        self,
        *,
        cube_unit_hwc: np.ndarray,
        leaf_mask: np.ndarray,
        question: str,
        targets: Sequence[str],
        image_id: str,
        visual_output_dir: str | Path,
    ) -> dict[str, Any]:
        self.load()
        assert self.runtime is not None
        return self.runtime.infer(
            cube_unit_hwc=cube_unit_hwc,
            leaf_mask=leaf_mask,
            question=question,
            targets=targets,
            image_id=image_id,
            visual_output_dir=visual_output_dir,
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "verified": True,
            "loaded": self.runtime is not None,
            "scientific_result": bool(self.manifest["scientific_result"]),
            "activation_allowed": bool(self.manifest["activation_allowed"]),
            "deployment_status": str(self.manifest["deployment_status"]),
            "bundle_dir": str(self.bundle_dir),
            "outer_fold": int(self.manifest["outer_fold"]),
            "model_seed": int(self.manifest["model_seed"]),
        }
