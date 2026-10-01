"""Clean, reproducible PhytoSpectra-VLM research package."""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "PhytoCore", "PhytoSpectraVLM", "load_config"]
__version__ = "3.0.0"


def __getattr__(name: str):
    if name in {"PhytoCore", "PhytoSpectraVLM"}:
        from .model import PhytoCore, PhytoSpectraVLM
        return {"PhytoCore": PhytoCore, "PhytoSpectraVLM": PhytoSpectraVLM}[name]
    raise AttributeError(name)
