"""Command-line interface for new-from-scratch experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_stage


STAGES = ("prepare", "stage42d-parity-check", "phytocore", "phytocore-evaluate", "evidence-alignment", "evidence-outer", "vqa", "vlm", "remediation", "cultivar-ranking", "fixed-interpolation", "contract-optimization", "vlm-evaluate", "ablation", "statistics", "report", "figures", "aggregate", "campaign-figures", "source-freeze", "all")


def main() -> int:
    package_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, default=package_root / "configs/experiment.yaml")
    parser.add_argument("--root", type=Path, default=package_root)
    parser.add_argument("--raw-data", type=Path, default=package_root / "data/raw")
    parser.add_argument("--run-id", default="final_3x5_v1")
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), default=2026)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, root=args.root, raw_data=args.raw_data, run_id=args.run_id, seed=args.seed, fold=args.fold)
    result = run_stage(config, args.stage, smoke=args.smoke, resume=args.resume, force=args.force)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
