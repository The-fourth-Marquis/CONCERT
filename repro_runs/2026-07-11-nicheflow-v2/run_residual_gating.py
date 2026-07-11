#!/usr/bin/env python3
"""CLI for the NicheFlow-v2 residual/gating ablation benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.config import ExperimentConfig
from nicheflow_v2.data import load_manifest
from nicheflow_v2.residual_run import run_residual_gating


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "residual_gating")
    parser.add_argument("--limit-tasks", type=int)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    cfg = replace(
        ExperimentConfig.default(),
        master_seed=int(settings["master_seed"]),
        reference_fractions=tuple(map(float, settings["reference_fractions"])),
        split_seeds=tuple(map(int, settings["split_seeds"])),
        split_types=tuple(map(str, settings["split_types"])),
        min_reference=int(settings["min_reference"]),
        min_evaluation=int(settings["min_evaluation"]),
        bootstrap_resamples=int(settings["bootstrap_resamples"]),
    )
    manifest = load_manifest(cfg=cfg)
    if args.limit_tasks is not None:
        manifest = manifest.head(args.limit_tasks)
    result = run_residual_gating(manifest=manifest, output_dir=args.output, config=cfg)
    print(f"wrote {len(result.metrics)} metric rows and {len(result.coverage)} coverage rows")


if __name__ == "__main__":
    main()
