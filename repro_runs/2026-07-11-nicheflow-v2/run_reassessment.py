#!/usr/bin/env python3
"""CLI for NicheFlow-v2 existing-method held-out reassessment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.artifacts import write_artifacts
from nicheflow_v2.config import ExperimentConfig
from nicheflow_v2.data import load_manifest
from nicheflow_v2.reassess import reassess_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "existing_method_reassessment")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--include-expensive", action="store_true")
    parser.add_argument("--exclude-expensive", action="store_true")
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
    metrics, coverage, splits = reassess_manifest(
        manifest,
        cfg,
        include_expensive=bool(args.include_expensive and not args.exclude_expensive),
    )
    write_artifacts(
        metrics,
        coverage,
        splits,
        args.output,
        {
            "claim_scope": "observed perturbation relocation with held-out target evaluation",
            "n_tasks_requested": int(manifest.shape[0]),
            "include_expensive": bool(args.include_expensive and not args.exclude_expensive),
        },
        config=cfg,
    )
    print(f"wrote {len(metrics)} metric rows and {len(coverage)} coverage rows")


if __name__ == "__main__":
    main()
