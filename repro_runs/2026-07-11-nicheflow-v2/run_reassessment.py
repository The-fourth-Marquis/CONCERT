#!/usr/bin/env python3
"""CLI for NicheFlow-v2 existing-method held-out reassessment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.artifacts import write_artifacts
from nicheflow_v2.config import ExperimentConfig
from nicheflow_v2.data import load_manifest
from nicheflow_v2.reassess import merge_fast_and_expensive, reassess_manifest


def _config(path: Path) -> ExperimentConfig:
    settings = json.loads(path.read_text())
    return replace(
        ExperimentConfig.default(),
        master_seed=int(settings["master_seed"]),
        reference_fractions=tuple(map(float, settings["reference_fractions"])),
        split_seeds=tuple(map(int, settings["split_seeds"])),
        split_types=tuple(map(str, settings["split_types"])),
        min_reference=int(settings["min_reference"]),
        min_evaluation=int(settings["min_evaluation"]),
        bootstrap_resamples=int(settings["bootstrap_resamples"]),
        spatialprop_k=int(settings.get("spatialprop_k", 8)),
        spatialprop_hidden_dim=int(settings.get("spatialprop_hidden_dim", 64)),
        spatialprop_latent_dim=int(settings.get("spatialprop_latent_dim", 32)),
        spatialprop_epochs=int(settings.get("spatialprop_epochs", 80)),
    )


def _merge(args: argparse.Namespace, cfg: ExperimentConfig) -> None:
    fast_dir = ROOT / "results" / "existing_method_reassessment"
    expensive_dir = args.expensive_output
    fast_metrics = pd.read_csv(fast_dir / "metrics.csv")
    fast_coverage = pd.read_csv(fast_dir / "coverage.csv")
    expensive_metrics = pd.read_csv(expensive_dir / "metrics.csv")
    expensive_coverage = pd.read_csv(expensive_dir / "coverage.csv")
    metrics, coverage = merge_fast_and_expensive(
        fast_metrics,
        fast_coverage,
        expensive_metrics,
        expensive_coverage,
    )
    splits = pd.read_csv(fast_dir / "task_manifest.csv")
    output = args.output or fast_dir
    write_artifacts(
        metrics,
        coverage,
        splits,
        output,
        {
            "claim_scope": "observed perturbation relocation with held-out target evaluation",
            "n_tasks_requested": int(splits["task_id"].nunique()),
            "include_expensive": True,
            "spatialprop_source": str(expensive_dir.relative_to(cfg.repo_root)),
        },
        config=cfg,
    )
    print(
        f"merged {len(metrics)} metric rows and {len(coverage)} coverage rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--include-expensive", action="store_true")
    parser.add_argument("--exclude-expensive", action="store_true")
    parser.add_argument("--only-expensive", action="store_true")
    parser.add_argument("--resume-cache", action="store_true")
    parser.add_argument("--merge-expensive", action="store_true")
    parser.add_argument(
        "--expensive-output",
        type=Path,
        default=ROOT / "results" / "spatialprop_reference_only",
    )
    args = parser.parse_args()
    cfg = _config(args.config)
    if args.merge_expensive:
        _merge(args, cfg)
        return
    if args.only_expensive and args.exclude_expensive:
        parser.error("--only-expensive conflicts with --exclude-expensive")
    include_expensive = bool(
        (args.include_expensive or args.only_expensive)
        and not args.exclude_expensive
    )
    manifest = load_manifest(cfg=cfg)
    if args.limit_tasks is not None:
        manifest = manifest.head(args.limit_tasks)
    output = args.output
    if output is None:
        output = (
            args.expensive_output
            if args.only_expensive
            else ROOT / "results" / "existing_method_reassessment"
        )
    metrics, coverage, splits = reassess_manifest(
        manifest,
        cfg,
        include_expensive=include_expensive,
        only_expensive=args.only_expensive,
    )
    write_artifacts(
        metrics,
        coverage,
        splits,
        output,
        {
            "claim_scope": "observed perturbation relocation with held-out target evaluation",
            "n_tasks_requested": int(manifest.shape[0]),
            "include_expensive": include_expensive,
            "only_expensive": bool(args.only_expensive),
            "resume_cache": True,
        },
        config=cfg,
    )
    print(f"wrote {len(metrics)} metric rows and {len(coverage)} coverage rows")


if __name__ == "__main__":
    main()
