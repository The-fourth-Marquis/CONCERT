#!/usr/bin/env python3
"""Independent artifact verifier for NicheFlow-v2 runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "task_manifest.csv", "metrics.csv", "method_ranking.csv",
    "coverage.csv", "metadata.json", "README.md",
}


def verify_directory(path: Path) -> None:
    missing = sorted(name for name in REQUIRED if not (path / name).exists())
    if missing:
        raise AssertionError(f"Missing artifacts in {path}: {missing}")
    metrics = pd.read_csv(path / "metrics.csv")
    ranking = pd.read_csv(path / "method_ranking.csv")
    coverage = pd.read_csv(path / "coverage.csv")
    if metrics.empty or coverage.empty:
        raise AssertionError(f"Empty metrics or coverage in {path}")
    if not ranking.empty and not ranking["eligible_for_primary"].astype(bool).all():
        raise AssertionError("Primary ranking contains ineligible rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("reassessment-fast", "residual-gating", "complete"),
        required=True,
    )
    args = parser.parse_args()
    reassessment = ROOT / "results" / "existing_method_reassessment"
    residual = ROOT / "results" / "residual_gating"
    if args.mode in {"reassessment-fast", "complete"}:
        verify_directory(reassessment)
    if args.mode in {"residual-gating", "complete"}:
        verify_directory(residual)
    if args.mode == "complete" and not (ROOT / "reports" / "NicheFlow_v2_report.md").exists():
        raise AssertionError("Missing final NicheFlow-v2 report")
    print("ok")


if __name__ == "__main__":
    main()
