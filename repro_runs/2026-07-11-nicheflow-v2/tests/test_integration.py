#!/usr/bin/env python3
"""End-to-end smoke test for residual/gating evaluation artifacts."""

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.config import ExperimentConfig
from nicheflow_v2.data import load_manifest
from nicheflow_v2.residual_run import run_residual_gating


def test_one_task_end_to_end(tmp_path: Path) -> None:
    cfg = replace(
        ExperimentConfig.default(),
        run_root=tmp_path,
        reference_fractions=(0.70,),
        split_seeds=(20260711,),
        split_types=("random_stratified",),
        bootstrap_resamples=100,
    )
    result = run_residual_gating(
        manifest=load_manifest().head(1),
        output_dir=tmp_path,
        config=cfg,
    )
    assert set(result.metrics["method"]) >= {
        "anchor_only",
        "residual_prediction_only",
        "fixed_80_20",
        "fixed_80_20_calibrated",
        "nicheflow_v2_gated",
        "nicheflow_v2_gated_calibrated",
    }
    assert result.metrics["heldout_edistance"].notna().all()
    assert result.metrics["leakage_mutation_check"].all()
    assert not result.gate_summary.empty
    assert (tmp_path / "gate_summary.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "coverage.csv").exists()
    assert (tmp_path / "README.md").exists()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_one_task_end_to_end(Path(tmp))
    print("ok")
