#!/usr/bin/env python3
"""Tests for NicheFlow-v2 evidence artifacts and statistical summaries."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.artifacts import write_artifacts
from nicheflow_v2.config import ExperimentConfig
from nicheflow_v2.report import clustered_bootstrap


def test_required_artifacts_and_ranking_eligibility(tmp_path: Path) -> None:
    manifest = pd.DataFrame([{"task_id": "t1", "split_seed": 20260711, "split_type": "random_stratified"}])
    metrics = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "method": "source_identity_l0",
                "access_level": "L0",
                "reference_fraction": 0.70,
                "split_seed": 20260711,
                "split_type": "random_stratified",
                "calibration": False,
                "uses_target_reference": False,
                "heldout_edistance": 2.0,
                "eligible_for_primary": True,
            },
            {
                "task_id": "t1",
                "method": "fixed_ineligible",
                "access_level": "fullpool",
                "reference_fraction": 0.70,
                "split_seed": 20260711,
                "split_type": "random_stratified",
                "calibration": True,
                "uses_target_reference": True,
                "heldout_edistance": 0.1,
                "eligible_for_primary": False,
            },
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "model_family": "a2_baseline",
                "method": "source_identity_l0",
                "eligibility_status": "exact",
                "reason": "",
                "eligible_for_primary": True,
            }
        ]
    )
    metadata = {"claim_scope": "observed perturbation relocation", "smoke": True}
    cfg = replace(ExperimentConfig.default(), run_root=tmp_path)
    outputs = write_artifacts(metrics, coverage, manifest, tmp_path, metadata, config=cfg)
    required = {
        "task_manifest.csv",
        "metrics.csv",
        "method_ranking.csv",
        "coverage.csv",
        "metadata.json",
        "README.md",
    }
    assert required.issubset({path.name for path in outputs})
    ranking = pd.read_csv(tmp_path / "method_ranking.csv")
    assert ranking["eligible_for_primary"].all()
    assert "fixed_ineligible" not in set(ranking["method"])
    assert "held-out" in (tmp_path / "README.md").read_text().lower()
    written_metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert isinstance(written_metadata["git_dirty"], bool)


def test_clustered_bootstrap_is_deterministic() -> None:
    comparison = pd.DataFrame(
        {
            "task_id": ["a", "a", "b", "b"],
            "delta_edistance": [-2.0, -1.0, 1.0, -0.5],
        }
    )
    one = clustered_bootstrap(
        comparison,
        value_col="delta_edistance",
        seed=20260711,
        n_resamples=1000,
    )
    two = clustered_bootstrap(
        comparison,
        value_col="delta_edistance",
        seed=20260711,
        n_resamples=1000,
    )
    assert one == two
    assert one["n_clusters"] == 2
    assert one["ci_low"] <= one["mean"] <= one["ci_high"]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_required_artifacts_and_ranking_eligibility(Path(tmp))
    test_clustered_bootstrap_is_deterministic()
    print("ok")
