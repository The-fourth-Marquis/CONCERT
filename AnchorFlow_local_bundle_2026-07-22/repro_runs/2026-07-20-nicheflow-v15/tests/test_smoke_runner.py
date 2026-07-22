#!/usr/bin/env python3
"""Smoke-run artifact contract for NicheFlow-v1.5."""

from pathlib import Path
import json
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_smoke import run_smoke


def test_smoke_run_writes_comparable_reference_only_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = run_smoke(
            output_dir=Path(tmp),
            task_index=0,
            reference_fraction=0.70,
            seed=20260711,
            split_type="random_stratified",
            latent_dim=4,
            hidden_dim=8,
        )
        summary = json.loads(output.read_text())

    assert summary["method"] == "nonlinear_monge_map"
    assert summary["uses_evaluation_target"] is False
    assert summary["leakage_mutation_check"] is True
    assert summary["n_reference"] > 0
    assert summary["n_evaluation"] > 0
    assert summary["heldout_edistance"] >= 0.0
    assert summary["fixed_80_20_heldout_edistance"] >= 0.0


if __name__ == "__main__":
    test_smoke_run_writes_comparable_reference_only_metrics()
    print("ok")
