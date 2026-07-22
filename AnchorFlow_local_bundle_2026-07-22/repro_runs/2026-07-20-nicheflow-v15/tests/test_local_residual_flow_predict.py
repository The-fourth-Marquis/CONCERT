#!/usr/bin/env python3
"""Task wrapper contract for local anchor plus source residual."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "2026-07-11-nicheflow-v2" / "src"))

from nicheflow_v15.local_residual_flow_predict import (
    predict_task_local_residual_flow,
)
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool


def test_local_residual_task_prediction_is_reference_only() -> None:
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(
        payload,
        reference_fraction=0.70,
        seed=20260711,
        split_type="random_stratified",
    )
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    settings = {
        "latent_dim": 4,
        "hidden_dim": 16,
        "epochs": 20,
        "integration_steps": 8,
        "integration_horizon": 0.25,
        "seed": 67,
    }
    original = predict_task_local_residual_flow(payload, split, **settings)
    changed = predict_task_local_residual_flow(altered, split, **settings)

    assert np.allclose(original.base_prediction, changed.base_prediction)
    assert np.allclose(original.prediction, changed.prediction)
    assert original.metadata["uses_evaluation_target"] is False
    assert original.metadata["query_self_exclusions"] == payload.query_expr.shape[0]


if __name__ == "__main__":
    test_local_residual_task_prediction_is_reference_only()
    print("ok")
