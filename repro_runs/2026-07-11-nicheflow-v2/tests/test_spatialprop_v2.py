#!/usr/bin/env python3
"""Tests for the reference-only SpatialProp-style comparator."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.spatialprop import predict_spatialprop_reference_only
from nicheflow_v2.splits import split_target_pool


def test_spatialprop_prediction_is_reference_only_and_query_aligned() -> None:
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
    first = predict_spatialprop_reference_only(
        payload, split, epochs=2, hidden_dim=8, latent_dim=4, seed=7
    )
    second = predict_spatialprop_reference_only(
        altered, split, epochs=2, hidden_dim=8, latent_dim=4, seed=7
    )
    assert first.prediction.shape == payload.query_expr.shape
    assert first.prediction.dtype == np.float32
    assert np.isfinite(first.prediction).all()
    assert np.allclose(first.prediction, second.prediction)
    assert first.access_level == "L2"
    assert first.calibration is True
    assert first.backend == "torch_spatialprop_style"


if __name__ == "__main__":
    test_spatialprop_prediction_is_reference_only_and_query_aligned()
    print("ok")
