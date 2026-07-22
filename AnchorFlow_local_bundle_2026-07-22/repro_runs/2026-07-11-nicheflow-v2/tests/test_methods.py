#!/usr/bin/env python3
"""Tests for explicit residual matching and reliability gating."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.methods import context_matched_delta, predict_method, residual_prediction
from nicheflow_v2.splits import split_target_pool


def _case():
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(
        payload,
        reference_fraction=0.70,
        seed=20260711,
        split_type="random_stratified",
    )
    return payload, split


def test_residual_contract_is_explicit() -> None:
    payload, split = _case()
    delta = context_matched_delta(payload, split)
    prediction = residual_prediction(payload, delta)
    assert delta.shape == payload.query_expr.shape
    assert np.allclose(prediction, payload.query_expr + delta)
    assert not np.allclose(delta, prediction)


def test_evaluation_expression_cannot_change_predictions() -> None:
    payload, split = _case()
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    for method in (
        "anchor_only",
        "residual_prediction_only",
        "fixed_80_20",
        "nicheflow_v2_gated",
        "nicheflow_v2_gated_calibrated",
    ):
        first = predict_method(method, payload, split)
        second = predict_method(method, altered, split)
        assert np.allclose(first.prediction, second.prediction), method


def test_gate_is_bounded_and_not_globally_constant() -> None:
    payload, split = _case()
    result = predict_method("nicheflow_v2_gated", payload, split)
    assert result.gate is not None
    assert np.all((result.gate >= 0.05) & (result.gate <= 0.80))
    assert np.std(result.gate) > 0
    assert result.components["residual_contract"] == "pure_delta_then_source_plus_delta"


def test_calibration_is_an_explicit_ablation() -> None:
    payload, split = _case()
    uncalibrated = predict_method("nicheflow_v2_gated", payload, split)
    calibrated = predict_method("nicheflow_v2_gated_calibrated", payload, split)
    assert uncalibrated.calibration is False
    assert calibrated.calibration is True
    assert not np.allclose(uncalibrated.prediction, calibrated.prediction)


if __name__ == "__main__":
    test_residual_contract_is_explicit()
    test_evaluation_expression_cannot_change_predictions()
    test_gate_is_bounded_and_not_globally_constant()
    test_calibration_is_an_explicit_ablation()
    print("ok")
