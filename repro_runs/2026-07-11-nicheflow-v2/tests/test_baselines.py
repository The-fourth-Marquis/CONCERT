#!/usr/bin/env python3
"""Tests for reference-only reconstructions of existing methods."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.baselines import calibrate_to_reference, predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
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


def test_baselines_ignore_evaluation_expression() -> None:
    payload, split = _case()
    methods = (
        "source_identity_l0",
        "target_mean_l1",
        "knn10_l1",
        "cauchy_l1",
        "legacy_nicheflow_lite_l3",
        "legacy_v1_fixed_80_20_l3",
    )
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    for method in methods:
        first = predict_existing_method(method, payload, split)
        second = predict_existing_method(method, altered, split)
        assert first.prediction.shape == payload.query_expr.shape
        assert np.allclose(first.prediction, second.prediction), method
        assert first.method == method
        assert first.access_level in {"L0", "L1", "L2", "L3"}


def test_reference_only_calibration_uses_no_evaluation_rows() -> None:
    payload, split = _case()
    prediction = np.zeros_like(payload.query_expr)
    first = calibrate_to_reference(prediction, payload.target_expr[split.reference_indices])
    changed = payload.target_expr.copy()
    changed[split.evaluation_indices] += 1000.0
    second = calibrate_to_reference(prediction, changed[split.reference_indices])
    assert np.allclose(first, second)


def test_source_identity_is_l0_and_target_free() -> None:
    payload, split = _case()
    result = predict_existing_method("source_identity_l0", payload, split)
    assert np.array_equal(result.prediction, payload.query_expr)
    assert result.access_level == "L0"
    assert result.uses_target_reference is False
    assert result.calibration is False


if __name__ == "__main__":
    test_baselines_ignore_evaluation_expression()
    test_reference_only_calibration_uses_no_evaluation_rows()
    test_source_identity_is_l0_and_target_free()
    print("ok")
