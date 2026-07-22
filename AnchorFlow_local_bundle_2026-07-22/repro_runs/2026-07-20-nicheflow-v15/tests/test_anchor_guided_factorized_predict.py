#!/usr/bin/env python3
"""Reference-only task wrapper contract for anchor-guided decoder."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "2026-07-11-nicheflow-v2" / "src"))

from nicheflow_v15.anchor_guided_factorized_predict import (
    predict_task_anchor_guided_factorized,
)
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool


def test_wrapper_ignores_evaluation_expression() -> None:
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
    settings = {"modulation_strength": 0.4, "anchor_temperature": 0.25}
    original = predict_task_anchor_guided_factorized(payload, split, **settings)
    changed = predict_task_anchor_guided_factorized(altered, split, **settings)
    assert np.allclose(original.prediction, changed.prediction)
    assert original.metadata["uses_evaluation_target"] is False


if __name__ == "__main__":
    test_wrapper_ignores_evaluation_expression()
    print("ok")
