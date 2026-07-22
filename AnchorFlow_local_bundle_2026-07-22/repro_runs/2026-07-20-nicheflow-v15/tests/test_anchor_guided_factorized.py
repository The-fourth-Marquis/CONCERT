#!/usr/bin/env python3
"""Contracts for anchor-guided factorized decoding."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.anchor_guided_factorized import (
    fit_predict_anchor_guided_factorized,
)


def test_zero_modulation_preserves_source_geometry() -> None:
    control = np.array(
        [[0.0, 0.0], [0.3, 1.0], [0.8, -1.0], [1.0, 0.5]],
        dtype=np.float32,
    )
    reference = control + np.array([1.0, -0.5], dtype=np.float32)
    query = np.array(
        [[0.2, 3.0], [0.7, -2.0], [1.2, 0.25]],
        dtype=np.float32,
    )
    context = np.linspace(0.0, 1.0, 4)[:, None].astype(np.float32)
    result = fit_predict_anchor_guided_factorized(
        control_expr=control,
        control_context=context,
        query_expr=query,
        query_context=np.array([[0.1], [0.5], [0.9]], dtype=np.float32),
        reference_expr=reference,
        reference_context=context,
        anchor_temperature=0.25,
        modulation_strength=0.0,
    )
    predicted_pairs = result.prediction[:, None, :] - result.prediction[None, :, :]
    source_pairs = query[:, None, :] - query[None, :, :]
    assert np.allclose(predicted_pairs, source_pairs, atol=1e-5)
    assert result.metadata["modulation_strength"] == 0.0
    assert result.metadata["output_mode"] == "anchor_guided_factorized_decoder"


def test_strength_is_bounded() -> None:
    rng = np.random.default_rng(12)
    control = rng.normal(size=(5, 3)).astype(np.float32)
    reference = (control + 0.5).astype(np.float32)
    context = rng.normal(size=(5, 2)).astype(np.float32)
    result = fit_predict_anchor_guided_factorized(
        control_expr=control,
        control_context=context,
        query_expr=rng.normal(size=(4, 3)).astype(np.float32),
        query_context=rng.normal(size=(4, 2)).astype(np.float32),
        reference_expr=reference,
        reference_context=context,
        modulation_strength=0.4,
    )
    assert 0.0 <= result.metadata["modulation_strength"] <= 1.0
    assert result.metadata["max_modulation_fraction"] <= 0.4 + 1e-6


if __name__ == "__main__":
    test_zero_modulation_preserves_source_geometry()
    test_strength_is_bounded()
    print("ok")
