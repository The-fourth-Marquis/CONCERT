#!/usr/bin/env python3
"""Intrinsic identity must survive when context modulation is disabled."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.model import fit_predict_nonlinear_transport


def test_zero_modulation_preserves_all_pairwise_source_differences() -> None:
    control_expr = np.array(
        [[0.0, 0.0], [0.3, 1.0], [0.8, -1.0], [1.0, 0.5]],
        dtype=np.float32,
    )
    reference_expr = control_expr + np.array([1.0, -0.5], dtype=np.float32)
    query_expr = np.array(
        [[0.2, 3.0], [0.7, -2.0], [1.2, 0.25]],
        dtype=np.float32,
    )
    control_context = np.linspace(0.0, 1.0, 4)[:, None].astype(np.float32)
    reference_context = control_context[::-1].copy()
    query_context = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)

    result = fit_predict_nonlinear_transport(
        control_expr=control_expr,
        control_context=control_context,
        query_expr=query_expr,
        query_context=query_context,
        reference_expr=reference_expr,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=8,
        context_weight=1.0,
        modulation_strength=0.0,
        seed=17,
    )

    predicted_pairs = result.prediction[:, None, :] - result.prediction[None, :, :]
    source_pairs = query_expr[:, None, :] - query_expr[None, :, :]
    assert np.allclose(predicted_pairs, source_pairs, atol=1e-5)
    assert result.metadata["modulation_strength"] == 0.0
    assert result.metadata["decoder_contract"] == (
        "intrinsic_identity_plus_global_shift_plus_centered_modulation"
    )


if __name__ == "__main__":
    test_zero_modulation_preserves_all_pairwise_source_differences()
    print("ok")
