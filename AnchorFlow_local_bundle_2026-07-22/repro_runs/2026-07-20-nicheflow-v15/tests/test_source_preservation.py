#!/usr/bin/env python3
"""Source-preservation contract for the low-rank nonlinear transport."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.model import fit_predict_nonlinear_transport


def test_expression_outside_transport_basis_is_preserved() -> None:
    control_expr = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    reference_expr = np.array(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    query_expr = np.array(
        [[0.5, 2.0], [1.5, -2.0]],
        dtype=np.float32,
    )
    control_context = np.zeros((3, 1), dtype=np.float32)
    reference_context = np.zeros((3, 1), dtype=np.float32)
    query_context = np.zeros((2, 1), dtype=np.float32)

    result = fit_predict_nonlinear_transport(
        control_expr=control_expr,
        control_context=control_context,
        query_expr=query_expr,
        query_context=query_context,
        reference_expr=reference_expr,
        reference_context=reference_context,
        latent_dim=1,
        hidden_dim=4,
        context_weight=0.0,
        seed=5,
    )

    assert np.allclose(result.prediction[:, 1], query_expr[:, 1], atol=1e-5)
    assert result.metadata["decoder_contract"] == "intrinsic_identity_plus_global_shift_plus_centered_modulation"


if __name__ == "__main__":
    test_expression_outside_transport_basis_is_preserved()
    print("ok")
