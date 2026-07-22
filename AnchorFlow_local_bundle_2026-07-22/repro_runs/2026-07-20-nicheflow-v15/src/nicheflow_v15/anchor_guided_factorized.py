"""Anchor-guided factorized decoder with an explicit bounded modulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .anchor_flow import _anchor_from_context
from .model import _matrix, _standardize


@dataclass(frozen=True)
class AnchorGuidedFactorizedResult:
    prediction: np.ndarray
    base_prediction: np.ndarray
    anchor_prediction: np.ndarray
    coupling: np.ndarray | None
    metadata: dict[str, Any]


def fit_predict_anchor_guided_factorized(
    *,
    control_expr: np.ndarray,
    control_context: np.ndarray,
    query_expr: np.ndarray,
    query_context: np.ndarray,
    reference_expr: np.ndarray,
    reference_context: np.ndarray,
    anchor_temperature: float = 0.25,
    modulation_strength: float = 0.4,
) -> AnchorGuidedFactorizedResult:
    """Apply a shared anchor-derived shift plus bounded context modulation.

    The base path is source expression plus one shared global shift. The local
    target anchor supplies a context-conditioned residual, scaled by a single
    bounded strength. At strength zero, all source pairwise geometry is exact.
    """

    control_expr = _matrix("control_expr", control_expr)
    query_expr = _matrix("query_expr", query_expr)
    reference_expr = _matrix("reference_expr", reference_expr)
    control_context = _matrix("control_context", control_context)
    query_context = _matrix("query_context", query_context)
    reference_context = _matrix("reference_context", reference_context)
    if len({control_expr.shape[1], query_expr.shape[1], reference_expr.shape[1]}) != 1:
        raise ValueError("expression matrices must share genes")
    if len({control_context.shape[1], query_context.shape[1], reference_context.shape[1]}) != 1:
        raise ValueError("context matrices must share features")
    if control_context.shape[0] != control_expr.shape[0]:
        raise ValueError("control context row count differs")
    if query_context.shape[0] != query_expr.shape[0]:
        raise ValueError("query context row count differs")
    if reference_context.shape[0] != reference_expr.shape[0]:
        raise ValueError("reference context row count differs")
    if float(anchor_temperature) <= 0.0:
        raise ValueError("anchor_temperature must be positive")
    if not 0.0 <= float(modulation_strength) <= 1.0:
        raise ValueError("modulation_strength must be between zero and one")

    control_context_z, query_context_z, reference_context_z = _standardize(
        control_context,
        query_context,
        reference_context,
    )
    control_anchor = _anchor_from_context(
        control_context_z,
        reference_context_z,
        reference_expr,
        temperature=float(anchor_temperature),
    )
    query_anchor = _anchor_from_context(
        query_context_z,
        reference_context_z,
        reference_expr,
        temperature=float(anchor_temperature),
    )

    global_shift = control_anchor.mean(axis=0, keepdims=True) - control_expr.mean(
        axis=0,
        keepdims=True,
    )
    base_control = control_expr + global_shift
    base_query = query_expr + global_shift
    control_residual = control_anchor - base_control
    query_residual = query_anchor - base_query
    prediction = base_query + float(modulation_strength) * query_residual
    return AnchorGuidedFactorizedResult(
        prediction=prediction.astype(np.float32),
        base_prediction=base_query.astype(np.float32),
        anchor_prediction=query_anchor.astype(np.float32),
        coupling=None,
        metadata={
            "output_mode": "anchor_guided_factorized_decoder",
            "decoder_contract": "source_expression_plus_shared_global_shift_plus_bounded_context_anchor_modulation",
            "uses_target_reference": True,
            "uses_evaluation_target": False,
            "anchor_temperature": float(anchor_temperature),
            "modulation_strength": float(modulation_strength),
            "global_shift_norm": float(np.sqrt(np.mean(global_shift**2))),
            "control_residual_norm": float(np.sqrt(np.mean(control_residual**2))),
            "query_residual_norm": float(np.sqrt(np.mean(query_residual**2))),
            "max_modulation_fraction": float(modulation_strength),
            "n_control": int(control_expr.shape[0]),
            "n_reference": int(reference_expr.shape[0]),
            "n_query": int(query_expr.shape[0]),
        },
    )
