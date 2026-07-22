"""Task wrapper for the anchor-guided factorized decoder."""

from __future__ import annotations

from typing import Any

import numpy as np

from .anchor_guided_factorized import (
    AnchorGuidedFactorizedResult,
    fit_predict_anchor_guided_factorized,
)
from .predict import _eligible_context


def predict_task_anchor_guided_factorized(
    payload: Any,
    split: Any,
    *,
    graph_k: int = 8,
    anchor_temperature: float = 0.25,
    modulation_strength: float = 0.4,
) -> AnchorGuidedFactorizedResult:
    if getattr(split, "status", None) != "ok":
        raise ValueError(
            f"Cannot predict with split status {getattr(split, 'status', None)}"
        )
    reference_indices = np.asarray(split.reference_indices, dtype=int)
    control_context, query_context, reference_context = _eligible_context(
        payload,
        reference_indices,
        graph_k=int(graph_k),
    )
    result = fit_predict_anchor_guided_factorized(
        control_expr=payload.control_expr,
        control_context=control_context,
        query_expr=payload.query_expr,
        query_context=query_context,
        reference_expr=payload.target_expr[reference_indices],
        reference_context=reference_context,
        anchor_temperature=float(anchor_temperature),
        modulation_strength=float(modulation_strength),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "task_id": str(payload.task_id),
            "access_level": "L3",
            "reference_fraction": float(split.reference_fraction),
            "split_seed": int(split.seed),
            "split_type": str(split.split_type),
            "graph_k": int(graph_k),
        }
    )
    return AnchorGuidedFactorizedResult(
        prediction=result.prediction,
        base_prediction=result.base_prediction,
        anchor_prediction=result.anchor_prediction,
        coupling=result.coupling,
        metadata=metadata,
    )
