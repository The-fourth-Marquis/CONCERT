"""Small dependency-free optimal-transport primitives."""

from __future__ import annotations

import numpy as np


def entropic_sinkhorn(
    cost: np.ndarray,
    *,
    epsilon: float = 0.1,
    iterations: int = 100,
) -> np.ndarray:
    """Return a uniform-marginal entropic OT coupling for a cost matrix."""

    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or min(cost.shape) == 0:
        raise ValueError("cost must be a non-empty two-dimensional array")
    if not np.isfinite(cost).all():
        raise ValueError("cost must contain only finite values")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")

    n_source, n_target = cost.shape
    source_mass = np.full(n_source, 1.0 / n_source, dtype=np.float64)
    target_mass = np.full(n_target, 1.0 / n_target, dtype=np.float64)
    shifted = cost - float(cost.min())
    kernel = np.exp(-shifted / float(epsilon))
    kernel = np.maximum(kernel, 1e-300)
    source_scale = np.ones(n_source, dtype=np.float64)
    target_scale = np.ones(n_target, dtype=np.float64)

    for _ in range(int(iterations)):
        source_scale = source_mass / np.maximum(kernel @ target_scale, 1e-300)
        target_scale = target_mass / np.maximum(
            kernel.T @ source_scale,
            1e-300,
        )

    coupling = source_scale[:, None] * kernel * target_scale[None, :]
    return coupling.astype(np.float64)
