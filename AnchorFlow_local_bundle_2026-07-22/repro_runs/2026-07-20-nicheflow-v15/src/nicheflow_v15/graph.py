"""Reference-only niche graph construction."""

from __future__ import annotations

import numpy as np


def build_niche_adjacency(
    position: np.ndarray,
    niche: np.ndarray,
    *,
    k: int = 8,
    niche_weight: float = 1.0,
    boundary_weight: float = 0.75,
    temperature: float = 1.0,
) -> np.ndarray:
    """Build a row-stochastic context graph from supplied, eligible rows only.

    The caller owns the information boundary. In held-out evaluation it must
    pass source/control, query, and target-reference rows, never target-
    evaluation rows.
    """

    position = np.asarray(position, dtype=np.float32)
    niche = np.asarray(niche, dtype=np.float32)
    if position.ndim != 2 or position.shape[0] == 0:
        raise ValueError("position must be a non-empty two-dimensional array")
    if niche.shape != (position.shape[0],):
        raise ValueError("niche must contain one value per position row")
    if int(k) < 1:
        raise ValueError("k must be positive")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    spatial = np.sqrt(
        ((position[:, None, :] - position[None, :, :]) ** 2).sum(axis=2)
    )
    positive = spatial[spatial > 0]
    scale = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    spatial = spatial / scale
    niche_distance = np.abs(niche[:, None] - niche[None, :])
    opposite_side = (niche[:, None] >= 0.5) != (niche[None, :] >= 0.5)
    cost = (
        spatial
        + float(niche_weight) * niche_distance
        + float(boundary_weight) * opposite_side.astype(np.float32)
    )

    n_rows = int(position.shape[0])
    k_eff = min(int(k), max(n_rows - 1, 0))
    adjacency = np.zeros((n_rows, n_rows), dtype=np.float64)
    for row in range(n_rows):
        if k_eff:
            candidates = np.argsort(cost[row])[: k_eff + 1]
        else:
            candidates = np.array([row], dtype=int)
        weights = np.exp(-cost[row, candidates] / float(temperature))
        adjacency[row, candidates] = weights

    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-12)
    return adjacency.astype(np.float32)
