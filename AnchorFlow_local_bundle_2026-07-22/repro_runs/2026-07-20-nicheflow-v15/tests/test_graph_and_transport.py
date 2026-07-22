#!/usr/bin/env python3
"""Contract tests for the first NicheFlow-v1.5 implementation slice."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.graph import build_niche_adjacency
from nicheflow_v15.transport import entropic_sinkhorn


def test_niche_adjacency_is_row_stochastic_and_context_sensitive() -> None:
    position = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.9, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    niche = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    adjacency = build_niche_adjacency(position, niche, k=2)

    assert adjacency.shape == (4, 4)
    assert np.all(adjacency >= 0.0)
    assert np.allclose(adjacency.sum(axis=1), 1.0)
    assert adjacency[0, 1] > adjacency[0, 2]
    assert adjacency[3, 2] > adjacency[3, 1]


def test_sinkhorn_coupling_has_requested_uniform_marginals() -> None:
    cost = np.array(
        [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    coupling = entropic_sinkhorn(cost, epsilon=0.2, iterations=200)

    assert coupling.shape == cost.shape
    assert np.all(coupling >= 0.0)
    assert np.allclose(coupling.sum(axis=1), 0.5, atol=1e-5)
    assert np.allclose(coupling.sum(axis=0), 1.0 / 3.0, atol=1e-5)
    assert coupling[0, 0] > coupling[0, 2]
    assert coupling[1, 1] > coupling[1, 0]


if __name__ == "__main__":
    test_niche_adjacency_is_row_stochastic_and_context_sensitive()
    test_sinkhorn_coupling_has_requested_uniform_marginals()
    print("ok")
