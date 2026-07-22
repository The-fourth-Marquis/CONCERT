#!/usr/bin/env python3
"""Tests for versioned NicheFlow-v2 spatial/context diagnostics."""

from pathlib import Path
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.metrics import (
    METRIC_REGISTRY,
    boundary_leakage_score,
    cross_niche_drop,
    distance_response_decay_score,
    source_individuality_score,
)


def test_boundary_leakage_increases_for_cross_boundary_collapse() -> None:
    target = np.array([[10.0], [10.0], [0.0], [0.0]], dtype=np.float32)
    labels = np.array(["tumor_like", "tumor_like", "boundary", "boundary"])
    preserved = target.copy()
    collapsed = np.full_like(target, 5.0)
    assert boundary_leakage_score(collapsed, target, labels) > boundary_leakage_score(preserved, target, labels)


def test_source_individuality_rewards_preserved_distances() -> None:
    source = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    collapsed = np.ones_like(source)
    preserved_value, preserved_reason = source_individuality_score(source.copy(), source)
    collapsed_value, collapsed_reason = source_individuality_score(collapsed, source)
    assert preserved_value > 0.99
    assert preserved_reason == ""
    assert np.isnan(collapsed_value)
    assert collapsed_reason == "constant_input"


def test_distance_response_decay_is_positive_for_decay() -> None:
    distance = np.array([0.0, 1.0, 2.0, 3.0])
    response = np.array([4.0, 3.0, 2.0, 1.0])
    value, reason = distance_response_decay_score(distance, response)
    assert value > 0.99
    assert reason == ""


def test_cross_niche_drop_is_in_niche_minus_cross_niche() -> None:
    rows = pd.DataFrame(
        {
            "task_type": ["niche_core", "cross_niche_core_to_border"],
            "delta_pcc": [0.8, 0.3],
        }
    )
    assert np.isclose(cross_niche_drop(rows), 0.5)


def test_spatial_metrics_are_versioned_and_noncausal() -> None:
    for name in ("boundary_leakage_score", "source_individuality_score", "distance_response_decay_score"):
        item = METRIC_REGISTRY[name]
        assert item["version"] == "v2.0"
        assert item["direction"] in {"lower", "higher"}
        assert item["causal_proof"] == "no"


if __name__ == "__main__":
    test_boundary_leakage_increases_for_cross_boundary_collapse()
    test_source_individuality_rewards_preserved_distances()
    test_distance_response_decay_is_positive_for_decay()
    test_cross_niche_drop_is_in_niche_minus_cross_niche()
    test_spatial_metrics_are_versioned_and_noncausal()
    print("ok")
