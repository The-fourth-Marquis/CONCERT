#!/usr/bin/env python3
"""Tests for held-out distribution, perturbation-effect, and DE metrics."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.metrics import (
    de_direction_metrics,
    distribution_metrics,
    perturbation_effect_metrics,
    safe_correlation,
    select_de_indices,
)


def test_correct_delta_beats_reversed_delta() -> None:
    source = np.zeros((4, 6), dtype=np.float32)
    target_delta = np.array([3, 2, 1, -1, -2, -3], dtype=np.float32)
    target = np.tile(target_delta, (5, 1))
    correct = np.tile(target_delta, (4, 1))
    reversed_pred = -correct
    good = perturbation_effect_metrics(correct, source, target)
    bad = perturbation_effect_metrics(reversed_pred, source, target)
    assert good["delta_pcc"] > bad["delta_pcc"]
    assert good["delta_spearman"] > bad["delta_spearman"]
    assert good["delta_mae"] < bad["delta_mae"]
    assert good["delta_rmse"] < bad["delta_rmse"]


def test_de_metrics_report_opposite_direction() -> None:
    reference_delta = np.array([5, 4, 3, 2, 1, -1, -2, -3, -4, -5], dtype=float)
    true_delta = reference_delta.copy()
    reversed_delta = -true_delta
    metrics = de_direction_metrics(reversed_delta, true_delta, reference_delta, top_k=10)
    assert metrics["de_top10_direction_accuracy"] == 0.0
    assert metrics["de_top10_opposite_direction_fraction"] == 1.0
    assert metrics["de_top10_delta_pcc"] == -1.0


def test_de_selection_uses_absolute_reference_delta() -> None:
    reference_delta = np.array([0.5, -9.0, 3.0, 7.0, -1.0])
    selected = select_de_indices(reference_delta, top_k=3)
    assert selected.tolist() == [1, 3, 2]


def test_undefined_correlation_is_nan_with_reason() -> None:
    value, reason = safe_correlation(np.ones(4), np.ones(4), method="pearson")
    assert np.isnan(value)
    assert reason == "constant_input"


def test_distribution_metrics_distinguish_heldout_and_fullpool() -> None:
    pred = np.array([[0.0], [1.0]], dtype=np.float32)
    heldout = np.array([[0.0], [1.0]], dtype=np.float32)
    fullpool = np.array([[0.0], [1.0], [100.0]], dtype=np.float32)
    metrics = distribution_metrics(pred, heldout, fullpool)
    assert np.isclose(metrics["heldout_edistance"], 0.0)
    assert metrics["fullpool_edistance"] > metrics["heldout_edistance"]


if __name__ == "__main__":
    test_correct_delta_beats_reversed_delta()
    test_de_metrics_report_opposite_direction()
    test_de_selection_uses_absolute_reference_delta()
    test_undefined_correlation_is_nan_with_reason()
    test_distribution_metrics_distinguish_heldout_and_fullpool()
    print("ok")
