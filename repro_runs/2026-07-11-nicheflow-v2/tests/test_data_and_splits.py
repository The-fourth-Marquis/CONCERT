#!/usr/bin/env python3
"""Tests for v2 task payloads and leakage-safe target splits."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool


def test_payload_uses_full_eligible_control_pool_and_selected_queries() -> None:
    task = load_manifest().iloc[0]
    payload = load_task_payload(task)
    assert payload.query_expr.shape[0] == int(task["n_selected_spots"])
    assert payload.control_expr.shape[0] >= payload.query_expr.shape[0]
    assert set(payload.query_indices.tolist()).issubset(set(payload.control_indices.tolist()))
    assert payload.target_expr.shape[0] == int(task["n_target_spots"])
    assert payload.query_expr.shape[1] == payload.target_expr.shape[1] == payload.gene_names.size


def test_random_splits_are_disjoint_deterministic_and_complete() -> None:
    payload = load_task_payload(load_manifest().iloc[0])
    n_target = payload.target_expr.shape[0]
    one = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    two = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    assert one.status == "ok"
    assert np.array_equal(one.reference_indices, two.reference_indices)
    assert np.array_equal(one.evaluation_indices, two.evaluation_indices)
    assert not set(one.reference_indices) & set(one.evaluation_indices)
    assert sorted(np.r_[one.reference_indices, one.evaluation_indices].tolist()) == list(range(n_target))
    one.validate(n_target)


def test_spatial_block_split_separates_blocks() -> None:
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="spatial_block")
    assert split.status == "ok"
    assert split.split_type == "spatial_block"
    assert split.reference_indices.size >= 5
    assert split.evaluation_indices.size >= 5
    assert split.evaluation_blocks
    split.validate(payload.target_expr.shape[0])


def test_small_target_pool_returns_explicit_insufficient_status() -> None:
    payload = load_task_payload(load_manifest().iloc[0])
    small = payload.with_target_subset(np.arange(8))
    split = split_target_pool(small, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    assert split.status == "insufficient_target_pool"
    assert split.reference_indices.size == 0
    assert split.evaluation_indices.size == 0
    assert split.reason


if __name__ == "__main__":
    test_payload_uses_full_eligible_control_pool_and_selected_queries()
    test_random_splits_are_disjoint_deterministic_and_complete()
    test_spatial_block_split_separates_blocks()
    test_small_target_pool_returns_explicit_insufficient_status()
    print("ok")
