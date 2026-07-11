#!/usr/bin/env python3
"""Tests for held-out existing-method reassessment and coverage."""

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.config import ExperimentConfig
from nicheflow_v2.data import load_manifest
from nicheflow_v2.reassess import merge_fast_and_expensive, reassess_manifest


def _smoke_config() -> ExperimentConfig:
    return replace(
        ExperimentConfig.default(),
        reference_fractions=(0.70,),
        split_seeds=(20260711,),
        split_types=("random_stratified",),
    )


def test_every_metric_row_has_protocol_metadata() -> None:
    rows, coverage, split_manifest = reassess_manifest(load_manifest().head(1), _smoke_config())
    required = {
        "task_id",
        "method",
        "access_level",
        "reference_fraction",
        "split_seed",
        "split_type",
        "calibration",
        "uses_target_reference",
        "heldout_edistance",
        "niche_stratified_edistance",
        "niche_wise_delta_pcc",
        "spatial_map_pcc",
        "delta_pcc",
        "de_top20_direction_accuracy",
        "eligible_for_primary",
    }
    assert not rows.empty
    assert required.issubset(rows.columns)
    assert not rows[["task_id", "method", "access_level", "split_seed", "split_type", "heldout_edistance"]].isna().any().any()
    assert split_manifest.shape[0] == 1
    assert split_manifest.iloc[0]["split_status"] == "ok"
    assert set(coverage["eligibility_status"]) <= {
        "exact",
        "approximate",
        "unavailable",
        "not_eligible_for_heldout_primary",
    }


def test_fixed_fullpool_concert_is_not_eligible_for_heldout_primary() -> None:
    rows, coverage, _ = reassess_manifest(load_manifest().head(1), _smoke_config())
    concert = coverage[coverage["model_family"] == "available_concert"]
    assert not concert.empty
    assert set(concert["eligibility_status"]) == {"not_eligible_for_heldout_primary"}
    assert not rows["method"].str.contains("concert", case=False).any()


def test_evaluation_mutation_guard_is_exercised() -> None:
    rows, coverage, _ = reassess_manifest(load_manifest().head(1), _smoke_config())
    exact = coverage[coverage["eligibility_status"] == "exact"]
    assert exact["leakage_mutation_check"].all()
    assert rows["eligible_for_primary"].all()


def test_include_expensive_records_spatialprop_when_split_is_unavailable() -> None:
    cfg = replace(
        _smoke_config(),
        min_reference=10000,
        min_evaluation=5,
    )
    rows, coverage, _ = reassess_manifest(
        load_manifest().head(1),
        cfg,
        include_expensive=True,
    )
    assert rows.empty
    spatial = coverage[
        coverage["method"] == "spatialprop_reference_only_retrain"
    ]
    assert spatial.shape[0] == 1
    assert set(spatial["eligibility_status"]) == {"unavailable"}


def test_merge_replaces_unavailable_spatialprop_coverage() -> None:
    import pandas as pd

    fast_metrics = pd.DataFrame([{"method": "anchor", "value": 1.0}])
    fast_coverage = pd.DataFrame([
        {"method": "anchor", "eligibility_status": "exact"},
        {
            "method": "spatialprop_reference_only_retrain",
            "eligibility_status": "unavailable",
        },
    ])
    expensive_metrics = pd.DataFrame([
        {"method": "spatialprop_reference_only_retrain", "value": 2.0}
    ])
    expensive_coverage = pd.DataFrame([
        {
            "method": "spatialprop_reference_only_retrain",
            "eligibility_status": "approximate",
        }
    ])
    metrics, coverage = merge_fast_and_expensive(
        fast_metrics,
        fast_coverage,
        expensive_metrics,
        expensive_coverage,
    )
    assert set(metrics["method"]) == {
        "anchor", "spatialprop_reference_only_retrain"
    }
    spatial = coverage[
        coverage["method"] == "spatialprop_reference_only_retrain"
    ]
    assert set(spatial["eligibility_status"]) == {"approximate"}


def test_include_expensive_runs_reference_only_spatialprop() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(
            _smoke_config(),
            run_root=Path(tmp),
            spatialprop_epochs=2,
            spatialprop_hidden_dim=8,
            spatialprop_latent_dim=4,
        )
        rows, coverage, _ = reassess_manifest(
            load_manifest().head(1),
            cfg,
            include_expensive=True,
            only_expensive=True,
        )
    assert set(rows["method"]) == {"spatialprop_reference_only_retrain"}
    assert rows["leakage_mutation_check"].all()
    spatialprop = coverage[
        coverage["method"] == "spatialprop_reference_only_retrain"
    ]
    assert set(spatialprop["eligibility_status"]) == {"approximate"}
    assert set(spatialprop["reason"]) == {"reference-only SpatialProp-style retraining"}


if __name__ == "__main__":
    test_every_metric_row_has_protocol_metadata()
    test_fixed_fullpool_concert_is_not_eligible_for_heldout_primary()
    test_evaluation_mutation_guard_is_exercised()
    test_include_expensive_records_spatialprop_when_split_is_unavailable()
    test_merge_replaces_unavailable_spatialprop_coverage()
    test_include_expensive_runs_reference_only_spatialprop()
    print("ok")
