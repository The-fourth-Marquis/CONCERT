#!/usr/bin/env python3
"""Frozen protocol and E-distance decision contracts."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "run_submission_smoke.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("anchorflow_submission", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load submission runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positive_summaries() -> list[dict[str, object]]:
    return [
        {
            "task_id": "task_a",
            "split_type": "random_stratified",
            "final_heldout_edistance": 8.0,
            "initial_heldout_edistance": 9.0,
            "fixed_80_20_heldout_edistance": 10.0,
            "leakage_mutation_check": True,
        },
        {
            "task_id": "task_b",
            "split_type": "random_stratified",
            "final_heldout_edistance": 10.0,
            "initial_heldout_edistance": 11.0,
            "fixed_80_20_heldout_edistance": 12.0,
            "leakage_mutation_check": True,
        },
        {
            "task_id": "task_a",
            "split_type": "spatial_block",
            "final_heldout_edistance": 9.0,
            "initial_heldout_edistance": 10.0,
            "fixed_80_20_heldout_edistance": 11.0,
            "leakage_mutation_check": True,
        },
        {
            "task_id": "task_b",
            "split_type": "spatial_block",
            "final_heldout_edistance": 11.0,
            "initial_heldout_edistance": 12.0,
            "fixed_80_20_heldout_edistance": 13.0,
            "leakage_mutation_check": True,
        },
    ]


def test_frozen_config() -> None:
    runner = _load_runner()
    config = runner.load_config()
    task_ids = {item["index"]: item["task_id"] for item in config["tasks"]}
    assert task_ids == {
        13: "GSM5808055_border_surrogate_KP_to_Tgfbr2",
        16: "GSM5808055_cross_slide_surrogate_KP_to_Jak2_to_GSM5808054",
    }
    assert config["split_types"] == ["random_stratified", "spatial_block"]
    assert config["settings"]["integration_horizon"] == 0.25
    assert config["settings"]["device"] == "cuda:0"
    assert config["settings"]["seed"] == 20260711
    assert config["runtime"]["python_executable"] == (
        "/root/code/nicheformer/.venv/bin/python"
    )
    assert config["runtime"]["required_cuda_arch"] == "sm_120"


def test_runtime_rejects_pytorch_without_blackwell_kernels() -> None:
    runner = _load_runner()
    try:
        runner.validate_cuda_architecture(["sm_90"], "sm_120")
    except RuntimeError as error:
        assert "sm_120" in str(error)
    else:
        raise AssertionError("unsupported PyTorch architecture unexpectedly passed")


def test_positive_decision_requires_all_frozen_gates() -> None:
    runner = _load_runner()
    result = runner.aggregate_decision(_positive_summaries())
    assert result["decision"] == "confirmation_positive"
    assert result["flow_wins_both_count"] == 4
    assert result["all_mutation_guards"] is True
    assert result["all_metrics_finite"] is True
    assert all(result["gates"].values())


def test_fewer_than_three_joint_wins_stops_confirmation() -> None:
    runner = _load_runner()
    rows = _positive_summaries()
    rows[0]["final_heldout_edistance"] = 11.0
    rows[1]["final_heldout_edistance"] = 13.0
    result = runner.aggregate_decision(rows)
    assert result["decision"] == "insufficient_confirmation_for_performance_claim"
    assert result["gates"]["at_least_three_joint_wins"] is False


if __name__ == "__main__":
    test_frozen_config()
    test_runtime_rejects_pytorch_without_blackwell_kernels()
    test_positive_decision_requires_all_frozen_gates()
    test_fewer_than_three_joint_wins_stops_confirmation()
    print("ok")
