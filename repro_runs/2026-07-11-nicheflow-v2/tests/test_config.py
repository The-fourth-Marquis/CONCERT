#!/usr/bin/env python3
"""Tests for the isolated NicheFlow-v2 experiment configuration."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.config import ExperimentConfig, assert_writable_output


def test_default_config_and_frozen_root_guard() -> None:
    cfg = ExperimentConfig.default()
    assert cfg.master_seed == 20260711
    assert cfg.reference_fractions == (0.25, 0.50, 0.70)
    assert cfg.split_seeds == (20260711, 20260712, 20260713, 20260714, 20260715)
    assert_writable_output(cfg.run_root / "results" / "smoke")
    try:
        assert_writable_output(cfg.frozen_root / "forbidden")
    except ValueError as exc:
        assert "frozen" in str(exc).lower()
    else:
        raise AssertionError("frozen output path was accepted")


def test_output_guard_rejects_paths_outside_run_root(tmp_path: Path) -> None:
    cfg = ExperimentConfig.default()
    try:
        assert_writable_output(tmp_path, cfg)
    except ValueError as exc:
        assert "run root" in str(exc).lower()
    else:
        raise AssertionError("outside output path was accepted")


if __name__ == "__main__":
    test_default_config_and_frozen_root_guard()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_output_guard_rejects_paths_outside_run_root(Path(tmp))
    print("ok")
