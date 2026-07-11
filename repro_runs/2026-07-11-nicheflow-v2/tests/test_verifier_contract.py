#!/usr/bin/env python3
"""Contract tests for the independent full-output verifier."""

from pathlib import Path
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from verify_full_outputs import verify_complete


def test_complete_verifier_checks_exact_gate_set_and_decision() -> None:
    verify_complete(ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp)
        shutil.copytree(ROOT / "results", copy / "results")
        shutil.copytree(ROOT / "reports", copy / "reports")
        gate_path = copy / "results" / "residual_gating" / "gate_summary.csv"
        gates = pd.read_csv(gate_path)
        gates[gates["gate_id"] != "G7"].to_csv(gate_path, index=False)
        try:
            verify_complete(copy)
        except AssertionError as exc:
            assert "G1-G7" in str(exc)
        else:
            raise AssertionError("Verifier accepted an incomplete gate set")


if __name__ == "__main__":
    test_complete_verifier_checks_exact_gate_set_and_decision()
    print("ok")
