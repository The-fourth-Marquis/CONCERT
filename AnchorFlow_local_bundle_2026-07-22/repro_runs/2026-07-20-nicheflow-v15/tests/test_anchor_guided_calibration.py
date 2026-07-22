#!/usr/bin/env python3
"""Reference-only strength selection contracts."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_anchor_guided_calibrated_smoke import _select_strength


def test_select_strength_uses_pseudo_gates_then_edistance() -> None:
    fixed = {
        "heldout_edistance": 10.0,
        "delta_pcc": 0.9,
        "boundary_leakage": 1.0,
        "source_individuality": 1.0,
    }
    rows = [
        {
            "modulation_strength": 0.2,
            "heldout_edistance": 8.0,
            "delta_pcc": 0.87,
            "boundary_leakage": 1.03,
            "source_individuality": 0.99,
        },
        {
            "modulation_strength": 0.7,
            "heldout_edistance": 6.0,
            "delta_pcc": 0.89,
            "boundary_leakage": 1.04,
            "source_individuality": 0.985,
        },
        {
            "modulation_strength": 1.0,
            "heldout_edistance": 4.0,
            "delta_pcc": 0.95,
            "boundary_leakage": 1.5,
            "source_individuality": 0.8,
        },
    ]
    selected = _select_strength(rows, fixed)
    assert selected == 0.7


if __name__ == "__main__":
    test_select_strength_uses_pseudo_gates_then_edistance()
    print("ok")


def test_select_strength_falls_back_to_zero_without_pseudo_gate() -> None:
    fixed = {
        "heldout_edistance": 10.0,
        "delta_pcc": 0.9,
        "boundary_leakage": 1.0,
        "source_individuality": 1.0,
    }
    rows = [
        {
            "modulation_strength": 0.0,
            "heldout_edistance": 12.0,
            "delta_pcc": 0.8,
            "boundary_leakage": 1.2,
            "source_individuality": 1.0,
        },
        {
            "modulation_strength": 1.0,
            "heldout_edistance": 5.0,
            "delta_pcc": 0.95,
            "boundary_leakage": 1.5,
            "source_individuality": 0.3,
        },
    ]
    assert _select_strength(rows, fixed) == 0.0
