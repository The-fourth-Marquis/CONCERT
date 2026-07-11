#!/usr/bin/env python3
"""Independent artifact verifier for NicheFlow-v2 runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "task_manifest.csv", "metrics.csv", "method_ranking.csv",
    "coverage.csv", "metadata.json", "README.md",
}
KEYS = ["task_id", "method", "reference_fraction", "split_seed", "split_type"]
PROTOCOL_COLUMNS = {
    "task_id", "method", "access_level", "reference_fraction", "split_seed",
    "split_type", "calibration", "uses_target_reference",
    "eligible_for_primary", "leakage_mutation_check", "claim_scope",
    "heldout_edistance", "fullpool_edistance", "niche_stratified_edistance",
    "delta_pcc", "de_top20_direction_accuracy", "boundary_leakage_score",
    "source_individuality_score", "spatial_map_pcc",
}


def _all_true(values: pd.Series) -> bool:
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1"}).all()


def verify_directory(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = sorted(name for name in REQUIRED if not (path / name).exists())
    if missing:
        raise AssertionError(f"Missing artifacts in {path}: {missing}")
    metrics = pd.read_csv(path / "metrics.csv")
    ranking = pd.read_csv(path / "method_ranking.csv")
    coverage = pd.read_csv(path / "coverage.csv")
    if metrics.empty or coverage.empty:
        raise AssertionError(f"Empty metrics or coverage in {path}")
    missing_columns = sorted(PROTOCOL_COLUMNS - set(metrics.columns))
    if missing_columns:
        raise AssertionError(f"Missing protocol metric columns in {path}: {missing_columns}")
    if metrics.duplicated(KEYS).any():
        raise AssertionError(f"Duplicate task/method/split metric keys in {path}")
    if not _all_true(metrics["eligible_for_primary"]):
        raise AssertionError(f"Ineligible rows present in primary metrics in {path}")
    if not _all_true(metrics["leakage_mutation_check"]):
        raise AssertionError(f"Metric leakage mutation gate failed in {path}")
    eligible_coverage = coverage[
        coverage["eligibility_status"].isin({"exact", "approximate"})
    ]
    if not eligible_coverage.empty and not _all_true(
        eligible_coverage["leakage_mutation_check"]
    ):
        raise AssertionError(f"Eligible coverage leakage gate failed in {path}")
    if not ranking.empty and not _all_true(ranking["eligible_for_primary"]):
        raise AssertionError("Primary ranking contains ineligible rows")
    if set(metrics["reference_fraction"].round(2)) != {0.25, 0.50, 0.70}:
        raise AssertionError(f"Reference-fraction coverage incomplete in {path}")
    if metrics["split_seed"].nunique() != 5:
        raise AssertionError(f"Split-seed coverage incomplete in {path}")
    if set(metrics["split_type"]) != {"random_stratified", "spatial_block"}:
        raise AssertionError(f"Split-type coverage incomplete in {path}")
    if metrics["task_id"].nunique() != 37:
        raise AssertionError(f"Task coverage incomplete in {path}")
    return metrics, ranking, coverage


def verify_gate_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AssertionError(f"Missing gate summary: {path}")
    gates = pd.read_csv(path)
    expected = [f"G{index}" for index in range(1, 8)]
    if gates["gate_id"].tolist() != expected:
        raise AssertionError("gate_summary.csv must contain exactly ordered G1-G7")
    required = {
        "gate_id", "criterion", "passed", "observed", "threshold",
        "candidate_method", "decision",
    }
    if not required.issubset(gates.columns):
        raise AssertionError("gate_summary.csv is missing formal gate columns")
    decisions = gates["decision"].dropna().unique()
    if decisions.size != 1:
        raise AssertionError("gate_summary.csv must contain one unique decision")
    if decisions[0] not in {"advance_to_graph_flow", "do_not_advance_to_graph_flow"}:
        raise AssertionError(f"Unknown graph/flow decision: {decisions[0]}")
    return gates


def verify_complete(root: Path = ROOT) -> None:
    reassessment = root / "results" / "existing_method_reassessment"
    residual = root / "results" / "residual_gating"
    reassessment_metrics, _, reassessment_coverage = verify_directory(reassessment)
    residual_metrics, _, residual_coverage = verify_directory(residual)
    if reassessment_metrics.shape[0] != 14430:
        raise AssertionError("Expected 14,430 existing-method metric rows")
    if reassessment_coverage.shape[0] != 15540:
        raise AssertionError("Expected 15,540 existing-method coverage rows")
    if residual_metrics.shape[0] != 6660 or residual_coverage.shape[0] != 6660:
        raise AssertionError("Expected 6,660 residual metric and coverage rows")
    if not {"L0", "L1", "L2", "L3"}.issubset(
        set(reassessment_metrics["access_level"])
    ):
        raise AssertionError("Existing-method access ladder is incomplete")
    spatialprop_metrics = reassessment_metrics[
        reassessment_metrics["method"] == "spatialprop_reference_only_retrain"
    ]
    spatialprop_coverage = reassessment_coverage[
        reassessment_coverage["method"] == "spatialprop_reference_only_retrain"
    ]
    if spatialprop_metrics.shape[0] != 1110:
        raise AssertionError("SpatialProp held-out metric coverage is incomplete")
    if set(spatialprop_coverage["eligibility_status"]) != {"approximate"}:
        raise AssertionError("SpatialProp coverage was not merged as approximate")
    if not {"L1", "L3", "L4"}.issubset(set(residual_metrics["access_level"])):
        raise AssertionError("Residual/gating access ladder is incomplete")
    gates = verify_gate_summary(residual / "gate_summary.csv")
    report = root / "reports" / "NicheFlow_v2_report.md"
    if not report.exists():
        raise AssertionError("Missing final NicheFlow-v2 report")
    report_text = report.read_text()
    decision = str(gates["decision"].iloc[0])
    if decision not in report_text:
        raise AssertionError("Final report does not state the gate-derived decision")
    if "Mainline A" not in report_text or "unseen/OOD" not in report_text:
        raise AssertionError("Final report is missing the Mainline A claim boundary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("reassessment-fast", "residual-gating", "complete"),
        required=True,
    )
    args = parser.parse_args()
    reassessment = ROOT / "results" / "existing_method_reassessment"
    residual = ROOT / "results" / "residual_gating"
    if args.mode == "reassessment-fast":
        verify_directory(reassessment)
    elif args.mode == "residual-gating":
        verify_directory(residual)
        verify_gate_summary(residual / "gate_summary.csv")
    else:
        verify_complete(ROOT)
    print("ok")


if __name__ == "__main__":
    main()
