"""Validated tabular and Markdown evidence artifacts for NicheFlow-v2."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from .config import ExperimentConfig, assert_writable_output
from .metrics import METRIC_REGISTRY


REQUIRED_METRIC_COLUMNS = {
    "task_id", "method", "access_level", "reference_fraction", "split_seed",
    "split_type", "calibration", "uses_target_reference", "heldout_edistance",
    "eligible_for_primary",
}


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty(repo_root: Path) -> bool:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def _validate_metrics(metrics: pd.DataFrame) -> None:
    missing = REQUIRED_METRIC_COLUMNS - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {sorted(missing)}")
    keys = ["task_id", "method", "reference_fraction", "split_seed", "split_type"]
    if metrics.duplicated(keys).any():
        raise ValueError(f"Duplicate metric keys: {keys}")
    if metrics["heldout_edistance"].isna().any():
        raise ValueError("Eligible metric rows require heldout_edistance")


def _method_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    eligible = metrics[metrics["eligible_for_primary"].astype(bool)].copy()
    keys = ["task_id", "reference_fraction", "split_seed", "split_type"]
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values(keys + ["heldout_edistance", "method"])
        .groupby(keys, as_index=False)
        .first()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _readme(metrics: pd.DataFrame, coverage: pd.DataFrame, metadata: dict[str, Any]) -> str:
    eligible = int(metrics["eligible_for_primary"].astype(bool).sum())
    return f"""# NicheFlow-v2 Held-out Evidence

This artifact evaluates observed perturbation relocation with disjoint target reference and held-out evaluation pools. It does not claim strict unseen/OOD perturbation prediction or causal proof.

- Metric rows: {len(metrics)}
- Primary-eligible rows: {eligible}
- Coverage rows: {len(coverage)}
- Claim scope: {metadata.get('claim_scope', 'observed perturbation relocation')}

`method_ranking.csv` excludes every row not eligible for the held-out primary protocol. Full-pool metrics, when present, are compatibility diagnostics only.
"""


def write_artifacts(
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    task_manifest: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    config: ExperimentConfig | None = None,
) -> list[Path]:
    config = config or ExperimentConfig.default()
    output_dir = Path(output_dir)
    assert_writable_output(output_dir, config)
    _validate_metrics(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking = _method_ranking(metrics)
    full_metadata = {
        **metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(config.repo_root),
        "git_dirty": _git_dirty(config.repo_root),
        "metric_registry": METRIC_REGISTRY,
        "reference_fractions": list(config.reference_fractions),
        "split_seeds": list(config.split_seeds),
        "split_types": list(config.split_types),
    }
    frames = {
        "task_manifest.csv": task_manifest,
        "metrics.csv": metrics,
        "method_ranking.csv": ranking,
        "coverage.csv": coverage,
    }
    outputs: list[Path] = []
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs.append(path)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(full_metadata, indent=2, sort_keys=True))
    outputs.append(metadata_path)
    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme(metrics, coverage, full_metadata))
    outputs.append(readme_path)
    return outputs
