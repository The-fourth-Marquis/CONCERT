"""Load frozen Mainline A tasks into explicit v2 array contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .config import ExperimentConfig


def _decode_strings(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O", "U"}:
        return values.astype(str)
    return values


def _load_raw_h5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "x": np.asarray(handle["X"], dtype=np.float32),
            "pos": np.asarray(handle["pos"], dtype=np.float32).T,
            "perturbation": _decode_strings(np.asarray(handle["perturbation"])),
            "tissue": _decode_strings(np.asarray(handle["tissue"])),
            "gene": _decode_strings(np.asarray(handle["gene"])),
        }


def _load_selected(path: Path) -> np.ndarray:
    selected = np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int)
    if selected.size == 0 or np.any(selected <= 0):
        raise ValueError(f"Selected-cell file must contain positive 1-based indices: {path}")
    return selected - 1


def _local_label_fraction(pos: np.ndarray, tissue: np.ndarray, label: str, k: int = 12) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    tissue = np.asarray(tissue).astype(str)
    if pos.shape[0] == 0:
        return np.array([], dtype=np.float32)
    k_eff = max(1, min(int(k), pos.shape[0]))
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)
    neighbors = np.argsort(d2, axis=1)[:, :k_eff]
    return (tissue[neighbors] == label).mean(axis=1).astype(np.float32)


def _resolve_repo_path(value: object, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _shared_gene_indexer(
    source_genes: np.ndarray,
    target_genes: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared_path = cfg.repo_root / "datasets" / "perturb_map_data_all.h5"
    with h5py.File(shared_path, "r") as handle:
        shared = _decode_strings(np.asarray(handle["gene"]))
    source_lookup = {gene: idx for idx, gene in enumerate(source_genes.tolist())}
    target_lookup = {gene: idx for idx, gene in enumerate(target_genes.tolist())}
    missing_source = [gene for gene in shared if gene not in source_lookup]
    missing_target = [gene for gene in shared if gene not in target_lookup]
    if missing_source or missing_target:
        raise ValueError(
            "Shared genes missing from task data: "
            f"source={len(missing_source)}, target={len(missing_target)}"
        )
    return (
        shared,
        np.asarray([source_lookup[gene] for gene in shared], dtype=int),
        np.asarray([target_lookup[gene] for gene in shared], dtype=int),
    )


@dataclass(frozen=True)
class TaskPayload:
    task: dict[str, Any]
    task_id: str
    gene_names: np.ndarray
    query_indices: np.ndarray
    query_expr: np.ndarray
    query_pos: np.ndarray
    query_niche: np.ndarray
    query_tissue: np.ndarray
    control_indices: np.ndarray
    control_expr: np.ndarray
    control_pos: np.ndarray
    control_niche: np.ndarray
    control_tissue: np.ndarray
    target_indices: np.ndarray
    target_expr: np.ndarray
    target_pos: np.ndarray
    target_niche: np.ndarray
    target_tissue: np.ndarray
    is_cross_slide: bool

    def with_target_subset(self, indices: np.ndarray) -> "TaskPayload":
        subset = np.asarray(indices, dtype=int)
        return replace(
            self,
            target_indices=self.target_indices[subset],
            target_expr=self.target_expr[subset],
            target_pos=self.target_pos[subset],
            target_niche=self.target_niche[subset],
            target_tissue=self.target_tissue[subset],
        )

    def with_target_evaluation_expr(self, split: object, values: np.ndarray) -> "TaskPayload":
        evaluation_indices = np.asarray(getattr(split, "evaluation_indices"), dtype=int)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != self.target_expr[evaluation_indices].shape:
            raise ValueError("Replacement evaluation expression has the wrong shape")
        updated = self.target_expr.copy()
        updated[evaluation_indices] = values
        return replace(self, target_expr=updated)


def load_manifest(path: Path | None = None, cfg: ExperimentConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ExperimentConfig.default()
    manifest_path = path or cfg.frozen_root / "task_manifest_mainline_a.csv"
    if not manifest_path.exists():
        manifest_path = cfg.frozen_root / "paper_metric_benchmark" / "task_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if manifest["task_id"].duplicated().any():
        raise ValueError(f"Duplicate task IDs in {manifest_path}")
    return manifest


def load_task_payload(task: pd.Series | dict[str, Any], cfg: ExperimentConfig | None = None) -> TaskPayload:
    cfg = cfg or ExperimentConfig.default()
    row = dict(task)
    source_data = _load_raw_h5(_resolve_repo_path(row["source_data_file"], cfg.repo_root))
    target_data = _load_raw_h5(_resolve_repo_path(row["target_data_file"], cfg.repo_root))
    query_indices = _load_selected(_resolve_repo_path(row["selected_cell_file"], cfg.repo_root))
    target_mask = (
        (target_data["perturbation"] == str(row["target_perturbation"]))
        & (target_data["tissue"] == str(row["target_tissue"]))
    )
    target_indices = np.flatnonzero(target_mask)
    control_mask = (
        (source_data["perturbation"] == str(row["source_perturbation"]))
        & (source_data["tissue"] == str(row["target_tissue"]))
    )
    control_indices = np.flatnonzero(control_mask)
    if query_indices.size == 0 or target_indices.size == 0 or control_indices.size == 0:
        raise ValueError(f"Task {row['task_id']} has an empty query, control, or target pool")
    if not set(query_indices.tolist()).issubset(set(control_indices.tolist())):
        raise ValueError(f"Selected queries are not eligible controls for task {row['task_id']}")
    genes, source_gene_idx, target_gene_idx = _shared_gene_indexer(
        source_data["gene"], target_data["gene"], cfg
    )
    label = str(row["target_tissue"])
    source_niche = _local_label_fraction(source_data["pos"], source_data["tissue"], label)
    target_niche = _local_label_fraction(target_data["pos"], target_data["tissue"], label)
    return TaskPayload(
        task=row,
        task_id=str(row["task_id"]),
        gene_names=genes,
        query_indices=query_indices,
        query_expr=source_data["x"][query_indices][:, source_gene_idx].astype(np.float32),
        query_pos=source_data["pos"][query_indices].astype(np.float32),
        query_niche=source_niche[query_indices],
        query_tissue=source_data["tissue"][query_indices],
        control_indices=control_indices,
        control_expr=source_data["x"][control_indices][:, source_gene_idx].astype(np.float32),
        control_pos=source_data["pos"][control_indices].astype(np.float32),
        control_niche=source_niche[control_indices],
        control_tissue=source_data["tissue"][control_indices],
        target_indices=target_indices,
        target_expr=target_data["x"][target_indices][:, target_gene_idx].astype(np.float32),
        target_pos=target_data["pos"][target_indices].astype(np.float32),
        target_niche=target_niche[target_indices],
        target_tissue=target_data["tissue"][target_indices],
        is_cross_slide=str(row["source_sample"]) != str(row["target_sample"]),
    )
