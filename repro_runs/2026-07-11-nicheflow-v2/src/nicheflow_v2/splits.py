"""Deterministic, disjoint target-reference/evaluation splitting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .data import TaskPayload


@dataclass(frozen=True)
class TargetSplit:
    reference_indices: np.ndarray
    evaluation_indices: np.ndarray
    reference_fraction: float
    seed: int
    split_type: str
    status: str
    reason: str = ""
    evaluation_blocks: tuple[int, ...] = ()

    def validate(self, n_target: int) -> None:
        if self.status != "ok":
            return
        reference = set(self.reference_indices.tolist())
        evaluation = set(self.evaluation_indices.tolist())
        if reference & evaluation:
            raise ValueError("Target leakage: reference/evaluation overlap")
        if reference | evaluation != set(range(int(n_target))):
            raise ValueError("Target split does not cover the observed target pool")


def _empty_split(reference_fraction: float, seed: int, split_type: str, reason: str) -> TargetSplit:
    empty = np.array([], dtype=int)
    return TargetSplit(
        reference_indices=empty,
        evaluation_indices=empty,
        reference_fraction=float(reference_fraction),
        seed=int(seed),
        split_type=split_type,
        status="insufficient_target_pool",
        reason=reason,
    )


def _spatial_blocks(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    x_high = pos[:, 0] >= np.median(pos[:, 0])
    y_high = pos[:, 1] >= np.median(pos[:, 1])
    return (2 * x_high.astype(int) + y_high.astype(int)).astype(int)


def _target_reference_size(n: int, fraction: float, min_reference: int, min_evaluation: int) -> int:
    requested = int(round(float(fraction) * n))
    return max(min_reference, min(requested, n - min_evaluation))


def _random_stratified_indices(
    payload: TaskPayload,
    n_reference: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks = _spatial_blocks(payload.target_pos)
    niche_side = (payload.target_niche >= 0.5).astype(int)
    strata = 10 * niche_side + blocks
    reference: list[int] = []
    evaluation: list[int] = []
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        indices = indices[rng.permutation(indices.size)]
        take = int(round(indices.size * n_reference / payload.target_expr.shape[0]))
        take = min(max(take, 0), indices.size)
        reference.extend(indices[:take].tolist())
        evaluation.extend(indices[take:].tolist())
    reference = list(dict.fromkeys(reference))
    evaluation = list(dict.fromkeys(evaluation))
    if len(reference) < n_reference:
        candidates = np.asarray(evaluation, dtype=int)
        chosen = candidates[rng.permutation(candidates.size)[: n_reference - len(reference)]].tolist()
        reference.extend(chosen)
        chosen_set = set(chosen)
        evaluation = [idx for idx in evaluation if idx not in chosen_set]
    elif len(reference) > n_reference:
        candidates = np.asarray(reference, dtype=int)
        moved = candidates[rng.permutation(candidates.size)[: len(reference) - n_reference]].tolist()
        moved_set = set(moved)
        reference = [idx for idx in reference if idx not in moved_set]
        evaluation.extend(moved)
    return np.sort(np.asarray(reference, dtype=int)), np.sort(np.asarray(evaluation, dtype=int))


def _spatial_block_indices(
    payload: TaskPayload,
    n_reference: int,
    seed: int,
    min_reference: int,
    min_evaluation: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]] | None:
    blocks = _spatial_blocks(payload.target_pos)
    present = np.unique(blocks).tolist()
    desired_evaluation = payload.target_expr.shape[0] - n_reference
    rng = np.random.default_rng(seed)
    tie_rank = {block: rank for rank, block in enumerate(rng.permutation(present).tolist())}
    candidates: list[tuple[int, int, tuple[int, ...], np.ndarray]] = []
    for size in range(1, len(present)):
        for selected in combinations(present, size):
            evaluation = np.flatnonzero(np.isin(blocks, selected))
            reference_size = payload.target_expr.shape[0] - evaluation.size
            if evaluation.size < min_evaluation or reference_size < min_reference:
                continue
            tie = sum(tie_rank[block] for block in selected)
            candidates.append((abs(evaluation.size - desired_evaluation), tie, tuple(sorted(selected)), evaluation))
    if not candidates:
        return None
    _, _, selected_blocks, evaluation = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    reference = np.setdiff1d(np.arange(payload.target_expr.shape[0]), evaluation, assume_unique=True)
    return reference.astype(int), evaluation.astype(int), selected_blocks


def split_target_pool(
    payload: TaskPayload,
    *,
    reference_fraction: float,
    seed: int,
    split_type: str,
    min_reference: int = 5,
    min_evaluation: int = 5,
) -> TargetSplit:
    n_target = int(payload.target_expr.shape[0])
    if not 0.0 < float(reference_fraction) < 1.0:
        raise ValueError("reference_fraction must be between zero and one")
    if n_target < min_reference + min_evaluation:
        return _empty_split(
            reference_fraction,
            seed,
            split_type,
            f"Need at least {min_reference + min_evaluation} target spots; found {n_target}",
        )
    n_reference = _target_reference_size(n_target, reference_fraction, min_reference, min_evaluation)
    evaluation_blocks: tuple[int, ...] = ()
    if split_type == "random_stratified":
        reference, evaluation = _random_stratified_indices(payload, n_reference, seed)
    elif split_type == "spatial_block":
        result = _spatial_block_indices(payload, n_reference, seed, min_reference, min_evaluation)
        if result is None:
            return _empty_split(reference_fraction, seed, split_type, "No feasible disjoint spatial block split")
        reference, evaluation, evaluation_blocks = result
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    split = TargetSplit(
        reference_indices=reference,
        evaluation_indices=evaluation,
        reference_fraction=float(reference_fraction),
        seed=int(seed),
        split_type=split_type,
        status="ok",
        evaluation_blocks=evaluation_blocks,
    )
    split.validate(n_target)
    if reference.size < min_reference or evaluation.size < min_evaluation:
        raise AssertionError("Split violated configured minimum pool sizes")
    return split
