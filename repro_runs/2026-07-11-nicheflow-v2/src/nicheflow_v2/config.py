"""Central configuration and output safety for NicheFlow-v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    run_root: Path
    frozen_root: Path
    master_seed: int = 20260711
    reference_fractions: tuple[float, ...] = (0.25, 0.50, 0.70)
    split_seeds: tuple[int, ...] = (20260711, 20260712, 20260713, 20260714, 20260715)
    split_types: tuple[str, ...] = ("random_stratified", "spatial_block")
    min_reference: int = 5
    min_evaluation: int = 5
    bootstrap_resamples: int = 10_000
    spatialprop_k: int = 8
    spatialprop_hidden_dim: int = 64
    spatialprop_latent_dim: int = 32
    spatialprop_epochs: int = 80

    @classmethod
    def default(cls) -> "ExperimentConfig":
        run_root = Path(__file__).resolve().parents[2]
        repo_root = run_root.parents[1]
        return cls(
            repo_root=repo_root,
            run_root=run_root,
            frozen_root=repo_root / "repro_runs" / "2026-06-22-full-repro",
        )


def assert_writable_output(path: Path, cfg: ExperimentConfig | None = None) -> None:
    cfg = cfg or ExperimentConfig.default()
    resolved = Path(path).resolve()
    frozen = cfg.frozen_root.resolve()
    run_root = cfg.run_root.resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError(f"Refusing to write into frozen experiment root: {resolved}")
    if resolved != run_root and run_root not in resolved.parents:
        raise ValueError(f"Output must stay inside NicheFlow-v2 run root: {resolved}")
