"""Statistical comparisons and evidence-gate summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def clustered_bootstrap(
    comparison: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str = "task_id",
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    source = comparison[[cluster_col, value_col]].dropna().copy()
    clusters = source[cluster_col].drop_duplicates().to_numpy()
    if clusters.size == 0:
        raise ValueError("No valid clusters for bootstrap")
    rng = np.random.default_rng(seed)
    samples = np.empty(int(n_resamples), dtype=np.float64)
    grouped = {
        cluster: source.loc[source[cluster_col] == cluster, value_col].to_numpy()
        for cluster in clusters
    }
    for index in range(int(n_resamples)):
        chosen = rng.choice(clusters, size=clusters.size, replace=True)
        values = np.concatenate([grouped[cluster] for cluster in chosen])
        samples[index] = values.mean()
    return {
        "mean": float(source[value_col].mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_clusters": int(clusters.size),
        "n_rows": int(source.shape[0]),
        "seed": int(seed),
        "n_resamples": int(n_resamples),
    }
