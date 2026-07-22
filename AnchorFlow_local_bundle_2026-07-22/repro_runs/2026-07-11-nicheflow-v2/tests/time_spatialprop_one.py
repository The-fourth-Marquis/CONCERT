#!/usr/bin/env python3
"""Time one real 80-epoch reference-only SpatialProp-style configuration."""

from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.spatialprop import predict_spatialprop_reference_only
from nicheflow_v2.splits import split_target_pool

payload = load_task_payload(load_manifest().iloc[0])
split = split_target_pool(
    payload,
    reference_fraction=0.70,
    seed=20260711,
    split_type="random_stratified",
)
started = perf_counter()
result = predict_spatialprop_reference_only(payload, split, epochs=80)
elapsed = perf_counter() - started
print(
    f"task={payload.task_id} reference={split.reference_indices.size} "
    f"query={payload.query_expr.shape[0]} genes={payload.query_expr.shape[1]} "
    f"epochs=80 seconds={elapsed:.3f} backend={result.backend}"
)
