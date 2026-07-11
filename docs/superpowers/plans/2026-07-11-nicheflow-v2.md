# NicheFlow-v2 Evaluation and Method Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-controlled NicheFlow-v2 evaluation pipeline, reassess existing methods with held-out and mechanism metrics, then implement explicit residual matching and reliability gating under the frozen Mainline A task set.

**Architecture:** Keep `repro_runs/2026-06-22-full-repro/` read-only and place all new code and outputs in `repro_runs/2026-07-11-nicheflow-v2/`. A focused `nicheflow_v2` package separates configuration, data loading, target splitting, metrics, prediction methods, reassessment, and artifact writing. Every method receives reference-only inputs; evaluation arrays are passed only to metric functions.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, h5py, scikit-learn, matplotlib, unittest-style executable test scripts, RTK-prefixed shell commands.

---

## File Map

- `repro_runs/2026-07-11-nicheflow-v2/README.md`: experiment entry point, claim boundary, and command index.
- `repro_runs/2026-07-11-nicheflow-v2/configs/default.json`: frozen seeds, split fractions, thresholds, and method settings.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/config.py`: path constants, typed configuration, and frozen-root write guard.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/data.py`: manifest loading, H5 loading, gene alignment, control pool, and query/target payloads.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/splits.py`: deterministic random-stratified and spatial-block target splits.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/metrics.py`: E-distance, delta, DE, and spatial/context metrics.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/baselines.py`: reference-only reconstructions of existing methods.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/methods.py`: explicit residual matching, calibration ablations, and reliability gating.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/reassess.py`: method eligibility, coverage, split-loop evaluation, and ranking.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/artifacts.py`: CSV/JSON/Markdown writers and artifact validation.
- `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/report.py`: paired comparisons, clustered bootstrap, gate table, and research report.
- `repro_runs/2026-07-11-nicheflow-v2/run_reassessment.py`: CLI for existing methods.
- `repro_runs/2026-07-11-nicheflow-v2/run_residual_gating.py`: CLI for NicheFlow-v2 ablations.
- `repro_runs/2026-07-11-nicheflow-v2/tests/test_*.py`: executable unit, integration, and artifact tests.

### Task 1: Create the Isolated Experiment Skeleton and Configuration Guard

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/README.md`
- Create: `repro_runs/2026-07-11-nicheflow-v2/configs/default.json`
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/__init__.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/config.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_config.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.config import ExperimentConfig, assert_writable_output


def test_default_config_and_frozen_root_guard():
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


if __name__ == "__main__":
    test_default_config_and_frozen_root_guard()
    print("ok")
```

- [ ] **Step 2: Run the test and verify the import failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_config.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'nicheflow_v2'`.

- [ ] **Step 3: Implement configuration and the write guard**

```python
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
    resolved = path.resolve()
    if resolved == cfg.frozen_root.resolve() or cfg.frozen_root.resolve() in resolved.parents:
        raise ValueError(f"Refusing to write into frozen experiment root: {resolved}")
    if resolved != cfg.run_root.resolve() and cfg.run_root.resolve() not in resolved.parents:
        raise ValueError(f"Output must stay inside NicheFlow-v2 run root: {resolved}")
```

Write `default.json` with the same seeds, fractions, split types, minimum sizes, `bootstrap_resamples: 10000`, and no absolute paths. The README must state Mainline A observed-perturbation relocation and identify the 2026-06-22 root as read-only evidence.

- [ ] **Step 4: Run and verify the test**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_config.py`

Expected: `ok`.

- [ ] **Step 5: Commit the skeleton**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2
rtk git commit -m "feat: scaffold isolated NicheFlow v2 experiments"
```

### Task 2: Load Full Control/Query/Target Data and Split Target Pools

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/data.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/splits.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_data_and_splits.py`

- [ ] **Step 1: Write failing tests for payload semantics and leakage-safe splits**

```python
def test_payload_uses_full_eligible_control_pool_and_selected_queries():
    task = load_manifest().iloc[0]
    payload = load_task_payload(task)
    assert payload.query_expr.shape[0] == int(task["n_selected_spots"])
    assert payload.control_expr.shape[0] >= payload.query_expr.shape[0]
    assert set(payload.query_indices.tolist()).issubset(set(payload.control_indices.tolist()))
    assert payload.target_expr.shape[0] == int(task["n_target_spots"])


def test_splits_are_disjoint_deterministic_and_complete():
    payload = load_task_payload(load_manifest().iloc[0])
    n_target = payload.target_expr.shape[0]
    one = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    two = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    assert np.array_equal(one.reference_indices, two.reference_indices)
    assert not set(one.reference_indices) & set(one.evaluation_indices)
    assert sorted(np.r_[one.reference_indices, one.evaluation_indices].tolist()) == list(range(n_target))


def test_spatial_block_split_separates_centroid_regions():
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="spatial_block")
    assert split.split_type == "spatial_block"
    assert split.reference_indices.size >= 5
    assert split.evaluation_indices.size >= 5
```

- [ ] **Step 2: Run the tests and verify missing-module failures**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_data_and_splits.py`

Expected: FAIL because `nicheflow_v2.data` and `nicheflow_v2.splits` do not exist.

- [ ] **Step 3: Implement typed payload loading**

Define `TaskPayload` with `task_id`, aligned gene names, query expression/position/niche/index arrays, full eligible control expression/position/niche/index arrays, and observed target expression/position/niche/index arrays. Reuse the frozen manifest and H5 schema, but copy the small loading and gene-alignment functions into the v2 package so old runners are not imported for writes or global paths.

```python
@dataclass(frozen=True)
class TaskPayload:
    task_id: str
    gene_names: np.ndarray
    query_indices: np.ndarray
    query_expr: np.ndarray
    query_pos: np.ndarray
    query_niche: np.ndarray
    control_indices: np.ndarray
    control_expr: np.ndarray
    control_pos: np.ndarray
    control_niche: np.ndarray
    target_indices: np.ndarray
    target_expr: np.ndarray
    target_pos: np.ndarray
    target_niche: np.ndarray
    is_cross_slide: bool

    def with_target_evaluation_expr(self, split: "TargetSplit", values: np.ndarray) -> "TaskPayload":
        updated = self.target_expr.copy()
        updated[split.evaluation_indices] = values
        return replace(self, target_expr=updated)
```

The eligible control mask is exactly:

```python
control_mask = (
    (source_data["perturbation"] == str(task["source_perturbation"]))
    & (source_data["tissue"] == str(task["target_tissue"]))
)
```

Validate that every selected query index belongs to this mask.

- [ ] **Step 4: Implement deterministic split algorithms**

`random_stratified` bins target niches at 0.5 and spatial coordinates into median-split quadrants, shuffles within combined strata, then rebalances to the requested total while retaining at least five rows on each side. `spatial_block` assigns target spots to four median-split quadrants, chooses evaluation blocks by seeded order until the requested evaluation size is reached, and records the selected block IDs. Return `insufficient_target_pool` rather than weakening minimum sizes.

```python
@dataclass(frozen=True)
class TargetSplit:
    reference_indices: np.ndarray
    evaluation_indices: np.ndarray
    reference_fraction: float
    seed: int
    split_type: str
    status: str
    reason: str = ""

    def validate(self, n_target: int) -> None:
        ref = set(self.reference_indices.tolist())
        eva = set(self.evaluation_indices.tolist())
        if ref & eva:
            raise ValueError("target leakage: reference/evaluation overlap")
        if ref | eva != set(range(n_target)):
            raise ValueError("target split does not cover observed pool")
```

- [ ] **Step 5: Run split tests**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_data_and_splits.py`

Expected: `ok` with real manifest loading and both split policies passing.

- [ ] **Step 6: Commit data and split contracts**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/data.py repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/splits.py repro_runs/2026-07-11-nicheflow-v2/tests/test_data_and_splits.py
rtk git commit -m "feat: add leakage-safe target pool splits"
```

### Task 3: Implement Distribution, Delta, and DE Metrics

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/metrics.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_metrics.py`

- [ ] **Step 1: Write tests that expose E-distance insufficiency and direction errors**

```python
def test_correct_delta_beats_reversed_delta():
    source = np.zeros((4, 6), dtype=np.float32)
    target = np.tile(np.array([3, 2, 1, -1, -2, -3], dtype=np.float32), (5, 1))
    correct = np.tile(target.mean(0), (4, 1))
    reversed_pred = -correct
    good = perturbation_effect_metrics(correct, source, target)
    bad = perturbation_effect_metrics(reversed_pred, source, target)
    assert good["delta_pcc"] > bad["delta_pcc"]
    assert good["delta_mae"] < bad["delta_mae"]


def test_de_metrics_report_opposite_direction():
    reference_delta = np.array([5, 4, 3, 2, 1, -1, -2, -3, -4, -5], dtype=float)
    true_delta = reference_delta.copy()
    reversed_delta = -true_delta
    metrics = de_direction_metrics(reversed_delta, true_delta, reference_delta, top_k=10)
    assert metrics["direction_accuracy"] == 0.0
    assert metrics["opposite_direction_fraction"] == 1.0


def test_undefined_correlation_is_nan_with_reason():
    value, reason = safe_correlation(np.ones(4), np.ones(4), method="pearson")
    assert np.isnan(value)
    assert reason == "constant_input"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_metrics.py`

Expected: FAIL because `nicheflow_v2.metrics` is missing.

- [ ] **Step 3: Implement the metric API**

Implement `energy_distance`, `safe_correlation`, `perturbation_effect_metrics`, `select_de_indices`, `de_direction_metrics`, and `distribution_metrics`. DE selection uses only `reference_target_mean - matched_control_mean`; evaluation truth uses `evaluation_target_mean - matched_control_mean`.

```python
def perturbation_effect_metrics(pred, source, evaluation_target):
    predicted_delta = np.asarray(pred).mean(0) - np.asarray(source).mean(0)
    true_delta = np.asarray(evaluation_target).mean(0) - np.asarray(source).mean(0)
    pcc, pcc_reason = safe_correlation(predicted_delta, true_delta, method="pearson")
    spearman, spearman_reason = safe_correlation(predicted_delta, true_delta, method="spearman")
    error = predicted_delta - true_delta
    return {
        "delta_mae": float(np.mean(np.abs(error))),
        "delta_rmse": float(np.sqrt(np.mean(error**2))),
        "delta_pcc": pcc,
        "delta_pcc_reason": pcc_reason,
        "delta_spearman": spearman,
        "delta_spearman_reason": spearman_reason,
        "delta_r2": safe_r2(predicted_delta, true_delta),
    }
```

Preserve NaN and a reason string for constant inputs, fewer than two valid values, and non-finite input.

- [ ] **Step 4: Run metric tests**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_metrics.py`

Expected: `ok`.

- [ ] **Step 5: Commit metric primitives**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/metrics.py repro_runs/2026-07-11-nicheflow-v2/tests/test_metrics.py
rtk git commit -m "feat: add held-out perturbation metrics"
```

### Task 4: Port Versioned Spatial/Context Diagnostics

**Files:**
- Modify: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/metrics.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_spatial_metrics.py`

- [ ] **Step 1: Write failing tests for spatial metric directionality**

Copy the existing synthetic tests for boundary leakage, source individuality, and distance-response decay, then add niche-wise delta and cross-niche drop checks:

```python
def test_cross_niche_drop_is_in_niche_minus_cross_niche():
    rows = pd.DataFrame({
        "task_type": ["niche_core", "cross_niche_core_to_border"],
        "delta_pcc": [0.8, 0.3],
    })
    assert np.isclose(cross_niche_drop(rows), 0.5)
```

- [ ] **Step 2: Run and verify failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_spatial_metrics.py`

Expected: FAIL for missing v2 spatial functions.

- [ ] **Step 3: Implement versioned diagnostics**

Port the formulas into `metrics.py`, change constant-input correlation behavior from zero to NaN plus reason, and add `METRIC_REGISTRY` entries containing metric name, version `v2.0`, direction, scope, and causal-proof disclaimer.

- [ ] **Step 4: Run spatial and core metric tests**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_metrics.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_spatial_metrics.py
```

Expected: both print `ok`.

- [ ] **Step 5: Commit spatial metrics**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/metrics.py repro_runs/2026-07-11-nicheflow-v2/tests/test_spatial_metrics.py
rtk git commit -m "feat: version NicheFlow spatial diagnostics"
```

### Task 5: Implement Reference-Only Existing-Method Baselines

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/baselines.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_baselines.py`

- [ ] **Step 1: Write failing reference-only baseline tests**

```python
def test_baselines_ignore_evaluation_expression():
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    methods = (
        "source_identity_l0", "target_mean_l1", "knn10_l1",
        "cauchy_l1", "legacy_nicheflow_lite_l3", "legacy_v1_fixed_80_20_l3",
    )
    for method in methods:
        first = predict_existing_method(method, payload, split)
        altered = payload.with_target_evaluation_expr(
            split, payload.target_expr[split.evaluation_indices] + 1000.0
        )
        second = predict_existing_method(method, altered, split)
        assert np.allclose(first.prediction, second.prediction)


def test_reference_only_calibration_uses_no_evaluation_rows():
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    pred = np.zeros_like(payload.query_expr)
    first = calibrate_to_reference(pred, payload.target_expr[split.reference_indices])
    changed = payload.target_expr.copy()
    changed[split.evaluation_indices] += 1000.0
    second = calibrate_to_reference(pred, changed[split.reference_indices])
    assert np.allclose(first, second)
```

- [ ] **Step 2: Run and verify failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_baselines.py`

Expected: FAIL because `nicheflow_v2.baselines` does not exist.

- [ ] **Step 3: Implement access-matched existing methods**

Implement:

- L0 source identity;
- L1 reference target mean, kNN-5, kNN-10, Gaussian, Cauchy, graph diffusion, and linear covariate ridge;
- a legacy-compatible NicheFlow-lite residual predictor using the original query-only control anchoring, but reference target rows only;
- A2.3/v1 fixed 0.80/0.20 ensembles with reference-only calibration and the frozen task-type anchor policy;
- a `PredictionResult` containing prediction, method, access level, calibration flag, and component metadata.

```python
@dataclass(frozen=True)
class PredictionResult:
    prediction: np.ndarray
    method: str
    access_level: str
    calibration: bool
    uses_target_reference: bool
    components: dict[str, object]


def reference_target_arrays(payload: TaskPayload, split: TargetSplit) -> tuple[np.ndarray, np.ndarray]:
    return (
        payload.target_expr[split.reference_indices],
        payload.target_pos[split.reference_indices],
    )
```

All target-based functions call `reference_target_arrays`; no function accepts evaluation expression as a separate parameter.

- [ ] **Step 4: Run baseline tests**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_baselines.py`

Expected: `ok`.

- [ ] **Step 5: Commit reference-only baselines**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/baselines.py repro_runs/2026-07-11-nicheflow-v2/tests/test_baselines.py
rtk git commit -m "feat: add reference-only comparison methods"
```

### Task 6: Build the Reassessment Loop and Coverage Audit

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/reassess.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/run_reassessment.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_reassess.py`

- [ ] **Step 1: Write failing reassessment tests**

```python
def test_every_metric_row_has_protocol_metadata():
    cfg = replace(
        ExperimentConfig.default(),
        reference_fractions=(0.70,),
        split_seeds=(20260711,),
        split_types=("random_stratified",),
    )
    rows, coverage = reassess_manifest(load_manifest().head(1), cfg)
    required = {
        "task_id", "method", "access_level", "reference_fraction", "split_seed",
        "split_type", "calibration", "uses_target_reference", "heldout_edistance",
    }
    assert required.issubset(rows.columns)
    assert not rows[list(required)].isna().any().any()
    assert set(coverage["eligibility_status"]) <= {"exact", "approximate", "unavailable", "not_eligible_for_heldout_primary"}


def test_fixed_fullpool_concert_is_not_eligible_for_heldout_primary():
    cfg = replace(
        ExperimentConfig.default(),
        reference_fractions=(0.70,),
        split_seeds=(20260711,),
        split_types=("random_stratified",),
    )
    _, coverage = reassess_manifest(load_manifest().head(1), cfg)
    concert = coverage[coverage["model_family"] == "available_concert"]
    assert set(concert["eligibility_status"]) == {"not_eligible_for_heldout_primary"}
```

- [ ] **Step 2: Run and verify failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_reassess.py`

Expected: FAIL because reassessment orchestration is missing.

- [ ] **Step 3: Implement method registry and eligibility**

Register the A2.1, NicheFlow-lite-equivalent, and A2.3/v1 methods from `baselines.py`. Mark available CONCERT fixed outputs as held-out-ineligible. Adapt SpatialProp-style training by passing only reference target expression and calibrating only to the reference subset; cache results by task, fraction, seed, split type, and k. Smoke mode excludes expensive GNN retraining, records eligibility as `unavailable`, and sets reason to `excluded_expensive_smoke`; the full release command must include it.

- [ ] **Step 4: Implement split-loop evaluation**

For each eligible task/fraction/seed/split/method, compute prediction from reference-only inputs, then pass evaluation targets to metric functions. Write long-format rows and a separate coverage row even when prediction fails or pool size is insufficient. Never rank an ineligible row.

- [ ] **Step 5: Run reassessment tests**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_reassess.py`

Expected: `ok` on one task, one seed, 0.70 random-stratified split.

- [ ] **Step 6: Commit orchestration**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/reassess.py repro_runs/2026-07-11-nicheflow-v2/run_reassessment.py repro_runs/2026-07-11-nicheflow-v2/tests/test_reassess.py
rtk git commit -m "feat: reassess methods under held-out targets"
```

### Task 7: Write and Validate Research Artifacts

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/artifacts.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/report.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_artifacts.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_required_artifacts_and_ranking_eligibility(tmp_path):
    manifest = pd.DataFrame([{"task_id": "t1", "split_seed": 20260711, "split_type": "random_stratified"}])
    metrics = pd.DataFrame([{
        "task_id": "t1", "method": "source_identity_l0", "access_level": "L0",
        "reference_fraction": 0.70, "split_seed": 20260711, "split_type": "random_stratified",
        "calibration": False, "uses_target_reference": False, "heldout_edistance": 2.0,
        "eligible_for_primary": True,
    }])
    coverage = pd.DataFrame([{
        "task_id": "t1", "model_family": "a2_baseline", "method": "source_identity_l0",
        "eligibility_status": "exact", "reason": "", "eligible_for_primary": True,
    }])
    metadata = {"claim_scope": "observed perturbation relocation", "smoke": True}
    cfg = replace(ExperimentConfig.default(), run_root=tmp_path)
    outputs = write_artifacts(metrics, coverage, manifest, tmp_path, metadata, config=cfg)
    required = {"task_manifest.csv", "metrics.csv", "method_ranking.csv", "coverage.csv", "metadata.json", "README.md"}
    assert required.issubset({p.name for p in outputs})
    ranking = pd.read_csv(tmp_path / "method_ranking.csv")
    assert ranking["eligible_for_primary"].all()
    assert "held-out" in (tmp_path / "README.md").read_text().lower()


def test_clustered_bootstrap_is_deterministic():
    comparison = pd.DataFrame({
        "task_id": ["a", "a", "b", "b"],
        "delta_edistance": [-2.0, -1.0, 1.0, -0.5],
    })
    one = clustered_bootstrap(comparison, value_col="delta_edistance", seed=20260711, n_resamples=1000)
    two = clustered_bootstrap(comparison, value_col="delta_edistance", seed=20260711, n_resamples=1000)
    assert one == two
```

- [ ] **Step 2: Run and verify failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_artifacts.py`

Expected: FAIL for missing artifact/report modules.

- [ ] **Step 3: Implement writers and validators**

Use `assert_writable_output` before directory creation. Validate required metric columns, disjoint split metadata, unique row keys, ranking eligibility, finite-or-explained metric values, and repository-relative paths. `metadata.json` embeds `METRIC_REGISTRY`, config, git commit, source artifact paths, and claim scope. Create `verify_full_outputs.py` here with modes `reassessment-fast`, `residual-gating`, and `complete`; modes whose artifacts do not exist yet must fail with an explicit missing-artifact message.

- [ ] **Step 4: Implement comparisons and gates**

Use task-clustered bootstrap with 10,000 resamples for the full report. Generate `gate_summary.csv` with every quantitative criterion from the approved design, its observed value, threshold, pass/fail status, and coverage. Do not collapse metrics into a composite score.

- [ ] **Step 5: Run artifact tests**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_artifacts.py`

Expected: `ok`.

- [ ] **Step 6: Commit artifacts and report logic**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/artifacts.py repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/report.py repro_runs/2026-07-11-nicheflow-v2/tests/test_artifacts.py repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py
rtk git commit -m "feat: add NicheFlow v2 evidence artifacts"
```

### Task 8: Run Existing-Method Reassessment Before Selecting the New Module

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/results/evaluation_contract/`
- Create: `repro_runs/2026-07-11-nicheflow-v2/results/existing_method_reassessment/`
- Create: `repro_runs/2026-07-11-nicheflow-v2/reports/existing_method_reassessment.md`

- [ ] **Step 1: Run the fast full-manifest evaluator without expensive GNN retraining**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/run_reassessment.py --config repro_runs/2026-07-11-nicheflow-v2/configs/default.json --exclude-expensive
```

Expected: 37 tasks attempted across 3 fractions, 5 seeds, and both split types; insufficient pools and fixed-output ineligibility appear in `coverage.csv`, not as crashes.

- [ ] **Step 2: Validate fast artifacts**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py --mode reassessment-fast`

Expected: `ok`, no reference/evaluation overlap, complete metadata, and no fixed full-pool outputs in the held-out ranking.

- [ ] **Step 3: Run expensive SpatialProp-style reassessment with cache**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/run_reassessment.py --config repro_runs/2026-07-11-nicheflow-v2/configs/default.json --only-expensive --device cpu --resume-cache
```

Expected: cached reference-only GNN predictions for every eligible task-split configuration; failures are recorded with exception type and message in coverage.

- [ ] **Step 4: Merge and report reassessment evidence**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/run_reassessment.py --merge-only`

Expected: `existing_method_reassessment/metrics.csv`, `coverage.csv`, `method_ranking.csv`, `metadata.json`, `README.md`, and the Markdown report. The report answers whether v1's full-pool ranking survives held-out evaluation and identifies which metric axis motivates the first module.

- [ ] **Step 5: Commit reproducible tabular summaries and report**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/results/existing_method_reassessment repro_runs/2026-07-11-nicheflow-v2/reports/existing_method_reassessment.md
rtk git commit -m "results: reassess NicheFlow under held-out targets"
```

Do not commit large prediction caches or generated binary figures.

### Task 9: Implement Explicit Residual Matching and Reliability Gating

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/methods.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_methods.py`

- [ ] **Step 1: Write failing method-contract and leakage tests**

```python
def test_residual_contract_is_explicit():
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    delta = context_matched_delta(payload, split)
    pred = residual_prediction(payload, delta)
    assert delta.shape == payload.query_expr.shape
    assert np.allclose(pred, payload.query_expr + delta)
    assert not np.allclose(delta, pred)


def test_evaluation_expression_cannot_change_predictions():
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    first = predict_method("nicheflow_v2_gated", payload, split)
    altered = payload.with_target_evaluation_expr(split, payload.target_expr[split.evaluation_indices] + 1000.0)
    second = predict_method("nicheflow_v2_gated", altered, split)
    assert np.allclose(first.prediction, second.prediction)


def test_gate_is_bounded_and_not_globally_constant():
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(payload, reference_fraction=0.70, seed=20260711, split_type="random_stratified")
    result = predict_method("nicheflow_v2_gated", payload, split)
    assert np.all((result.gate >= 0.05) & (result.gate <= 0.80))
    assert np.std(result.gate) > 0
```

- [ ] **Step 2: Run and verify failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_methods.py`

Expected: FAIL because `nicheflow_v2.methods` is missing.

- [ ] **Step 3: Implement access-matched prediction inputs**

Prediction methods receive `TaskPayload` and `TargetSplit`; they slice reference rows internally and never receive evaluation expressions as an argument. Implement L0 source identity; L1 target mean, kNN-5, kNN-10, Gaussian, Cauchy, graph diffusion, and reference-only linear covariate ridge; L2 reference-only calibration variants; L3 fixed ensemble; and L4 gated ensemble.

- [ ] **Step 4: Implement perturbation-invariant soft matching and pure delta**

Build context matrices from slide-normalized coordinates, niche fraction, boundary distance `abs(niche - 0.5)`, tissue/region agreement, and cross-slide indicator. Match each reference target to the full eligible control pool to form residual examples, then match query contexts to reference-target contexts to transport those residuals.

```python
target_control_weights = softmax(-context_distance(target_reference, control_pool) / 0.75)
control_anchor = target_control_weights @ payload.control_expr
residual_examples = target_reference_expr - control_anchor
query_target_weights = softmax(-context_distance(query, target_reference) / 0.75)
delta_context = query_target_weights @ residual_examples
```

- [ ] **Step 5: Implement deterministic reliability gating**

For every query, compute normalized matching entropy, nearest scaled match distance, local reference density, boundary uncertainty, cross-slide penalty, and normalized anchor/residual disagreement. Define:

```python
quality = (
    (1.0 - normalized_entropy)
    * np.exp(-nearest_distance)
    * np.exp(-disagreement)
    * (1.0 - 0.25 * boundary_uncertainty)
    * np.where(is_cross_slide, 0.75, 1.0)
)
gate = np.clip(0.05 + 0.75 * quality, 0.05, 0.80)
prediction = (1.0 - gate[:, None]) * anchor + gate[:, None] * residual_pred
```

Return gate and component diagnostics with every prediction. Calibration uses reference targets only and is a separate method suffix `_calibrated`.

- [ ] **Step 6: Run method tests**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_methods.py`

Expected: `ok`, including the evaluation-expression mutation test.

- [ ] **Step 7: Commit methods**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2/src/nicheflow_v2/methods.py repro_runs/2026-07-11-nicheflow-v2/tests/test_methods.py
rtk git commit -m "feat: add explicit residual and reliability gating"
```

### Task 10: Run Residual/Gating Ablations and Complete the v2 Report

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/run_residual_gating.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/test_integration.py`
- Modify: `repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/results/residual_gating/`
- Create: `repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_report.md`

- [ ] **Step 1: Write a failing end-to-end smoke test**

```python
def test_one_task_end_to_end(tmp_path):
    result = run_residual_gating(
        manifest=load_manifest().head(1),
        output_dir=tmp_path,
        reference_fractions=(0.70,),
        split_seeds=(20260711,),
        split_types=("random_stratified",),
        config=replace(ExperimentConfig.default(), run_root=tmp_path),
    )
    assert set(result.metrics["method"]) >= {
        "anchor_only", "residual_prediction_only", "fixed_80_20",
        "nicheflow_v2_gated", "nicheflow_v2_gated_calibrated",
    }
    assert result.metrics["heldout_edistance"].notna().all()
    assert (tmp_path / "gate_summary.csv").exists()
```

- [ ] **Step 2: Run and verify failure**

Run: `rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_integration.py`

Expected: FAIL until the residual/gating runner and verifier are connected.

- [ ] **Step 3: Implement the runner and verifier**

The runner evaluates required calibrated/uncalibrated ablations for every feasible split, writes component diagnostics including gate mean/std, matching entropy, match distance, disagreement, and calibration status, and calls the shared artifact writer. The verifier independently reads outputs and asserts all design-required files, methods, fractions, seeds, split types, eligibility flags, and gate criteria are represented.

- [ ] **Step 4: Run all unit and integration tests**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_config.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_data_and_splits.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_metrics.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_spatial_metrics.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_baselines.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_methods.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_reassess.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_artifacts.py
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/test_integration.py
```

Expected: every command prints `ok`.

- [ ] **Step 5: Run the full residual/gating benchmark**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/run_residual_gating.py --config repro_runs/2026-07-11-nicheflow-v2/configs/default.json
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py --mode residual-gating
```

Expected: 37 tasks attempted across all configured splits, `ok` from the verifier, and explicit pass/fail rows for every success gate.

- [ ] **Step 6: Write the final report and graph/flow decision**

The report must include full-pool versus held-out conclusions, method/access coverage, delta and DE results, spatial diagnostics, calibration ablation, reduced-reference sensitivity, gate variability, failures, and claim limits. End with one evidence-based decision: `advance_to_graph_flow` or `do_not_advance_to_graph_flow`, derived from `gate_summary.csv` without overriding failed gates narratively.

- [ ] **Step 7: Run final completion audit**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py --mode complete
rtk git status --short repro_runs/2026-07-11-nicheflow-v2 docs/superpowers/specs/2026-07-11-nicheflow-v2-design.md docs/superpowers/plans/2026-07-11-nicheflow-v2.md
```

Expected: verifier prints `ok`; status lists only intentional v2 files and no changes under the frozen 2026-06-22 root.

- [ ] **Step 8: Commit the complete v2 evidence package**

```bash
rtk git add -f repro_runs/2026-07-11-nicheflow-v2
rtk git commit -m "feat: complete NicheFlow v2 held-out evaluation"
```

