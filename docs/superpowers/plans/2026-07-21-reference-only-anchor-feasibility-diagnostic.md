# Reference-Only Anchor Feasibility Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, without scoring outer-evaluation rows, whether the current support-aware anchor direction is intrinsically inadequate or only becomes useful beyond the frozen 5% global distortion budget.

**Architecture:** Reuse the frozen first-task outer reference pools and create the same-type pseudo-fit/pseudo-validation splits inside them. Evaluate `beta_max=1` at predeclared global distortion budgets `0, 0.05, 0.10, 0.20, 0.40` plus the raw unprojected endpoint, compare every point with fixed 80/20 on pseudo-validation only, and emit a conservative per-split and aggregate diagnosis. This is a diagnostic Pareto frontier, not model selection; it must not change the prior outer decision.

**Tech Stack:** Python, NumPy, existing `nicheflow_v2` task/split/baseline utilities, existing bounded-anchor operator and metrics, JSON/CSV/Markdown artifacts.

---

### Task 1: Freeze diagnosis contracts

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_feasibility_diagnostic.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/diagnostics.py`

- [ ] **Step 1: Write failing tests for the fixed grid and decision labels**

Test that the public diagnostic grid is exactly `(0.0, 0.05, 0.10, 0.20, 0.40)` and that per-split classification obeys these predeclared rules:

```python
assert diagnose_frontier(frontier_location_pass=False, minimum_location_budget=None) == "anchor_direction_inadequate"
assert diagnose_frontier(frontier_location_pass=True, minimum_location_budget=0.20) == "large_global_distortion_required"
assert diagnose_frontier(frontier_location_pass=True, minimum_location_budget=0.05) == "small_global_budget_can_locate"
```

Also test that two inadequate splits aggregate to `abandon_current_anchor_family`, two large-budget splits aggregate to `redesign_geometry_by_niche`, two small-budget splits aggregate to `investigate_projection_or_selection`, and mixed splits aggregate conservatively to `inconclusive_across_splits_do_not_advance`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```text
rtk uv run python repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_feasibility_diagnostic.py
```

Expected: fail because `bounded_anchor.diagnostics` does not exist.

- [ ] **Step 3: Implement the minimal pure decision functions**

Create `diagnostics.py` with:

```python
DISTORTION_BUDGETS = (0.0, 0.05, 0.10, 0.20, 0.40)

def diagnose_frontier(*, frontier_location_pass: bool, minimum_location_budget: float | None) -> str:
    ...

def aggregate_diagnoses(per_split: dict[str, str]) -> str:
    ...
```

Use a numerical tolerance of `1e-6` around the 5% boundary. Never convert a mixed two-split result into a positive structural conclusion.

- [ ] **Step 4: Run the test and verify GREEN**

Run the same command and expect `ok`.

### Task 2: Build the reference-only frontier evaluator

**Files:**
- Modify: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_feasibility_diagnostic.py`
- Modify: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/diagnostics.py`

- [ ] **Step 1: Add a failing synthetic evaluator test**

Construct a small synthetic payload and outer/pseudo split. Assert that the evaluator:

- exposes one row for every fixed budget plus `raw_unprojected` and fixed 80/20;
- uses only the outer-reference subset and pseudo-validation target rows;
- reports E-distance, Delta PCC, boundary, individuality, measured distortion, and deletion stability;
- records location gates separately from the full mechanism gates;
- returns byte-equivalent scientific rows after extreme mutation of outer-evaluation expression.

- [ ] **Step 2: Run the test and verify RED**

Expected: fail because `evaluate_reference_only_frontier` is missing.

- [ ] **Step 3: Implement the evaluator**

For each budget call the existing operator with `beta_max=1`. Use `epsilon=budget` for projected points and the operator's `unprojected_prediction` for the raw endpoint. Compute fixed 80/20 and leave-one-reference-out stability on pseudo-fit only. Define location gates as:

```text
heldout_edistance <= fixed_80_20_heldout_edistance
boundary_leakage <= 1.05 * fixed_80_20_boundary_leakage
```

Record the existing Delta, individuality, and deletion noninferiority checks but do not use them to relabel a failed location direction as successful. Do not select a production configuration.

- [ ] **Step 4: Run the focused test and all existing tests**

Run:

```text
rtk uv run python repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_feasibility_diagnostic.py
rtk uv run python repro_runs/2026-07-21-bounded-target-anchor-base/tests/run_all.py
```

Expected: all test files pass.

### Task 3: Add the real first-task two-split diagnostic runner

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/run_reference_only_feasibility_diagnostic.py`
- Modify: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_feasibility_diagnostic.py`
- Modify: `repro_runs/2026-07-21-bounded-target-anchor-base/README.md`

- [ ] **Step 1: Add a failing runner artifact test**

Assert that a synthetic run writes `summary.json` and `metrics.csv`, includes both required split types, labels `diagnostic_only=true`, `uses_outer_evaluation_target=false`, includes a real mutation guard, and contains no outer-evaluation metric rows or promotion decision.

- [ ] **Step 2: Run the test and verify RED**

Expected: fail because the runner does not exist.

- [ ] **Step 3: Implement the runner and README command**

Use task 0, seed `20260711`, outer reference fraction `0.70`, bandwidth `0.75`, support prior `4.0`, and the frozen budget grid. Recreate the two outer splits only to identify their reference pools, generate the same-type pseudo splits inside each pool, run the evaluator, rerun after `+1000` mutation of outer evaluation expression, and fail if scientific rows differ.

Write to:

```text
results/reference_only_feasibility_diagnostic/summary.json
results/reference_only_feasibility_diagnostic/metrics.csv
```

- [ ] **Step 4: Run focused and full tests**

Expected: all pass.

### Task 4: Run the diagnostic and write the decision report

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/results/reference_only_feasibility_diagnostic/summary.json`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/results/reference_only_feasibility_diagnostic/metrics.csv`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/reports/reference_only_anchor_feasibility_diagnostic_zh.md`

- [ ] **Step 1: Run the real diagnostic once**

Run:

```text
rtk uv run python repro_runs/2026-07-21-bounded-target-anchor-base/run_reference_only_feasibility_diagnostic.py
```

- [ ] **Step 2: Inspect compact aggregates and write the report**

Report the fixed baseline, each frontier endpoint, measured rather than requested distortion, location gates, full mechanism gates, per-split diagnosis, aggregate decision, mutation result, and claim boundary. Explicitly state that this diagnostic does not calibrate or promote a candidate and does not revise the frozen outer result.

- [ ] **Step 3: Re-run and verify deterministic artifacts**

Hash JSON and CSV, rerun the command, and require identical hashes. Run all tests fresh and read back the report to verify numbers and the absence of outer-based tuning claims.

No git commit is created because the project instructions require explicit user authorization for commits.
