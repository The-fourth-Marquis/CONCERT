# Support-Aware Low-Rank Reference Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and evaluate kernel and Bayesian low-rank reference-conditioned operators in a new isolated experiment directory.

**Architecture:** Both heads share context-only reference residual construction, a shared global shift, a rank-limited SVD gene basis, support shrinkage, and a local-correction cap. The kernel head uses RBF interpolation in coefficient space; the Bayesian head uses Bayesian ridge coefficient prediction and posterior uncertainty. Existing NicheFlow-v2 data, splits, metrics, and baselines remain read-only dependencies.

**Tech Stack:** Python 3.10, NumPy, SciPy/scikit-learn, pandas, existing NicheFlow-v2 evaluator, unittest-style executable tests.

---

No git commit is created because repository instructions require explicit user
authorization before committing.

### Task 1: Create the operator contracts

**Files:**
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_operator.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/__init__.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/operator.py`

- [ ] Write failing tests for shared-shift fallback, bounded rank, finite shrinkage, support monotonicity, far-query shrinkage, and correction clipping.
- [ ] Run `rtk uv run python repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_operator.py` and confirm import failure because the package does not yet exist.
- [ ] Implement matrix validation, context standardization, RBF weights, residual SVD, correction clipping, and the kernel head.
- [ ] Re-run the test and confirm all kernel contracts pass.
- [ ] Add a failing Bayesian-head test asserting finite predictions, finite posterior uncertainty, and shrinkage in `[0, 1]`.
- [ ] Implement Bayesian ridge coefficient heads with uncertainty-aware shrinkage.
- [ ] Re-run the test and confirm all operator contracts pass.

### Task 2: Adapt the operator to Mainline A tasks

**Files:**
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_task_adapter.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/task_adapter.py`

- [ ] Write a failing synthetic `TaskPayload` test for context construction and evaluation-expression mutation invariance.
- [ ] Run the test and confirm failure because `predict_task` is absent.
- [ ] Implement cross-slide-aware context normalization and `predict_task` using reference rows only.
- [ ] Re-run the test and confirm the mutation guard passes for both heads.

### Task 3: Add metric and output contracts

**Files:**
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_evaluation.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/evaluation.py`

- [ ] Write failing tests for pairwise distortion and the first-task advancement rule.
- [ ] Run the test and confirm failure because evaluation helpers are absent.
- [ ] Implement the metric bundle using NicheFlow-v2 metrics and a decision helper with the frozen thresholds.
- [ ] Re-run the test and confirm the expected advance/stop decisions.

### Task 4: Run a real-task smoke experiment

**Files:**
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_smoke_runner.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/run_smoke.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/results/first_task_smoke/summary.json`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/results/first_task_smoke/predictions.npz`

- [ ] Write a failing runner test using injected synthetic payload/split inputs.
- [ ] Run it and confirm the runner is missing.
- [ ] Implement the runner, mutation guard, fixed 80/20 comparison, JSON metadata, and compressed predictions.
- [ ] Re-run the runner test.
- [ ] Run `rtk uv run python repro_runs/2026-07-21-support-aware-low-rank-operator/run_smoke.py` on the first real task.

### Task 5: Report and verify the first decision

**Files:**
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/README.md`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/reports/first_task_smoke_zh.md`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/run_all.py`

- [ ] Write the report from `summary.json`, separating diagnostic results from calibrated promotion claims.
- [ ] Run all new tests through `tests/run_all.py`.
- [ ] Re-run the real smoke and verify deterministic summary metrics.
- [ ] Read back the report and verify key numbers, claim boundary, file paths, and decision.
- [ ] Run `rtk git diff --check -- repro_runs/2026-07-21-support-aware-low-rank-operator docs/superpowers/specs/2026-07-21-support-aware-low-rank-operator-design.md docs/superpowers/plans/2026-07-21-support-aware-low-rank-operator.md`.

