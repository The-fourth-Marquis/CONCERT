# NicheFlow Mainline A Next Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the A2.3 anchored NicheFlow baseline into a defensible NicheFlow-v1 research package by auditing the source of its gains, adding mechanism metrics, and producing an interpretable comparison report.

**Architecture:** Keep all executable experiments under `repro_runs/2026-06-22-full-repro/`. Treat A2.3 as a strong transductive baseline, not yet the final method. Add one audit runner, one NicheFlow-v1 runner, one mechanism-metric runner, and one report directory; every runner writes CSVs, metadata, README, and has a smoke test.

**Tech Stack:** Python via `rtk uv run python`, pandas, numpy, scipy/sklearn helpers already used by the CONCERT reproduction, matplotlib for figures, existing paper-metric utilities in `repro_runs/2026-06-22-full-repro/`.

---

### Task 1: A2.4 Audit The A2.3 Gain Source

**Files:**
- Create: `repro_runs/2026-06-22-full-repro/nicheflow_anchored_audit.py`
- Create: `repro_runs/2026-06-22-full-repro/test_nicheflow_anchored_audit.py`
- Read: `repro_runs/2026-06-22-full-repro/nicheflow_anchored_baseline.py`
- Output: `repro_runs/2026-06-22-full-repro/nicheflow_anchored_audit/`

- [ ] **Step 1: Write the failing smoke test**

Create `test_nicheflow_anchored_audit.py` with tests requiring five method families:

```python
#!/usr/bin/env python3
"""Smoke tests for A2.4 anchored NicheFlow gain audit."""

from pathlib import Path

import pandas as pd

from paper_metric_benchmark import build_task_manifest
from nicheflow_anchored_audit import (
    OUTDIR,
    AUDIT_VARIANTS,
    compare_audit_variants,
    evaluate_audit_manifest,
    evaluate_audit_task,
    write_outputs,
)


def test_audit_task_contains_anchor_residual_and_calibration_ablation():
    task = build_task_manifest().iloc[0]
    rows = evaluate_audit_task(task)
    methods = {row["method"] for row in rows}
    assert methods == AUDIT_VARIANTS
    assert {"anchor_only", "residual_only", "anchor_residual_no_calibration", "anchor_only_calibrated", "anchor_residual_calibrated"} <= {
        row["audit_family"] for row in rows
    }
    assert all(pd.notna(row["edistance"]) for row in rows)


def test_audit_outputs_include_contribution_tables(tmp_path):
    manifest = build_task_manifest().head(2)
    metrics = evaluate_audit_manifest(manifest)
    outputs = write_outputs(manifest, metrics, tmp_path, write_figures=False)
    expected = {
        "metrics.csv",
        "best_by_task.csv",
        "audit_family_summary.csv",
        "gain_decomposition.csv",
        "compare_to_a2_best.csv",
        "metadata.json",
        "README.md",
    }
    assert expected.issubset({path.name for path in outputs})
    decomp = compare_audit_variants(metrics)
    assert {"anchor_gain", "residual_gain", "calibration_gain"}.issubset(decomp.columns)


def test_default_output_directory_names_a2_4():
    assert OUTDIR.name == "nicheflow_anchored_audit"


if __name__ == "__main__":
    import tempfile

    test_audit_task_contains_anchor_residual_and_calibration_ablation()
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_outputs_include_contribution_tables(Path(tmp))
    test_default_output_directory_names_a2_4()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/test_nicheflow_anchored_audit.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nicheflow_anchored_audit'`.

- [ ] **Step 3: Implement the audit runner**

Create `nicheflow_anchored_audit.py` by reusing functions from `nicheflow_anchored_baseline.py`. It must evaluate these exact audit families:

```python
AUDIT_VARIANTS = {
    "nicheflow_audit_anchor_only",
    "nicheflow_audit_residual_only",
    "nicheflow_audit_anchor_residual_no_calibration",
    "nicheflow_audit_anchor_only_calibrated",
    "nicheflow_audit_anchor_residual_calibrated",
}
```

Implementation rules:
- `anchor_only`: use the best of `knn10`, `knn5`, `cauchy`, `gaussian` target-pool anchors without residual or calibration.
- `residual_only`: use `nicheflow_lite_full` exactly.
- `anchor_residual_no_calibration`: use `0.80 * anchor + 0.20 * residual`, no moment alignment.
- `anchor_only_calibrated`: use anchor plus the same target-pool calibration used in A2.3.
- `anchor_residual_calibrated`: reproduce A2.3.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/test_nicheflow_anchored_audit.py
```

Expected: `ok`.

- [ ] **Step 5: Run full audit**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/nicheflow_anchored_audit.py --outdir repro_runs/2026-06-22-full-repro/nicheflow_anchored_audit
```

Expected: 37 tasks, audit tables written, and README states whether the 37/37 A2.3 gain comes mostly from anchor, calibration, residual branch, or their interaction.

### Task 2: Promote A2.3 Into NicheFlow-v1 With Explicit Components

**Files:**
- Create: `repro_runs/2026-06-22-full-repro/nicheflow_v1.py`
- Create: `repro_runs/2026-06-22-full-repro/test_nicheflow_v1.py`
- Read: `repro_runs/2026-06-22-full-repro/nicheflow_anchored_audit/`
- Output: `repro_runs/2026-06-22-full-repro/nicheflow_v1/`

- [ ] **Step 1: Write the failing NicheFlow-v1 test**

Create `test_nicheflow_v1.py`:

```python
#!/usr/bin/env python3
"""Smoke tests for Mainline A NicheFlow-v1."""

from pathlib import Path

import pandas as pd

from paper_metric_benchmark import build_task_manifest
from nicheflow_v1 import OUTDIR, evaluate_manifest_task, evaluate_nicheflow_v1, write_outputs


def test_v1_rows_report_interpretable_components():
    task = build_task_manifest().iloc[0]
    rows = evaluate_manifest_task(task)
    assert len(rows) >= 3
    for row in rows:
        assert row["method"].startswith("nicheflow_v1")
        assert row["anchor_method"] in {"knn10", "knn5", "cauchy", "gaussian", "adaptive"}
        assert 0.0 <= row["anchor_weight"] <= 1.0
        assert 0.0 <= row["residual_weight"] <= 1.0
        assert row["uses_observed_target_pool"] is True
        assert pd.notna(row["edistance"])


def test_v1_outputs_mainline_comparisons(tmp_path):
    manifest = build_task_manifest().head(2)
    metrics = evaluate_nicheflow_v1(manifest)
    outputs = write_outputs(manifest, metrics, tmp_path, write_figures=False)
    expected = {
        "metrics.csv",
        "method_ranking.csv",
        "task_type_summary.csv",
        "compare_to_a2_best.csv",
        "compare_to_spatialprop_style.csv",
        "compare_to_stage4_nicheflow_lite.csv",
        "metadata.json",
        "README.md",
    }
    assert expected.issubset({path.name for path in outputs})


def test_default_output_directory_names_v1():
    assert OUTDIR.name == "nicheflow_v1"


if __name__ == "__main__":
    import tempfile

    test_v1_rows_report_interpretable_components()
    with tempfile.TemporaryDirectory() as tmp:
        test_v1_outputs_mainline_comparisons(Path(tmp))
    test_default_output_directory_names_v1()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/test_nicheflow_v1.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nicheflow_v1'`.

- [ ] **Step 3: Implement NicheFlow-v1**

Create `nicheflow_v1.py` with these variants:

```python
V1_VARIANTS = {
    "nicheflow_v1_adaptive_anchor_residual",
    "nicheflow_v1_cauchy_anchor_residual",
    "nicheflow_v1_knn10_anchor_residual",
}
```

Implementation rules:
- `adaptive_anchor_residual` chooses `cauchy` for border, periphery, cross-slide, and exact patch; chooses `knn10` for patch and cross-niche; chooses `knn5` only if A2.4 shows a task-type gain.
- Use the A2.4 audit result to set whether calibration remains enabled by default.
- Keep `anchor_weight` and `residual_weight` explicit in every metric row.
- Do not claim strict unseen prediction.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/test_nicheflow_v1.py
```

Expected: `ok`.

- [ ] **Step 5: Run full NicheFlow-v1**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/nicheflow_v1.py --outdir repro_runs/2026-06-22-full-repro/nicheflow_v1
```

Expected: 37 tasks, ranking tables, comparison to A2.1/A2.2/Stage4, and a README that explains why v1 is more than a metric-hacking baseline.

### Task 3: Add Mechanism Metrics For Mainline A

**Files:**
- Create: `repro_runs/2026-06-22-full-repro/mainline_a_mechanism_metrics.py`
- Create: `repro_runs/2026-06-22-full-repro/test_mainline_a_mechanism_metrics.py`
- Read: `repro_runs/2026-06-22-full-repro/nicheflow_v1/method_ranking.csv`
- Read: `repro_runs/2026-06-22-full-repro/nicheflow_anchored_baseline/metrics.csv`
- Output: `repro_runs/2026-06-22-full-repro/mainline_a_mechanism_metrics/`

- [ ] **Step 1: Write failing metric unit tests**

Create `test_mainline_a_mechanism_metrics.py` with tiny deterministic arrays:

```python
#!/usr/bin/env python3
"""Tests for Mainline A mechanism metrics."""

import numpy as np

from mainline_a_mechanism_metrics import (
    boundary_leakage_score,
    source_individuality_score,
    distance_response_decay_score,
)


def test_boundary_leakage_score_increases_for_cross_boundary_smoothing():
    pred = np.array([[10.0], [10.0], [0.0], [0.0]], dtype=np.float32)
    target = np.array([[10.0], [10.0], [0.0], [0.0]], dtype=np.float32)
    tissue = np.array(["tumor", "tumor", "normal", "normal"])
    mixed = np.array([[5.0], [5.0], [5.0], [5.0]], dtype=np.float32)
    assert boundary_leakage_score(mixed, target, tissue) > boundary_leakage_score(pred, target, tissue)


def test_source_individuality_score_rewards_noncollapsed_predictions():
    source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    collapsed = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    preserved = source.copy()
    assert source_individuality_score(preserved, source) > source_individuality_score(collapsed, source)


def test_distance_response_decay_score_returns_finite_number():
    distances = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    response = np.array([3.0, 2.0, 1.0, 0.5], dtype=np.float32)
    assert np.isfinite(distance_response_decay_score(distances, response))


if __name__ == "__main__":
    test_boundary_leakage_score_increases_for_cross_boundary_smoothing()
    test_source_individuality_score_rewards_noncollapsed_predictions()
    test_distance_response_decay_score_returns_finite_number()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/test_mainline_a_mechanism_metrics.py
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement mechanism metric functions and runner**

Create `mainline_a_mechanism_metrics.py` with:
- `boundary_leakage_score(pred, target, tissue)`: difference between within-tissue error and cross-tissue collapsed error, higher means more leakage.
- `source_individuality_score(pred, source)`: Spearman or Pearson similarity of pairwise distance matrices, higher means local heterogeneity is preserved.
- `distance_response_decay_score(distances, response)`: correlation between distance and response magnitude with finite fallback.
- A full runner that computes these metrics for A2.3, NicheFlow-v1, A2.1 winners, and SpatialProp-style where predictions are available or reconstructable.

- [ ] **Step 4: Run test and full mechanism analysis**

Run:

```bash
rtk uv run python repro_runs/2026-06-22-full-repro/test_mainline_a_mechanism_metrics.py
rtk uv run python repro_runs/2026-06-22-full-repro/mainline_a_mechanism_metrics.py --outdir repro_runs/2026-06-22-full-repro/mainline_a_mechanism_metrics
```

Expected: mechanism metric CSVs and figures exist, and README explains whether NicheFlow-v1's advantage is only global E-distance or also mechanism-consistent.

### Task 4: Produce Mainline A Research Summary

**Files:**
- Create: `repro_runs/2026-06-22-full-repro/mainline_a_report/`
- Create: `repro_runs/2026-06-22-full-repro/mainline_a_report/NicheFlow_Mainline_A_summary.md`

- [ ] **Step 1: Create report with fixed sections**

Write `NicheFlow_Mainline_A_summary.md` with these sections:
- Scientific question: observed perturbation spatial relocation.
- Protocol: target-pool access is allowed and reported.
- Baselines: A2.1 strong transductive, A2.2 SpatialProp-style, available CONCERT, A2.3 anchored NicheFlow, NicheFlow-v1.
- Main results: global E-distance and task-type summary.
- Mechanism results: boundary, niche, distance-response, source individuality.
- Caveats: surrogate tasks, incomplete CONCERT coverage, not strict unseen/OOD.
- Next paper-facing figures.

- [ ] **Step 2: Add machine-readable result index**

Create `mainline_a_report/result_index.csv` with columns:

```text
artifact,kind,path,primary_question,include_in_paper
```

- [ ] **Step 3: Verify links and counts**

Run:

```bash
rtk uv run python -c "from pathlib import Path; p=Path('repro_runs/2026-06-22-full-repro/mainline_a_report/NicheFlow_Mainline_A_summary.md'); text=p.read_text(); required=['A2.1','A2.2','A2.3','NicheFlow-v1','strict unseen','CONCERT']; missing=[x for x in required if x not in text]; assert not missing, missing; print('ok')"
```

Expected: `ok`.

### Task 5: Decide Whether To Retrain Or Only Compare Available CONCERT

**Files:**
- Create: `repro_runs/2026-06-22-full-repro/concert_retraining_decision.md`

- [ ] **Step 1: Write decision memo**

Create a short memo answering:
- Is full CONCERT retraining necessary for Mainline A claims?
- Which 37 tasks can CONCERT cover without changing model semantics?
- Which tasks require retraining, and which are impossible without new perturbation-embedding design?
- What is the minimum fair CONCERT comparison for the first NicheFlow report?

- [ ] **Step 2: Use current evidence**

Include these already observed facts:
- Available CONCERT covers 11/37 tasks.
- No available cross-slide CONCERT outputs.
- Existing available CONCERT does not beat A2.1 on covered tasks.
- A2.3 beats available SpatialProp-style and original NicheFlow-lite, but must be interpreted under target-pool access.

- [ ] **Step 3: Verify memo has concrete decision**

Run:

```bash
rtk uv run python -c "from pathlib import Path; text=Path('repro_runs/2026-06-22-full-repro/concert_retraining_decision.md').read_text(); assert 'Decision:' in text; assert 'Minimum fair comparison' in text; print('ok')"
```

Expected: `ok`.

### Self-Review

- This plan starts with auditing A2.3 before making stronger claims.
- It promotes only audited components into NicheFlow-v1.
- It adds mechanism metrics so the project is not judged only by global E-distance.
- It keeps strict unseen/OOD as a separate appendix line, not the Mainline A claim.
- It avoids requiring immediate CONCERT retraining before the current NicheFlow evidence is organized.
