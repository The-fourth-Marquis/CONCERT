# SPAC-seq ELRR Model and One-Shot Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen 113-parameter ELRR learned retrieval kernel, strong nonlearned and learned-null comparators, grouped development training, hierarchical W/C gates, and transaction-locked external/final runners without ever using reserve values.

**Architecture:** The model consumes only 12-dimensional query/reference context pairs and returns convex retrieval weights over observed support residuals. Deterministic LOGO training optimizes biased E-distance on development episodes; frozen baseline/null selections and hierarchical metrics decide advancement. Stage-aware transaction runners separate guide metadata, reference expression, prediction, and query scoring so external and Lung 993 can each be evaluated once after explicit authorization.

**Tech Stack:** Python 3.10+, NumPy, pandas, SciPy, PyTorch CPU float64, h5py/anndata through Plan A adapters, hashlib/json/csv/npz/PT artifacts, assertion-based executable tests.

---

## Scope, prerequisite, and authorization checkpoints

This is Plan B of two. It may start only when Plan A verifier confirms:

```text
results/data_readiness/data_readiness_summary.json: status == "data_ready"
results/data_readiness/seal_activation.json: valid
external_value_reads == 0
final_value_reads == 0
reserve_value_reads == 0
```

Frozen scientific source:

```text
docs/superpowers/specs/2026-07-21-spacseq-elrr-design.md
design commit = 1535ab1b645ceec9644730d961507e45dc8b41cf
```

This planning turn creates no git commit. During execution, each commit step requires explicit commit authorization.

Authorization boundaries:

1. Tasks 1–9 may use synthetic fixtures and development values only.
2. Task 10 implements and tests the external runner but must not execute real Lung 994/OT1 Day7 rep2 without a new explicit user authorization.
3. Task 11 implements and tests the final runner but must not execute real Lung 993 without a second explicit authorization after external success.
4. No task or command may read OT1 Day10 rep2 values.

## File map

- `src/elrr/features.py`: 12D pair features and W/C D4 marginalization.
- `src/elrr/model.py`: the 113-parameter tanh scorer and global temperature.
- `src/elrr/losses.py`: biased energy V-statistic.
- `src/elrr/sampling.py`: stable-hash support dropout.
- `src/elrr/baselines.py`: control, uniform, nearest, and strong frozen RBF retrieval.
- `src/elrr/nulls.py`: no-query and query-permuted learned mechanism nulls.
- `src/elrr/data.py`: immutable Plan A artifact loading and role-stage views.
- `src/elrr/prediction.py`: residual construction, D4-averaged weights, and convex prediction.
- `src/elrr/training.py`: deterministic LOGO fold training, epoch selection, and all-development refit.
- `src/elrr/metrics.py`: E-distance, Delta PCC, context strata, and undefined handling.
- `src/elrr/evaluation.py`: exact support/view/target/specimen/screen hierarchy.
- `src/elrr/gates.py`: readiness, baseline/null selection, and internal/external/final decisions.
- `src/elrr/transactions.py`: Stage 0–3 hashes, sentinels, abort states, and replay.
- `src/elrr/artifacts.py`: atomic JSON/CSV/NPZ/PT writes and complete freeze manifests.
- `run_development.py`: development training, comparator selection, internal gate, and Stage-0 freeze.
- `run_external.py`: explicitly authorized one-shot Lung 994 and OT1 Day7 rep2 evaluation.
- `run_final.py`: explicitly authorized one-shot Lung 993 evaluation; reserve denied.
- `verify_model_artifacts.py`: cross-artifact hashes, metrics, decisions, and claim-boundary checks.

### Task 1: 12D pair features, D4 marginalization, and 113-parameter scorer

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/features.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/model.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_model.py`

- [ ] **Step 1: Write the failing model contract test**

```python
import numpy as np
import torch

from elrr.features import pair_features
from elrr.model import ELRRScorer

q = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.2, -0.1]])
r = np.array([[0.0, 0.1, 0.2, 0.3], [0.9, 0.8, 0.4, 0.2], [0.4, 0.4, 0.1, 0.0]])
phi = pair_features(q, r)
assert phi.shape == (2, 3, 12)

model = ELRRScorer(seed=20260721).to(dtype=torch.float64)
assert sum(p.numel() for p in model.parameters()) == 113
weights, metadata = model(torch.tensor(phi, dtype=torch.float64))
assert weights.shape == (2, 3)
assert torch.all(weights >= 0)
assert torch.allclose(weights.sum(dim=1), torch.ones(2, dtype=torch.float64))
assert float(metadata["temperature"]) > 0.05
assert model.output.bias is None
```

Add tests for support permutation invariance, one-reference weight 1, D4 transforms around `(0.5,0.5)`, W eight-transform averaging, C independent 8×8 averaging, and residual-value mutation leaving logits/weights unchanged.

- [ ] **Step 2: Run RED**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_model.py
```

Expected: import failure for `elrr.model`.

- [ ] **Step 3: Implement exact pair features and model**

```python
# src/elrr/features.py
def pair_features(query_context, reference_context):
    q = np.asarray(query_context, dtype=np.float64)
    r = np.asarray(reference_context, dtype=np.float64)
    q3 = np.broadcast_to(q[:, None, :], (q.shape[0], r.shape[0], 4))
    r3 = np.broadcast_to(r[None, :, :], (q.shape[0], r.shape[0], 4))
    return np.concatenate([q3, r3, np.abs(q3 - r3)], axis=2)

def d4_contexts(context):
    base = np.asarray(context, dtype=np.float64)
    x = base[:, 0] - 0.5
    y = base[:, 1] - 0.5
    xy_variants = (
        np.column_stack([x, y]),
        np.column_stack([-x, y]),
        np.column_stack([x, -y]),
        np.column_stack([-x, -y]),
        np.column_stack([y, x]),
        np.column_stack([-y, x]),
        np.column_stack([y, -x]),
        np.column_stack([-y, -x]),
    )
    transformed = []
    for xy in xy_variants:
        item = base.copy()
        item[:, :2] = xy + 0.5
        transformed.append(item)
    return tuple(transformed)
```

```python
# src/elrr/model.py
import math
import torch


class ELRRScorer(torch.nn.Module):
    def __init__(self, seed=20260721):
        super().__init__()
        torch.manual_seed(int(seed))
        self.hidden = torch.nn.Linear(12, 8, bias=True)
        self.output = torch.nn.Linear(8, 1, bias=False)
        theta = math.log(math.expm1(0.95))
        self.theta_temperature = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, pair_tensor):
        h = torch.tanh(self.hidden(pair_tensor))
        logits = self.output(h).squeeze(-1)
        temperature = 0.05 + torch.nn.functional.softplus(self.theta_temperature)
        weights = torch.softmax(logits / temperature, dim=1)
        return weights, {"logits": logits, "temperature": temperature}
```

Keep all parameters float64 by calling `.to(dtype=torch.float64)` before training. `Linear(8,1,bias=False)` supplies eight effective output weights and no softmax-cancelled bias. The frozen count is `12*8 + 8 + 8 + 1 = 113`.

- [ ] **Step 4: Run GREEN**

Expected: model tests pass and parameter count is exactly 113.

- [ ] **Step 5: Commit**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/features.py repro_runs/2026-07-21-spacseq-elrr/src/elrr/model.py repro_runs/2026-07-21-spacseq-elrr/tests/test_model.py
rtk git commit -m "feat: add 113-parameter ELRR scorer"
```

### Task 2: Biased energy loss and deterministic reference dropout

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/losses.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/sampling.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_losses_sampling.py`

- [ ] **Step 1: Write failing hand-calculated tests**

```python
import numpy as np
import torch

from elrr.losses import energy_v_statistic
from elrr.sampling import reference_dropout_ids

pred = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
true = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
loss = energy_v_statistic(pred, true)
expected = (
    2 * torch.cdist(pred, true).mean()
    - torch.cdist(pred, pred).mean()
    - torch.cdist(true, true).mean()
)
assert torch.allclose(loss, expected)

ids = tuple(f"id-{i:02d}" for i in range(28))
kept_a = reference_dropout_ids(ids, 0.20, 20260721, 3, "unit")
kept_b = reference_dropout_ids(tuple(reversed(ids)), 0.20, 20260721, 3, "unit")
assert kept_a == kept_b
assert len(kept_a) >= 8
```

Test finite gradients, diagonal inclusion, exact `m*n/m*m/n*n` means, stable ID order, and the “fewer than eight Bernoulli survivors keeps eight smallest hashes” rule.

- [ ] **Step 2: Run RED**

Expected: import failures.

- [ ] **Step 3: Implement the exact functions**

```python
def energy_v_statistic(predicted, observed):
    if predicted.ndim != 2 or observed.ndim != 2:
        raise ValueError("energy inputs must be matrices")
    if predicted.shape[1] != observed.shape[1]:
        raise ValueError("gene dimensions differ")
    return (
        2.0 * torch.cdist(predicted, observed, p=2).mean()
        - torch.cdist(predicted, predicted, p=2).mean()
        - torch.cdist(observed, observed, p=2).mean()
    )
```

For dropout, hash `seed|epoch|unit_id|observation_id` with SHA-256, convert the first eight bytes to a uniform integer, retain values above the 0.20 threshold, and force the eight smallest hashes when fewer survive. Return IDs sorted lexicographically so input row order cannot alter the result.

- [ ] **Step 4: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_losses_sampling.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/losses.py repro_runs/2026-07-21-spacseq-elrr/src/elrr/sampling.py repro_runs/2026-07-21-spacseq-elrr/tests/test_losses_sampling.py
rtk git commit -m "feat: add ELRR energy loss and sampling"
```

### Task 3: Strong nonlearned baselines and learned mechanism nulls

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/baselines.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/nulls.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_baselines_nulls.py`

- [ ] **Step 1: Write failing baseline tests**

Require uniform, nearest, and RBF weights to be finite/nonnegative/sum-one. For RBF:

```python
d2 = (
    (dx / s_xy) ** 2
    + (dy / s_xy) ** 2
    + (db / s_boundary) ** 2
    + (dd / s_density) ** 2
)
expected = softmax(-d2 / (2 * bandwidth**2))
```

Require exactly 81 configurations, specimen-balanced W-overall selection, and the total-order key:

```python
(
    validation_loss,
    log2(s_xy)**2 + log2(s_boundary)**2 + log2(s_density)**2,
    -bandwidth,
    s_xy,
    s_boundary,
    s_density,
)
```

Within `1e-12`, ignore the first tuple element and apply the remaining order. Require the global comparator tie order `uniform_residual`, `nearest_residual`, `fixed_rbf_retrieval`.

- [ ] **Step 2: Write failing null tests**

```python
phi = no_query_features(query_context, reference_context)
assert np.all(phi[:, :, :4] == 0)
assert np.all(phi[:, :, 8:] == 0)
assert np.allclose(phi[:, :, 4:8], reference_context[None, :, :])

permutations = query_context_permutations(query_ids, count=20, seed=20260721, unit_id="u")
assert len(permutations) == 20
assert permutations == query_context_permutations(tuple(reversed(query_ids)), 20, 20260721, "u")
```

Require anchor, residuals, truth, and observation IDs to remain fixed for the permuted scorer input. Select one global mechanism null from `ELRR_no_query` and mean `ELRR_query_permuted` by the frozen development rule.

- [ ] **Step 3: Run RED**

Expected: import failures for `elrr.baselines` and `elrr.nulls`.

- [ ] **Step 4: Implement comparators and null builders**

Implement baseline weights independently from the neural model. `ELRR_no_query` reuses `ELRRScorer` and training but zeros query/difference inputs. Query permutation hashes `study_seed|unit_id|permutation_index|query_id`; it must never permute the anchor or observed truth.

- [ ] **Step 5: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_baselines_nulls.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/baselines.py repro_runs/2026-07-21-spacseq-elrr/src/elrr/nulls.py repro_runs/2026-07-21-spacseq-elrr/tests/test_baselines_nulls.py
rtk git commit -m "feat: add ELRR comparators and mechanism nulls"
```

### Task 4: Immutable data bundle loading and residual prediction

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/data.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/prediction.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_prediction.py`
- Read: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/gene_panel.json`
- Read: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/section_transforms.npz`
- Read: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/development_splits.csv`
- Read: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/processed/development/{section_id}.rna.h5ad`
- Read: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/processed/development/{section_id}.guide.h5ad`

- [ ] **Step 1: Write failing Plan A artifact-loader tests**

Require `load_development_bundle` to verify the Plan A summary, seal hash, gene order, section-transform hashes, observation IDs, and W/C unit IDs before returning immutable arrays. Mutating any H5AD or split artifact after the manifest hash is written must fail before training.

- [ ] **Step 2: Write failing prediction tests**

```python
residuals = reference_expression - reference_anchor
prediction, meta = predict_from_weights(query_anchor, residuals, weights)
assert np.allclose(prediction, query_anchor + weights @ residuals)
assert np.allclose(meta["weight_sum"], 1.0)

residual_min = residuals.min(axis=0)
residual_max = residuals.max(axis=0)
pred_residual = prediction - query_anchor
assert np.all(pred_residual >= residual_min - 1e-12)
assert np.all(pred_residual <= residual_max + 1e-12)
```

Also require support permutation invariance, one-reference degeneracy, uniform-logit equivalence, W 8-transform logit averaging, C 64-transform logit averaging, and no target-evaluation expression parameter in any fit/predict signature.

- [ ] **Step 3: Run RED**

Expected: import failures.

- [ ] **Step 4: Implement immutable loading and prediction**

```python
def predict_from_weights(query_anchor, reference_residuals, weights):
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        raise PredictionError("weights do not sum to one")
    predicted_residual = weights @ reference_residuals
    return query_anchor + predicted_residual, {
        "weight_sum": weights.sum(axis=1),
        "maximum_weight": weights.max(axis=1),
        "effective_references": 1.0 / np.square(weights).sum(axis=1),
    }
```

For D4 inference, transform contexts, obtain logits for each orientation pair, average logits, then apply one softmax. Never average already-softmaxed weights.

- [ ] **Step 5: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_prediction.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/data.py repro_runs/2026-07-21-spacseq-elrr/src/elrr/prediction.py repro_runs/2026-07-21-spacseq-elrr/tests/test_prediction.py
rtk git commit -m "feat: add ELRR residual prediction"
```

### Task 5: Deterministic LOGO training and all-development refit

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/training.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_training.py`

- [ ] **Step 1: Write failing synthetic training tests**

Create at least three specimen groups with known context-conditioned residual mixtures. Require:

- leave-one-`biological_specimen_id`-out folds;
- no held-out query expression in gradient inputs;
- held-out W support used only in validation forward;
- C validation directions only `training support -> held-out query`;
- specimen-balanced loss;
- Adam `lr=0.003`, `weight_decay=0.0001`;
- CPU float64, seed 20260721, clip norm 1.0;
- maximum 400 epochs, patience 40;
- earliest epoch within `1e-12` of the curve minimum;
- final reinitialization and refit on all development groups for exactly that epoch;
- identical loss history/state hash across two runs.

- [ ] **Step 2: Run RED**

Expected: import failure for `elrr.training`.

- [ ] **Step 3: Implement fold training and epoch selection**

Use one optimizer step per specimen-balanced collection of W/C episodes. Store fold-level and specimen-level losses separately. Build one validation curve by averaging held-out specimen losses at each epoch. Select:

```python
minimum = min(curve)
selected_epoch = min(i for i, value in enumerate(curve, start=1) if value <= minimum + 1e-12)
```

Recreate model/optimizer from seed before all-development refit. Reject CLI overrides for width, seed, learning rate, dropout, temperature, epochs, or gene features.

- [ ] **Step 4: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_training.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/training.py repro_runs/2026-07-21-spacseq-elrr/tests/test_training.py
rtk git commit -m "feat: add deterministic ELRR training"
```

### Task 6: Metrics, context strata, and hierarchical aggregation

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/metrics.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/evaluation.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_metrics_evaluation.py`

- [ ] **Step 1: Write failing metric tests**

```python
delta_true = observed.mean(axis=0) - anchor.mean(axis=0)
delta_pred = predicted.mean(axis=0) - anchor.mean(axis=0)
expected = np.corrcoef(delta_pred, delta_true)[0, 1]
assert delta_pcc(predicted, observed, anchor) == pytest.approx(expected)
```

Require `metric_undefined` when either vector variance is zero, fewer than three genes remain, or the result is nonfinite. Require boundary-distance tertiles with at least four query bins each and no target expression input.

- [ ] **Step 2: Write failing hierarchy tests**

Feed duplicate support directions/task views and require exact equal-weight order:

```text
view/support-direction replicate
-> section-target
-> query-specimen-target
-> query specimen
-> screen
```

Bins and task views may not change the screen scalar when duplicated. W overall, W spatial, and C transfer remain separate columns and never average into a single result.

- [ ] **Step 3: Run RED**

Expected: import failures.

- [ ] **Step 4: Implement metrics and hierarchy**

Use the same `energy_v_statistic` convention for NumPy reporting. Every metric difference is `ELRR - frozen comparator`. Store numerator unit IDs, denominator counts, missing reasons, and screen scalars. Target resampling may produce only `target_task_resampling_sensitivity_interval`; implement no bin bootstrap.

- [ ] **Step 5: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_metrics_evaluation.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/metrics.py repro_runs/2026-07-21-spacseq-elrr/src/elrr/evaluation.py repro_runs/2026-07-21-spacseq-elrr/tests/test_metrics_evaluation.py
rtk git commit -m "feat: add ELRR hierarchical evaluation"
```

### Task 7: Frozen comparator selection and promotion gates

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/gates.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_gates.py`

- [ ] **Step 1: Write failing readiness and selection tests**

Require at least three frozen targets per external/final screen, and for every target:

```text
W overall
W spatial
W context-stratified
C transfer
C context-stratified
Delta PCC
```

Any missing view, undefined Delta, or fewer than three targets returns `data_not_ready` for the whole screen.

Require one global nonlearned baseline selected only from grouped development W overall, and one global mechanism null selected only from grouped development W spatial/C. External/final rows must not change either selection.

- [ ] **Step 2: Write failing gate tests**

Require all conditions simultaneously:

```python
assert summary["W_overall_vs_nonlearned"] < 0
assert summary["W_spatial_vs_nonlearned"] < 0
assert summary["C_transfer_vs_nonlearned"] < 0
assert summary["W_spatial_vs_mechanism_null"] < 0
assert summary["C_transfer_vs_mechanism_null"] < 0
assert summary["W_context_vs_nonlearned"] < 0
assert summary["C_context_vs_nonlearned"] < 0
assert summary["W_delta_pcc_difference"] >= -0.02
assert summary["C_delta_pcc_difference"] >= -0.02
```

A single failed condition must fail the screen. A large gain in one screen cannot offset failure in another.

- [ ] **Step 3: Run RED**

Expected: import failure for `elrr.gates`.

- [ ] **Step 4: Implement immutable decisions**

Return typed decisions and a full per-condition boolean map. Never select a baseline/null per target, screen, or view. Never filter an unfavorable required unit. `data_not_ready` is procedural, not algorithm failure.

- [ ] **Step 5: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_gates.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/gates.py repro_runs/2026-07-21-spacseq-elrr/tests/test_gates.py
rtk git commit -m "feat: add frozen ELRR gates"
```

### Task 8: Stage 0–3 transactions, complete hashes, and abort handling

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/transactions.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/artifacts.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_transactions.py`

- [ ] **Step 1: Write failing staged-access tests**

Inject guide, NTC/support, and query loaders with counters. Require:

```text
Stage 0: no raw value loader called
Stage 1: guide loader only
Stage 2: NTC/support loader only; query target loader count remains zero
Stage 3: query target loader once; prediction function count remains zero
```

Require a content hash between every stage, atomic `started/completed` sentinel transitions, and role/source/model/stage binding. Changing output directory must not permit another raw read.

- [ ] **Step 2: Write failing abort/replay tests**

Simulate a crash after raw read but before artifact completion. Require `external_aborted_no_result` or `final_aborted_no_result`, `access_action=keep_next_role_sealed`, and refusal of a second raw read. If a completed hash-verified stage artifact exists, allow replay from that artifact only.

Require every Stage 1 mode to deny reserve, including a callback assertion proving the reserve loader is not called.

- [ ] **Step 3: Run RED**

Expected: import failures.

- [ ] **Step 4: Implement transactions and atomic artifacts**

Use sentinel key:

```text
study_id + data_role + transaction_stage + source_checksums + model_hash
```

Write JSON/CSV/NPZ/PT artifacts to a temporary sibling path, fsync, then atomically replace. Stage 2 serializes predictions before Stage 3. Stage 3 accepts prediction artifact + query truth only; its function signature has no model or fitting callback.

- [ ] **Step 5: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_transactions.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/transactions.py repro_runs/2026-07-21-spacseq-elrr/src/elrr/artifacts.py repro_runs/2026-07-21-spacseq-elrr/tests/test_transactions.py
rtk git commit -m "feat: lock ELRR evaluation transactions"
```

### Task 9: Development runner, comparator freeze, and internal decision

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/run_development.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_development_runner.py`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/model_state.pt`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/loss_history.csv`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/baseline_selection.json`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/mechanism_null_selection.json`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/unit_metrics.csv`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/hierarchical_summary.csv`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/development_decision.json`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/development/freeze_manifest.json`

- [ ] **Step 1: Write a failing injected development-runner test**

Require Plan A artifact verification before load, zero external/final/reserve callbacks, deterministic model/null retraining, exact global comparator selections, all internal gate fields, complete dynamic-dependency hashes, and no freeze manifest when any gate fails.

- [ ] **Step 2: Run RED**

Expected: runner import failure.

- [ ] **Step 3: Implement the runner**

Order:

1. verify Plan A artifacts and `data_ready`;
2. load development bundle;
3. run LOGO training and select epoch;
4. refit ELRR and no-query null;
5. evaluate 20 query permutations and nonlearned grid;
6. freeze best nonlearned and mechanism null;
7. compute W/C metrics and internal gates;
8. run query-expression mutation and permutation guards;
9. write artifacts atomically;
10. write Stage-0 freeze manifest only when every internal gate passes.

Hash all local source, dynamically imported Plan A code, environment lock, model state, configs, manifests, splits, gene order, selections, and development artifacts.

- [ ] **Step 4: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_development_runner.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/run_development.py repro_runs/2026-07-21-spacseq-elrr/tests/test_development_runner.py
rtk git commit -m "feat: add ELRR development freeze"
```

- [ ] **Step 5: Run real development only**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/run_development.py
```

Expected: either `advance_to_external_gate` with a complete Stage-0 freeze manifest, or a frozen development failure. Do not run Task 10 real commands in the same authorization scope.

### Task 10: External one-shot runner implementation and authorization checkpoint

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/run_external.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_external_runner.py`

- [ ] **Step 1: Write failing injected runner tests**

Require exact query sections `lung_994`, `ot1_day7_rep2`; exact C supports `lung_992 -> lung_994` and `ot1_day7_rep1 -> ot1_day7_rep2`; at least three shared targets; separate per-screen gates; no Lung 993/reserve callback; and `do_not_unseal_lung_993` if either external screen fails or is not ready.

- [ ] **Step 2: Run RED**

Expected: import failure for `run_external`.

- [ ] **Step 3: Implement CLI and staged execution**

CLI must require:

```text
--external-evaluation
--authorization-token-file repro_runs/2026-07-21-spacseq-elrr/authorizations/external_authorization.json
```

The token file is content-hashed into the sentinel and must equal the runtime expectation:

```python
expected_token = {
    "authorization_kind": "external_evaluation",
    "section_ids": ["lung_994", "ot1_day7_rep2"],
    "model_state_sha256": freeze_manifest["model_state_sha256"],
    "user_authorized": True,
}
if authorization_token != expected_token:
    raise AccessDenied("external authorization token mismatch")
```

Implement `--replay` to read immutable artifacts only. No CLI option may change target thresholds, supports, baseline/null, model, or metrics.

- [ ] **Step 4: Run injected tests only**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_external_runner.py
```

Expected: all injected tests pass and real external loader counters remain zero.

- [ ] **Step 5: Commit implementation**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/run_external.py repro_runs/2026-07-21-spacseq-elrr/tests/test_external_runner.py
rtk git commit -m "feat: add one-shot ELRR external runner"
```

- [ ] **Step 6: STOP for explicit user authorization**

Do not run this command during ordinary plan execution:

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/run_external.py --external-evaluation --authorization-token-file repro_runs/2026-07-21-spacseq-elrr/authorizations/external_authorization.json
```

Expected after separate authorization: both external screens pass all gates, or `do_not_unseal_lung_993`. A crash after raw read produces `external_aborted_no_result`, not an algorithm failure and not a retry.

### Task 11: Lung 993 final runner, reserve denial, and final decision

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/run_final.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_final_runner.py`

- [ ] **Step 1: Write failing injected final-runner tests**

Require:

- exact final query `lung_993`;
- C support only `lung_992`;
- frozen baseline/null/model/gene/splits;
- final target denominator at least three;
- all W/C/context/Delta/mechanism gates;
- external success artifact and hash;
- zero calls to `ot1_day10_rep2`;
- `final_gate_failed` for a metric failure;
- `final_aborted_no_result` for a procedural crash;
- no biological confidence interval or bin bootstrap fields.

- [ ] **Step 2: Run RED**

Expected: import failure for `run_final`.

- [ ] **Step 3: Implement CLI and final transaction**

Require:

```text
--final-evaluation
--authorization-token-file repro_runs/2026-07-21-spacseq-elrr/authorizations/final_authorization.json
```

The token must equal:

```python
expected_token = {
    "authorization_kind": "final_evaluation",
    "section_ids": ["lung_993"],
    "external_success_sha256": external_decision["artifact_sha256"],
    "model_state_sha256": freeze_manifest["model_state_sha256"],
    "user_authorized": True,
}
if authorization_token != expected_token:
    raise AccessDenied("final authorization token mismatch")
```

The runner writes per-target metrics, one Lung screen scalar, target-task resampling sensitivity only, mutation evidence, final decision, and immutable sentinel. `--replay` reads artifacts only.

- [ ] **Step 4: Run injected tests only**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_final_runner.py
```

Expected: tests pass; real Lung 993 and reserve loader counters remain zero.

- [ ] **Step 5: Commit implementation**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/run_final.py repro_runs/2026-07-21-spacseq-elrr/tests/test_final_runner.py
rtk git commit -m "feat: add one-shot ELRR final runner"
```

- [ ] **Step 6: STOP for a second explicit user authorization**

Do not run this command unless the external decision is successful and the user separately authorizes Lung 993:

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/run_final.py --final-evaluation --authorization-token-file repro_runs/2026-07-21-spacseq-elrr/authorizations/final_authorization.json
```

Expected after authorization: `final_success`, `final_gate_failed`, or `final_aborted_no_result`. Every outcome keeps OT1 Day10 rep2 sealed.

### Task 12: Full test aggregator, artifact verifier, and decision report

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/verify_model_artifacts.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_model_artifact_verifier.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/run_model_tests.py`
- Modify: `repro_runs/2026-07-21-spacseq-elrr/README.md`

- [ ] **Step 1: Write failing verifier tests**

Require failure for:

- model parameter count not 113;
- output bias present;
- W/C metrics pooled together;
- fewer than three targets;
- stale baseline/null selections;
- missing query mutation evidence;
- incomplete dynamic-dependency hash;
- an external/final sentinel keyed only by output directory;
- reserve value reads above zero;
- a `final_success` report using “statistically significant”, “cross-animal stable”, “optimal transport”, or “geometry-preserving”.

- [ ] **Step 2: Run RED**

Expected: import failure for `verify_model_artifacts`.

- [ ] **Step 3: Implement fresh-process aggregator**

```python
TEST_FILES = (
    "test_model.py",
    "test_losses_sampling.py",
    "test_baselines_nulls.py",
    "test_prediction.py",
    "test_training.py",
    "test_metrics_evaluation.py",
    "test_gates.py",
    "test_transactions.py",
    "test_development_runner.py",
    "test_external_runner.py",
    "test_final_runner.py",
    "test_model_artifact_verifier.py",
)
```

Run each test in a fresh Python process and stop on first failure. The verifier cross-checks raw unit rows, hierarchical summaries, decisions, sentinels, hashes, and report language.

- [ ] **Step 4: Run all synthetic/injected tests**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/run_data_tests.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/run_model_tests.py
```

Expected:

```text
PASS 11 data test files
PASS 12 model test files
```

- [ ] **Step 5: Verify development artifacts**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/verify_data_artifacts.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/verify_model_artifacts.py --role development
```

Expected: complete hashes and zero external/final/reserve value reads. Do not verify external/final roles until separately authorized runs exist.

- [ ] **Step 6: Commit verifier and README**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/verify_model_artifacts.py repro_runs/2026-07-21-spacseq-elrr/tests/test_model_artifact_verifier.py repro_runs/2026-07-21-spacseq-elrr/tests/run_model_tests.py repro_runs/2026-07-21-spacseq-elrr/README.md
rtk git commit -m "docs: add ELRR execution and verification guide"
```

## Plan B completion evidence before any real unseal

Run:

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/run_data_tests.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/run_model_tests.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/verify_data_artifacts.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/verify_model_artifacts.py --role development
rtk git status --short repro_runs/2026-07-21-spacseq-elrr
```

The development handoff must report:

- Plan A `data_ready`;
- selected epoch and model hash;
- selected global nonlearned baseline;
- selected global mechanism null;
- internal W overall/W spatial/C/context/Delta gate values;
- mutation/permutation/hash guard status;
- zero external/final/reserve value reads;
- an explicit request for external authorization rather than running it automatically.

## Spec coverage for Plan B

| Design sections | Implementation tasks |
| --- | --- |
| §12 pair features, 113-parameter scorer, convex prediction | Tasks 1 and 4 |
| §13 E-distance, dropout, LOGO training | Tasks 2 and 5 |
| §14 strong baselines and mechanism nulls | Task 3 |
| §15 W/C metrics, Delta PCC, hierarchy | Task 6 |
| §16 denominators and development/external/final gates | Tasks 7, 9–11 |
| §17 staged unseal, hashes, sentinel, replay | Tasks 8, 10–11 |
| §18 typed decisions and abort outcomes | Tasks 7–11 |
| §20 model/protocol tests | Tasks 1–12 |
| §21 model, metric, guard, and decision artifacts | Tasks 8–12 |
| §22 AAAI claim verifier | Task 12 |

