# AnchorFlow GPU Submission Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit CUDA device support and run the four frozen AnchorFlow submission-smoke views on physical GPU 2, producing verified decision artifacts.

**Architecture:** Preserve the existing CPU NumPy/SVD/Sinkhorn path and make only the PyTorch velocity network device-selectable. A new isolated runner imports the existing smoke function, executes the two frozen task indices under two frozen split types sequentially, and aggregates only held-out E-distance into the predeclared decision.

**Tech Stack:** Python 3.11 via `uv`, NumPy, pandas, PyTorch 2.1/CUDA 12.1, existing NicheFlow-v1.5/v2 modules.

---

### Task 1: Explicit CUDA Device Propagation

**Files:**
- Modify: `repro_runs/2026-07-20-nicheflow-v15/src/nicheflow_v15/anchor_flow.py`
- Modify: `repro_runs/2026-07-20-nicheflow-v15/src/nicheflow_v15/anchor_flow_predict.py`
- Modify: `repro_runs/2026-07-20-nicheflow-v15/run_anchor_flow_smoke.py`
- Create: `repro_runs/2026-07-20-nicheflow-v15/tests/test_anchor_flow_device.py`

- [ ] **Step 1: Write the failing device test**

The test calls `fit_predict_anchor_residual_flow(..., device="cpu")` on the
existing two-gene synthetic example and asserts:

```python
assert result.metadata["device"] == "cpu"
assert result.metadata["precision"] == "float32"
```

It also calls a new `_resolve_torch_device("cuda:0")` under a monkeypatched
`torch.cuda.is_available == False` and expects `RuntimeError` containing
`CUDA device requested but CUDA is unavailable`.

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
rtk uv run python repro_runs/2026-07-20-nicheflow-v15/tests/test_anchor_flow_device.py
```

Expected: import or keyword failure because device selection is not implemented.

- [ ] **Step 3: Implement the minimal device contract**

Add to `anchor_flow.py`:

```python
def _resolve_torch_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device
```

Add `device: str | torch.device = "cpu"` to
`fit_predict_anchor_residual_flow`, replace the hard-coded CPU device with the
resolver, seed CUDA when selected, and record `device` and `precision` in
metadata. Add `device: str = "cpu"` to the task wrapper and smoke function and
pass it through unchanged. Existing callers retain CPU behavior by default.

- [ ] **Step 4: Run focused and full v1.5 tests**

```bash
rtk uv run python repro_runs/2026-07-20-nicheflow-v15/tests/test_anchor_flow_device.py
rtk uv run python repro_runs/2026-07-20-nicheflow-v15/tests/run_all.py
```

Expected: focused test prints `ok`; full runner prints `PASS 24 test files`.

### Task 2: Frozen Submission Runner Contracts

**Files:**
- Create: `repro_runs/2026-07-22-anchorflow-submission-smoke/config.json`
- Create: `repro_runs/2026-07-22-anchorflow-submission-smoke/run_submission_smoke.py`
- Create: `repro_runs/2026-07-22-anchorflow-submission-smoke/tests/test_contract.py`
- Create: `repro_runs/2026-07-22-anchorflow-submission-smoke/README.md`

- [ ] **Step 1: Write failing frozen-config and decision tests**

The config test asserts exact task mappings:

```python
assert task_ids == {
    13: "GSM5808055_border_surrogate_KP_to_Tgfbr2",
    16: "GSM5808055_cross_slide_surrogate_KP_to_Jak2_to_GSM5808054",
}
assert split_types == ["random_stratified", "spatial_block"]
assert settings["integration_horizon"] == 0.25
assert settings["device"] == "cuda:0"
assert settings["seed"] == 20260711
```

The decision test provides four synthetic summary dictionaries and asserts that
`aggregate_decision` returns `confirmation_positive` only when flow beats both
comparators on aggregate, in both split strata, in at least three views, with
all mutation and finiteness guards true.

- [ ] **Step 2: Run the contract test and observe RED**

```bash
rtk uv run python repro_runs/2026-07-22-anchorflow-submission-smoke/tests/test_contract.py
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement the isolated runner**

The runner must:

1. load and validate `config.json` before importing task data;
2. require `CUDA_VISIBLE_DEVICES=2`, `torch.cuda.is_available()`, and one
   process-visible CUDA device;
3. set the per-process CUDA memory fraction to 0.10;
4. run task indices 13 and 16 across the two split types sequentially;
5. pass `device="cuda:0"` and all frozen settings to
   `run_anchor_flow_smoke`;
6. refuse to overwrite an existing `results/` directory;
7. write twelve metric rows for fixed, anchor, and flow;
8. compute the decision from `final_heldout_edistance`,
   `initial_heldout_edistance`, and `fixed_80_20_heldout_edistance` only;
9. record CUDA/PyTorch/device metadata and all mutation/finiteness guards.

- [ ] **Step 4: Run the contract test and observe GREEN**

```bash
rtk uv run python repro_runs/2026-07-22-anchorflow-submission-smoke/tests/test_contract.py
```

Expected: `ok`.

### Task 3: GPU Sanity Check and Frozen Real Runs

**Files:**
- Create: `repro_runs/2026-07-22-anchorflow-submission-smoke/results/*`

- [ ] **Step 1: Verify the shared GPU mapping**

```bash
rtk nvidia-smi
rtk env CUDA_VISIBLE_DEVICES=2 uv run python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Expected: one visible CUDA device and an NVIDIA RTX PRO 6000 Blackwell name.

- [ ] **Step 2: Run the four frozen views once**

```bash
rtk env CUDA_VISIBLE_DEVICES=2 uv run python repro_runs/2026-07-22-anchorflow-submission-smoke/run_submission_smoke.py
```

Expected: four completed view directories followed by paths to
`metrics.csv` and `aggregate_summary.json`. No rerun is authorized after output
publication.

### Task 4: Independent Artifact Verification and Report

**Files:**
- Create: `repro_runs/2026-07-22-anchorflow-submission-smoke/verify_results.py`
- Modify: `repro_runs/2026-07-22-anchorflow-submission-smoke/README.md`

- [ ] **Step 1: Write the verifier**

The verifier independently reads the four per-view summaries and checks task
IDs, split types, settings, CUDA metadata, mutation guards, finite metrics, row
count 12, and exact recomputation of every decision boolean and mean.

- [ ] **Step 2: Run verification**

```bash
rtk env CUDA_VISIBLE_DEVICES=2 uv run python repro_runs/2026-07-22-anchorflow-submission-smoke/verify_results.py
```

Expected: a JSON object with `verified: true` and the frozen decision.

- [ ] **Step 3: Write the decision-oriented README**

Report the claim boundary, frozen protocol, per-view E-distance table,
aggregate differences, exact decision, mutation evidence, descriptive
Delta/boundary/individuality diagnostics, and whether the provisional abstract
may be strengthened. Do not describe a negative confirmation as method success.

- [ ] **Step 4: Run final scoped verification**

```bash
rtk uv run python repro_runs/2026-07-20-nicheflow-v15/tests/run_all.py
rtk env CUDA_VISIBLE_DEVICES=2 uv run python repro_runs/2026-07-22-anchorflow-submission-smoke/tests/test_contract.py
rtk env CUDA_VISIBLE_DEVICES=2 uv run python repro_runs/2026-07-22-anchorflow-submission-smoke/verify_results.py
rtk git status --short repro_runs/2026-07-20-nicheflow-v15 repro_runs/2026-07-22-anchorflow-submission-smoke
```

Expected: all tests pass, verifier reports `verified: true`, and scoped status
contains only the intended implementation and experiment artifacts.

