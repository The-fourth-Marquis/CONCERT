# NicheFlow-v1.5 Nonlinear Transport Prototype

This directory is an isolated experiment root for replacing the current
anchor/residual convex blend with a reference-only, niche-conditioned nonlinear
transport model.

## Scientific scope

The experiment remains inside Mainline A: observed-perturbation spatial
relocation with explicitly reported target-reference access. It is not a strict
unseen/OOD perturbation experiment and does not support spot-level causal
counterfactual claims.

The frozen evidence tree at `repro_runs/2026-06-22-full-repro/` and the v2
experiment at `repro_runs/2026-07-11-nicheflow-v2/` are read-only inputs.

## First implementation slice

1. Build a reference-only niche graph without evaluation-target rows.
2. Construct an entropic Sinkhorn coupling between source/control and target
   reference representations.
3. Fit a small nonlinear latent transport map while retaining the existing
   anchor as a reported fallback rather than blending it into the primary
   prediction.
4. Reuse the v2 held-out splits, metrics, and evaluation-expression mutation
   guard.

## Promotion boundary

The prototype is not promoted unless it improves the uncalibrated fixed 80/20
baseline under held-out evaluation, including the 0.50 reference-fraction
spatial-block condition, without weakening the existing delta, DE-direction,
boundary, individuality, coverage, or leakage gates.

## Commands

Run unit tests:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/tests/run_all.py

Run one real-task smoke comparison:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_smoke.py


Run the representative base pilot:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_pilot.py

Run the intrinsic-preserving modulation pilot:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_pilot.py --config-set intrinsic --output-dir repro_runs/2026-07-20-nicheflow-v15/results/pilot_intrinsic_preserving

## Current decision

The closed-form nonlinear Monge map is not approved for multiseed or full-task
scaling. The source-preserving modulation pilot restored individuality but failed
E-distance, delta-PCC, and boundary gates.

Detailed findings: reports/pilot_findings.md


## Conditional residual flow

Run the trainable flow smoke:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_flow_smoke.py

The direct control-to-target flow is numerically stable and leakage-safe, but it
does not pass the first real-task distribution, delta, or boundary gates. It is
not approved for the 6-task pilot. The next implementation target is an
anchor-initialized residual flow.

Detailed findings: reports/conditional_flow_smoke.md


## Anchor-initialized residual flow

Run the anchor-flow smoke:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_anchor_flow_smoke.py --integration-horizon 0.25

Anchor initialization fixes the direct flow target-location failure and yields a
promising E-distance/delta/boundary result at horizon 0.25. No tested setting
also passes the individuality gate, so the method is not yet approved for the
6-task pilot.

Detailed findings: reports/anchor_flow_smoke.md


## Structured source-shift decoder

Run the structured decoder smoke:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_structured_smoke.py

The structured decoder preserves source individuality but cannot replace local
anchor information with a shared global shift. It is not approved for the
multi-task pilot.

Detailed findings: reports/structured_flow_smoke.md


## Local anchor plus source-context residual flow

Run the local residual smoke:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_local_residual_smoke.py

The leave-one-out source-context residual preserves source individuality, but
the target-local anchor has poor target-distribution location on the first real
task. All tested integration horizons fail the fixed 80/20 E-distance, delta,
and boundary gates. The method is not approved for multi-task scaling.

Detailed findings: reports/local_residual_flow_smoke.md

The next decoder target is a shared global flow shift plus bounded
context-conditioned modulation, with the modulation cap selected by reference
pseudo-validation.


## Anchor-guided factorized decoder

Run the anchor-guided smoke:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_anchor_guided_smoke.py

This decoder uses the mean target-reference anchor displacement as a shared
global shift and scales the target-local anchor residual by a bounded
modulation strength. On the first real task, strength 0.77 passes the four
mechanism gates, but this is still a single-task smoke result. The strength
must be selected with reference-only pseudo-validation before a multi-task
pilot.

Detailed findings: reports/anchor_guided_factorized_smoke.md


## Reference-calibrated anchor-guided pilot

Run the calibrated six-task pilot:

    rtk uv run python repro_runs/2026-07-20-nicheflow-v15/run_anchor_guided_calibrated_pilot.py

The first task passes all four mechanism gates after reference-only calibration
selects strength 0.76. The six-task pilot (random-stratified and spatial-block)
does not pass distribution, delta, or boundary promotion gates, although
individuality is preserved. The decoder is therefore not promoted to 37-task or
multi-seed evaluation.

Detailed findings: reports/anchor_guided_calibrated_pilot.md
