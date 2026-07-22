# Existing-Method Reassessment Under Held-out Target Evaluation

Date: 2026-07-11

## Scope

This reassessment covers 37 Mainline A observed-perturbation relocation tasks, reference fractions 0.25, 0.50, and 0.70, five seeds, and random-stratified plus spatial-block target splits. Reference targets may be used for prediction and labeled calibration; held-out evaluation targets are metric-only. Every reconstructed metric row passed the evaluation-expression mutation guard.

## Coverage

- Task/split units: 1,110
- Metric rows: 14,430
- Coverage rows: 15,540
- Access levels represented: L0, L1, L2, L3
- Fixed CONCERT output: not_eligible_for_heldout_primary
- SpatialProp reference-only retraining: 1,110 approximate rows, all leakage guards passed

## Mean held-out results

| Method | Access | Held-out E-distance | Niche-stratified E-distance | Delta PCC | DE direction | Boundary | Individuality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy v1 fixed 80/20 calibrated | L3 | 55.3383 | 57.4979 | 0.8914 | 0.9686 | 0.7025 | 0.6403 |
| adaptive calibrated anchor | L2 | 71.3564 | 72.9853 | 0.8897 | 0.9673 | 0.6430 | 0.0661 |
| legacy v1 fixed 80/20 | L3 | 72.9115 | 75.8099 | 0.8737 | 0.9677 | 0.6803 | 0.9134 |
| Gaussian anchor | L1 | 92.9950 | 93.9899 | 0.9012 | 0.9695 | 0.5187 | 0.0951 |
| Cauchy anchor | L1 | 94.4492 | 95.3460 | 0.9025 | 0.9695 | 0.5119 | 0.0680 |
| source identity | L0 | 110.5040 | 127.7295 | undefined | 0.0000 | 1.5530 | 1.0000 |
| SpatialProp reference-only retrain | L2 | 146.8739 | 151.5784 | 0.6666 | 0.8954 | 1.6402 | 0.0761 |
| legacy NicheFlow-lite residual | L3 | 151.9095 | 169.8000 | 0.5596 | 0.8808 | 1.6128 | 0.9926 |

Undefined correlations remain missing with explicit reason fields.

## Access-matched conclusions

Calibrated legacy v1 versus the independent L2 calibrated anchor wins 1,101/1,110 paired units, with mean held-out E-distance delta -16.0182 and task-clustered 95% CI [-18.2811, -13.9934].

Uncalibrated legacy v1 versus the strongest uncalibrated A2.1 reference-only baseline wins 587/1,110 units and has mean delta +4.3425.

The held-out ranking therefore preserves a calibrated v1 advantage, but the calibration-free residual evidence is insufficient. This motivated explicit residual semantics, invariant matching, and a separately labeled reliability-gating ablation.

## Claim boundary

These are held-out evaluations for observed-perturbation spatial relocation with reported target-reference access. They do not establish strict unseen/OOD perturbation prediction or spot-level causal counterfactual accuracy.

See reports/NicheFlow_v2_report.md for the final v2 results and graph/flow decision.
