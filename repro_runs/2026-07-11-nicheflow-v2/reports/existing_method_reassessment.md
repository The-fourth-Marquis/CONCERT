# Existing-Method Reassessment Under Held-out Target Evaluation

Date: 2026-07-11

## Scope

This reassessment freezes the evidence available before implementing the new NicheFlow-v2 residual/gating module. It covers 37 Mainline A tasks, three target-reference fractions (0.25, 0.50, 0.70), five split seeds, and random-stratified plus spatial-block splits.

The observed target pool is divided into disjoint reference and evaluation subsets. Reference rows may be used for anchor, residual construction, or calibration; evaluation rows are used only for metrics. Every reconstructed prediction passed an evaluation-expression mutation test. Fixed available CONCERT outputs are not eligible for the held-out primary ranking because their training-time target access cannot be reconstructed from the fixed files.

## Coverage

- Tasks: 37
- Task/split configurations: 1,110
- Eligible method rows: 12,210
- Coverage rows: 14,430
- Shared genes: 1,053
- SpatialProp-style reference-only retraining: excluded from this fast reassessment and recorded as unavailable, pending the separate expensive lane

## Mean Results Across Task/Split Configurations

| Method | Held-out E-distance | Delta PCC | Top-20 DE direction accuracy | Boundary leakage | Source individuality | Distance-response decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy v1 fixed 80/20 calibrated | 55.3383 | 0.8914 | 0.9686 | 0.7025 | 0.6403 | -0.0576 |
| legacy v1 fixed 80/20 uncalibrated | 72.9115 | 0.8737 | 0.9677 | 0.6803 | 0.9134 | -0.0650 |
| Gaussian anchor | 92.9950 | 0.9012 | 0.9695 | 0.5187 | 0.0951 | -0.0858 |
| Cauchy anchor | 94.4492 | 0.9025 | 0.9695 | 0.5119 | 0.0680 | -0.0823 |
| kNN-10 anchor | 94.4636 | 0.8523 | 0.9618 | 0.6796 | 0.0740 | -0.0879 |
| Graph diffusion | 95.8992 | 0.9013 | 0.9695 | 0.5148 | 0.0954 | -0.0854 |
| Target mean | 98.9969 | 0.9067 | 0.9692 | 0.4957 | undefined | -0.0819 |
| kNN-5 anchor | 103.2961 | 0.8102 | 0.9481 | 0.8097 | 0.0629 | -0.0982 |
| Source identity | 110.5040 | undefined | 0.0000 | 1.5530 | 1.0000 | undefined |
| Legacy NicheFlow-lite residual prediction | 151.9095 | 0.5596 | 0.8808 | 1.6128 | 0.9926 | 0.1134 |
| Linear covariate ridge | 516.8672 | 0.4174 | 0.7373 | 4.0081 | 0.1058 | -0.5783 |

Undefined values are preserved as missing metrics with reason fields; they are not replaced by zero.

## Primary Comparison

The calibrated legacy-v1 reconstruction is compared with the best access-matched A2 baseline separately for each task, fraction, seed, and split type.

- Wins: 930/1,110
- Mean held-out E-distance delta: -13.2307
- Task-clustered bootstrap 95% CI: [-17.9271, -8.3320]

| Reference fraction | Split type | Wins | Configurations | Mean delta vs best baseline |
| ---: | --- | ---: | ---: | ---: |
| 0.25 | random-stratified | 167 | 185 | -18.2287 |
| 0.25 | spatial-block | 149 | 185 | -8.9818 |
| 0.50 | random-stratified | 173 | 185 | -18.4978 |
| 0.50 | spatial-block | 115 | 185 | +0.2595 |
| 0.70 | random-stratified | 169 | 185 | -17.8026 |
| 0.70 | spatial-block | 157 | 185 | -16.1326 |

The aggregate held-out advantage is statistically supported at the task-cluster level, but it is not uniform. The 0.50 spatial-block setting is the clear sensitivity case and must remain visible in later reporting.

## Calibration Audit

- Uncalibrated v1 versus best access-matched baseline: 587/1,110 wins
- Mean uncalibrated delta versus baseline: +4.3425
- Calibrated versus uncalibrated v1: 1,082/1,110 wins
- Mean calibration improvement: -17.5732 E-distance

Calibration remains the dominant source of held-out E-distance improvement. The result is more defensible than the earlier full-pool evaluation because calibration uses only reference targets, but it still does not establish that the residual mechanism is independently strong.

## Metric Trade-offs

The calibrated model has the best mean held-out E-distance, while simple anchors have slightly stronger mean delta PCC, DE direction accuracy, and boundary leakage. The uncalibrated 80/20 ensemble preserves much more source individuality than simple anchors but does not improve mean E-distance over the best baseline. Legacy residual transfer alone preserves source individuality and has positive distance-response decay, but its distribution and delta accuracy are poor.

These trade-offs justify retaining separate metrics rather than using a composite score.

## Decision for the First v2 Module

Proceed with the approved lightweight module:

1. explicit pure `delta_context` separated from full residual prediction;
2. perturbation-invariant matching against the full eligible control pool;
3. reliability gating between reference anchor and residual prediction;
4. calibrated and uncalibrated variants reported separately.

Do not proceed directly to Graph Transformer or latent flow. The immediate scientific question is whether better residual semantics and gating can retain the held-out E-distance advantage without depending on calibration and without sacrificing delta, DE, and boundary metrics.

## Claim Boundary

These results support observed-perturbation spatial relocation with held-out target evaluation. They do not support never-seen perturbation prediction, strict causal-effect estimation, or mechanism discovery from E-distance alone.
