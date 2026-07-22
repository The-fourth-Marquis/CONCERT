# AnchorFlow Frozen Submission Confirmation

## Scope

This isolated experiment evaluates a reference-anchored conditional residual
flow for few-shot, observed-perturbation gene-expression distribution
generation. The method receives an explicitly reported target-reference pool
and a requested spatial context.

The experiment does not claim per-source-spot causal counterfactual correctness,
strict unseen/OOD perturbation prediction, source-geometry preservation, or
perturbation spatial-distribution completion.

## Frozen protocol

Two untouched target pools were each evaluated under random-stratified and
spatial-block reference/evaluation splits, giving four diagnostic views.
These views represent two biological target pools, not four independent
biological replicates.

The model used reference fraction 0.70, seed 20260711, latent dimension 8,
hidden dimension 32, 60 epochs, 16 integration steps, horizon 0.25, anchor
temperature 0.25, source skip 0.20, distribution weight 0.25, and identity
weight 0.05. There was no post-result tuning or rescue run.

E-distance was the sole promotion endpoint. Delta PCC, boundary leakage, and
source individuality were retained only as descriptive diagnostics.

## Compute contract

The four views ran sequentially on shared physical GPU 2. The process saw that
device as cuda:0 and was limited to 10% of GPU memory. The run used an NVIDIA
RTX PRO 6000 Blackwell Workstation Edition, PyTorch 2.12.0+cu130, CUDA 13.0,
sm_120 kernels, and FP32 without mixed precision.

The project default environment was not used because its PyTorch build did not
contain sm_120 kernels. The compatible interpreter was frozen in config.json
and validated before results were created.

## Primary results

Lower E-distance is better.

| Target / split | Fixed 80/20 | Initial anchor | AnchorFlow | Flow - anchor | Flow - fixed | Wins both |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Tgfbr2 border / random | 62.009 | 248.573 | 203.125 | -45.448 | +141.116 | No |
| Tgfbr2 border / spatial block | 117.674 | 283.542 | 249.824 | -33.719 | +132.149 | No |
| Jak2 cross-slide / random | 46.102 | 37.826 | 35.300 | -2.526 | -10.801 | Yes |
| Jak2 cross-slide / spatial block | 48.243 | 44.528 | 42.575 | -1.953 | -5.668 | Yes |
| Mean | 68.507 | 153.617 | 132.706 | -20.911 | +64.199 | 2 / 4 |

AnchorFlow improved on its initial anchor in all four views and by 20.911
E-distance on average. It did not beat the strong fixed linear comparator:
its mean E-distance was 64.199 higher, and it jointly beat both comparators in
only two of four views.

The split-stratum mean differences were:

| Split stratum | Flow - anchor | Flow - fixed |
| --- | ---: | ---: |
| Random-stratified | -23.987 | +65.157 |
| Spatial block | -17.836 | +63.240 |

## Frozen decision

Decision: insufficient_confirmation_for_performance_claim.

The following gates passed:

- mean AnchorFlow E-distance below the initial anchor;
- all evaluation-expression mutation guards;
- all metrics and saved prediction arrays finite.

The following gates failed:

- mean AnchorFlow E-distance below fixed 80/20;
- improvement over both comparators in each split stratum;
- joint wins in at least three of four views.

The run therefore cannot support a performance claim over the strong linear
baseline. No architecture, horizon, loss, task, split, or seed was changed
after observing the result.

## Descriptive diagnostics

On the Tgfbr2 border task, residual learning moved Delta PCC toward the target
relative to the initial anchor, but remained far behind fixed 80/20. Its
boundary leakage and source individuality were also worse than the fixed
comparator. On the Jak2 cross-slide task, AnchorFlow slightly improved Delta PCC
over both the anchor and fixed comparator, while source individuality remained
below fixed 80/20. These diagnostics do not alter the E-distance decision.

All four evaluation-expression mutation guards passed, every exposed
prediction array was finite, and the independent verifier exactly reproduced
the 12 metric rows, aggregate means, split gates, joint-win count, and final
decision.

## Submission implication

Keep the conservative title and abstract. The result supports the narrow
statement that learned residual dynamics improve the initial anchor on these
four views. It does not support saying that AnchorFlow outperforms linear or
access-matched baselines.

The method story should therefore be framed as a task and diagnostic
contribution plus a principled anchor-residual model, with the strong linear
baseline treated as an empirical few-shot ceiling that current nonlinear
learning has not yet surpassed. The title can remain method-forward, but the
abstract must not imply state-of-the-art predictive performance.
