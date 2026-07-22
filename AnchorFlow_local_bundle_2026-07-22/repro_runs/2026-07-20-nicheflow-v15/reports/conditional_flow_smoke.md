# Conditional Residual Flow Smoke

日期：2026-07-20

## 实现

新增了可训练的低维 conditional residual flow：

    source/control expression
    + graph-derived context
    + target-reference Sinkhorn coupling
    -> conditional velocity network
    -> Euler integration
    -> query expression plus integrated latent delta

训练只使用 target reference rows。target evaluation rows 不进入 latent basis、context normalization、Sinkhorn coupling、loss 或推理。

训练目标包含：

- conditional velocity smooth-L1 loss；
- barycentric endpoint loss；
- endpoint covariance loss；
- source pairwise-distance preservation loss。

## 验证

已通过以下独立契约：

1. 合成非线性数据上训练 loss 下降且预测不是 identity；
2. expression 全局缩放 100 倍时，标准化 latent 的训练 loss 保持一致，预测按相同比例缩放；
3. 真实 task 中修改 evaluation target expression 不改变 prediction；
4. prediction、coupling 和训练 loss 均为有限值。

## 首个真实任务

任务：

    GSM5808054_patchclose_tumor_Jak2

协议：

- reference fraction 0.70；
- random-stratified；
- seed 20260711；
- latent dimension 8；
- hidden dimension 32；
- 60 epochs；
- 16 Euler integration steps；
- distribution weight 0.25；
- identity weight 0.05；
- no calibration。

### 数值稳定性修复前

训练直接使用未标准化 PCA latent：

| 指标 | Flow | Fixed 80/20 |
| --- | ---: | ---: |
| Training loss first | 792648.6 | - |
| Training loss last | 83288.5 | - |
| Held-out E-distance | 126.33 | 52.93 |
| Source individuality | 0.9924 | 1.0000 |

### 标准化修复后

| 指标 | Flow | Fixed 80/20 | Difference |
| --- | ---: | ---: | ---: |
| Training loss first | 0.4635 | - | - |
| Training loss last | 0.2654 | - | - |
| Held-out E-distance | 113.63 | 52.93 | +60.70 |
| Delta PCC | 0.8178 | 0.9846 | -0.1667 |
| Delta MAE | 0.9073 | 0.4516 | +0.4557 |
| Boundary leakage | 1.3547 | 0.4934 | +0.8612 |
| Source individuality | 0.9839 | 1.0000 | -0.0161 |

Mutation guard 通过。

## 解释

训练稳定性已经不是主要问题。flow 能够学习 velocity field，也基本保留 source individuality，但 direct control-to-target barycentric supervision 在真实任务上产生错误的 perturbation direction 和 boundary behavior。

当前 reference pool 很小，OT coupling 将完整 target state 当成 flow endpoint。对于 query context，网络必须同时学习全局 target shift、spatial anchor 和 context-specific residual，监督过弱且目标混杂。这使模型接近 source identity，却没有复现强 spatial anchor 的 target-state location。

## 决策

当前 direct conditional flow 不进入 6-task pilot：

    do_not_scale_direct_flow

下一变体应为 anchor-initialized residual flow：

    initial state = reference-only spatial anchor
    learned field = source/context-conditioned correction
    endpoint = integrate(initial state, conditional residual velocity)

训练时应把 anchor error 而不是完整 target expression 作为 flow target，并在 reference 内部划分 pseudo-validation rows，显式优化：

- held-out distribution loss；
- group delta and DE-direction；
- boundary consistency；
- source pairwise identity。

只有 anchor-initialized 版本在首个 task 同时达到 E-distance 非劣、delta PCC 非劣和 individuality 非劣，才进入 6-task pilot。
