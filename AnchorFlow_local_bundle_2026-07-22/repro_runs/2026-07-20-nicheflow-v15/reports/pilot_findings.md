# NicheFlow-v1.5 代表性 Pilot 结论

日期：2026-07-20

## 结论

当前 reference-only nonlinear Monge map 不应扩大到五个 seeds 或完整 37-task benchmark。

实验显示，低 E-distance 可以通过强制预测贴近 target-reference distribution 获得，但这种改进会显著破坏 source individuality 和 boundary behavior。加入严格 intrinsic identity 路径后，individuality 得到恢复，但 E-distance、delta PCC 和 boundary 指标整体恶化。因此问题不是简单的 latent dimension 或输出 decoder，而是当前 OT coupling 与训练目标没有同时约束分布接近、空间边界和 source identity。

最终决策：

    do_not_scale_current_monge

下一阶段应停止继续扫描当前 closed-form Monge map，转向带显式多目标损失的 trainable conditional flow 或 residual field。

## 协议

Pilot 使用 6 个代表性 Mainline A 任务：

- exact patch；
- border；
- cross-niche core-to-border；
- cross-slide；
- niche core；
- periphery。

每个任务使用：

- reference fraction 0.50；
- seed 20260711；
- random-stratified 与 spatial-block；
- target evaluation-expression mutation guard；
- uncalibrated fixed 80/20 作为比较基线。

每轮包含 6 tasks × 2 split types × 3 configurations = 36 units。

Mainline A 的 claim boundary 保持不变：已观测 perturbation 的空间重定位，允许明确报告的 target-reference access，不是严格 unseen/OOD perturbation prediction。

## 第一轮：基础 nonlinear Monge map

最佳配置为 context_l16：

| 指标 | 结果 |
| --- | ---: |
| Mean E-distance delta vs fixed 80/20 | -1.2927 |
| Spatial-block mean delta | -0.3459 |
| Wins | 8/12 |
| Mean delta-PCC difference | -0.0010 |
| Mean boundary difference | +0.0510 |
| Mean individuality difference | -0.7967 |

E-distance 略有优势，但 source individuality 大幅下降，因此不能晋级。

## 第二轮：保留低维正交残差

将输出从完整 PCA reconstruction 改为 source expression 加 latent delta 后，context_l16 的分布结果进一步改善：

| 指标 | 结果 |
| --- | ---: |
| Mean E-distance delta vs fixed 80/20 | -10.1736 |
| Spatial-block mean delta | -9.7976 |
| Wins | 8/12 |
| Mean delta-PCC difference | -0.0031 |
| Mean boundary relative difference | +0.7343 |
| Mean individuality difference | -0.8333 |

这说明原始 decoder 的确损失了部分表达信息，但它不是 individuality collapse 的主要原因。真正的问题是 query-specific Monge delta 本身强烈扭曲了 spots 之间的相对结构。

## 第三轮：intrinsic identity 加有界 context modulation

预测被改写为：

    prediction
    = source expression
    + global reference perturbation shift
    + modulation strength × centered nonlinear context delta

modulation strength 取 0、0.25 和 1.0。

最佳配置为 modulation_025_l16：

| 指标 | 结果 |
| --- | ---: |
| Mean E-distance delta vs fixed 80/20 | +13.3683 |
| Spatial-block mean delta | +13.2864 |
| Wins | 3/12 |
| Mean delta-PCC difference | -0.0515 |
| Mean boundary relative difference | +1.2443 |
| Mean individuality difference | -0.00085 |
| Leakage checks | 12/12 passed |

严格 identity path 基本恢复了 individuality，但分布距离、扰动方向和边界指标均未通过。因此当前 Monge map 的分布优势与机制完整性之间存在实质冲突。

## 科学解释

1. 当前 Sinkhorn cost 主要在低维 expression 和手工 context features 上匹配，不能识别真正可迁移的 perturbation response。
2. OT barycentric targets 倾向于压缩 heterogeneous source states，使 E-distance 下降，却牺牲 spot-level individuality。
3. 单纯增加 latent dimension 能改善 distribution matching，但不会自动学习正确空间机制。
4. 全局 perturbation mean shift 可以保留 individuality，却不能描述 border、periphery 和 niche-dependent response。
5. 因此下一模型必须把 source identity、target distribution 和 spatial mechanism 同时写入训练目标，而不是事后用一个 modulation scalar 折中。

## 下一实现方向

停止扩大当前 closed-form Monge map。下一原型应实现 trainable conditional residual flow：

    velocity = f(expression latent, graph context, perturbation, time)

训练目标至少包含：

- conditional flow-matching loss；
- reference-only Sinkhorn or MMD distribution loss；
- source pairwise-distance preservation loss；
- boundary consistency loss；
- group-level perturbation delta and DE-direction loss。

模型仍需使用同一 held-out split、access labeling 和 mutation guard。只有在 6-task pilot 同时通过 E-distance、delta、boundary 和 individuality 后，才进入五 seeds 和完整 37 tasks。
