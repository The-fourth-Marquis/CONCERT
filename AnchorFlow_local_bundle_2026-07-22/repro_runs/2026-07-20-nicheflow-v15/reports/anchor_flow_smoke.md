# Anchor-Initialized Residual Flow Smoke

日期：2026-07-20

## 目的

Direct conditional flow 从 source expression 出发，必须同时学习 target-state location、perturbation shift 和 context-specific correction。首个真实任务上训练数值稳定，但 E-distance、delta 和 boundary 全部失败。

本轮将初始状态改为 reference-only context anchor，并保留 source skip：

    initial = (1 - source skip) × target-reference context anchor
              + source skip × source expression

conditional flow 只学习 initial anchor 到 OT barycentric target 的残差场。

## 验证

- 合成数据上，context anchor 比 source 更接近匹配 target；
- residual velocity loss 随训练下降；
- integration horizon 为零时，prediction 严格等于 initial anchor；
- 修改 target evaluation expression 不改变 initial anchor 或 final prediction；
- 全部训练和预测在标准化 latent 中进行。

## 首个真实任务

任务：GSM5808054_patchclose_tumor_Jak2  
Reference fraction：0.70  
Split：random-stratified  
Seed：20260711

Fixed 80/20：

| E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: |
| 52.9304 | 0.9846 | 0.4934 | 1.0000 |

### 初始 anchor 与完整 flow

| 方法 | E-distance | Delta PCC | Boundary | Individuality |
| --- | ---: | ---: | ---: | ---: |
| Initial anchor | 45.9050 | 0.9931 | 0.5072 | 0.9717 |
| Full flow, horizon 1.0 | 31.2464 | 0.9824 | 0.6620 | 0.5124 |

Anchor initialization 已解决 direct flow 无法定位 target distribution 的问题。完整 correction 进一步降低 E-distance，但重新破坏 boundary 和 individuality。

### Integration horizon

Identity weight 固定为 0.05，source skip 固定为 0.20：

| Horizon | E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: | ---: |
| 0.00, initial | 45.9050 | 0.9931 | 0.5072 | 0.9717 |
| 0.25 | 38.8719 | 0.9923 | 0.4941 | 0.9255 |
| 0.50 | 33.3947 | 0.9900 | 0.5255 | 0.7801 |
| 0.75 | 30.4041 | 0.9866 | 0.5828 | 0.5498 |
| 1.00 | 31.2464 | 0.9824 | 0.6620 | 0.5124 |

Horizon 0.25 是当前最佳 distribution/boundary 折中：

- E-distance 比 fixed 80/20 低 14.06；
- delta PCC 高 0.0077；
- boundary 只高 0.0007；
- individuality 低 0.0745，未通过非劣门。

### Identity loss

完整 horizon 下：

| Identity weight | E-distance | Boundary | Individuality |
| ---: | ---: | ---: | ---: |
| 0.05 | 31.2464 | 0.6620 | 0.5124 |
| 0.50 | 51.8658 | 0.9602 | 0.7442 |
| 2.00 | 87.0544 | 1.1759 | 0.8494 |

增加 identity loss 虽提高 individuality，却同时放大 correction norm 并破坏其他指标。普通 loss reweighting 不能解决结构冲突。

### Source skip

Horizon 固定为 0.25：

| Source skip | E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: | ---: |
| 0.20 | 38.8719 | 0.9923 | 0.4941 | 0.9255 |
| 0.30 | 44.3612 | 0.9934 | 0.5671 | 0.9649 |
| 0.40 | 54.3920 | 0.9926 | 0.6591 | 0.9911 |

更高 source skip 可以恢复 individuality，但会把 prediction 拉回错误 source distribution，导致 E-distance 和 boundary 重新失败。

## 结论

Anchor initialization 是有效方向：

- 相比 direct flow，target distribution location、delta direction 和 boundary 显著修复；
- flow correction 在短 horizon 下提供额外 E-distance 增益；
- mutation guard 全部通过。

但当前没有配置同时通过 E-distance、delta、boundary 和 individuality：

    promising_but_not_advance_to_multitask

当前瓶颈是 query-specific correction field 改变了 spot 间的相对距离结构。训练集上的 pairwise identity loss不能保证该性质泛化到 query spots。

## 下一结构改动

下一版不应继续扫描 identity loss 或 source skip。需要把 individuality 写入 decoder contract：

    correction = global flow shift
                 + bounded context-specific modulation

其中 global shift 对所有 query spots 相同，因此严格保留 pairwise source geometry；context modulation 需要在 reference pseudo-validation 中选择强度，并加入 boundary-aware supervision。

只有该结构在当前 task 同时满足：

- E-distance 不高于 fixed 80/20；
- delta PCC difference 不低于 -0.02；
- boundary relative difference 不高于 5%；
- individuality difference 不低于 -0.02；

才进入 6-task pilot。
