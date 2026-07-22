# Local Anchor + Source-Context Residual Flow Smoke

日期：2026-07-20

## 目的

本轮验证 decoder：

base = target local anchor + (source expression - source-context anchor)
prediction = base + bounded residual modulation

source anchor 对 control/query 使用 leave-one-out，避免 query 的自身
control row 让 residual 被压成零。训练和推理只使用 target reference rows，
并用 evaluation-expression mutation guard 检查泄漏。

## 首个真实任务

任务：GSM5808054_patchclose_tumor_Jak2
Reference fraction：0.70
Split：random-stratified
Seed：20260711

Fixed 80/20 指标：

| E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: |
| 52.9304 | 0.9846 | 0.4934 | 1.0000 |

### Horizon 对照

| 方法 | Horizon | E-distance | Delta PCC | Boundary | Individuality |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base prediction | 0.00 | 77.7020 | 0.9032 | 1.2512 | 0.9980 |
| Local residual flow | 0.25 | 70.6734 | 0.9032 | 1.1873 | 0.9916 |
| Local residual flow | 0.50 | 69.7014 | 0.8922 | 1.1529 | 0.9815 |
| Local residual flow | 1.00 | 84.2580 | 0.8288 | 1.1512 | 0.9745 |

所有 horizon 的 mutation guard 均通过，且 uses_evaluation_target=False。

## 解释

这次结果没有证明 local anchor 结构可用，反而定位了新的冲突：

1. target local anchor 的 target location 不够稳定。它由 context 最近邻
   reference expression 给出，单独作为 base 时 E-distance 已为 77.70，显著
   差于 fixed 80/20 的 52.93。
2. 加入 source expression - source-context anchor 后 individuality 仍接近
   1，但 source-specific residual 也把 source distribution 的误差带回了
   target 空间。因此短 horizon 只能小幅改善 E-distance，不能修复 target
   location；长 horizon 进一步损害 delta PCC。
3. 这不是 identity loss 权重不足的问题。identity 已经接近满分，但
   E-distance、delta PCC 和 boundary 同时失败，说明需要约束 correction 的
   形式，而不是继续重加 loss。

## 决策

当前 local residual flow 不进入 multi-task 或 multi-seed scaling：

- distribution gate：失败（所有 horizon 的 E-distance 均高于 fixed）；
- delta gate：失败（最高仅 0.9032）；
- boundary gate：失败（约为 fixed 的 2.3 倍以上）；
- individuality：通过，但不足以抵消前三项失败；
- leakage guard：通过。

下一步应回到结构化 decoder：先学习对所有 query 共享的 global flow shift，
再叠加有界、以 context 为条件的 modulation，并在 reference pseudo-validation
上选择 modulation cap。共享 shift 负责 target location，有界 modulation 只
提供局部差异，从结构上保护 source pairwise geometry。

原始数据：results/local_residual_smoke/summary.json。
