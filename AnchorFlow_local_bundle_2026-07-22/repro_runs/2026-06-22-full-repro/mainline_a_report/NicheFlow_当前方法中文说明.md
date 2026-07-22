# NicheFlow 当前方法中文说明

生成日期：2026-07-09

## 1. 一句话定位

NicheFlow 当前版本是一个用于空间转录组扰动数据的 **observed perturbation spatial relocation** 方法：当某个扰动已经在组织中的一部分位置被观测到，但实验无法把该扰动精确施加到用户关心的位置时，NicheFlow 预测“如果该扰动发生在指定 spot、区域或生态位，会产生怎样的表达状态”。

更短地说：

> NicheFlow 将已观测扰动响应迁移到指定空间生态位，同时尽量保留局部组织上下文和 spot 个体差异。

这一定义非常重要。当前 NicheFlow-v1 **不是** strict unseen perturbation predictor，也不应被表述为“预测从未观测过的扰动”。它的主线问题是：目标扰动已经在数据中出现过，如何把这个扰动状态迁移到新的空间位置。

## 2. 为什么需要 NicheFlow

空间扰动技术的现实限制是：我们通常能观测到某些扰动在组织中的表达响应，但还不能任意控制扰动发生在特定空间位置、组织边界、肿瘤核心、肿瘤边缘或跨切片区域。

因此，当前更现实的计算问题不是马上做完全 OOD 的扰动预测，而是：

```text
已观测目标扰动池 + 用户指定源位置/生态位
    -> 目标扰动在这些位置上的表达预测
```

NicheFlow 要解决的是这个“扰动迁移”问题。它和传统插值或简单 target-state matching 的区别在于：NicheFlow 不只希望预测结果接近目标扰动总体分布，还希望预测结果保留 source spot 的空间上下文、边界属性和个体结构。

## 3. 当前版本：NicheFlow-v1

当前推荐汇报的版本是 `NicheFlow-v1`。它来自 A2.3 anchored NicheFlow baseline，并经过 A2.4 audit 后整理成一个显式组件模型。

核心形式是：

```text
prediction = 0.80 * target_pool_anchor
           + 0.20 * context_matched_residual

prediction = target_pool_calibration(prediction)
```

其中：

- `target_pool_anchor`：从已观测目标扰动池中，为每个 source spot 找到空间或上下文上合适的目标扰动参考状态。
- `context_matched_residual`：估计与局部 niche、boundary、空间上下文相关的残差响应，用作可解释的扰动迁移层。
- `target_pool_calibration`：将预测结果的矩分布校准到目标扰动池，使整体表达尺度与目标状态一致。

当前实现包含三个 NicheFlow-v1 变体：

| 变体 | 说明 |
| --- | --- |
| `nicheflow_v1_adaptive_anchor_residual` | 根据 task type 自动选择 anchor，当前主推荐版本。 |
| `nicheflow_v1_cauchy_anchor_residual` | 固定使用 Cauchy smoothing anchor。 |
| `nicheflow_v1_knn10_anchor_residual` | 固定使用 kNN-10 anchor。 |

adaptive 版本当前的经验策略是：

- exact patch、border、periphery、cross-slide、niche_core：优先使用 `cauchy` anchor；
- patch、cross_niche：优先使用 `knn10` anchor；
- 所有版本默认启用 target-pool calibration。

## 4. 输入与输出

### 输入

NicheFlow-v1 使用以下信息：

- source spots：用户关心的待预测位置，例如某个 patch、边界区域、肿瘤核心区域或跨切片 source spots。
- observed target perturbation pool：数据中已经观测到的目标扰动 spots。
- spatial coordinates：spot 空间坐标。
- tissue/niche/boundary context：例如 tumor、normal、边界距离、局部邻域组成等上下文信息。
- expression matrix：对齐后的基因表达矩阵，当前 Mainline A 使用 1053 个 Perturb-map shared genes。

### 输出

NicheFlow 输出每个 source spot 在目标扰动状态下的预测表达：

```text
n_source_spots x n_genes
```

这些预测会被拿来与真实目标扰动 selected cells 或 surrogate target cells 比较，主指标是 E-distance。

## 5. 方法实现详解

当前 NicheFlow-v1 不是一个重新训练的深度模型，而是一个无训练、可解释的 relocation pipeline。主入口在：

```text
repro_runs/2026-06-22-full-repro/nicheflow_v1.py
```

它复用两部分已有实现：

```text
nicheflow_anchored_baseline.py
    - target_pool_anchor
    - calibrate_to_target_pool

nicheflow_lite_benchmark.py
    - load_task_payload
    - predict_variant(..., variant="nicheflow_lite_full")
```

整体执行流程是：

```text
task manifest
    -> load_task_payload
    -> target_pool_anchor
    -> nicheflow_lite_full residual transfer
    -> 0.80 anchor + 0.20 residual prediction
    -> target-pool calibration
    -> metric_row / E-distance evaluation
```

### 5.1 数据读取与任务 payload

每个任务首先由 `load_task_payload(task)` 转成统一 payload。这个函数位于：

```text
repro_runs/2026-06-22-full-repro/nicheflow_lite_benchmark.py
```

它做了几件事：

1. 根据 manifest 中的 `source_data_file` 和 `target_data_file` 读取 source/target h5 数据。
2. 读取 `selected_cell_file`，得到本任务要预测的 source spots。
3. 用 `observed_target_indices(...)` 找到 target sample 中真实观测到的目标扰动 spots。
4. 检查 source selected spots 的 perturbation 和 tissue 是否与任务定义一致。
5. 用 `gene_indexer(...)` 将 source 和 target 表达矩阵对齐到共同基因空间。
6. 用 `local_label_fraction(...)` 计算每个 spot 的局部 niche score。

payload 中最重要的字段是：

| 字段 | 含义 |
| --- | --- |
| `source_expr` | 待预测 source spots 的表达矩阵。 |
| `target_expr` | 已观测目标扰动池的表达矩阵，也是评价时的目标分布。 |
| `source_pos` | source spots 空间坐标。 |
| `target_pos` | observed target perturbation spots 空间坐标。 |
| `source_niche` | source spots 的局部 target tissue 邻域比例。 |
| `target_niche` | target perturbation spots 的局部 target tissue 邻域比例。 |
| `n_genes` | 对齐后的基因数，当前 Mainline A 为 1053。 |

`local_label_fraction(pos, tissue, label, k=12)` 的含义是：对每个 spot 找到空间上最近的 12 个邻居，计算这些邻居中属于目标 tissue label 的比例。这个比例被用作一个简单的 niche/context 表征。例如当目标 tissue 是 tumor 时，`source_niche` 越接近 1，表示该 spot 越处在 tumor-dominant 局部环境中；越接近 0，表示它更接近 non-tumor 或边界外侧环境。

### 5.2 Residual transfer：上下文匹配残差迁移

残差分支来自 `nicheflow_lite_full`，实现入口是：

```text
predict_variant(payload, variant="nicheflow_lite_full")
```

它先构造一个 target perturbation residual pool。具体步骤是：

1. 对每个 observed target perturbation spot，在 source spots 中找到空间最近的 source anchor。
2. 计算该 target spot 相对最近 source anchor 的表达差：

```text
residual_j = target_expr_j - source_expr_nearest_source(j)
```

这一步由 `nearest_anchor_residual(payload)` 完成。得到的 residual pool 可以理解为：目标扰动池中每个 target spot 相对局部 source-like baseline 的表达偏移。

然后，对每个待预测 source spot，NicheFlow 使用 `context_distance(payload, variant="nicheflow_lite_full")` 计算 source-query 到 target-residual 的上下文距离：

```text
distance(i, j)
  = scaled_spatial_distance(source_i, target_j)
  + abs(source_niche_i - target_niche_j)
  + 0.75 * boundary_side_mismatch(i, j)
```

其中：

- `scaled_spatial_distance` 会用正距离的中位数做归一化；
- `abs(source_niche_i - target_niche_j)` 鼓励 source 和 target 拥有相似局部生态位；
- `boundary_side_mismatch` 根据 `niche >= 0.5` 将 spot 分到边界两侧，如果 source 和 target 位于不同侧，就额外加 0.75 惩罚。

随后用 softmax 将距离转成权重：

```text
weight(i, j) = softmax_j(-distance(i, j) / 0.75)
```

每个 source spot 的 residual 是 target residual pool 的加权平均：

```text
delta_i = sum_j weight(i, j) * residual_j
```

残差分支的预测为：

```text
residual_prediction_i = source_expr_i + delta_i
```

这就是 `context_matched_residual`。它的生物学解释是：把 observed target perturbation pool 中与当前 source spot 空间位置、niche score 和边界侧更相似的扰动残差，迁移到该 source spot 上。

### 5.3 Target-pool anchor：目标扰动池锚定

anchor 分支由 `target_pool_anchor(payload, anchor=...)` 实现，位于：

```text
repro_runs/2026-06-22-full-repro/nicheflow_anchored_baseline.py
```

它先构造一个 combined spatial object：

```text
x   = [target_expr; source_expr]
pos = [target_pos;  source_pos]
train_indices = target perturbation pool
query_indices = source spots
```

也就是说，anchor 分支把 observed target perturbation spots 当作训练参考点，把 source spots 当作 query points，然后用空间邻近或核平滑直接预测 source 位置上的目标扰动表达。

当前支持四种 anchor：

| anchor | 实现含义 |
| --- | --- |
| `knn10` | 使用 target pool 中最近 10 个空间邻居做 kNN 预测。 |
| `knn5` | 使用 target pool 中最近 5 个空间邻居做 kNN 预测。 |
| `gaussian` | 使用 Gaussian kernel smoothing。 |
| `cauchy` | 使用 Cauchy smoothing，尾部更重，对距离较远但相关的 target spots 更宽容。 |

NicheFlow-v1 默认只暴露三个主变体：

```text
nicheflow_v1_adaptive_anchor_residual
nicheflow_v1_cauchy_anchor_residual
nicheflow_v1_knn10_anchor_residual
```

其中 adaptive 规则由 `adaptive_anchor_for_task(task)` 决定：

```text
exact patch                -> cauchy
border                     -> cauchy
periphery                  -> cauchy
cross_slide                -> cauchy
niche_core                 -> cauchy
patch surrogate            -> knn10
cross_niche_core_to_border -> knn10
fallback                   -> cauchy
```

这个规则来自 A2.4 之后的经验整理，不是学习出来的参数。

### 5.4 Anchor 和 residual 的组合

NicheFlow-v1 的核心组合发生在 `predict_v1_variant(...)` 中：

```text
anchor   = target_pool_anchor(payload, anchor=selected_anchor)
residual = predict_variant(payload, variant="nicheflow_lite_full")

mixed = 0.80 * anchor + 0.20 * residual
```

这里的含义是：

- `anchor` 提供强 target-state relocation baseline；
- `residual` 保留 source spot 本身的表达结构，并注入 context-matched perturbation residual；
- 0.80/0.20 是当前固定权重，不是训练得到的参数；
- A2.4 audit 显示 anchor 是当前主导增益来源，因此 v1 将 anchor 放在主要位置。

需要注意：`residual` 不是单纯的 target mean，也不是直接复制 target pool；它是：

```text
source_expr + context-weighted target residual
```

所以 residual 分支承担的是“保留 source individuality + 搬运目标扰动残差”的角色。

### 5.5 Target-pool calibration：目标状态矩校准

组合后的 `mixed` 还会进入：

```text
calibrate_to_target_pool(mixed, target_expr, strength=0.25)
```

该函数做的是温和的 per-gene moment alignment：

1. 计算 prediction 每个基因的均值和标准差；
2. 计算 observed target pool 每个基因的均值和标准差；
3. 将 prediction 标准化后重缩放到 target pool 的均值和标准差；
4. 用 `strength=0.25` 将原 prediction 和 scaled prediction 混合；
5. 最后按 target pool 每个基因的 1% 和 99% 分位数裁剪。

公式可以写成：

```text
scaled = (pred - mean(pred)) / std(pred) * std(target) + mean(target)
calibrated = 0.75 * pred + 0.25 * scaled
calibrated = clip(calibrated, target_1pct, target_99pct)
```

这一步的目的不是生成新机制，而是让预测表达的尺度和 observed target perturbation state 更一致。A2.4 audit 显示 calibration 在 10/37 个任务中是主导增益来源，因此必须在方法说明中显式报告。

### 5.6 评价流程

每个 task 会对三个 NicheFlow-v1 变体都生成预测，并调用 `metric_row(...)` 计算：

- E-distance；
- MAE；
- PCC；
- R2。

之后 `method_ranking(metrics)` 会在每个 task 内按照：

```text
edistance ascending, then mae ascending
```

选择 best NicheFlow-v1 variant。`compare_to_a2_best(...)` 再将这个 best variant 与 A2.1 strong transductive best 对齐比较，生成：

```text
delta_vs_a2_best = nicheflow_v1_best_edistance - a2_best_edistance
beats_a2_best = delta_vs_a2_best < 0
```

最终输出包括：

| 文件 | 内容 |
| --- | --- |
| `metrics.csv` | 所有 task x variant 的原始指标。 |
| `method_ranking.csv` | 每个 task 的 best NicheFlow-v1 variant。 |
| `compare_to_a2_best.csv` | 与 A2.1 strong transductive best 的逐任务比较。 |
| `task_type_summary.csv` | 按 task type 聚合的结果。 |
| `ablation_summary.csv` | 按 variant/anchor 聚合的结果。 |
| `metadata.json` | protocol、权重、校准、任务数等元信息。 |

### 5.7 实现上的关键边界

当前实现有几个必须明确的边界：

1. **无训练**  
   NicheFlow-v1 当前没有神经网络训练过程，也没有梯度优化。所有权重、anchor 策略和校准强度都是固定规则。

2. **使用 observed target pool**  
   anchor、residual 和 calibration 都直接或间接使用 observed target perturbation pool。因此它属于 transductive relocation，不属于 strict unseen prediction。

3. **adaptive anchor 是规则，不是学习器**  
   `adaptive_anchor_for_task` 根据 task type 选择 anchor，这来自 audit 后的人为策略。

4. **A2.4 audit 显示 residual 不是主导增益来源**  
   residual 分支是方法解释的核心之一，但当前性能主要由 anchor 和 calibration 驱动。

5. **prediction 的目标是分布和结构兼顾**  
   E-distance 衡量预测分布接近 target pool；机制指标用于检查 boundary leakage、source individuality 和 distance-response decay。

## 6. 与 CONCERT、SpatialProp 的关系

NicheFlow、CONCERT 和 SpatialProp 都可以放在“空间扰动预测/迁移”这个大问题下，但当前定位不同。

| 方法 | 当前理解 |
| --- | --- |
| CONCERT | 更接近学习 perturbation-specific spatial kernel 或生成式 counterfactual prediction；当前本地可用输出覆盖有限。 |
| SpatialProp | 论文关注 spatially resolved single-cell transcriptomics 下的 tissue perturbation modeling；当前项目中实现的是 SpatialProp-style spot-GNN adaptation，而不是完整重训复现。 |
| NicheFlow | 更轻量、可解释，强调 target-pool anchor、context residual transfer 和 calibration 的组合。 |

当前实验中，SpatialProp-style spot-GNN 没有超过 A2.1 strong transductive baseline；可用 CONCERT 输出只覆盖 11/37 个任务，并且在覆盖任务上没有超过 A2.1 best。因此，NicheFlow-v1 的主要比较对象应首先是强 transductive relocation baseline，而不是弱 baseline。

## 7. 当前 benchmark 结果

Mainline A 当前冻结了 37 个任务：

- exact available patch task：1 个；
- surrogate patch：6 个；
- border：6 个；
- periphery：6 个；
- niche_core：6 个；
- cross_niche_core_to_border：6 个；
- cross_slide：6 个。

主指标是 E-distance，越低越好。

| 方法族 | 任务数 | 平均 E-distance | 相对 A2.1 best 胜场 |
| --- | ---: | ---: | ---: |
| A2.1 strong transductive best | 37 | 48.2582 | reference |
| SpatialProp-style spot-GNN | 37 | 256.3295 | 0/37 |
| A2.3 anchored NicheFlow | 37 | 24.9837 | 37/37 |
| NicheFlow-v1 | 37 | 25.1836 | 37/37 |

按 task type 看，NicheFlow-v1 在所有 task type 上都超过 A2.1 best：

| Task type | 任务数 | NicheFlow-v1 平均 E-distance | A2.1 平均 E-distance | 平均差值 |
| --- | ---: | ---: | ---: | ---: |
| exact patch | 1 | 18.3163 | 46.6524 | -28.3361 |
| periphery | 6 | 24.9218 | 53.0710 | -28.1492 |
| border | 6 | 26.0744 | 52.5523 | -26.4779 |
| cross-niche core-to-border | 6 | 28.4788 | 52.0639 | -23.5851 |
| niche core | 6 | 28.4249 | 50.9792 | -22.5543 |
| cross-slide | 6 | 17.4227 | 39.7238 | -22.3012 |
| patch surrogate | 6 | 26.9237 | 41.4265 | -14.5029 |

## 8. A2.4 audit：NicheFlow 的增益来自哪里

A2.4 audit 将 anchored NicheFlow 拆成五类：

| 组件族 | 平均 E-distance | 胜场 |
| --- | ---: | ---: |
| anchor + residual + calibration | 24.9837 | 36 |
| anchor only + calibration | 34.6885 | 0 |
| anchor + residual, no calibration | 41.9256 | 0 |
| anchor only | 52.4261 | 0 |
| residual only | 125.2747 | 1 |

dominant gain source 的任务计数：

| 主导增益来源 | 任务数 |
| --- | ---: |
| anchor | 27 |
| calibration | 10 |

这说明当前 NicheFlow-v1 的强性能主要来自：

1. target-pool anchor；
2. target-pool calibration；
3. residual branch 的辅助增强。

因此，当前不能把 NicheFlow-v1 的胜出解释成“残差机制模型单独学到了扰动因果机制”。更稳妥的解释是：

> NicheFlow-v1 在 observed target-pool access 协议下，将强 target-state anchor 与上下文残差迁移结合起来，从而得到比简单空间迁移 baseline 更好的 relocation 表现。

## 9. 机制指标如何理解

为了避免只看 E-distance，当前额外计算了三个机制 proxy：

| 指标 | 越大/越小 | 含义 |
| --- | --- | --- |
| `boundary_leakage_score` | 越低越好 | 预测是否过度跨越组织边界、造成边界状态混合。 |
| `source_individuality_score` | 越高越好 | 预测是否保留 source spots 之间的个体结构。 |
| `distance_response_decay_score` | 越高越好 | response magnitude 是否随距离目标扰动池变远而衰减。 |

当前结果：

| 方法族 | 平均 E-distance | Boundary leakage | Source individuality | Distance-response decay |
| --- | ---: | ---: | ---: | ---: |
| A2.3 anchored NicheFlow | 24.9837 | 0.5863 | 0.5542 | -0.0479 |
| NicheFlow-v1 | 25.1836 | 0.5704 | 0.5941 | -0.0458 |
| A2.1 strong transductive best | 48.2582 | 0.7055 | 0.2233 | -0.1184 |
| SpatialProp-style spot-GNN | 256.3295 | 3.3612 | 0.0656 | -0.0266 |

解释：

- NicheFlow-v1 相比 A2.1 best 有更低 boundary leakage；
- NicheFlow-v1 更好地保留 source individuality；
- distance-response decay 仍然较弱，不能据此宣称模型已经学到清晰的空间剂量响应规律。

## 10. 当前应该如何表述 NicheFlow 的贡献

推荐表述：

> NicheFlow-v1 is a transductive spatial perturbation relocation method that combines observed target-pool anchoring, context-matched residual transfer, and target-state calibration. On the frozen Mainline A benchmark, it improves over strong spatial and target-state access baselines while better preserving boundary and source-spot individuality diagnostics.

中文表述：

> NicheFlow-v1 是一个转导式空间扰动迁移方法。它结合已观测目标扰动池锚定、上下文匹配残差迁移和目标状态校准，在 frozen Mainline A benchmark 上超过强空间/目标状态基线，并在边界泄漏和 source spot 个体性指标上表现更好。

不推荐表述：

- NicheFlow 已经解决 strict OOD perturbation prediction。
- NicheFlow 可以预测从未观测过的扰动。
- NicheFlow 的性能主要证明了残差分支学到扰动因果机制。
- NicheFlow 已经证明了真实空间剂量响应规律。

## 11. 当前局限

1. **目标扰动池依赖明显**  
   当前主线 A 允许使用 observed target perturbation pool。NicheFlow-v1 的强结果不能外推为 strict unseen/OOD 能力。

2. **残差分支还不是主性能来源**  
   A2.4 audit 显示 anchor 和 calibration 是当前主导增益来源。残差分支更像机制层和辅助层。

3. **surrogate tasks 仍占多数**  
   37 个任务中只有 1 个 exact available task，其余是 deterministic surrogate tasks。

4. **CONCERT 对比覆盖不完整**  
   当前可用 CONCERT 输出只覆盖 11/37 个任务，没有 cross-slide 输出。

5. **机制指标是 proxy，不是因果证明**  
   boundary leakage、source individuality、distance-response decay 能帮助诊断模型行为，但不能直接证明因果机制。

## 12. 下一步工作

建议优先推进三条线：

### A3：target-pool access ladder

系统比较不同信息访问级别：

```text
no target pool
uncalibrated target anchor
calibrated target anchor
anchor + residual
anchor + residual + calibration
```

目标是定量说明：NicheFlow 的性能到底依赖多少 observed target-pool 信息。

### A4：强化 residual mechanism

下一版 NicheFlow 不应只追求更低 E-distance，而应让 residual branch 对机制指标产生更明确贡献，尤其是：

- boundary-aware residual；
- niche-conditioned residual；
- distance-aware response decay；
- cross-slide residual alignment。

### A5：strict unseen/OOD appendix

strict unseen 仍然重要，但建议作为 appendix 或 stress test，而不是当前主线 headline。

该实验回答的是另一个问题：

```text
完全禁止使用目标扰动观测池时，模型还能不能预测目标扰动？
```

当前证据显示这个问题非常难，不应和 Mainline A 的 observed relocation claim 混在一起。

## 13. 相关产物

| 产物 | 路径 |
| --- | --- |
| Mainline A 英文研究总结 | `repro_runs/2026-06-22-full-repro/mainline_a_report/NicheFlow_Mainline_A_summary.md` |
| 结果索引 | `repro_runs/2026-06-22-full-repro/mainline_a_report/result_index.csv` |
| NicheFlow-v1 结果 | `repro_runs/2026-06-22-full-repro/nicheflow_v1/` |
| A2.4 gain audit | `repro_runs/2026-06-22-full-repro/nicheflow_anchored_audit/` |
| 机制指标 | `repro_runs/2026-06-22-full-repro/mainline_a_mechanism_metrics/` |
| A2.1 strong baseline | `repro_runs/2026-06-22-full-repro/mainline_a_transductive_baselines/` |

