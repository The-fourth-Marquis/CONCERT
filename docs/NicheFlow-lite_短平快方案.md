# NicheFlow-lite：基于 CONCERT 的短平快空间扰动预测方案

## 1. 一句话定位

NicheFlow-lite 是一个从 CONCERT 出发的轻量空间扰动预测框架。它保留 CONCERT 的空间上下文反事实预测思想，但暂时不做完整生成式大模型，而是用扰动不变生态位匹配、残差转移和边界感知平滑，在论文已有指标 E-distance 上快速验证是否能超过强空间基线。

当前阶段的核心目标不是提出最复杂模型，而是尽快回答：

> 在公开 Perturb-map 数据和可复现 surrogate 子任务上，是否能用一个更可解释、更少泄漏风险的轻量方法，在论文指标 E-distance 上优于 CONCERT 和强空间基线？

## 2. 合并两份文档后的取舍

### 2.1 从 CONCERT 研究方案中继承

- 先诊断 CONCERT 是否存在 target-state distribution fitting、局部空间平滑插值和任务采样过易的问题。
- 建立强简单空间基线，而不是只和 scGEN、CPA、BioLORD 等非空间扰动模型比较。
- 改进方向优先考虑参数共享、边界感知和扰动响应分解。
- 避免一开始堆更复杂的深度生成模型。

### 2.2 从 NicheFlow/mainV2 方案中继承

- 明确 same-slide unpaired spatial perturbation 的设定：control 和 perturbation-labeled spots 并不是 spot-level before-after 配对。
- 使用 perturbation-invariant niche matching 构造 control-treated pseudo-counterfactual coupling。
- 使用 held-out perturbation distribution evaluation，而不是不可获得的 spot-level counterfactual ground truth。
- 以 energy distance / E-distance 作为主评价指标。
- 引入 boundary-aware smoothing，避免跨组织边界过度平滑。
- 保留 patch、border、niche、cross-niche 等任务结构。

### 2.3 当前阶段暂时砍掉

为了短平快，第一版不做：

- 完整 VAE / latent flow；
- graph Transformer；
- Best-of-K reranking；
- DE、GRN、pathway prior；
- 新指标体系；
- 大规模 CONCERT 全量重训。

这些内容放到第二阶段，前提是 NicheFlow-lite 已经在 E-distance 上站得住。

## 3. 方法核心：NicheFlow-lite

### 3.1 基本形式

NicheFlow-lite 不直接生成 target expression，而是预测扰动残差：

```text
predicted_expression = source_expression + predicted_perturbation_residual
```

其中 residual 来自生态位匹配后的 observed perturbed spots：

```text
delta_i^p = sum_j pi(j|i) * residual_j^p
```

最小实现中，`residual_j^p` 可以定义为：

```text
residual_j^p = x_j^p - mean_control_context(j)
```

或者：

```text
residual_j^p = x_j^p - x_anchor(j)^control
```

第一版建议先用 `mean_control_context(j)`，实现更稳、更快。

### 3.2 匹配权重

对 source spot `i` 和 observed perturbed spot `j`，定义上下文距离：

```text
d(i,j) =
  alpha_s * spatial_distance(i,j)
  + alpha_t * tissue_mismatch(i,j)
  + alpha_n * niche_score_distance(i,j)
  + alpha_b * boundary_crossing(i,j)
  + alpha_q * batch_mismatch(i,j)
```

soft matching 权重为：

```text
pi(j|i) = exp(-d(i,j) / tau) / sum_j exp(-d(i,j) / tau)
```

第一版不学习这些参数，而是使用少量固定网格：

- `tau`: 0.5, 1.0, 2.0
- `alpha_s`: 1.0
- `alpha_n`: 0.5, 1.0
- `alpha_b`: 0.0, 1.0, 2.0

用验证任务选择最优组合，避免训练成本。

### 3.3 边界感知

用已有 `tissue` 标签和邻域 tumor/normal fraction 近似组织边界。

边界任务中：

- tumor-neighbor fraction 接近 0.5 的 spots 定义为 border；
- tumor-neighbor fraction 高的 spots 定义为 tumor core；
- normal-neighbor fraction 高的 spots 定义为 periphery / normal-facing。

边界惩罚：

```text
boundary_crossing(i,j) = 1 if local_context_bin(i) != local_context_bin(j) else 0
```

这比纯空间距离更符合组织结构，也直接回应 CONCERT 可能 across boundary over-smoothing 的问题。

## 4. 任务设计

当前使用已生成的 surrogate selected-cell files：

```text
repro_runs/2026-06-22-full-repro/surrogate_select_cells/
```

共 36 个任务，覆盖：

| 子任务 | 定义 | 目的 |
| --- | --- | --- |
| patch | 选择离 observed target perturbation spots 最近的 KP tumor spots | 测同一局部上下文内的扰动响应 |
| border | 选择 tumor-neighbor fraction 接近 0.5 的 KP tumor spots | 测组织边界附近预测 |
| niche_core | 选择 tumor-neighbor fraction 最高的 KP tumor spots | 测 tumor core 上下文 |
| periphery | 选择 normal-neighbor fraction 最高的 KP tumor spots | 测 normal-facing/periphery 上下文 |
| cross_niche_core_to_border | 用 tumor core source 构造 border-context benchmark | 测跨生态位泛化 |
| cross_slide | 在没有目标 perturbation 的样本中选 source spots，目标来自另一样本 | 测跨切片泛化 |

所有 surrogate task 必须标注：

```text
is_exact_author_file = false
```

不能混同为作者原始 Shiny/lasso 选点。

## 5. 评价策略

当前阶段只使用论文已有主指标：

```text
E-distance lower is better
```

可以保留 `MAE / PCC / R2` 作为诊断列，但不用于主排序，不作为论文主结论。

暂时不加入：

- delta PCC；
- DE direction accuracy；
- pathway score；
- marker spatial map correlation；
- Moran's I / hotspot overlap。

这些指标进入第二阶段，触发条件是 NicheFlow-lite 在 E-distance 上优于强基线。

## 6. 对照方法

第一轮只跑快速方法：

| 方法 | 作用 |
| --- | --- |
| source mean | 最弱 control |
| target mean | 检查 target distribution matching 上限 |
| niche target mean | 检查 niche-level mean 是否足够 |
| spatial kNN | 检查局部复制能力 |
| kernel smoothing | 检查距离衰减平滑是否足够 |
| graph diffusion | 检查空间图传播是否足够 |
| NicheFlow-lite no-boundary | 消融边界项 |
| NicheFlow-lite no-niche | 消融生态位匹配 |
| NicheFlow-lite spatial-only | 检查是否只是空间平滑 |
| NicheFlow-lite full | 主方法 |

CONCERT 对照放在第二轮，只跑已有模型可直接 eval 的任务，暂不全量重训。

## 7. 最短执行路线

### Step 1：冻结任务表

输入：

```text
surrogate_select_cells/surrogate_selected_cells_manifest.csv
```

输出：

```text
paper_metric_benchmark/task_manifest.csv
```

关键字段：

- `task_id`
- `sample`
- `subtask_type`
- `selected_cell_file`
- `target_perturbation`
- `target_tissue`
- `is_exact_author_task`
- `is_surrogate_task`
- `can_run_baseline`
- `can_run_concert_eval`
- `can_compare_to_paper_reported`

### Step 2：实现 paper-metric evaluator

新增脚本：

```text
repro_runs/2026-06-22-full-repro/nicheflow_lite_benchmark.py
```

输出：

```text
paper_metric_benchmark/
  task_manifest.csv
  metrics.csv
  summary.csv
  method_ranking.csv
  fig_edistance_by_task.png
  fig_method_ranking.png
  README.md
```

### Step 3：跑 baseline + NicheFlow-lite

先不跑 CONCERT 训练，只跑快速 baseline 和 NicheFlow-lite。

成功标准：

- NicheFlow-lite full 平均 E-distance 低于 best simple baseline；
- 在 patch、border、niche_core、periphery、cross_niche、cross_slide 中多数任务类型胜出；
- full 优于 no-boundary；
- full 优于 no-niche；
- full 优于 spatial-only。

### Step 4：有限 CONCERT 对照

只对已有模型和已有输出可覆盖的任务跑 CONCERT eval：

- GSM5808054 Jak2 exact / surrogate task；
- repo reference output；
- repro output；
- 可直接 eval 的 selected-cell files。

不在第一轮补全所有样本训练。

### Step 5：阶段报告

输出：

```text
paper_metric_benchmark/README.md
```

报告必须分开：

- exact available task；
- surrogate reconstructed tasks；
- simple baselines；
- NicheFlow-lite；
- CONCERT limited comparison；
- unavailable paper tasks due to missing author selected-cell files。

## 8. 风险与规避

| 风险 | 规避 |
| --- | --- |
| surrogate 任务被质疑不是论文原始任务 | 明确标注 surrogate，并保留 exact available task 单独比较 |
| NicheFlow-lite 只是在做空间平滑 | 必须包含 no-niche、no-boundary、spatial-only 消融 |
| target mean 很强 | 把它作为核心 baseline；若目标均值已赢，说明任务本身不够区分 |
| CONCERT 对照不完整 | 第一版只做 limited comparison，不声称完整复现所有论文任务 |
| 新指标缺失 | 当前阶段目标是 paper metric first；新指标进入第二阶段 |

## 9. 预期论文故事

第一阶段可以形成如下故事：

> CONCERT 证明了空间扰动反事实预测的重要性，但公开任务依赖手工 selected-cell files，且复杂模型可能混合了 target-state distribution matching 与局部空间平滑。我们构建了一个可复现的 paper-metric-only surrogate benchmark，并提出 NicheFlow-lite：一种基于扰动不变生态位匹配、残差转移和边界感知平滑的轻量方法。该方法不依赖复杂生成网络，却能在 E-distance 上更稳地预测 patch、border、niche 和 cross-niche 空间扰动响应。

## 10. 后续升级路径

只有当 NicheFlow-lite 在 E-distance 上胜出后，再进入第二阶段：

1. 加入 delta / DE / pathway / spatial map 指标；
2. 将固定 matching 权重改成可学习参数；
3. 引入 shared kernel parameterization；
4. 加入轻量 graph encoder；
5. 再考虑 latent flow 或生成式模型；
6. 扩展到 DSS、STROKE 和 transplant case studies。

## 11. 当前最重要的下一步

立即实现：

```text
repro_runs/2026-06-22-full-repro/nicheflow_lite_benchmark.py
```

并先跑：

```text
surrogate selected-cell tasks + strong baselines + NicheFlow-lite variants
```

第一版只以 `E-distance` 排名。只要 NicheFlow-lite full 能在 E-distance 上稳定优于 best baseline，就进入第二阶段。
