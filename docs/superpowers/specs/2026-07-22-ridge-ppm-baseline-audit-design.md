# Ridge PPM Strong-Baseline Audit 设计

## 1. 目标与范围

本实验补齐 few-shot spatial perturbation distribution completion 的关键统计控制组：

> 在现有 Cd52、K=4、40 个 held-out events 的冻结 development episode 上，运行 ridge-regularized conditional Poisson point-process GLM，并与现有神经 context-only 和 support-conditioned candidate 做同协议比较。

实验只回答低容量 log-linear spatial field 是否已足够，以及四个 support events 是否在 guide-blind tissue context 之外提供增量信息。它不扩大到多 episode、多 seed、external/final/reserve，也不恢复 spot-level counterfactual expression 路线。

新实验目录固定为：

```text
repro_runs/2026-07-22-ridge-ppm-baseline-audit/
```

## 2. 冻结输入与访问边界

复用已完成 first smoke 的以下合同：

- section：`ot1_day4_rep2`；
- target：`Cd52`；
- accepted events：44；
- visible support：4；
- held-out evaluation events：40；
- support seed 和 event split 与原 first smoke 完全一致；
- 训练 episode：同 section 内 12 个非 Cd52 development perturbations；
- selected Cd52 的 held-out counts 不参与拟合、特征选择、lambda 选择或回退选择；
- external、final、reserve value access 保持硬关闭。

数据适配复用 `/tmp/CONCERT-spacseq-elrr` 中已经通过 assignment、坐标对齐和 mutation guard 的 first-smoke loader。新实验必须记录依赖文件 SHA256、源 worktree commit、dirty-state 摘要、Table S1 SHA256、section inventory hash 和原 split hash。新 runner 不修改原 worktree或冻结结果。

若依赖 worktree、原 summary 或输入文件发生漂移，runner fail closed，不重新抽 support。

## 3. 统计对象

对每个训练 perturbation episode `p` 和 eligible location `i`：

```text
eta_pi = z_pi^T beta
p_pi = softmax_i(eta_pi + log exposure_i)

conditional NLL_p
= - sum_i y_pi * log p_pi / sum_i y_pi
```

本轮 exposure 冻结为 1。RNA/cell density 仅作为 guide-blind context covariate，不作为 exposure offset。因此结论只涉及 eligible locations 上的条件空间分布，不涉及绝对 perturbation abundance。

训练目标为 pseudo-fit episodes 的事件数加权平均 conditional NLL，加 ridge penalty：

```text
objective(beta)
= sum_p n_p * NLL_p / sum_p n_p
+ 0.5 * lambda * ||beta||_2^2
```

不拟合截距，因为 episode 内 softmax 会消去常数项。所有非截距标准化系数使用同一 ridge penalty。

## 4. 特征合同

### 4.1 共享坐标基函数

所有 learned PPM comparator 共享以下低容量 nuisance basis：

```text
x, y, x^2, x*y, y^2
```

坐标使用原 first smoke 的 guide-blind tissue geometry。每列只用 development tissue geometry 计算均值和标准差。

### 4.2 Context features

Context 使用 first smoke 完全相同的 guide-blind frozen panel genes 加 `n_genes`，不重新用 guide truth 选择基因。每列标准化参数只由允许的 development tissue context 计算。

### 4.3 Support features

每个 episode 只由其 K=4 visible support 构造：

```text
d_min(i, S_p) / D

log(epsilon + mean_s exp(-d(i,s)^2 / (2*h^2)))
for h / D in {0.02, 0.05, 0.10, 0.20}

euclidean distance between C_i and mean support context
cosine similarity between C_i and mean support context
```

`D` 为 eligible tissue bounding-box diagonal。`epsilon`、尺度集合和生成规则在 target evaluation 前冻结。不加入 context×support interaction。

## 5. Baseline 定义

为使 support gain 是严格嵌套比较，坐标 basis 作为所有 learned PPM 的共享 nuisance component：

- B3 diagnostic：coordinates only；
- B4：coordinates + guide-blind context；
- B5：coordinates + support summaries；
- B6：coordinates + context + support summaries。

主要比较：

```text
support gain = NLL(B4) - NLL(B6)
```

`support gain > 0` 才说明 support 在 tissue context 和共享大尺度坐标趋势之外提供增量信息。B3 只用于解释结果，不改变本轮主范围。

## 6. 选型协议

12 个非 Cd52 training perturbations 按稳定 target ID hash 分为 pseudo-fit 和 pseudo-validation，数量冻结为 8/4。不同模型族分别从报告预注册网格中选择 lambda：

```text
{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000}
```

选择规则：

1. 在 pseudo-fit 上拟合；
2. 在四个 pseudo-validation episodes 上计算事件数加权 conditional NLL；
3. 最小 NLL 胜出；完全相同时选择更大的 lambda；
4. 用选中的 lambda 在全部 12 个非 Cd52 episodes 上重拟合；
5. Cd52 held-out events 只评价一次。

不得因 Cd52 结果为负而更改 split、特征、lambda 网格或优化容差。

## 7. 开源代码与数值后端复用

### 7.1 B4–B6 主后端

使用已安装 SciPy 的 `L-BFGS-B` 优化精确 conditional softmax objective 和解析梯度。这一 objective 与 Warton–Shepherd PPM、Renner–Warton MaxEnt 等价关系中的条件位置 likelihood 同构，同时避免为约 47 万个固定 locations 构造随机 background 或连续 quadrature。

不直接使用 `glmnet`、`maxnet` 或 `spatstat::ppm` 作为主后端，原因是它们没有现成表达“多个 episode 各自条件化事件总数、共享系数、episode-specific support features、selected target 完全隔离”的接口。强行包装会引入巨大数据展开或不必要近似。

### 7.2 B7 触发与 `mgcv` 复用

系统 R 已安装 `mgcv 1.8-39`。预注册触发条件为：

```text
NLL(B6) <= 12.599395462034924 + 0.02
```

触发后运行 B7。B7 直接调用 `mgcv::bam` 的 penalized smooth 实现，并保留相同 episode、support、context、exposure 和 held-out evaluation 边界。冻结公式为 episode-specific intercept、`te(x, y, k=16)` 二维 smooth，以及线性 context/support 项，使用 Poisson log link、`offset(log(exposure))`、`method="fREML"` 和 `discrete=TRUE`。episode intercept 消去各 perturbation 的未知 abundance；smoothness 只由 12 个非 Cd52 training episodes 估计。

B7 只在预注册条件满足时运行；未触发时 summary 明确记录 `not_triggered`，不安装或扫描 LGCP/INLA。由于是否运行 B7 取决于已经观察到的 Cd52 B6 结果，B7 必须标记为 `post_trigger_diagnostic`，不能用于本轮方法 promotion 或替换预注册 B4–B6 主结论。

### 7.3 隔离环境

优先使用仓库现有 Python 环境和系统 `mgcv`，并记录版本。只有缺失必需包时，才在新实验目录创建隔离环境并安装；不得修改全局环境。用户已授权为本实验创建该环境。

## 8. 最小 RED→GREEN 合约

实现前必须观察以下最小测试的有效 RED：

1. 三位置手算 conditional NLL；
2. 解析梯度与有限差分一致；
3. lambda=0 的 toy conditional-softmax 与直接枚举一致；
4. B6 support 系数固定为零时精确恢复 B4；
5. 输出有限、非负、mask 外为零、mask 内概率和为 1；
6. support/evaluation disjoint，selected target 不进入 fit/selection；
7. 替换 Cd52 held-out counts 后 B3–B6 predictions bitwise 不变；
8. 同 seed、同输入、同版本产生相同 lambda 和预测；
9. 原 first-smoke episode、support 和 NLL 常数复现。

在这些合同通过后立即运行一个真实 Cd52 audit，不补 exhaustive CLI 或异常输入测试。

## 9. 结果与判定

必须写出：

- `results/ridge_ppm_audit/metrics.csv`；
- `results/ridge_ppm_audit/summary.json`；
- `results/ridge_ppm_audit/probability_fields.npz`；
- `results/ridge_ppm_audit/coefficients.npz`；
- `results/ridge_ppm_audit/selection_metrics.csv`；
- `reports/ridge_ppm_audit_zh.md`；
- verifier 输出。

与以下冻结结果比较：

```text
neural context-only = 12.599395462034924
neural candidate    = 12.608464102647213
best RBF            = 13.126427974206555
```

决策解释：

- `NLL(B4) <= 12.599395462034924`：神经 context head 未显示超出低容量 log-linear field 的方向性增益；
- B6 优于 B4：K=4 support 有条件增量信号；
- B6 不优于 B4：记录 `do_not_scale_no_support_gain`，不增加 seed、episode 或网格；
- B7 触发且优于 ridge：作为 post-trigger diagnostic，提示平滑非线性可能有价值，但仍未证明需要 neural；
- neural 稳定优于 B6/B7 且 support gain 为正：只有这种模式才支持继续 neural routing。

单个 Cd52 smoke 只报告方向和原始差值，不把 40 个 held-out events 当作独立生物重复，不宣称统计显著性。

## 10. 验证与停止规则

verifier 重新计算：

- 文件集合和 SHA256；
- source/split provenance；
- selected target isolation；
- lambda selection；
- conditional NLL；
- probability normalization；
- mutation invariance；
- B7 trigger；
- decision code；
- external/final/reserve 零读取。

若 B6 不优于 B4，首轮 audit 停止。任何后续实验必须提出直接针对 support acquisition、shrinkage 或非线性残差结构的新假设，不能用追加 seed、任务或超参数营救负结果。
