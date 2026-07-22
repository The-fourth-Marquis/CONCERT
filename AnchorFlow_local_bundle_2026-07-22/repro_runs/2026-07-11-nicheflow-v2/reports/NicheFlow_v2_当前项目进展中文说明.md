# NicheFlow-v2 当前项目进展中文说明

日期：2026-07-13  
面向读者：具备机器学习、生物信息学或空间转录组背景的技术研究人员  
对应实验目录：`repro_runs/2026-07-11-nicheflow-v2/`

## 1. 一页式结论

### 1.1 项目当前处于什么阶段

当前已经完成四件核心工作：

1. 重新建立了无评价泄漏的 held-out（留出）评价协议；
2. 在 E-distance 之外加入扰动方向、差异表达和空间行为指标；
3. 重新评价了 NicheFlow-v1、常见空间锚点方法和 SpatialProp-style 模型；
4. 实现并评价了 NicheFlow-v2 的 explicit residual（显式残差）与 reliability gating（可靠性门控）。

本轮共覆盖 37 个 Mainline A 任务，每个任务使用 3 个 target reference fraction、5 个随机种子和 2 种拆分方式，形成：

```text
37 × 3 × 5 × 2 = 1,110
```

个独立评价单元。

### 1.2 最重要的研究结论

第一，原来的 E-distance 不是无效，而是单独使用不够。它能评价预测表达分布是否接近目标分布，却不能保证扰动方向正确、差异表达基因方向正确，也不能保证局部空间结构合理。

第二，NicheFlow-v1 的最佳平均 held-out E-distance 为 55.3383，但这一优势高度依赖 calibration（校准）。未校准的 legacy v1 对最强未校准 A2.1 baseline 的平均差值为 +4.3425，说明旧 residual 本身没有稳定超过强空间锚点。

第三，v2 的显式残差设计有效。residual-only 的 held-out E-distance 从旧 residual 的 151.9095 降到 69.4766，说明把“纯扰动变化量”和“完整预测表达”分开是正确方向。

第四，当前 reliability gate 没有成功。未校准固定 80/20 组合的 E-distance 是 66.8581，而门控组合是 85.9374。gate 权重均值只有 0.051562，标准差只有 0.000948，几乎全部贴在预设下界附近，没有形成真正的逐 spot 自适应。

第五，当前阶段决策是：

`do_not_advance_to_graph_flow`

这表示现在不进入 Graph Transformer 或 latent flow 开发。原因不是认定这些复杂模块永远无效，而是现有轻量方案尚未满足跨任务显著性和 reduced-reference 稳健性要求。先解决评价、matching 和 gate collapse，能避免用更大的模型掩盖基础问题。

### 1.3 当前最值得保留的方案

如果只看未校准、可解释的机制版本，目前最值得保留的是：

```text
P_fixed = 0.80 × P_anchor + 0.20 × P_residual
```

它不是最终模型，但可作为下一轮 gate 改造的可靠对照。校准版本能达到更低的 E-distance（54.1277），但不能把该结果直接解释成 residual mechanism 已经学得更好。

---

## 2. 项目究竟在预测什么

### 2.1 Mainline A 的任务定义

Mainline A 研究的是 observed-perturbation spatial relocation，即“已观测扰动状态的空间重定位”。

对于一个任务，我们已在某个 target sample 中观察到特定 perturbation 的一部分 spots。模型需要利用这些已观测 target reference spots，以及 source sample 中的控制状态和空间上下文，估计 selected source/query spots 在该已观测 perturbation 状态下可能呈现的表达分布。

输入可以概括为三部分：

| 输入 | 含义 | 是否可以用于预测 |
| --- | --- | --- |
| selected source/query spots | 真正需要生成预测的 source spots | 可以 |
| eligible control pool | source sample 中符合 perturbation/tissue 协议的完整控制候选池 | 可以 |
| target reference pool | observed target perturbation pool 中划给 reference 的部分 | 可以，且必须记录 access level |
| target evaluation pool | observed target pool 中独立留出的部分 | 不可以，只能用于计算指标 |

输出不是每个 source spot 的已知配对真值，而是一组预测表达向量，用来近似 source spots 在 target perturbation 状态下的表达分布。

### 2.2 本项目明确不主张什么

Mainline A 不是严格 unseen/OOD 扰动预测。这里的 unseen/OOD 指训练和 reference 数据中从未出现过目标扰动，模型仍要依靠扰动语义进行外推。当前协议允许访问同一目标扰动的 reference spots，因此不能声称模型学会了从未见扰动的生物学规律。

当前结果也不是 spot-level causal counterfactual。source spot 与 target spot 没有一一配对，因此 delta 指标评价的是 group-level 或 niche-level 的平均扰动效应，而不是单个细胞“如果被扰动会怎样”的真实因果结果。

这两个边界必须始终保留：

- 支持：已观测扰动的空间重定位；
- 不支持：严格 unseen/OOD 扰动预测；
- 不支持：spot-level 因果反事实准确性；
- 不支持：仅凭空间相关性证明生物机制。

---

## 3. 数据与评价单元

### 3.1 为什么要拆 target pool

旧 full-pool 评价中，同一个 observed target pool 既可能参与预测、锚点选择或 calibration，又整体作为评估目标。即使没有直接复制 target spot，这也会让指标偏向使用 target pool 信息较多的方法。

v2 将每个 observed target perturbation pool 拆成互斥的两部分：

- reference subset：模型允许访问；
- evaluation subset：模型完全不可访问，只在预测结束后计算指标。

代码还执行 evaluation-expression mutation guard：把 evaluation expression 加上一个很大的扰动后重新预测。如果预测发生变化，就说明 evaluation 信息泄漏进了模型。当前所有可评价方法均通过该检查。

### 3.2 三个 reference fraction

reference fraction 分别为 0.25、0.50 和 0.70。

例如 reference fraction = 0.70 表示 observed target pool 的 70% 允许模型访问，剩余 30% 只用于 held-out 评价。较低 fraction 用来检查模型在 target reference 较少时是否仍稳定，而不是只在 reference 很充足时有效。

### 3.3 两种拆分方式

random-stratified（随机分层拆分）尽量保持 niche 和空间象限构成相近，方差较低，但 reference 与 evaluation 中可能存在距离很近的空间邻居。

spatial-block（空间块拆分）按空间区域留出完整 block，更接近“预测未覆盖区域”的压力测试。它更困难，也更容易暴露模型只会局部插值的问题。

因此 random-stratified 是常规评价，spatial-block 是必要的空间敏感性分析。两者不能相互替代。

### 3.4 Access ladder

为了避免把获得不同 target 信息的方法混在同一含糊排名中，结果使用以下 access level：

| Level | 可使用的信息 | 代表方法 |
| --- | --- | --- |
| L0 | 不使用 target reference | source identity |
| L1 | 使用未校准 target anchor | Gaussian、Cauchy、kNN |
| L2 | 使用 reference-only calibration | calibrated anchor、SpatialProp |
| L3 | target anchor 加 residual branch | fixed 80/20 |
| L4 | gated anchor/residual | reliability-gated model |

Access level 不是模型优劣等级，而是信息使用合同。L2 或 L3 方法优于 L1 时，必须说明增益中可能包含更多 target reference 信息。

---

## 4. 为什么原来的评价不够

### 4.1 Full-pool E-distance 的问题

full-pool E-distance 将预测分布与完整 observed target pool 比较。由于完整 pool 包含模型已经访问过的 reference rows，分数通常会比真正 held-out 更低。

例如：

| 方法 | Held-out E-distance ↓ | Full-pool E-distance ↓ |
| --- | ---: | ---: |
| legacy v1 fixed 80/20 calibrated | 55.3383 | 35.3975 |
| adaptive calibrated anchor | 71.3564 | 50.3612 |
| legacy v1 fixed 80/20 | 72.9115 | 53.1820 |
| Gaussian anchor | 92.9950 | 71.8525 |

箭头“↓”表示越低越好。

可以得出的结论：full-pool 可继续作为与旧实验兼容的传导式（transductive）指标，即允许使用 target reference 信息的兼容性评价。

不能得出的结论：不能因为 full-pool 很低，就断言模型能泛化到没有参与预测的 target spots，更不能断言 residual mechanism 学到了正确扰动效应。

### 4.2 为什么不能只保留一个总分

假设一个模型通过 calibration 把预测均值和方差对齐到 target reference，它可能获得很好的 E-distance；但该模型仍可能：

- 对某些关键基因预测反方向；
- 抹掉 source spot 之间的个体差异；
- 在 tumor/normal 边界产生混合泄漏；
- 只在随机拆分中有效，在空间块拆分中失败。

因此 v2 不建立单一 composite score。不同指标保持分开，避免一个优秀的分布指标遮蔽其他失败。

---

## 5. 指标如何理解

### 5.1 分布指标

#### Held-out E-distance

E-distance 比较两组多维样本的整体分布距离。这里比较预测 spots 与 held-out target spots，越低越好。

它回答：“预测表达分布整体上是否接近未参与预测的 observed target 分布？”

它不能回答：“每个 source spot 是否预测正确？”也不能单独证明扰动方向和空间机制正确。

#### Full-pool E-distance

定义相同，但目标是完整 observed target pool。它是兼容性指标，不是 v2 的主选择指标。

#### Niche-stratified E-distance

先按 niche 分层，在共享 niche 内计算 E-distance，再按 target 数量汇总。它能减少一种假象：模型只要预测对总体均值，就掩盖在某个 niche 中的失败。越低越好。

### 5.2 扰动效应指标

对 source group 定义：

```text
Delta_pred = mean(P_pred) - mean(X_source)
Delta_true = mean(X_eval-target) - mean(X_source)
```

其中每个 delta 都是以 gene 为维度的向量。

| 指标 | 方向 | 直观含义 |
| --- | --- | --- |
| delta PCC | 越高越好 | 预测和真实扰动幅度是否呈线性一致 |
| delta Spearman | 越高越好 | 基因扰动排序是否一致 |
| delta MAE | 越低越好 | 平均绝对扰动误差 |
| delta RMSE | 越低越好 | 对大误差更敏感的扰动误差 |

这些指标比 E-distance 更直接地检查“扰动是否朝正确方向变化”，但仍然是 group-level effect，不是单 spot 因果效应。

### 5.3 差异表达指标

DE 是 differential expression（差异表达）的缩写。为了避免泄漏，top-20 和 top-50 DE genes 只根据 reference target 与 matched control 的差异选择，然后用 held-out target 检查：

- direction accuracy：预测上调/下调方向正确的比例，越高越好；
- opposite-direction fraction：预测成完全相反方向的比例，越低越好；
- DE delta PCC 与 MSE：关键基因上的效应相关性和误差。

这能防止总体分布很好、关键响应基因却方向错误的模型被误判为成功。

### 5.4 空间与上下文指标

| 指标 | 方向 | 回答的问题 | 局限 |
| --- | --- | --- | --- |
| boundary leakage | 越低越好 | 不同边界类别是否被错误混合 | 是代理指标（proxy），不是组织学真值 |
| source individuality | 越高越好 | 预测是否保留 source spots 间的相对差异 | 高值也可能保留无关 source variation |
| distance-response decay | 当前定义下越高越好 | 响应强度是否随参考距离呈合理衰减 | 相关性不是因果关系 |
| niche-wise delta PCC | 越高越好 | 各 niche 内扰动方向是否一致 | 小 niche 时方差较大 |
| spatial-map PCC | 越高越好 | 同一注册坐标系内 marker/DE 空间图是否相似 | cross-slide 坐标不可直接比较 |

spatial-map PCC 只在 same-slide 且坐标可注册的单元计算。v2 结果中 5,580/6,660 行有定义；另外 1,080 个 cross-slide 行明确标记为 `unregistered_cross_slide`，而不是用零代替缺失值。

---

## 6. 从 NicheFlow-v1 到 v2

### 6.1 v1 实际组合了什么

实现中的 legacy v1 可写成：

```text
P_legacy
= 0.80 × P_anchor
+ 0.20 × P_legacy-residual-prediction
```

其中 anchor 是根据空间位置和任务类型选择的 target reference 预测；legacy residual branch 返回的并不是“纯 residual”，而是已经包含 source expression 的完整预测表达。

早期文档中曾出现“减去 context-matched residual”的写法。如果 residual 的正负号和返回值合同没有严格定义，“加 residual”与“减 correction”可能表达同一意图，也可能代表完全相反的运算。代码中真正执行的是对两个完整预测的加权相加。这个命名和符号歧义本身就是 v1 的缺陷之一。

### 6.2 v1 的主要缺陷

v1 存在四个需要分开的风险：

1. residual contract 不清楚：函数名像是返回变化量，实际返回完整表达；
2. control anchor 候选曾容易被限制在 selected query spots，而不是完整 eligible control pool；
3. matching 特征可能混入受 perturbation 强烈影响的表达信号，导致匹配本身携带 response 信息；
4. calibration 的改善与 residual mechanism 的改善混在同一个最终分数中。

这些问题意味着“v1 的 E-distance 很好”不能直接推导为“v1 residual 学得好”。

### 6.3 v2 的 explicit residual contract

v2 先定义纯扰动变化量：

```text
Delta_context(i)
= sum_j pi(j | i) × [x_target(j) - x_control-anchor(j)]
```

然后才构造完整 residual prediction：

```text
P_residual(i) = x_source(i) + Delta_context(i)
```

这里：

- `i` 是待预测 source/query spot；
- `j` 是 target reference spot；
- `pi(j | i)` 是 source spot 与 target reference spot 的软匹配权重；
- `x_control-anchor(j)` 来自完整 eligible control pool；
- `Delta_context` 只表示变化量；
- `P_residual` 才是完整预测表达。

这种拆分让代码、结果表和科学叙述使用同一语义，避免把 delta 与 expression 混为一谈。

### 6.4 Perturbation-invariant matching

perturbation-invariant matching 可译为“尽量不受扰动响应影响的匹配”。匹配主要使用空间坐标、niche composition、boundary score、slide/batch 信息和控制侧表达表示，而不是直接用 target response expression 决定谁和谁匹配。

目标不是声称匹配表示完全去除了扰动信息，而是减少“使用答案的一部分寻找最相似答案”的风险。

### 6.5 Calibration 到底是什么

calibration 是 reference-only moment alignment，即利用 target reference 调整预测的均值、方差和合理范围。它可以修正预测分布的尺度偏移，因此往往显著改善 E-distance。

但 calibration 不等于机制学习。它可能让整体分布更接近 target，同时削弱 source individuality，或者并没有改善逐基因的真实扰动变化。因此所有 calibration 结果必须单独标记，不能和未校准机制结果混为一谈。

---

## 7. 主要实验结果

### 7.1 Existing methods 的 held-out 重评

| 方法 | Access | Held-out E-distance ↓ | Delta PCC ↑ | DE direction ↑ | Boundary ↓ | Individuality ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| legacy v1 fixed 80/20 calibrated | L3 | 55.3383 | 0.8914 | 0.9686 | 0.7025 | 0.6403 |
| adaptive calibrated anchor | L2 | 71.3564 | 0.8897 | 0.9673 | 0.6430 | 0.0661 |
| legacy v1 fixed 80/20 | L3 | 72.9115 | 0.8737 | 0.9677 | 0.6803 | 0.9134 |
| Gaussian anchor | L1 | 92.9950 | 0.9012 | 0.9695 | 0.5187 | 0.0951 |
| Cauchy anchor | L1 | 94.4492 | 0.9025 | 0.9695 | 0.5119 | 0.0680 |
| source identity | L0 | 110.5040 | 未定义 | 0.0000 | 1.5530 | 1.0000 |
| SpatialProp reference-only | L2 | 146.8739 | 0.6666 | 0.8954 | 1.6402 | 0.0761 |
| legacy NicheFlow-lite residual | L3 | 151.9095 | 0.5596 | 0.8808 | 1.6128 | 0.9926 |

校准 legacy v1 对独立 L2 calibrated anchor 获得 1,101/1,110 次胜出，paired mean E-distance difference 为 -16.0182，task-clustered 95% CI 为 [-18.2811, -13.9934]。

但是未校准 legacy v1 对最强未校准 A2.1 baseline 只有 587/1,110 次胜出，平均差值为 +4.3425。由于 E-distance 越低越好，正差值表示平均更差。

可以得出的结论：reference-only calibration 后，legacy v1 的 held-out 分布表现确实很强；该结果不是直接使用 evaluation targets 得到的。

不能得出的结论：不能把这 55.3383 全部归因于 residual mechanism，也不能据此声称 v1 能预测从未见扰动。

SpatialProp-style reference-only 模型完成了全部 1,110 个配置，但平均 E-distance 为 146.8739，明显弱于简单 anchor 和 v2 轻量方法。这说明在当前小 reference pool 和任务合同下，增加 spot-GNN 并不会自动带来更好结果。

### 7.2 v2 residual/gating 结果

| 方法 | Held-out E-distance ↓ | Niche E-distance ↓ | Delta PCC ↑ | DE direction ↑ | Boundary ↓ | Individuality ↑ | Spatial-map PCC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed 80/20 calibrated | 54.1277 | 56.2992 | 0.8933 | 0.9685 | 0.6944 | 0.6625 | 0.6109 |
| gated calibrated | 63.9482 | 65.4745 | 0.8909 | 0.9678 | 0.6482 | 0.4978 | 0.6335 |
| fixed 80/20 | 66.8581 | 69.7634 | 0.8848 | 0.9675 | 0.6458 | 0.9211 | 0.6031 |
| residual prediction only | 69.4766 | 87.3574 | 0.8198 | 0.9611 | 1.3844 | 0.9996 | 0.3336 |
| gated | 85.9374 | 87.0608 | 0.8832 | 0.9663 | 0.5825 | 0.7222 | 0.6477 |
| anchor only | 93.8489 | 94.8701 | 0.8817 | 0.9660 | 0.5777 | 0.0653 | 0.6505 |

可以得出的结论：

- explicit residual 将 residual-only 从 151.9095 改善到 69.4766；
- fixed 80/20 未校准版本达到 66.8581，优于 anchor-only 和 gated；
- residual branch 很好地保留 source individuality；
- calibration 仍能进一步降低 E-distance。

不能得出的结论：

- residual-only 的 boundary leakage 为 1.3844，spatial-map PCC 只有 0.3336，不能说空间机制已经解决；
- calibrated fixed 80/20 的 54.1277 不表示未校准 mechanism 已经稳定显著；
- gated calibrated 较好不代表 gate 学会了自适应，因为其权重本身已经塌缩。

### 7.3 Calibration dependency

calibration 对 fixed 80/20 的平均 E-distance 改善为 -12.7304，对 gated 的改善为 -21.9893。负值表示校准后 E-distance 更低。

这证明 calibration 在当前结果中仍占很大比例。因此 promotion candidate 选择未校准 fixed 80/20，而不是最低绝对 E-distance 的 calibrated fixed 80/20。成功门要回答的是“机制是否独立有效”，而不是“允许更多 target moment alignment 后能否取得最低分”。

### 7.4 Reference fraction 与 spatial-block 敏感性

下表是未校准 fixed 80/20 相对最强未校准 A2.1 baseline 的 paired E-distance difference。负值表示 fixed 80/20 更好。

| Reference fraction | Split type | Mean difference ↓ | Win rate |
| ---: | --- | ---: | ---: |
| 0.25 | random-stratified | -9.0637 | 0.7459 |
| 0.25 | spatial-block | -0.9476 | 0.6486 |
| 0.50 | random-stratified | -6.3413 | 0.7622 |
| 0.50 | spatial-block | +10.9391 | 0.5135 |
| 0.70 | random-stratified | -5.6430 | 0.7351 |
| 0.70 | spatial-block | +0.7916 | 0.6432 |

可以得出的结论：方法在 random-stratified 下普遍有优势，但空间块拆分显著削弱优势；0.50 spatial-block 是最清楚的失败组合。

不能得出的结论：不能把 0.25 的负均值解释为“reference 越少反而越好”。不同 fraction 的 evaluation pool 大小和任务方差不同，需要 paired sensitivity 分析，而不是直接比较绝对均值。

---

## 8. 为什么 gating 没有成功

### 8.1 Gate 原本想解决什么

固定 80/20 对所有 source spots 使用相同权重。理论上，不同 spot 的 residual reliability 不同：

- reference 很密、匹配熵低时，可以多信 residual；
- reference 很远、anchor 与 residual 分歧大时，应少信 residual；
- boundary 或 cross-slide 情况可能需要更保守。

因此 v2 定义：

```text
P_v2(i) = [1 - g(i)] × P_anchor(i) + g(i) × P_residual(i)
0 <= g(i) <= 1
```

`g(i)` 越大，表示模型越信任 residual branch。

### 8.2 实际发生了 gate collapse

未校准 gated 模型的统计是：

| Gate statistic | 数值 |
| --- | ---: |
| mean | 0.051562 |
| within-run std | 0.000948 |
| mean minimum | 0.050110 |
| mean maximum | 0.053932 |

gate 的预设下界约为 0.05，而实际权重几乎都在 0.05 附近。这就是 gate collapse：门控函数没有根据 spot 条件产生足够变化，而是退化为“几乎只使用 anchor”。

因此 gated 的 E-distance 为 85.9374，接近 anchor-only 的 93.8489，远差于固定 80/20 的 66.8581。

### 8.3 可能原因

当前证据支持优先检查三类原因：

1. gate 特征尺度不匹配：entropy、distance、density 和 disagreement 的数值范围可能使 S 形映射函数（sigmoid）很快饱和；
2. 保守规则过强：多个不确定性项叠加后，大部分 spot 都被判定为低 reliability；
3. feature 与真实 residual error 对应不足：当前可审计特征未必能预测 residual 何时真正优于 anchor。

现阶段不应直接把 gate 换成更大的神经网络。应先画出 gate feature、真实 residual advantage 与 task type 的关系，确认问题来自标定、特征还是 residual 本身。

---

## 9. 成功门与当前决策

### 9.1 为什么需要成功门

如果每次实验后再凭直觉选择有利指标，容易发生 narrative override，即用叙事覆盖预先约定的失败条件。G1–G7 在实验前定义，用来约束是否进入更复杂的 graph/flow 阶段。

候选方法是未校准 fixed 80/20，比较对象是对应 task/split 下最强的未校准 A2.1 reference-only baseline。机制空间指标则与同协议下可重建的 NicheFlow-v1 比较。

### 9.2 正式结果

| Gate | 判定 | 观察值 | 如何理解 |
| --- | --- | --- | --- |
| G1：held-out E-distance 跨任务优势 | FAIL | 749/1,110 wins；mean -1.7108；95% CI [-9.0668, 6.0859] | 平均和 win rate 有利，但 CI 跨 0，跨任务显著性不足 |
| G2：delta 非劣 | PASS | PCC -0.0102；Spearman +0.0066；MAE -9.97%；RMSE -3.27% | 扰动效应没有超过预设劣化界限 |
| G3：DE direction 非劣 | PASS | direction +0.2675；opposite +0.0163 | 关键基因方向总体满足要求 |
| G4：boundary 与 individuality 非劣 | PASS | boundary -6.00%；individuality +0.0077 | 相对 v1 没有牺牲这两项机制指标 |
| G5：reduced-reference 支持 | FAIL | fraction 0.50 mean +2.2989；fraction 0.25 mean -5.0057 | 0.50 reference setting 未保持优势 |
| G6：calibration-free gain | PASS | mean -1.7108；win rate 0.6748 | 改善不完全依赖 calibration |
| G7：coverage 与 variability | PASS | 37 tasks、5 seeds、2 split types | 报告覆盖满足约定 |

### 9.3 为什么 G1 失败，即使 749/1,110 次胜出

1,110 行并不是 1,110 个独立生物学任务，而是 37 个任务在不同 fraction、seed 和 split 下的重复评价。同一 task 内的多行相关，不能把它们当成独立样本计算置信区间。

因此 bootstrap 按 task 聚类。得到的 95% CI 上界是 +6.0859，跨过零。这表示当前 37 个任务不足以排除“平均改善主要由部分任务驱动”的可能性。

### 9.4 当前决策的准确含义

`do_not_advance_to_graph_flow` 是阶段性工程门控：

- 当前不投入 Graph Transformer、conditional latent flow 或 Best-of-K reranking；
- 先解决 spatial-block 失败和 gate collapse；
- 保留 explicit residual 与 fixed 80/20 作为可靠基线；
- 当轻量方案重新满足 G1–G7 时，可以恢复复杂模型研究。

它不是对 graph/flow 长期价值的否定，也不是说所有 v2 工作失败。相反，explicit residual 已经给出明确正结果，只是整体证据还不足以支持扩大模型复杂度。

---

## 10. 当前进度与下一步

### 10.1 已完成

| 工作项 | 状态 | 产物 |
| --- | --- | --- |
| target reference/evaluation 无泄漏拆分 | 完成 | `splits.py`、split manifest |
| full control/query/target payload | 完成 | `data.py` |
| E-distance、delta、DE 新指标 | 完成 | `metrics.py` |
| 空间与 context diagnostics | 完成 | metric registry 与结果 CSV |
| existing methods held-out 重评 | 完成 | 14,430 metric rows |
| SpatialProp reference-only 重训 | 完成 | 1,110 metric rows |
| explicit residual 与 invariant matching | 完成 | `methods.py` |
| fixed/gated/calibrated ablations | 完成 | 6,660 metric rows |
| G1–G7 与 clustered bootstrap | 完成 | `gate_summary.csv` |
| 完整 artifact verifier | 完成 | complete mode 输出 `ok` |

### 10.2 当前未解决

第一，0.50 spatial-block 下 mean difference 为 +10.9391，是最重要的稳健性失败。需要检查 reference block 是否覆盖 query context、soft matching 是否跨越空间空洞，以及少数 task 是否主导正误差。

第二，gate collapse 尚未解决。需要校准 feature scale，并直接评价 gate score 与“residual 相对 anchor 的真实 held-out advantage”是否相关。

第三，residual-only 的 boundary leakage 和 spatial-map PCC 仍弱，说明保留 individuality 与保持局部 target context 之间存在冲突。

第四，严格 unseen/OOD perturbation 仍未进入主线。若未来开展，应作为独立协议，不能复用 Mainline A 的 target reference access 结论。

### 10.3 推荐的下一轮顺序

1. 对 0.50 spatial-block 做 task-level error decomposition；
2. 输出 gate feature 分布、饱和比例及其与 residual advantage 的关系；
3. 重新标定 deterministic gate，并保留 fixed 80/20 作为对照；
4. 只重跑必要 ablation，重新计算 G1–G7；
5. 只有全部成功门通过后，再设计 graph/flow。

---

## 11. 术语、公式符号与文件索引

### 11.1 术语表

| 术语 | 中文解释 |
| --- | --- |
| anchor | 目标锚点预测；直接根据 target reference 构造的表达估计 |
| residual | 残差或变化量；v2 中专指相对 control anchor 的扰动变化 |
| residual prediction | source expression 加上 residual 后的完整预测 |
| calibration | 使用 target reference 进行均值、方差和范围对齐 |
| held-out | 留出评价；目标数据未参与预测，只用于指标 |
| full-pool | 完整 target pool，包括模型可能访问过的 reference rows |
| gating | 门控；为 anchor 与 residual prediction 分配逐 spot 权重 |
| gate collapse | gate 权重几乎为常数或贴住上下界 |
| DE | differential expression，差异表达 |
| PCC | Pearson correlation coefficient，皮尔逊相关系数 |
| OOD | out-of-distribution，分布外条件 |
| spatial-block | 按完整空间区域留出的拆分方式 |
| task-clustered bootstrap | 任务聚类自助重采样；以 task 为重采样单位并保留 task 内相关性 |

### 11.2 公式符号

| 符号 | 含义 |
| --- | --- |
| `i` | 待预测的 source/query spot |
| `j` | target reference spot |
| `x_source(i)` | source spot 的观测表达 |
| `x_target(j)` | target reference spot 的表达 |
| `x_control-anchor(j)` | 与 target reference 匹配的 control 表达 |
| `pi(j | i)` | source spot 到 target reference 的软匹配权重 |
| `Delta_context(i)` | 纯 context-conditioned perturbation residual |
| `P_anchor(i)` | anchor branch 完整预测 |
| `P_residual(i)` | source 加 residual 后的完整预测 |
| `g(i)` | residual branch 的门控权重 |

### 11.3 主要文件

- 实验入口：`repro_runs/2026-07-11-nicheflow-v2/README.md`
- 本说明：`repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_当前项目进展中文说明.md`
- 正式精简报告：`repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_report.md`
- Existing-method metrics：`repro_runs/2026-07-11-nicheflow-v2/results/existing_method_reassessment/metrics.csv`
- v2 residual/gating metrics：`repro_runs/2026-07-11-nicheflow-v2/results/residual_gating/metrics.csv`
- 正式成功门：`repro_runs/2026-07-11-nicheflow-v2/results/residual_gating/gate_summary.csv`
- SpatialProp metrics：`repro_runs/2026-07-11-nicheflow-v2/results/spatialprop_reference_only/metrics.csv`
- 已批准设计：`docs/superpowers/specs/2026-07-11-nicheflow-v2-design.md`
- 实施与复现计划：`docs/superpowers/plans/2026-07-11-nicheflow-v2.md`

### 11.4 最后的阅读建议

如果只需要把握项目决策，阅读第 1、8、9、10 节即可。

如果要审查评价是否公平，重点阅读第 2、3、4、5 节，并直接检查 split manifest、metrics.csv 和 gate_summary.csv。

如果要继续开发方法，重点阅读第 6、7、8 节。下一轮不应先增加模型容量，而应先确认 spatial-block failure 与 gate collapse 的具体来源。
