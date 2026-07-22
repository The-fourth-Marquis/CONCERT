# 2026-07-20 工作归档与 HRCGT 方法规划审查

## 0. 执行摘要

本文归档 2026-07-20 自文献调研以来的主要工作，并审查
`Hierarchical Reference-Conditioned Geometry-Preserving Transport`（HRCGT）
是否适合作为下一阶段和 AAAI 投稿的方法主线。

审查结论为：

> HRCGT 有明确的领域范式基础，但当前仅是工作假设，必须在术语、模型容量、
> 泛化机制和验证协议上修订后才能成为论文主方法。

今天已经完成的核心科学进展不是得到最终模型，而是通过多轮可证伪实验确认：

1. 强 target-distribution matching 可以降低 E-distance，但容易压缩 source
   heterogeneity；
2. shared global shift 可以保护 source individuality，却不足以描述 border、
   periphery、cross-slide 和 spatial-block 中的异质响应；
3. identity loss、source skip 或单一 scalar modulation 只能移动折中点，不能消除
   target localization 与 source geometry 的结构冲突；
4. reference-only calibration 能防止直接使用 evaluation rows 调参，但目前只有
   4/12 个代表性 task/split unit 找到安全的非零局部修正；
5. 当前模型不应扩展到 37 tasks 或多 seed，下一步必须先提高 conditional
   transport prior 的跨 context 泛化能力。

HRCGT 的推荐修订方向是：

```text
confidence-weighted reference conditional estimator
+ shared low-rank conditional shift with partial pooling
+ uncertainty-aware bounded residual gate
+ explicit distributional transport objective
+ reference-only nested model selection
```

不建议直接实现“每个 niche 一个完整 gene-dimensional shift”。

---

## 1. 科学范围与 claim boundary

当前工作始终属于 Mainline A：

> observed-perturbation spatial relocation with explicitly reported
> target-reference access。

当前结果不支持以下更强主张：

- strict unseen/OOD perturbation prediction；
- 同一真实 spot 扰动前后的 causal correspondence；
- spot-level causal counterfactual；
- 细胞在组织中的真实物理迁移；
- 无 target-reference 条件下的零样本预测。

所有方法比较均应明确 reference fraction、split type、seed、calibration 状态和
target access level。target evaluation expression 不得进入表示学习、coupling、
超参数选择或预测。

---

## 2. 文献调研归纳

### 2.1 直接支持的已有范式

#### Conditional transport

Conditional Monge Gap（arXiv:2504.08328）将 OT map 条件化于 drug、dose、
cell type 等 covariates，并通过跨条件共享模型测试 unseen drug/dosage。

它支持：

- conditional transport map；
- cross-task parameter sharing；
- unpaired source/target distribution modeling。

它不直接支持：

- spatial niche-specific response；
- source pairwise geometry preservation；
- inference-time target-reference shrinkage。

#### Spatial microenvironment modeling

CONCERT、SpatialProp 和 NicheFlow 共同支持以下判断：

- perturbation response 依赖 intrinsic state 和 local microenvironment；
- 一个全组织共享的作用矩阵或 shared stimulus 假设过粗；
- niche/microenvironment 可以作为比单个 cell 更合理的空间条件单元；
- spatial model 需要显式处理边界、邻域传播和局部组织组成。

其中：

- CONCERT 使用 perturbation disentanglement、GP spatial kernels 和 generative
  reconstruction；
- SpatialProp 使用 GNN 预测邻域作用，并包含 calibration framework；
- NicheFlow 使用 microenvironment point clouds、entropic OT 和 variational
  flow matching 建模时空组织演化。

#### Intrinsic/context disentanglement

Cellina 将表达分解为 intrinsic cell identity 与 extrinsic spatial context，使用
cell-type/spatial-domain supervision 作为生物归纳偏置，并在 held-out spatial
domains 中验证。

它支持：

- source identity 与 spatial response 应分离；
- 单纯靠无监督 factorization 不可识别；
- 生物标签可以作为 soft inductive bias；
- OOD spatial-domain validation 是必要证据。

#### Distribution and uncertainty control

scDFM 使用 conditional flow matching + MMD 强调 population-level endpoint
alignment。SPATIA 使用 confidence-aware OT reweighting，降低不可靠弱配对对
flow supervision 的影响。

它们支持：

- 不能只优化 pointwise reconstruction；
- OT coupling 不是确定的 cell-to-cell truth；
- coupling uncertainty 应进入训练权重或 residual gate；
- distribution fidelity 与 local dynamics 需要分别约束。

#### Niche transferability

QueST 将 niche 表示为 subgraph，通过 contrastive learning 与 adversarial batch
correction 在跨样本、跨平台条件下检索相似 niche。

它支持 niche 可以是具有跨样本意义的生物单元，但不能直接证明“每个 niche 应有
独立 perturbation shift”。

### 2.2 没有直接文献支持的部分

目前没有发现工作完整提出并验证：

```text
reference conditional target estimator
+ geometry-preserving shared translation
+ uncertainty-aware shrinkage residual
+ spatial perturbation relocation
```

因此 HRCGT 应被表述为对已有 conditional transport、spatial disentanglement、
partial pooling 和 confidence-aware weak supervision 的任务化统一，而不是一个
已经成熟的标准领域术语。

不应声称：

> first semiparametric conditional transport model

更稳妥的表述是：

> We formulate observed-perturbation spatial relocation as a
> reference-conditioned transport problem and introduce a structured transport
> map that controls context adaptation while preserving source geometry.

---

## 3. 今日实验进展

### 3.1 Closed-form nonlinear Monge map

第一阶段建立 reference-only niche graph、Sinkhorn coupling 和 nonlinear Monge
map。

代表性 6-task pilot 中：

| 版本 | Mean E-distance delta vs fixed | Boundary relative difference | Individuality difference |
| --- | ---: | ---: | ---: |
| Base nonlinear Monge | -1.2927 | 未通过 | -0.7967 |
| Source residual reconstruction | -10.1736 | +0.7343 | -0.8333 |
| Bounded intrinsic modulation | +13.3683 | +1.2443 | -0.00085 |

结论：

- 强 nonlinear Monge delta 可以改善 distribution matching；
- 改善主要来自压缩 heterogeneous source states；
- 显式 source identity path 可以恢复 individuality，但 distribution、delta 和
  boundary 随之失败；
- 继续扫描 latent dimension 或 modulation scalar 没有依据。

### 3.2 Direct conditional residual flow

实现了标准化 latent 中的 conditional velocity network、Sinkhorn barycentric
targets 和 Euler integration。

首个真实 task：

| 指标 | Direct flow | Fixed 80/20 |
| --- | ---: | ---: |
| E-distance | 113.63 | 52.93 |
| Delta PCC | 0.8178 | 0.9846 |
| Boundary | 1.3547 | 0.4934 |
| Individuality | 0.9839 | 1.0000 |

结论：训练稳定性不是主要问题。direct flow 同时承担 global target shift、local
spatial localization 和 source-specific correction，监督目标混杂，无法定位正确
target state。

### 3.3 Anchor-initialized residual flow

将 reference-only local anchor 作为初始状态，再学习 residual flow。

| 方法 | E-distance | Delta PCC | Boundary | Individuality |
| --- | ---: | ---: | ---: | ---: |
| Fixed 80/20 | 52.9304 | 0.9846 | 0.4934 | 1.0000 |
| Initial anchor | 45.9050 | 0.9931 | 0.5072 | 0.9717 |
| Flow, horizon 0.25 | 38.8719 | 0.9923 | 0.4941 | 0.9255 |
| Flow, horizon 1.00 | 31.2464 | 0.9824 | 0.6620 | 0.5124 |

Identity loss 和 source skip 消融显示：

- 增大 identity loss 会同时恶化 E-distance 和 boundary；
- 增大 source skip 能恢复 individuality，但把预测拉回错误 source distribution；
- query-specific correction field 本身在改变 pairwise geometry；
- 训练集 pairwise identity loss 不能保证 query geometry 泛化。

### 3.4 Shared global shift

将 correction 限制为 source expression + shared global shift + bounded centered
modulation。

首个 task 结果：

| E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: |
| 81.8397 | 0.8833 | 1.2477 | 0.9989 |

结论：shared translation 严格保护 pairwise source geometry，但不能单独替代
target local location。

### 3.5 Local anchor + source-context residual

尝试：

```text
target local anchor
+ (source expression - source-context anchor)
+ bounded residual flow
```

首个 task 最佳 horizon=0.5：

| E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: |
| 69.7014 | 0.8922 | 1.1529 | 0.9815 |

结论：leave-one-out source residual 能保留 individuality，但也将 source
distribution error 带回 target space，不能修复 target location。

### 3.6 Anchor-guided bounded factorization

将 local anchor 仅作为有界 context modulation：

```text
base = source expression + shared global shift
prediction = base + strength * (target local anchor - base)
```

未经 calibration 的细网格显示 strength=0.77 在首个 task 上同时通过四项机制门。
随后实现 reference-only pseudo-validation，最终选择 strength=0.76：

| E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: |
| 31.8327 | 0.9905 | 0.5119 | 0.9829 |

相对 fixed 80/20：

- E-distance：-21.0977；
- Delta PCC：+0.0059；
- Boundary relative difference：+3.74%；
- Individuality difference：-0.0170。

首个 random-stratified task 通过四项门，mutation guard 通过。

### 3.7 Reference-calibrated 6-task pilot

运行 6 个代表性 tasks × random-stratified/spatial-block，共 12 units。

| 汇总指标 | 结果 |
| --- | ---: |
| Mean E-distance difference vs fixed | +17.3936 |
| Mean Delta PCC difference | -0.0345 |
| Mean boundary relative difference | +103.62% |
| Mean individuality difference | +0.0027 |
| Distribution promotion | Failed |
| Delta promotion | Failed |
| Boundary promotion | Failed |
| Individuality promotion | Passed |

只有 4/12 units 的 pseudo-validation 找到安全非零 strength。其余 unit 使用
strength=0 的保守 fallback，individuality 得到保护，但 target localization 在
border、cross-slide、periphery 和部分 spatial-block 条件失败。

最终决定：

```text
do_not_advance_to_37_tasks_or_multiseed
```

---

## 4. 当前主要问题与顾虑

### 4.1 “Transport”术语可能过强

当前表现最好的 anchor-guided predictor 最终预测路径主要是 context kernel、
shared translation 和 bounded interpolation，并没有直接求解最终 OT objective。

只有当下一版保留以下至少一项时，`Transport` 才有充分方法依据：

- explicit Sinkhorn/Monge coupling；
- endpoint MMD/Wasserstein/Sinkhorn distribution loss；
- 明确的 pushforward distribution objective。

否则应改称 `Mapping` 或 `Geometry-Regularized Decoder`，避免暗示严格 Optimal
Transport。

### 4.2 “Geometry-Preserving”需要限定

shared translation 满足：

```text
(x_i + shift) - (x_j + shift) = x_i - x_j
```

但一旦使用 context-dependent shift 或 local modulation，完整模型不再严格保持
全局 pairwise geometry。

下一版必须明确：

- 精确保持的是 shared-shift base path；
- 条件 shift 的变化由低秩和平滑性限制；
- local residual 引起的 distortion 由 hard cap 控制；
- 给出 pairwise distortion bound 或至少可验证的 contract test。

若无法给出严格性质，方法名宜使用 `Geometry-Regularized` 而非
`Geometry-Preserving`。

### 4.3 “Hierarchical”目前尚未实现

当前实现只有一个 task-level global shift 和一个 scalar strength。没有：

- learned group/context hierarchy；
- partial pooling；
- low-rank context-conditioned shift；
- uncertainty-aware instance gate。

因此 HRCGT 当前是下一阶段设计，不是已完成方法。

### 4.4 每组独立 shift 会严重过拟合

以下实现不可接受：

```text
one full gene-dimensional free vector per niche/spatial block
```

原因包括：

- gene dimension 远大于每组 reference 数量；
- group label 可能是连续生物过程的粗离散化；
- group 数量和定义容易被 evaluation results 反向选择；
- unseen group 无自然预测路径；
- 容易记忆 slide/batch，而非 perturbation response。

### 4.5 Reference pseudo-validation 的可迁移性有限

当前 nested calibration 防止了直接 evaluation leakage，但 4/12 的安全候选率
说明 pseudo-validation risk 与真实 held-out risk 并不稳定一致。

下一版不能继续依赖单一 scalar grid。需要：

- explicit support/confidence estimator；
- multiple pseudo splits 或更稳定的风险估计；
- conservative fallback；
- report selection frequency 与 calibration error。

### 4.6 生物学分组尚未证明

文献支持 niche 是有意义的空间单元，但不证明 niche-specific perturbation shift
必然存在。下一版需要：

- 使用与 target evaluation 无关的 niche/cell-type composition；
- random-group label negative control；
- replicate/slide consistency；
- pathway 与 DE-direction consistency；
- boundary/core/periphery 的效应差异分析。

### 4.7 AAAI 方法贡献仍不充分

当前 scalar shrinkage/interpolation 作为单独算法过于简单，且 6-task 泛化失败。

若投稿主方法仍只包含：

```text
global shift + scalar alpha * anchor residual
```

审稿人可能将其视为启发式 convex blending，而不是新的 AI 方法。

论文方法至少需要：

- shared low-rank conditional component；
- support/uncertainty-aware adaptive gate；
- explicit distributional learning objective；
- structural geometry bound；
- spatial/cross-slide 泛化证据。

---

## 5. HRCGT 规划审查结论

### 5.1 可保留的核心思想

1. reference-conditioned target distribution estimation；
2. source-preserving base path；
3. bounded local adaptation；
4. reference-only model selection；
5. confidence-aware fallback；
6. 以 niche/context 为连续条件，而非任意 group lookup。

### 5.2 必须修订的部分

原规划中的：

```text
one shared shift per niche/spatial group
```

应改为：

```text
global shift
+ partially pooled low-rank context-conditioned shift
```

即：

~~~text
Delta(c_i) = Delta_0 + rho_i * U * h(c_i)
~~~

其中：

- Delta_0：task-level global shift；
- h(c_i)：低维 spatial/niche context representation；
- U：所有 contexts 共享的 low-rank gene basis；
- rho_i：由 reference support 决定的 partial-pooling coefficient。

推荐：

~~~text
rho_i = n_eff(c_i) / (n_eff(c_i) + tau)
~~~

低支持 context 自动收缩到 global shift。

### 5.3 修订后的统一模型

#### Reference conditional estimator

~~~text
a_i = sum over j in R of pi_ij * y_j^R
~~~

其中 pi_ij 由 context similarity 和 confidence-aware soft coupling 得到。

#### Geometry-preserving base path

~~~text
b_i = x_i + Delta_0 + rho_i * U * h(c_i)
~~~

#### Uncertainty-aware bounded adaptation

~~~text
r_i = a_i - b_i
~~~

~~~text
lambda_i = lambda_max * rho_i * q_i
~~~

~~~text
y_hat_i = b_i + lambda_i * Clip_kappa(r_i)
~~~

其中 q_i 表示 coupling/anchor confidence。低 reference support 或高 coupling
entropy 时，lambda_i 自动减小。

### 5.4 推荐训练目标

~~~text
L = L_dist
  + beta * L_delta
  + gamma * L_boundary
  + eta * L_reg
~~~

- L_dist：confidence-weighted Sinkhorn/MMD endpoint loss；
- L_delta：group-level perturbation direction；
- L_boundary：pseudo-reference niche/boundary consistency；
- L_reg：low-rank、smoothness、partial pooling 和 gate penalty。

不建议继续将 pairwise identity loss 作为主要保护机制。geometry 应主要由 base
path 与 bounded residual 的结构性质保证。

### 5.5 推荐术语

若保留 explicit distributional transport objective，可暂用：

> Hierarchical Reference-Conditioned Geometry-Preserving Transport

但正文必须限定 preservation 的范围并给出 distortion bound。

若最终仍是 deterministic reference interpolation，推荐改为：

> Hierarchical Reference-Conditioned Geometry-Regularized Mapping

当前阶段不应在摘要中声称 HRCGT 已经验证。

---

## 6. 后续工作规划

### 阶段 A：方法最小实现

目标：只增加能直接回答当前失败模式的容量，不建设完整 Transformer/GNN。

实现顺序：

1. 固定现有 reference conditional estimator；
2. 在低维 expression basis 中实现 Delta_0 + rho * U * h(c)；
3. 首先使用 ridge/linear context head，不直接使用大 MLP；
4. 加入 reference density、coupling entropy 和 anchor variance；
5. 实现 lambda_i = lambda_max * rho_i * q_i；
6. 保留 strength=0/global-shift fallback；
7. 只有低容量模型通过 6-task 后，才考虑更复杂 encoder。

### 阶段 B：方法契约测试

在真实 pilot 前必须通过：

1. evaluation-expression mutation guard；
2. lambda=0 时 shared base 的 pairwise geometry invariance；
3. reference support 减小时 rho 和 lambda 单调减小；
4. unseen/empty group 自动回退；
5. low-rank dimension 不超过预先设定上限；
6. random group labels 不产生系统性增益；
7. expression scaling 不改变 latent behavior；
8. 所有输出与 losses 有限。

### 阶段 C：首个 task 机制门

相对 fixed 80/20，必须同时满足：

- held-out E-distance 不高于 fixed；
- Delta PCC difference 不低于 -0.02；
- Boundary relative difference 不高于 5%；
- Individuality difference 不低于 -0.02；
- mutation guard 通过；
- 所有超参数在 reference pseudo-validation 中冻结。

新模型至少应达到当前 reference-calibrated strength=0.76 的水平，否则停止。

### 阶段 D：6-task 双 split pilot

使用现有 6 个代表性 tasks：

- exact patch；
- border；
- cross-niche；
- cross-slide；
- niche core；
- periphery。

每个 task 同时运行：

- random-stratified；
- spatial-block。

在运行前冻结聚合 promotion rule。至少要求：

- mean E-distance delta < 0；
- spatial-block mean E-distance delta < 0；
- mean Delta PCC difference >= -0.02；
- mean Boundary relative difference <= 5%；
- mean Individuality difference >= -0.02；
- 所有 leakage guards 通过；
- 非零 adaptive correction 不是只出现在单一 task type。

若未通过，不运行 37-task 或多 seed。

### 阶段 E：泛化与生物学证据

只有阶段 D 通过后运行：

1. 3-5 split seeds；
2. reference fractions 0.25/0.50/0.70；
3. whole spatial-block holdout；
4. cross-slide evaluation；
5. random niche/group negative control；
6. replicate consistency；
7. group-wise DE-direction；
8. pathway enrichment；
9. ligand-receptor/cell-composition association；
10. reference support-error calibration curve。

### 阶段 F：论文方法叙事

推荐主线：

> Reference-conditioned geometry-regularized transport for observed spatial
> perturbation relocation under unpaired target-reference access.

推荐贡献结构：

1. problem formulation：reference-conditioned spatial relocation；
2. method：partial-pooled low-rank conditional transport prior；
3. method：uncertainty-aware bounded local adaptation；
4. property：source geometry distortion control；
5. protocol：reference-only nested selection and leakage guard；
6. evidence：spatial-block/cross-slide/biological validation。

不应把 scalar shrinkage 本身作为唯一算法创新。

---

## 7. 时间优先级建议

在投稿时间有限的前提下：

### 最高优先级

1. low-rank conditional shift；
2. support/confidence-aware gate；
3. 首个 task 与 6-task 双 split；
4. negative control 和跨 slide；
5. 方法公式、distortion property 和 claim boundary。

### 可延后

- 完整 Transformer/GNN backbone；
- 37-task × 5-seed；
- 大规模 hyperparameter search；
- strict unseen perturbation；
- 复杂多层 group hierarchy；
- 额外可视化和次要指标。

### 停止条件

若低秩 conditional shift + adaptive gate 仍不能在 6-task 中同时改善 distribution、
delta、boundary 和 individuality，应停止继续增加模型容量，并重新评估论文是否应
以方法贡献为主，避免用复杂度掩盖任务不可识别性或 reference coverage 不足。

---

## 8. 今日生成的主要工件

实验根目录：

```text
repro_runs/2026-07-20-nicheflow-v15/
```

主要报告：

- `reports/pilot_findings.md`
- `reports/conditional_flow_smoke.md`
- `reports/anchor_flow_smoke.md`
- `reports/structured_flow_smoke.md`
- `reports/local_residual_flow_smoke.md`
- `reports/anchor_guided_factorized_smoke.md`
- `reports/anchor_guided_calibrated_pilot.md`

主要最终结果：

- `results/anchor_guided_calibrated_smoke_final/summary.json`
- `results/anchor_guided_calibrated_pilot_final/metrics.csv`
- `results/anchor_guided_calibrated_pilot_final/summary.json`

主要新增方法原型：

- direct conditional flow；
- anchor-initialized residual flow；
- structured global-shift flow；
- local anchor + source-context residual flow；
- factorized global-shift flow；
- anchor-guided factorized decoder；
- reference-calibrated smoke/pilot runners。

验证状态：

```text
PASS 23 test files
```

---

## 9. 最终决策记录

### 已确认

- reference-only protocol 与 mutation guard 可复用；
- target local anchor 对 localization 有价值；
- shared translation 能结构性保护 source geometry；
- unrestricted query-specific correction 会损害 individuality；
- scalar bounded modulation 在单一 task 存在有效折中；
- 当前 scalar 模型不能跨 6-task 泛化。

### 尚未确认

- hierarchical conditional shift 是否优于 global shift；
- niche/context effect 是否可跨 slide 稳定迁移；
- support-aware gate 是否能提高 pseudo-validation 与 evaluation 的一致性；
- low-rank context model 是否具有生物学可解释性；
- HRCGT 是否能通过 6-task promotion。

### 下一步唯一优先方法问题

> 在不引入 per-group high-dimensional lookup 的情况下，partial-pooled low-rank
> context shift 与 uncertainty-aware residual gate 能否同时修复 target
> localization、boundary 和 cross-task generalization，同时保持 source geometry？

在该问题得到首个 task 和 6-task 双 split 的肯定答案前，不扩展实验规模，也不将
HRCGT 写成已验证的论文主方法。
