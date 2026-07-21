# SPAC-seq ELRR 设计规范

Date: 2026-07-21

## 0. 决策与本规范的状态

本阶段只设计一条新的 AI 路线：

```text
Episodic Learned Reference Residual Retrieval (ELRR)
```

ELRR 学习的是“给定 query context，应该从哪些已观测 target-reference
残差中取值以及取多少权重”，而不是用高维神经 decoder 自由生成一个新的
gene-dimensional correction。

本规范完成以下冻结：

- 科学问题和 claim boundary；
- SPAC-seq 数据清洗、join、guide assignment 与 episode readiness contract；
- 113 参数 ELRR 架构、训练目标和基线；
- 主指标、诊断指标、外部开发 gate、sealed final 与第二层 reserve；
- 泄漏防护、hash freeze、错误处理、artifact 与测试边界。

本轮不实现模型、不清洗表达矩阵、不读取 sealed expression、不运行实验。
只有用户复核并批准本规范后，才进入 implementation plan。

## 1. 与旧停止决策的关系

旧 Perturb-map 路线 B 已在
`repro_runs/2026-07-21-cross-pool-shared-film/reports/final_decision_zh.md`
预注册并执行：

```text
stop_all_algorithm_development
pivot_to_benchmark_or_empirical_study
```

该决定继续有效。ELRR 不得：

- 根据已经暴露的 GSM5808056/8057 holdout 修改架构、损失或阈值；
- 把 Perturb-map 重新包装成一个“新 final”；
- 用旧 holdout 选择 ELRR 的 feature、epoch、temperature 或 baseline；
- 事后修改旧路线的失败结论。

ELRR 之所以构成一个新的、可比较的阶段，而不是破坏旧 seal，是因为它使用：

1. 新的数据协议：SPAC-seq 直接 guide assignment；
2. 新的独立开发、external gate、sealed final 和 reserve 分层；
3. 新的主问题：小样本条件下学习 reference-residual 检索核；
4. 预先收窄的 distribution-level 指标合同。

Perturb-map 从本阶段开始只作为已经暴露的 backward-compatibility benchmark。
其结果可在 ELRR 完全冻结后描述性报告，但永不用于模型选择。

## 2. 科学问题与 claim boundary

### 2.1 科学问题

给定已经观测到的 perturbation、直接 NTC controls、少量 target references
和 target-expression-free spatial context，能否学习一个跨 perturbation、
跨 section episodically shared 的检索核，使 held-out target expression
distribution 的 E-distance 优于强固定 RBF/kNN 与 shared residual shift，同时
Delta PCC 的层级均值下降不超过 0.02？

本阶段分开回答两个 estimands，任何指标和结论都不得把二者混为一个总体：

```text
W: within-section few-reference estimation
C: matched cross-section reference transfer
```

W estimand：

```text
对 query section q 和 target t，
条件于 q 内冻结的 70% target support 与 q 内 direct NTC，
估计 q 内剩余 30% query 的 empirical expression distribution。
```

C estimand：

```text
对预先冻结的有向单元
(support specimen, query specimen, target)，
条件于指定 support section 的全部 eligible target references
与 query-section direct NTC，
估计完整 query section；
query section 不贡献任何 target reference。
```

W 是 primary Mainline A estimand；C 是单独的、预注册的 transfer stress
estimand。C 仍然使用已观测 perturbation references，因此不是 strict
unseen-perturbation OOD。

### 2.2 允许的主张

本阶段仍属于：

```text
observed perturbation spatial relocation
with explicitly reported target-reference access
```

更精确地说，ELRR 是一个：

```text
reference-conditioned distributional expression mapper
```

它只允许声称：

- W：有同 section target references 时的 held-out distribution prediction；
- C：有预先指定的其他 matching specimen references 时的 transfer prediction；
- 在预注册的有限 screens 和 reference-access regimes 上得到的 pilot evidence。

W 与 C 各自报告、各自过 gate；不得平均成一个更好看的总分。

### 2.3 明确不允许的主张

ELRR 不声称：

- strict unseen/OOD perturbation prediction；
- 没有任何 target reference 的 zero-shot prediction；
- 同一真实 spot 在扰动前后的 causal correspondence；
- spot-level causal counterfactual；
- 细胞在组织中的真实物理移动；
- optimal transport 或 transport map；
- source geometry preservation；
- 对所有组织、平台或未观测动物的普适泛化。

没有显式 coupling、pushforward objective 或 OT loss，因此方法名和正文只能使用
`retrieval`、`mapping` 或 `reference-conditioned prediction`。

## 3. 统一失败机制与 ELRR 的可证伪假设

### 3.1 已观察到的统一失败

现有 Perturb-map 记录形成了一条一致的结构性前沿：

1. shared global shift 或强 geometry-preserving base 能保存 source
   individuality，但不能恢复 target distribution、Delta 与 boundary；
2. local anchor、flow、Monge-style map 或自由 query-specific correction
   可以移动 E-distance，却容易压缩 source heterogeneity 或破坏 pairwise
   geometry；
3. per-task low-rank、affine 或 bounded residual 受限于约十个左右的
   pseudo-fit references，无法稳定识别 context interaction；
4. reference-only calibration 虽阻止直接泄漏，却频繁找不到 safe nonzero
   candidate；
5. 最终 cross-pool FiLM 的 20/20 task/split views 均为
   `no_safe_candidate`，回退到 alpha=0；它改善 E-distance，但 Delta 与
   boundary 仍失败。

这些结果与以下候选机制一致，但不能单独证明它是唯一原因：

```text
small target-reference pool
+ latent, unpaired distributional supervision
+ high-dimensional value generation
+ per-task calibration
```

对应的可检验工作假设是：

```text
global model underfit
<-> free local correction overfit
<-> calibration collapse
```

旧 Perturb-map 与新 SPAC-seq protocol 不同；本研究不得把旧失败外推成
“所有高维生成模型在 SPAC-seq 上必然失败”。

### 3.2 ELRR 的新假设

小数据可能不足以支持“生成一个新高维 residual”，但可能足以支持：

```text
在已测 reference residual prototypes 之间学习低容量的相似度与组合权重
```

ELRR 将高维值固定为真实 reference residual，仅学习 113 个全局共享参数的
pair scorer。它因此具有：

- 比固定 RBF/kNN 更灵活的 context-dependent matching；
- 比 FiLM、flow、GNN decoder 更小的可学习输出空间；
- 跨 perturbation/section 的 episodic parameter sharing；
- 不需要 per-task alpha calibration；
- 预测 residual 位于 observed residual convex hull 内的硬结构约束。

这条路线可能走出上述失败机制，但没有数学保证。若冻结的 ELRR 不能稳定优于
强 fixed retrieval 与 mechanism-null baselines，允许的结论仅是：

```text
本次 113 参数 learned-kernel 实现没有提供 learned retrieval gain 的证据
```

不得推出所有 learned retrieval 不可行，也不得通过增加容量继续追逐同一
external/final。

## 4. 与旧方法的精确关系

ELRR 不是与 kNN/RBF 完全无关的新方法族，而是对旧 retrieval skeleton 的
受控 learned-kernel extension。旧代码
`repro_runs/2026-07-20-nicheflow-v15/src/nicheflow_v15/anchor_flow.py:24-42`
已经实现固定 RBF softmax 后的 `weights @ reference_expr`。

ELRR 的新增结构只包括：

1. value 从 reference expression 改为 matched-control residual；
2. 固定 RBF kernel 改为跨 episode 共享的 113 参数 pair scorer；
3. 用 convex-hull residual contract 禁止自由高维 value generation；
4. 用 query-independent learned null 与强 RBF baseline 检验 learned
   query conditioning 是否真实贡献。

| 方法族 | 权重如何得到 | residual value 如何得到 | 小数据风险 |
| --- | --- | --- | --- |
| shared shift | uniform | reference residual mean | context underfit |
| kNN/RBF | 固定或开发集冻结的距离核 | observed residuals | 相似度受限 |
| per-task low-rank operator | 每 task 拟合 | 低秩生成 | reference 太少 |
| cross-pool FiLM | 神经网络 | latent residual 自由生成 | value model 仍过强 |
| flow/GNN | 神经场或消息传播 | 高维自由生成 | 过拟合、geometry 改写 |
| ELRR | 113 参数 episodic pair scorer | observed residuals 的凸组合 | 只需识别相对检索权重 |

ELRR 的贡献点必须限定为：

```text
episodically learned residual retrieval under sparse spatial references
```

不得把普通 attention、softmax 或 MLP 本身声称为新颖。

## 5. 数据分层与双层封存合同

### 5.1 数据角色

| 层级 | 数据 | 用途 |
| --- | --- | --- |
| development-train | Lung 992 / M001 | 训练、内部 grouped validation |
| development-train | eligible MC38 sections (`libHD274E6`–`libHD277E9`) | 训练；连续切片按 specimen 分组 |
| development-train | OT1 Day4 rep1/rep2、Day7 rep1、Day10 rep1 | 训练、内部 grouped validation |
| external development gate | Lung 994 / M002、OT1 Day7 rep2 | 模型冻结后一次性 gate |
| sealed final | Lung 993 / M003 | external gate 通过后一次性 final |
| second-layer reserve | OT1 Day10 rep2 | 下一独立阶段；不得救援本阶段 |
| quarantine | `subq` | 不进入主结果、训练、选择或 gate |
| exposed compatibility | Perturb-map | ELRR 冻结后描述性评估 |
| optional auxiliary | CONCERT Gut/Brain | 第一版不使用；以后只能作单独预训练消融 |

`M001=992`、`M002=994`、`M003=993` 在 manifest 中显式保存，不能依赖目录顺序。

### 5.2 prospective seal，而非伪装成从未查看

SPAC-seq 文件名、shape、feature schema 和部分 barcode/coordinate 结构已在
设计前被只读盘点。因此不得声称 Lung 993 或 OT1 Day10 rep2 是“从未接触过的
原始文件”。

本设计 commit 只声明封存意图。只有生成唯一的 `seal_activation.json` 且
access tests 全部通过后，才允许写“prospectively sealed”。该 artifact 必须包含：

```text
design_commit
activation_timestamp
data-role mapping
source checksums
pre_freeze_access_log hash
access-policy hash
access-test artifact hash
```

冻结前允许的 schema 访问写入 `pre_freeze_access_log.json`。seal activation
之后：

- 可读取文件存在性、字节数和 checksum；
- 不可读取 sealed RNA/guide value arrays；
- 不可生成 sealed assignment counts、target support 或 outcome metrics；
- external、final 与 reserve 使用不同 access token、runner 和 sentinel；
- 每次 role-specific 解封都按第 17 节的分阶段事务执行。

### 5.3 reserve 的含义

OT1 Day10 rep2 不是 final 失败后的补考集。Stage 1 runner 永久无权访问它。
无论 Lung 993 成功或失败，reserve 都保持 sealed。

未来 Stage 2 只有在以下条件同时满足时才能解封：

1. 新科学问题、不同 estimand 和新 claim boundary 先写成独立规范；
2. Stage 2 规范引用 Stage 1 decision artifact 与 hash；
3. 当前 ELRR 结果已归档且不再修改、重标或覆盖；
4. 新代码、模型和协议重新 hash freeze；
5. 共同论文同时报告 Stage 1 outcome，并把 Stage 2 标为
   `post-Stage-1 independent study`。

因此保留第二层 reserve 不会使早期探索失去意义；它使两个阶段的结论可独立解释。

## 6. 作者代码只读审计与复用边界

作者仓库 `/root/code/SPAC-seq` 保持只读。当前 `main` 主要是精简后的 Figure 3
脚本，完整 spatial assignment 逻辑并不存在；部分实验 notebook 只能从 revision
`f7c8d4e` 读取。

### 6.1 alignment 审计

已确认的风险包括：

- `fig_3_utils.py:119-138` 先缩到交集，再用全部 RNA keys 重排 guide；
  guide 缺坐标时会在补零前报错，补零后也未重新排序；
- 同一错误复制于 `fig_3_perturbation_preprocessing.py:50-68`；
- key 由 float coordinate string 拼接，样本名含 `_`、负坐标或浮点格式变化时
  不可靠；
- `fig_3_load_stereoseq.py:57-68` 对每个文件独立减坐标最小值，并用当前文件
  `max(xb)` 生成 `bin_ID`，RNA 与 guide 的 key 不稳定；
- `fig_6_mapping_sc_to_spatial.r:29-35` 存在按行盲拼 expression、metadata 和
  coordinates；
- 当前仓库没有完整 Visium HD H5 ingestion、CID-to-chip whitelist、MAPQ/UMI
  聚合与 provenance contract。

结论：作者 loader 不能作为 ELRR 的数据入口。

### 6.2 guide assignment 审计

已确认的代码语义与缺口包括：

- `fig_3_perturbation_preprocessing.py:23` 只按 `sg` 名称前缀拆 guide；
- `fig_3_utils.py:62-74` 仅以 feature 名第一个 `_` 前缀合并 target，且会
  dense materialize、丢失 guide metadata；
- 历史 notebook 最强证据为 `MIDCount > 1`，即单 guide 行至少 2 UMI，
  不是至少两个不同 guide；
- 历史 notebook 调用的 `preprocessing.filter_guide_reads` 源码缺失；
- 手写 `groupby(['x','y']).max()` 可能将不同 row 的 gene 与 count 拼接，
  不是合法 argmax；
- 当前 spatial 代码没有通用 tie/multi-target 删除；
- Perturb-seq 代码硬编码 70 guides、对 log-normalized matrix 做 argmax，
  且没有可靠 tie contract；
- NTC 通过名称硬编码合并，intergenic control 没有 ontology；
- OT1 `>=100 bins` 在当前代码中没有实现；
- KDE 只足以作描述性空间统计，不能证明直接 label；没有可复用的
  DBSCAN/KDE rescue API。

### 6.3 可复用与必须重写

可复用的只有：

- guide counts 按 target gene 汇总的思想；
- 明确验证后的 RNA/guide coordinate join 概念；
- KDE 作为 secondary visualization。

必须重写：

- Visium 与 Stereo ingestion；
- manifest-based `guide_id -> target_gene/control_type`；
- raw UMI filter 与 assignment 状态机；
- OT1 support gate；
- tissue mask、坐标单位、registration 与 provenance；
- 全部 mutation/access/hash guards。

## 7. 数据 manifest 与 join contract

### 7.1 frozen manifest

每个 section 必须有：

```text
screen_id
sample_id
mouse_id
biological_specimen_id
section_id
timepoint
tissue
perturbed_cell_type
platform
capture_mode
library_reference_id
bin_size_um
coordinate_system
image_id
tissue_mask_id
registration_id
data_role
```

如果 `mouse_id`、specimen/continuous-section 关系、library mapping 或真实
bin size 无法从 accession metadata 确认，该 section 记为 `data_not_ready`；
不得根据文件名猜测后进入独立样本统计。

### 7.2 nullable platform-union schema

RNA/guide 共享 observation identity，但平台字段不是伪造的全集必填项。

所有平台必填：

```text
sample_id, section_id, platform
x_um, y_um, bin_size_um
in_tissue, coordinate_system, registration_id
total_rna_umi, n_genes
guide_total_umi, n_guides_detected
assignment_status, target_gene, control_type
```

Visium HD 额外必填：

```text
barcode
array_row, array_col
pxl_row_fullres, pxl_col_fullres
```

Stereo 额外必填：

```text
cid
x_raw, y_raw
```

`segmented_cell_id` 为 nullable。Visium 的 `cid/x_raw/y_raw` 与 Stereo 的
`barcode/array/pixel` 在不可用时必须为 null，不能生成占位字符串或假坐标。

每个 feature 至少保存：

```text
feature_id, feature_name, feature_type
guide_id, target_gene, control_type
```

raw counts 永久保存在 `layers["counts"]` 或等价只读 sparse artifact；任何
gene-level 汇总不能覆盖 guide-level evidence。

### 7.3 Visium HD join

主合同为单 section 内：

```text
key = (section_id, exact_barcode)
```

要求：

- RNA 与 guide barcode 唯一；
- barcode 集合完全一致；
- guide 显式重排到 RNA 顺序；
- join 后逐行断言 identity 相等；
- 不按 H5 column order、coordinate 或近似字符串拼接。

若分开的 RNA/guide H5 barcode 集合不一致，默认 `join_mismatch` 并停止。
只有 accession metadata 明确证明二者来自同一完整 barcode universe、且差异只
代表 sparse-zero dropping 时，才允许在独立 secondary adapter 中补零；该规则
不得在 primary pipeline 自动触发。

### 7.4 Stereo join

CID reads 到 chip whitelist：

```text
(section_id, cid) -> (x_raw, y_raw)
```

必须是 many-to-one 显式验证。RNA/guide GEM 聚合后使用：

```text
key = (section_id, int64_x_raw, int64_y_raw)
```

要求：

- RNA 与 guide 两侧不各自归零；
- feature/count 先按 key 聚合，再做 one-to-one observation join；
- RNA tissue coordinates 必须全部在 guide coordinate universe 中；
- guide-only coordinates 可丢弃，但数量与比例必须记录；
- `bin_ID` 不能由单文件 `max(x)` 计算；
- 不跨 section 比较原始坐标，除非有显式 registration transform。

### 7.5 tissue 与 coordinate contract

- primary episode 必须有可验证 tissue mask；
- `in_tissue=1` 不能由 Cartesian grid 自动伪造；
- x/y 轴序、pixel/array/physical units 分列保存；
- Stereo bin20 约 10 um、Visium HD bin4 约 8 um 只能在 run metadata 证实后
  写入；
- 连续 sections 只有在 transform、reference section 和 registration QC
  全部存在时才使用 registered coordinate。

## 8. 保守 perturbation assignment

### 8.1 primary 状态机

对每个 raw `(bin_id, guide_id, raw_umi)`：

1. 使用 frozen library manifest 映射 `guide_id, target_gene, control_type`；
2. guide-row 只有在 `raw_umi > 1` 时 eligible；
3. 在同一 target/control identity 内对 eligible guides 求和；
4. 同 target 多 guide 仍可 accepted；
5. 保存 guide-level `top_guide_count` 与 target-sum `top_count`；
6. 没有 eligible identity 时记 `unassigned_low_signal`；
7. 恰好一个 eligible target/control identity 时记对应 accepted 状态；
8. 多 identity 且 target-sum top tie 时优先记 `ambiguous_tie`；
9. 其余多 identity 无论 top ratio 多高均记 `conflicting_multi_target`；
10. target、NTC、intergenic、sgCtrl 中任意多个 identity 同时出现，均按上述冲突规则排除。

不得使用 guide 名的 `_` 前缀替代 library mapping。低信号或冲突只改变派生
`assignment_status`，不删除 raw evidence。

observation-level `assignment_status` 枚举：

```text
accepted_target
accepted_non_targeting
accepted_intergenic
accepted_sgctrl
unassigned_low_signal
ambiguous_tie
conflicting_multi_target
```

### 8.2 controls

- `non_targeting`、`intergenic`、`sgCtrl` 保留独立 `control_type`；
- primary control anchor 只使用 `accepted_non_targeting` bins；
- `accepted_intergenic` 与 `accepted_sgctrl` 只作为各自预注册 sensitivity
  analysis，绝不进入 primary NTC anchor；
- 可以为分析构建 aggregate NTC view，但不能覆盖 guide-level counts；
- NTC 不假设空间均匀，所有 anchor 均在 section 内局部构造。

### 8.3 禁止 label rescue

DBSCAN、KDE、邻近 guide 扩散或 image-neighbor voting：

- 不得生成 primary assignment；
- 不得填补 unassigned bins；
- 不得进入 ELRR supervision；
- 只可输出 secondary/descriptive score，并与 direct labels 分表保存。

### 8.4 support gates

assignment 完成后才计算：

```text
screen-level accepted bins per target >= 100
section-target accepted bins >= 40
section direct NTC bins >= 50
```

论文的 OT1 规则是在六切片 cohort 内按 perturbation 汇总。为避免通过 support
count 偷看 external/reserve，本研究采用更严格的 role-scoped 重写：

- development eligible target list 只用 development-train sections 统计
  `>=100 bins`；
- external/final 只评估该 frozen target list 中、且在当前 section 仍满足
  `section-target >=40` 的 targets；
- external、final 或 reserve counts 不得回填或改变 development target list；
- 每个允许访问的 data role 均输出 per-section counts。

这项规则明确标为 ELRR 的保守 protocol，不声称复现作者未公开实现的六切片
筛选代码。不满足阈值时为 episode-level `insufficient_target_support`；
不得降低 UMI、100-bin、40-bin 或 50-NTC 阈值来扩大样本。

## 9. RNA preprocessing 与 evaluation-expression 隔离

primary expression transform 严格为：

1. 保留 raw sparse counts；
2. 通过 frozen annotation 删除 mitochondrial、rRNA、housekeeping 与
   non-coding features；
3. 使用 sealed schema 中的 feature IDs 求所有 primary platforms 的交集；
4. 每个平台内要求 gene 在 development direct NTC aggregate 中至少有
   10 个 nonzero bins；
5. library-size normalize 到 10,000 后 `log1p`；
6. 对每个 development specimen 以 `ddof=0` 计算 direct-NTC
   within-specimen variance，再对 specimens 等权平均；
7. 按 specimen-balanced variance 降序、`feature_id` 升序稳定 tie-break，
   选择前 2,000 genes；
8. 全局 development NTC variance 为零的 genes 在排序前排除；
9. gene list 与顺序在 external gate 前冻结。

所有 variance 与 standard deviation 均固定使用 `ddof=0`。每个 section
使用该 section direct NTC 的 log1p mean 与 standard deviation 做 control
z-score。若某 gene 的 section NTC standard deviation `<1e-6`，使用：

```text
sigma_global[g] =
  sqrt(mean_over_development_specimens(
    Var_ddof0(NTC_log1p_expression[specimen, g])
  ))
```

全局 scale 在第 8 步已保证非零。

ELRR residual、training E-distance、evaluation E-distance 与 Delta PCC 全部在：

```text
frozen 2,000-gene
log1p
section-NTC-standardized space
```

计算。external、final 和 reserve target-evaluation expression 不进入 gene
selection、scale fit 或 feature filtering。若 common eligible genes 少于 500，
stage 为 `data_not_ready`，不临时改用另一个 gene panel。

## 10. W/C episode 与 split contract

### 10.1 W: within-section

一个 W unit：

```text
W = (screen_id, biological_specimen_id, section_id, target_gene)
```

direct target bins 按 frozen algorithm 分成：

```text
70% in-section target-reference support
30% in-section target-evaluation query
```

最低 40 个 accepted bins 对应至少 28 support 与 12 query。

random-stratified view 使用 2x2 normalized-coordinate quadrants；每个至少
4 bins 的 quadrant 按 70/30 分配，小于 4 bins 的 quadrants 合并为一个
`sparse` stratum。随机顺序由 `seed + unit_id` 的稳定 hash 决定。

spatial-block view 在四个 quadrants 中选择 target-bin 比例最接近 30% 的一个
作为 query；ties 由稳定 hash 决定。若没有 candidate 落在 20%–40%，该 target
缺少 required spatial view，整个 screen 的 required-unit gate 按第 16 节失败，
不改用看过表达后的 block。

### 10.2 C: matched cross-section

一个 C unit 是有向的：

```text
C = (support_section_id, query_section_id, target_gene)
```

要求 support/query 匹配：

```text
screen_id
tissue
timepoint
perturbed_cell_type
platform
capture_mode
```

且属于不同 `biological_specimen_id`。support 使用该 section 的全部 eligible
target references；query 使用完整 query section 的 target bins，且不提供任何
in-section target reference。

support directions 冻结为：

```text
development:
  所有满足 matching keys 的不同 development specimens 有向组合；
  continuous sections of one specimen 禁止形成 C unit

external:
  Lung M001 -> Lung M002
  OT1 Day7 rep1 -> OT1 Day7 rep2

final:
  Lung M001 -> Lung M003
```

Lung M994 不加入 final C support。无预注册 support direction 或少于三个共同
eligible targets 时，C view 为 screen-level `data_not_ready`，不得临时换
support section。

独立统计单位永远是 query `biological_specimen_id`。共享同一 query pool 的多个
development support directions 先在 `query specimen x target` 内等权平均，
不能当作独立样本。

### 10.3 split 与 fold 泄漏规则

- split algorithm 与 seed 在 role unseal 前 hash freeze；
- external/final 先只解封 guide/metadata，生成 target set 与 split IDs，
  随即写入并 hash `frozen_splits.csv`，之后才允许 reference-expression phase；
- random/spatial split 只由 observation ID、coordinates 与 direct target label
  生成，不能查看 target expression；
- C support directions 由本节固定映射生成；
- grouped validation held-out query specimen 的 target-evaluation expression
  不得出现在任何 training support 或 query；
- held-out specimen 的 W support 只可在 validation forward 中作为显式
  few-shot input，不参与 gradient；
- C validation 只允许 `training specimen -> held-out query specimen` 方向；
- 相同 raw target pool 的不同 task views必须 deduplicate。

## 11. NTC control anchor 与四维 context

对每个 direct target bin `i`，从同 section direct NTC bins 构造固定 control
anchor `a_i`。所有 coordinate calculation 使用 tissue-mask physical coordinate
system。

### 11.1 context 的精确定义

设 tissue-mask bounding box 为 `[x_min,x_max] x [y_min,y_max]`：

```text
x_norm = (x_um - x_min) / (x_max - x_min)
y_norm = (y_um - y_min) / (y_max - y_min)
diameter = sqrt((x_max-x_min)^2 + (y_max-y_min)^2)
```

任一 range 或 diameter 非正、非有限时，section 为
`coordinate_contract_failure`。

`boundary_distance_um` 使用 platform tissue-mask raster 在 physical-um
transform 下的 Euclidean distance transform；不得从 target labels 或 target
expression 推断 boundary。

令 `z_i=(x_um,y_um)`，`h` 为每个 direct NTC 到其第 16 个最近 NTC 的 physical
distance 中位数：

```text
density_i =
  (1 / (n_NTC * h^2))
  * sum_k exp(-||z_i-z_k||^2 / (2*h^2))
```

计算 NTC 自身 density 时 leave-one-out，并以 NTC leave-one-out densities 的
mean/std z-score 所有 bins。若 std `<1e-12`，density z-score 固定为零。

最终：

```text
c_i = [
  x_norm,
  y_norm,
  boundary_distance_um / diameter,
  NTC_density_zscore
]
```

### 11.2 anchor

`h_c` 为 direct NTC 在四维 context 中第 16-NN distance 的中位数。candidate
NTC 按 `(context_distance, stable_observation_id)` 升序，恰好选择前 16 个；
cutoff distance tie 不依赖输入行序：

```text
q_ik = exp(-||c_i-c_k||^2 / (2*h_c^2))
a_i  = sum_k q_ik * x_k^NTC / sum_k q_ik
```

`x_k^NTC` 是第 9 节冻结空间中的 expression。全部 anchor metadata 暴露最近
distance、maximum weight 与 effective support。

若 `h`、`h_c` 或权重和非有限或不大于零，episode 为
`insufficient_ntc_support`；不得跨 section 借 control 或注入任意 epsilon
bandwidth。anchor 是局部 baseline estimator，不是真实未扰动 counterfactual。

## 12. ELRR prediction contract

### 12.1 reference residual

对 support target-reference `j`：

```text
r_j = y_j^reference - a_j
```

`r_j` 是 observed reference residual value，不是网络生成 target。

### 12.2 12 维 pair features 与 orientation

query `i` 与 support `j`：

```text
phi(i,j) = concat(c_i, c_j, abs(c_i-c_j))
```

维度严格为 `4+4+4=12`。不输入 target-evaluation expression、residual value、
gene one-hot、slide lookup、target-specific embedding 或 rescue score。

D4 transform 围绕 normalized center `(0.5,0.5)`，只变换 context 的 x/y
两维，boundary 与 density 不变。

- W：同 section query/support 使用同一个 D4 transform；推理平均 8 组 logits；
- C：query 与 support sections 独立 orientation marginalization；推理平均
  `8 x 8 = 64` 组 logits。

共同 D4 不能解决两个 sections 的相对旋转，因此 C 禁止只用 8 组共同 transform。
该 marginalization 不等价于 registration，也不授权 anatomical alignment claim。

### 12.3 113 参数 scorer

```text
h_ij  = tanh(W1 * phi(i,j) + b1)
u_ij  = w2^T * h_ij
tau   = 0.05 + softplus(theta_tau)
pi_ij = softmax_j(u_ij / tau)
```

容量冻结：

```text
input dimension = 12
hidden width = 8
hidden layers = 1
activation = tanh
one global temperature
```

参数总数：

```text
12*8 + 8 + 8 + 1 = 113
```

softmax-support 维中的 output bias 会严格抵消，因此明确删除，不把无效参数计入
方法容量。`W1/w2` 使用 seed `20260721` 的 Xavier-uniform 初始化，`b1=0`；
`theta_tau` 初始化为使 `tau=1.0` 的唯一值。

不得加入 gene decoder、attention value projection、per-target bias、per-section
embedding、GNN、flow 或第二隐藏层。

### 12.4 prediction

```text
y_hat_i = a_i + sum_j pi_ij * r_j
```

结构性质：

- `pi_ij >=0` 且 `sum_j pi_ij=1`；
- predicted residual 位于 support residual convex hull；
- support 顺序置换不改变 prediction；
- 一个 support 时退化为该 residual；
- uniform logits 时严格退化为 shared residual mean；
- 网络只改变 retrieval weights，不生成 residual values。

这些性质不等价于 source pairwise geometry preservation。

## 13. Episodic training

### 13.1 loss

训练只优化 episode-level biased E-distance V-statistic：

```text
L_episode =
  2 * mean_{p in Y_hat, q in Y} ||p-q||_2
  - mean_{p,p' in Y_hat} ||p-p'||_2
  - mean_{q,q' in Y} ||q-q'||_2

L = mean_over_query-specimen-balanced_episodes(L_episode)
```

within-sample means 包含 diagonal zero-distance，分母为 `m*n`、`m*m`、`n*n`。
不得切换 unbiased U-statistic。无 pointwise MSE、correspondence、boundary、
individuality、geometry 或 Delta optimization loss。

### 13.2 deterministic reference dropout

每个训练 W episode 在 70% support 内以概率 0.20 dropout，至少保留 8 个。
对 observation IDs 计算 `seed+epoch+unit_id+observation_id` 稳定 hash；
Bernoulli 后不足 8 个时保留 hash 最小的 8 个。因此 support row permutation
不改变保留 ID 集合。C support 使用相同规则。

### 13.3 optimization 与 epoch selection

```text
framework = PyTorch CPU deterministic mode
dtype = float64
seed = 20260721
optimizer = Adam
learning_rate = 0.003
weight_decay = 0.0001
maximum_epochs = 400
early_stopping_patience = 40
gradient_clipping_norm = 1.0
```

使用 leave-one-`biological_specimen_id`-out folds。每个 fold：

1. held-out group 不参与 gradient、gene ranking 或 model fit；
2. W validation 可读取 held-out group 的冻结 70% support 作为 few-shot input；
3. C validation 只允许 training support specimen 指向 held-out query specimen；
4. 每 epoch 先在 unit/target 内平均，再对 held-out query specimen 等权；
5. 汇总所有 folds 得到唯一 validation-loss curve。

curve 的最小值在 `1e-12` 内并列时取最早 epoch；patience 只监控这条 curve。
随后从同一初始化 seed 在全部 development groups 重训恰好该 epoch 数并冻结
state-dict hash。没有 architecture、feature、learning-rate、dropout、
temperature 或 epoch 之外的 sweep。若少于三个独立 groups，stage 为
`data_not_ready`。

## 14. 强基线与 mechanism-null controls

所有方法使用相同 assignment、anchor、split、gene space 和 W/C support。

### 14.1 nonlearned baselines

1. `control_anchor_only`: `y_hat_i=a_i`；
2. `uniform_residual`: support residual 等权，即 shared shift；
3. `nearest_residual`: 四维 context 最近 reference；
4. `fixed_rbf_retrieval`: diagonal-scaled context RBF。

强 RBF 的开发集 grid 预先冻结：

```text
s_xy       in {0.5, 1.0, 2.0}
s_boundary in {0.5, 1.0, 2.0}
s_density  in {0.5, 1.0, 2.0}
bandwidth  in {0.5, 1.0, 2.0}
```

令 `dx,dy,db,dd` 分别为 query-reference 的 x、y、boundary 与 density
context differences。RBF 严格为：

```text
d2 = (dx/s_xy)^2 + (dy/s_xy)^2
     + (db/s_boundary)^2 + (dd/s_density)^2
weight_j = softmax_j(-d2_j / (2*bandwidth^2))
```

81 个组合只在 grouped development folds 上比较，以 specimen-balanced W
overall E-distance 最小者冻结。validation loss 在 `1e-12` 内并列时，按以下
tuple 升序形成完整 total order：

```text
(
  log2(s_xy)^2 + log2(s_boundary)^2 + log2(s_density)^2,
  -bandwidth,
  s_xy,
  s_boundary,
  s_density
)
```

随后从 `uniform_residual`、`nearest_residual`、冻结的 `fixed_rbf_retrieval`
中选择一个全局 best nonlearned baseline。选择量仍是 grouped-development
specimen-balanced W overall E-distance；`1e-12` 内 tie 固定优先序：

```text
uniform_residual
nearest_residual
fixed_rbf_retrieval
```

禁止按 target、screen、W/C 或 external/final 重新选择。`control_anchor_only`
永远报告，但不参与 reference-conditioned baseline 选择。

### 14.2 learned mechanism nulls

`ELRR_no_query` 使用与 ELRR 相同初始化、训练、113 个存储参数与 optimizer，
但输入强制为：

```text
phi_no_query(i,j) = concat(zeros(4), c_j, zeros(4))
```

它只能学习 reference prior，不能看到 query context 或 query-reference
difference。

`ELRR_query_permuted` 不重训模型。anchor、support residual、query truth 和
query observation ID 全部保持不变，只在 scorer 输入端按冻结 seed 在每个
query section 内置换 `c_i`。使用 20 个由
`study_seed + unit_id + permutation_index` 生成的稳定 permutations，报告其
E-distance 均值。

在 grouped development 中，用 W spatial-block 与 C 两个 screen-level
E-distance 的等权均值，从 `ELRR_no_query` 与 `ELRR_query_permuted` 中选择
数值更低者作为唯一 `best_mechanism_null`；`1e-12` 内 tie 选择
`ELRR_no_query`。该选择在 external/final 前冻结。

### 14.3 其他固定消融

- `ELRR_no_reference_dropout`；
- `ELRR_uniform_logits`，必须严格等价于 uniform residual；
- `ELRR_reference_context_permuted`：固定 query，置换 `c_j` 与 `r_j` 的对应；
- direct NTC 与 intergenic-control sensitivity，二者不合并。

如果 ELRR 只优于 nonlearned baseline、但不优于 `best_mechanism_null`，只能写
“learned reweighting improved the finite query distribution”，不得写
“query-conditioned similarity was learned”。

Perturb-map 旧方法只在 ELRR 完全冻结后作 exposed compatibility comparison，
不得进入任何 baseline 或 null 选择。

## 15. 指标与层级聚合合同

### 15.1 primary 与 transfer metrics

唯一优化和排序指标是 held-out E-distance，但 estimand 分开：

```text
W overall = within-section random-stratified E-distance
W spatial = within-section spatial-block E-distance
C transfer = matched cross-section E-distance
```

W overall 是 primary metric；W spatial 与 C 是 required gates。三者不得混成
一个 aggregate。

primary E-distance 使用第 9 节冻结的 NTC-standardized expression space 和第
13 节 biased V-statistic。

### 15.2 独立单位与等权顺序

所有 metric difference 定义为：

```text
ELRR - frozen comparator
```

聚合顺序固定为：

```text
view/support-direction replicate
-> section-target
-> query-specimen-target
-> query biological specimen
-> screen
```

每层等权。共享同一 query target pool 的不同 support directions、task views
或 D4 evaluations 必须在进入下一层前平均。bins、targets、连续 sections 和
重复 task views均不能伪装成独立动物。

external/final gate 只使用最终 query-specimen/screen scalar。由于 Lung final
只有一个 query specimen，不产生 biological confidence interval。允许按
target task 重采样的 interval 只能命名为：

```text
target-task resampling sensitivity interval
```

禁止 bootstrap bins，禁止把该 interval 解释为动物层不确定性。

### 15.3 required context/niche views

target-expression-free context strata 固定为 query bins 的
`boundary_distance` tertiles。每个 tertile 分别计算 E-distance，再对可用
tertiles 等权；任一 required target 的 tertile 少于 4 bins，则该 target 缺少
context-stratified view并触发第 16 节 screen-level readiness failure。

有可靠 provenance 的 biological niche strata也必须报告。若 niche label 的来源、
版本或 target-expression isolation 无法证明，标记 `niche_label_unavailable`；
不得用 rescue label 补齐，也不得作 niche-level claim。

### 15.4 Delta PCC tolerance gate

每个 W/C unit 定义：

```text
Delta_true = mean_query(y_i - a_i)
Delta_pred = mean_query(y_hat_i - a_i)
Delta_PCC  = Pearson(Delta_pred, Delta_true) across frozen genes
```

若任一 Delta vector 在 frozen genes 上方差为零、有效 genes 不足 3 或结果非有限，
该 required unit 为 `metric_undefined`，整个 screen 为 `data_not_ready`；
不得填 0、删掉该 unit 或只删不利 genes。

按第 15.2 节聚合后：

```text
Delta_PCC(ELRR) - Delta_PCC(best_nonlearned) >= -0.02
```

W 与 C 分开过 gate。`-0.02` 是从冻结 Mainline A 继承的 operational tolerance，
不是置信区间或统计显著性声明，因此只称 `tolerance gate`，不称正式统计
noninferiority。

### 15.5 diagnostics，不作 promotion gate

以下完整报告，但在当前 claim 下不强制非劣：

- boundary leakage；
- source individuality；
- pairwise geometry distortion；
- DE direction；
- maximum reference weight；
- effective number of references；
- NTC coverage 与 anchor distance；
- guide-level assignment confidence。

原因是 ELRR 的 estimand 是 held-out distribution，不假定 paired source cell；
boundary 又依赖待验证 annotation。若论文主动声称 boundary fidelity、identity
或 geometry preservation，对应指标必须升级为 gate；当前规范禁止这些主张。

## 16. Development、external 与 final 决策

### 16.1 required denominator 与 readiness

训练前必须同时满足：

- 所有 primary sections 通过 join、feature、mask 与 manifest contracts；
- 至少三个独立 development specimen groups；
- 至少八个 eligible W section-target episodes；
- 至少三个 targets 可形成 development C units；
- frozen common genes 至少 500；
- seal/access tests 通过。

external 的每个 screen 与 final 必须在 guide/metadata phase 冻结至少 3 个共同
eligible targets。每个 target 必须同时具有：

```text
W overall
W spatial
W context-stratified
C transfer
C context-stratified
Delta PCC
```

target ID denominator 在读取 query expression 前写入 split manifest。任一
required view 缺失、target 少于 3 或 metric undefined 时，整个 screen 为
`data_not_ready`；不得从均值分母移除后继续 gate。

### 16.2 internal development gate

进入 external 前必须同时满足：

- W overall: mean paired E-distance vs best nonlearned `<0`；
- W spatial: mean paired E-distance vs best nonlearned `<0`；
- C transfer: mean paired E-distance vs best nonlearned `<0`；
- W spatial: mean paired E-distance vs best mechanism null `<0`；
- C transfer: mean paired E-distance vs best mechanism null `<0`；
- W/C context-stratified mean difference vs best nonlearned 各 `<0`；
- W/C Delta PCC tolerance difference 各 `>=-0.02`；
- 全部 mutation、permutation、finite 与 access guards 通过。

内部结果只决定是否进入 external，不作为最终论文证据。

### 16.3 one-shot external gate

模型、epoch、代码、gene list、baseline/null selections、split algorithm 和
environment 全部冻结后，按第 17 节一次性解封：

```text
Lung 994 / M002
OT1 Day7 rep2
```

每个 screen 独立应用第 16.2 节全部 E-distance、mechanism、context 与 Delta
gates。两个 screens 都必须通过；不允许一个 screen 的大 gain 抵消另一个失败。

任一 screen `data_not_ready` 或 gate 失败：

```text
do_not_unseal_lung_993
```

不得根据 external 结果修改模型后再开第二次 external gate。

### 16.4 one-shot sealed final

external 两个 screens 全部通过后，才允许解封：

```text
Lung 993 / M003
```

final 对同一个 frozen baseline/null 应用第 16.2 节全部 gates。由于只有一个
Lung query specimen，允许的结论只是：

```text
在该预注册 prospective Lung screen 的预定义层级均值上通过 pilot gates
```

不得写“统计显著”“跨动物稳定”或“普适泛化”。同时报告每 target paired
effect、screen scalar 与 target-task resampling sensitivity interval。

final 失败后：

- 不修改或重跑 ELRR；
- 不重跑 Lung 993；
- Stage 1 runner 永久不解封 OT1 Day10 rep2；
- 阶段结论改为 benchmark/negative empirical result。

## 17. Leakage、freeze 与 role-specific transactions

### 17.1 staged unseal

external 与 final 每个 data role 都执行四阶段事务：

```text
Stage 0: hash-frozen code/model/config/split algorithm; raw values inaccessible
Stage 1: guide+metadata only -> assignment, target denominator, split IDs -> hash
Stage 2: NTC+support expression only -> anchors, residuals, predictions -> hash
Stage 3: query expression only -> score exactly once -> immutable decision
```

Stage 2 完成前 query expression 必须物理/逻辑不可访问。Stage 3 不能重算
prediction。C query section 在 Stage 2 只提供 NTC expression，不提供任何 target
expression。

### 17.2 mutation 与 order guards

将 query target expression 替换为极端值后，assignment/splits、anchors、
features、logits、weights、predictions、baseline/null predictions、fit metadata
与 selections 必须不变。

另外要求：

- support rows 置换不改变 prediction；
- duplicate observation key 直接失败；
- 删除 reference 后只允许相应 retrieval set 变化；
- mutation reference residual values 时 logits/weights 不变，但 prediction
  按 value contract 改变；
- query label 可选择正确 support target，query expression 不可见；
- identical raw pools只加载一次。

### 17.3 complete freeze manifest

Stage 0 hash：

```text
all ELRR source and runners
all dynamically imported evaluator/data/split code
environment lock and package versions
model state_dict
frozen config
gene list and feature order
library/section manifests
support-direction mapping
baseline and mechanism-null selections
source-file checksums
development artifacts
access policy
pre-freeze access log
seal_activation.json
```

Stage 1 授权读取 guide/metadata 后，把 exact target IDs 与 split IDs追加到
content-addressed role manifest 并再次 hash；Stage 2 只有该 hash 验证后才能开始。
这避免在 seal 前偷看 assignment，又消除 final split 的实现自由度。

### 17.4 transactional sentinel

external 与 final 分别使用绑定以下内容的 sentinel：

```text
study_id + data_role + transaction_stage + source checksums + model hash
```

第一次 raw-array access 前原子写 `started`。每个阶段成功后写 immutable artifact
hash 和 `completed`。崩溃后只能从已经落盘、hash-verified 的 stage artifact
恢复；若 raw 已读取但 artifact 未完成，该 role 记 `aborted_no_result`，不得再次
读取 raw arrays。

更换 output directory 不能绕过 sentinel。deterministic replay 只能读取已有
prediction/metric artifacts。reserve 使用独立 study ID 与权限，所有 Stage 1
runner 均无 reserve token。

## 18. Typed status 与 fail-closed handling

### 18.1 observation assignment status

```text
accepted_target
accepted_non_targeting
accepted_intergenic
accepted_sgctrl
unassigned_low_signal
ambiguous_tie
conflicting_multi_target
```

`ambiguous_tie` 优先于 `conflicting_multi_target`。

### 18.2 episode/stage status

```text
ok
manifest_incomplete
join_mismatch
duplicate_observation_key
feature_schema_mismatch
coordinate_contract_failure
insufficient_target_support
insufficient_ntc_support
metric_undefined
niche_label_unavailable
sealed_access_denied
hash_mismatch
nonfinite_output
insufficient_required_units
aborted_no_result
```

### 18.3 stage decision

```text
data_not_ready
advance_to_external_gate
do_not_unseal_lung_993
final_success
final_gate_failed
external_aborted_no_result
final_aborted_no_result
```

ingestion、assignment、episode readiness 和 transactions 全部 fail closed。
`external_aborted_no_result` 与 `final_aborted_no_result` 是 procedural outcomes，
不解释为算法失败，并强制 `access_action=keep_next_role_sealed`。不得 catch
异常后输出全零 prediction；不得降低阈值、删除不利 required units 或开启
reserve 来填充分母。每个排除对象保存原始计数与精确原因。

## 19. Future implementation boundaries

用户批准后，implementation plan 将把工作限制在：

```text
repro_runs/2026-07-21-spacseq-elrr/
```

建议模块边界：

```text
src/elrr/schema.py          manifest 与 typed contracts
src/elrr/io_visium.py       exact-barcode ingestion
src/elrr/io_stereo.py       CID/integer-coordinate ingestion
src/elrr/assignment.py      conservative guide state machine
src/elrr/preprocessing.py   frozen expression transform
src/elrr/episodes.py        W/C splits、directions 与 eligibility
src/elrr/anchors.py         local NTC anchors
src/elrr/features.py        4D context 与 12D pair features
src/elrr/model.py           113-parameter scorer
src/elrr/losses.py          episodic E-distance
src/elrr/baselines.py       strong nonlearned 与 mechanism nulls
src/elrr/evaluation.py      hierarchical aggregation
src/elrr/guards.py          mutation/access/hash contracts
src/elrr/artifacts.py       decision-oriented outputs
```

作者仓库、raw SPAC-seq tree、旧 frozen evidence 和旧 final artifacts 全部只读。

## 20. Required contract tests

### 20.1 ingestion/schema

- Visium barcode row permutation后仍精确对齐；
- missing/extra/duplicate barcode 按合同失败；
- Stereo coordinate 必须 int64 且 section-scoped；
- RNA/guide 各自归零会被测试捕获；
- matrix rows 与 observation IDs 置换不改变结果；
- feature type 不依赖 `sg` 前缀；
- Visium/Stereo nullable-union required fields 正确；
- 不适用字段为 null，而非伪造 ID/coordinate。

### 20.2 assignment

- raw UMI 1 被拒绝，raw UMI 2 被接受；
- 同 target 多 guides 正确求和并 accepted；
- `top_guide_count` 与 target-sum `top_count` 分开；
- top ratio 很高的 multi-target 仍拒绝；
- tie precedence、target+NTC、NTC+intergenic 状态明确；
- 单一 sgCtrl、sgCtrl+target、sgCtrl+NTC 状态正确；
- raw guide counts 不被 gene-level view 覆盖；
- KDE/DBSCAN 输出不能进入 primary label。

### 20.3 context/anchor

- bounding-box normalization、physical diameter 与 mask distance 可复现；
- 所有 expression/context variance 与 std 使用 `ddof=0`；
- NTC KDE 对 NTC 自身 leave-one-out；
- density zero-variance feature 精确变为零；
- NTC row permutation 与 16-NN cutoff tie 不改变 anchor IDs；
- 非有限/零 bandwidth fail closed；
- W D4 八组与 C 独立 D4 六十四组 deterministic；
- D4 只改变 x/y 且围绕 `(0.5,0.5)`。

### 20.4 model/loss

- trainable parameter count 严格为 113；
- 不存在 support-softmax 中抵消的 output bias；
- weights finite、非负、和为 1；
- support permutation invariance；
- one-reference 与 uniform-logit degeneracy 正确；
- predicted residual 位于 support convex hull；
- residual value mutation 不改变 logits/weights；
- `ELRR_no_query` 无法访问 `c_i` 或 differences；
- query permutation 只改变 scorer input；
- biased E-distance 与手算 fixture 相同且 gradient finite；
- dropout support ID set 对 row permutation 不变。

### 20.5 scientific protocol

- W/C estimands、support directions 与 aggregations 分开；
- LOGO fold held-out expression 不进入 training；
- continuous sections 不作为独立 specimen；
- exact global baseline/null selection 与 tie-break；
- 至少 3 targets 和 required-view denominator gate；
- Delta PCC 公式、零方差与层级聚合；
- evaluation-expression mutation guard；
- failed readiness 不会降低阈值或删 unit；
- development runner 无权读取 external/final/reserve；
- external runner 无权读取 final/reserve；
- final runner 无权读取 reserve；
- Stage 0–3 access ordering 与 transaction sentinel；
- full dependency hash 变化拒绝运行；
- 更换 output directory 不能绕过 one-shot sentinel。

## 21. Required artifacts

每个允许阶段输出：

```text
manifests/data_manifest.csv
manifests/library_manifest.csv
manifests/pre_freeze_access_log.json
manifests/seal_activation.json
manifests/freeze_manifest.json
manifests/support_directions.csv
qc/join_audit.csv
qc/assignment_audit.csv
qc/episode_readiness.csv
splits/frozen_splits.csv
model/model_state.pt
model/model_hash.json
metrics/unit_metrics.csv
metrics/hierarchical_summary.csv
metrics/mechanism_null_summary.csv
metrics/reference_diagnostics.csv
guards/mutation_guard.json
guards/access_guard.json
guards/transaction_sentinel.json
reports/decision_zh.md
```

`decision_zh.md` 只能给出第 18.3 节一个 stage decision，并列出：

- frozen target denominator；
- W/C 各 gate scalar；
- best nonlearned 与 best mechanism null；
- missing/undefined unit 原因；
- exact model/config/data hashes；
- reserve 保持 sealed 的状态；
- transaction abort 时的 `access_action=keep_next_role_sealed`。

“训练稳定”“loss 下降”或“某个 target E-distance 很好”均不等于方法成功。

## 22. 阶段性 AAAI 故事的成立条件

只有 external 两个 screens 与 Lung final 全部通过时，允许的阶段性故事是：

1. 既有 Perturb-map 失败为 value-constrained learned retrieval 提供动机，
   但本实验不证明高维生成方法普遍不可行；
2. ELRR 是旧 fixed-retrieval skeleton 的受控 113 参数 learned-kernel
   extension，而不是全新 attention 原理；
3. 在预注册的 prospective screens 上，W overall、W spatial 与 C transfer
   的预定义 screen means 均优于强 nonlearned baseline；
4. ELRR 同时优于 query-independent/permuted mechanism null，才支持
   “query-conditioned retrieval”；
5. W/C Delta PCC 层级均值下降均不超过 0.02；
6. assignment、独立单位、seal、mutation 与 transactions 全部可审计。

由于 final 只有一个 Lung specimen，正文必须称为 prospective pilot evidence，
不得使用“跨动物稳定改善”或 population-level significance。

如果普通 E-distance 通过但 mechanism gate 失败，只能报告 learned reweighting，
不能报告 query-conditioned matching。如果任一 external/final gate 失败，允许的
结论仅是：

```text
a leakage-audited benchmark and a negative result for this frozen ELRR
```

不得把它扩大为“所有 learned retrieval 均失败”，也不得用 reserve 或模型扩容
救援同一阶段。这使 ELRR 成为一次可证伪的 AI 探索，而不是继续在小 holdout 上
追逐可投稿数字。
