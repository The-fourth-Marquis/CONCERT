# NicheFlow-v2 无泄漏评价与轻量模块报告

日期：2026-07-11

## 结论

在冻结的 37 个 Mainline A 任务上，explicit residual 明显修复了旧 residual branch，但 reliability gate 几乎塌缩到固定下界，未形成有效的逐 spot 自适应。正式成功门为 G1、G5 失败，G2、G3、G4、G6、G7 通过。

最终决策：do_not_advance_to_graph_flow。

当前证据不支持增加 Graph Transformer 或 latent flow；应先解决跨 task 显著性、0.50 reference fraction 的 spatial-block 退化和 gate 动态范围不足。

## 协议与 claim boundary

本报告只讨论 Mainline A：已观测 perturbation 的空间重定位，并允许明确报告的 target-reference pool 访问。它不是从未见 perturbation 的严格 unseen/OOD 预测，也不把 group-level delta、DE 或空间诊断解释为 spot-level 因果反事实。

每个 observed target pool 拆成互斥的 reference 和 evaluation 子集。使用 0.25、0.50、0.70 三个 reference fractions，5 个 seeds，random-stratified 与 spatial-block 两种 split，共 1,110 个 task-split-seed 单元。evaluation expression 只进入 metric functions；所有可重建方法均通过 evaluation-expression mutation guard。旧 2026-06-22 目录保持冻结只读。

## Full-pool 与 held-out

full-pool E-distance 在主要方法上系统性低于 held-out E-distance，因为 full pool 同时包含 reference rows，属于兼容性或 transductive upper-reference 指标。它适合与旧结果对齐，但不适合作为 v2 主选择门。

| Existing method | Access | Held-out | Full-pool | Niche-stratified |
| --- | --- | ---: | ---: | ---: |
| legacy v1 fixed 80/20 calibrated | L3 | 55.3383 | 35.3975 | 57.4979 |
| adaptive calibrated anchor | L2 | 71.3564 | 50.3612 | 72.9853 |
| legacy v1 fixed 80/20 | L3 | 72.9115 | 53.1820 | 75.8099 |
| Gaussian anchor | L1 | 92.9950 | 71.8525 | 93.9899 |
| Cauchy anchor | L1 | 94.4492 | 73.3223 | 95.3460 |
| source identity | L0 | 110.5040 | 99.5504 | 127.7295 |
| SpatialProp reference-only retrain | L2 | 146.8739 | 127.5305 | 151.5784 |
| legacy NicheFlow-lite residual | L3 | 151.9095 | 137.3586 | 169.8000 |

新增 niche-stratified E-distance、delta/DE、boundary、individuality、distance-response、cross-niche drop 和 registered spatial-map correlation 后，结论是 E-distance 仍必要，但单独使用会掩盖 perturbation direction、source individuality 与局部空间行为。

## Existing-method held-out 重评

全量重评包含 14,430 个 metric rows 和 15,540 个 coverage rows。Access ladder 已显式区分 L0 无 target reference、L1 未校准 anchor、L2 校准 anchor、L3 anchor 加 residual、L4 gated anchor/residual。

校准 legacy v1 对独立 L2 校准 anchor：

- wins：1,101/1,110；
- paired mean held-out E-distance delta：-16.0182；
- task-clustered 95% CI：[-18.2811, -13.9934]。

未校准 legacy v1 对最强未校准 A2.1 reference-only baseline：

- wins：587/1,110；
- paired mean delta：+4.3425。

因此 held-out 协议保留了“校准 v1 很强”的结论，但机制解释改变了：优势主要依赖 reference-only moment calibration，不能归因于旧 residual branch 本身。

固定 CONCERT outputs 因无法排除 evaluation-target access，被标记为 not_eligible_for_heldout_primary。SpatialProp-style reference-only retraining 已完成 1,110 个配置并标记为 approximate：mean held-out E-distance 146.8739、delta PCC 0.6666、DE direction 0.8954、boundary leakage 1.6402、individuality 0.0761。它明显弱于主要 anchor 与 v2 候选。作为 A2.2 comparator，它不改变以 A2.1 access-matched baseline 定义的 promotion gates。

## Residual 与 gating

| Method | Held-out | Niche-stratified | Delta PCC | DE direction | Boundary | Individuality | Spatial-map PCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed 80/20 calibrated | 54.1277 | 56.2992 | 0.8933 | 0.9685 | 0.6944 | 0.6625 | 0.6109 |
| gated calibrated | 63.9482 | 65.4745 | 0.8909 | 0.9678 | 0.6482 | 0.4978 | 0.6335 |
| fixed 80/20 | 66.8581 | 69.7634 | 0.8848 | 0.9675 | 0.6458 | 0.9211 | 0.6031 |
| residual prediction only | 69.4766 | 87.3574 | 0.8198 | 0.9611 | 1.3844 | 0.9996 | 0.3336 |
| gated | 85.9374 | 87.0608 | 0.8832 | 0.9663 | 0.5825 | 0.7222 | 0.6477 |
| anchor only | 93.8489 | 94.8701 | 0.8817 | 0.9660 | 0.5777 | 0.0653 | 0.6505 |

Explicit invariant residual 将 residual-only held-out E-distance 从 legacy residual 的 151.9095 降至 69.4766，并保留 source individuality 0.9996。但它的 niche-stratified E-distance 为 87.3574、boundary leakage 为 1.3844、spatial-map PCC 为 0.3336，说明局部分层与空间对齐仍不足。

fixed 80/20 比未校准 legacy v1 的均值低 6.0534，说明 explicit residual contract 和 perturbation-invariant matching 是有效改进。

## Gate collapse 与 calibration dependency

未校准 gated 模型的 gate mean 为 0.051562，mean within-run std 为 0.000948，mean minimum 为 0.050110，mean maximum 为 0.053932。gate 基本贴住 0.05 下界，接近 anchor-only 的弱 residual 混合；因此 gated held-out E-distance 85.9374，明显差于 fixed 80/20 的 66.8581。

Calibration 对 held-out E-distance 的平均作用仍很大：

- fixed 80/20：-12.7304；
- gated：-21.9893。

校准版本给出最低绝对 E-distance，但 promotion candidate 选未校准 fixed 80/20，以检验增益是否独立于 moment alignment。

## Reduced-reference 与 split sensitivity

| Fraction | Split | Mean delta vs uncalibrated A2.1 | Win rate |
| ---: | --- | ---: | ---: |
| 0.25 | random-stratified | -9.0637 | 0.7459 |
| 0.25 | spatial-block | -0.9476 | 0.6486 |
| 0.50 | random-stratified | -6.3413 | 0.7622 |
| 0.50 | spatial-block | +10.9391 | 0.5135 |
| 0.70 | random-stratified | -5.6430 | 0.7351 |
| 0.70 | spatial-block | +0.7916 | 0.6432 |

0.50 spatial-block 是主要失败组合，并使 0.50 fraction 综合 mean delta 变为 +2.2989。0.25 fraction 综合 mean delta 为 -5.0057，但不足以弥补 mandatory 0.50 gate 失败。

## 正式成功门

| Gate | Result | Evidence |
| --- | --- | --- |
| G1 held-out superiority with clustered CI | FAIL | 749/1,110 wins；mean -1.7108；95% CI [-9.0668, 6.0859] |
| G2 delta noninferiority | PASS | PCC -0.0102；Spearman +0.0066；MAE -9.97%；RMSE -3.27% |
| G3 DE direction noninferiority | PASS | direction +0.2675；opposite direction +0.0163 |
| G4 boundary 与 individuality 相对 v1 非劣 | PASS | boundary -6.00%；individuality +0.0077 |
| G5 reduced-reference support | FAIL | fraction 0.50 mean +2.2989；fraction 0.25 mean -5.0057 |
| G6 calibration-free gain | PASS | mean -1.7108；win rate 0.6748 |
| G7 coverage 与 variability | PASS | 37 tasks、5 seeds、2 split types |

G1 的均值与 win rate 有利，但 task-clustered CI 跨 0；不能描述为跨 task 显著优势。G5 明确失败，不能用叙事覆盖 quantitative gate。

## 空间诊断限制

registered spatial-map PCC 只在坐标可比的 same-slide 单元计算：5,580/6,660 个 v2 rows 有定义；1,080 个 cross-slide rows 标记为 unregistered_cross_slide。它只是最近邻 registered-map proxy，不是因果空间机制证明。

cross-niche delta-PCC drop 对 fixed 80/20 为 -0.0146，对 residual-only 为 -0.0692。该诊断受小型 task composition 影响，不作为 promotion gate。

## 完成范围与下一步

本轮已完成无泄漏 evaluator、新指标、L0-L4 access labeling、explicit residual、gating ablation、SpatialProp reference-only expensive comparator，以及 G1-G7 决策。SpatialProp 预测按 task/fraction/seed/split 缓存，训练只使用 reference target expression；缓存二进制不作为报告证据提交，汇总 CSV 保留完整 provenance。

下一步：

1. 修正 gate 特征标定，避免下界塌缩；
2. 分析 0.50 spatial-block 的 reference coverage 与 matching failure；
3. 只有重新通过 G1-G7 后才进入 graph/flow。

## 最终决策

do_not_advance_to_graph_flow
