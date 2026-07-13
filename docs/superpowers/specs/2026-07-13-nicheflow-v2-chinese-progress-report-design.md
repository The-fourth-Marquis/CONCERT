# NicheFlow-v2 中文项目进展说明文档设计

日期：2026-07-13

## 1. 目标与读者

为技术研究人员编写一份可独立阅读的中文进展说明。读者可以不了解此前的 NicheFlow 对话，但具备机器学习、生物信息学或空间转录组的基础背景。

文档不替代现有复现实验报告，而是解释这些实验为什么进行、各指标回答什么问题、数字如何支持当前决策。

## 2. 核心写作原则

1. 先给问题和结论，再给公式和数字。
2. 首次出现的英文术语必须给出中文解释；保留必要的代码方法名以便检索结果文件。
3. 每个指标都说明定义、优化方向、直观含义和局限。
4. 每张结果表后必须写出可得出的结论和不能得出的结论。
5. 明确区分未校准机制改善与 calibration 带来的分布对齐改善。
6. 始终保留 Mainline A 的 claim boundary：已观测扰动的空间重定位，不是严格 unseen/OOD 扰动预测。

## 3. 文档结构

采用“问题—证据—判断—下一步”结构：

1. 一页式执行摘要；
2. 项目任务、输入输出和不主张的能力；
3. 数据、37 个任务和 1,110 个评价单元如何形成；
4. full-pool 评价的缺陷与 held-out 协议；
5. E-distance、delta、DE 和空间指标的通俗技术解释；
6. NicheFlow-v1 的组成、原公式及已发现的问题；
7. v2 explicit residual、perturbation-invariant matching 和 gating 的设计；
8. existing methods、SpatialProp 与 v2 的结果比较；
9. calibration、reference fraction 和 spatial-block 敏感性；
10. G1–G7 的逐项解释；
11. 当前已完成、尚未解决和下一阶段优先级；
12. 术语表、公式符号表和结果文件索引。

## 4. 事实来源

数字只使用以下已验证工件：

- `repro_runs/2026-07-11-nicheflow-v2/results/existing_method_reassessment/metrics.csv`
- `repro_runs/2026-07-11-nicheflow-v2/results/residual_gating/metrics.csv`
- `repro_runs/2026-07-11-nicheflow-v2/results/residual_gating/gate_summary.csv`
- `repro_runs/2026-07-11-nicheflow-v2/results/spatialprop_reference_only/metrics.csv`
- `repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_report.md`

旧 `2026-06-22-full-repro` 目录只用于解释历史上下文，不产生新写入。

## 5. 输出与验收标准

正式文档保存为：

`repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_当前项目进展中文说明.md`

同时在实验根目录 README 中增加链接。验收要求：

- 所有关键数字与 CSV 一致；
- 能在不阅读旧报告的情况下理解项目目标、方法和当前决定；
- 解释固定 80/20、校准、gate collapse、G1/G5 失败的因果链条；
- 明确 `do_not_advance_to_graph_flow` 是阶段性工程决定，而非否定 graph/flow 的长期价值；
- 不包含占位文本或未解释缩写。
