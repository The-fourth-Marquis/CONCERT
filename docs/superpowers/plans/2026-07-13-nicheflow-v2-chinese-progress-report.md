# NicheFlow-v2 中文项目进展说明 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份面向技术研究人员、可独立阅读且所有数字可追溯的 NicheFlow-v2 中文项目进展说明。

**Architecture:** 正文按“问题—证据—判断—下一步”组织，只引用已经通过完整 verifier 的 CSV 和正式报告。新增一个轻量 Python 验收脚本检查章节、关键结论、关键数字与 claim boundary，并在实验 README 中增加入口链接。

**Tech Stack:** Markdown、Python 3.10、pandas、RTK-prefixed shell commands。

---

## File Map

- Create: `repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_当前项目进展中文说明.md`：正式中文说明。
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/verify_chinese_progress_report.py`：检查结构、关键数字与 CSV 一致性。
- Modify: `repro_runs/2026-07-11-nicheflow-v2/README.md`：增加中文说明入口。

### Task 1: 建立可验证的中文说明合同

**Files:**
- Create: `repro_runs/2026-07-11-nicheflow-v2/tests/verify_chinese_progress_report.py`
- Create: `repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_当前项目进展中文说明.md`

- [ ] **Step 1: 编写失败的验收脚本**

脚本读取目标 Markdown，并检查以下标题：

```python
required_headings = [
    "# NicheFlow-v2 当前项目进展中文说明",
    "## 1. 一页式结论",
    "## 2. 项目究竟在预测什么",
    "## 3. 数据与评价单元",
    "## 4. 为什么原来的评价不够",
    "## 5. 指标如何理解",
    "## 6. 从 NicheFlow-v1 到 v2",
    "## 7. 主要实验结果",
    "## 8. 为什么 gating 没有成功",
    "## 9. 成功门与当前决策",
    "## 10. 当前进度与下一步",
    "## 11. 术语、公式符号与文件索引",
]
```

脚本还必须读取以下 CSV：

```python
existing = pd.read_csv(ROOT / "results/existing_method_reassessment/metrics.csv")
v2 = pd.read_csv(ROOT / "results/residual_gating/metrics.csv")
gates = pd.read_csv(ROOT / "results/residual_gating/gate_summary.csv")
spatialprop = pd.read_csv(ROOT / "results/spatialprop_reference_only/metrics.csv")
```

从 CSV 计算并检查文档包含下列四位小数值：

```python
expected_values = {
    "legacy_v1_calibrated": existing.loc[
        existing["method"] == "legacy_v1_fixed_80_20_calibrated_l3",
        "heldout_edistance",
    ].mean(),
    "fixed_80_20": v2.loc[
        v2["method"] == "fixed_80_20", "heldout_edistance"
    ].mean(),
    "gated": v2.loc[
        v2["method"] == "nicheflow_v2_gated", "heldout_edistance"
    ].mean(),
    "spatialprop": spatialprop["heldout_edistance"].mean(),
}
for value in expected_values.values():
    assert f"{value:.4f}" in text
```

最后检查文档包含 `Mainline A`、`不是严格 unseen/OOD 扰动预测`、`do_not_advance_to_graph_flow`，以及 G1–G7 每个 gate ID。

- [ ] **Step 2: 运行验收脚本并确认失败**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/verify_chinese_progress_report.py
```

Expected: FAIL，原因是中文说明文档尚不存在。

- [ ] **Step 3: 创建只含标题的文档骨架**

创建目标 Markdown，写入 Step 1 中的全部标题，不填入占位语句。

- [ ] **Step 4: 再次运行验收脚本**

Expected: FAIL，原因是缺少关键数字、claim boundary 或 gate 内容。

### Task 2: 撰写技术研究人员可读的完整正文

**Files:**
- Modify: `repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_当前项目进展中文说明.md`

- [ ] **Step 1: 完成任务与协议说明**

正文必须明确：

- 输入是 selected source/query spots、eligible control pool 和 observed target perturbation pool；
- 输出是 source spots 在已观测 target perturbation 状态下的表达分布估计；
- Mainline A 是 observed-perturbation spatial relocation；
- reference targets 可用于预测或明确标记的 calibration，evaluation targets 只用于指标；
- 37 tasks × 3 fractions × 5 seeds × 2 split types = 1,110 个评价单元；
- 这不是严格 unseen/OOD 扰动预测，也不是 spot-level causal counterfactual。

- [ ] **Step 2: 解释指标而非只列名称**

每个指标按“定义、方向、回答的问题、不能证明什么”解释：

- held-out/full-pool/niche-stratified E-distance；
- delta PCC、Spearman、MAE、RMSE；
- DE direction accuracy 和 opposite-direction fraction；
- boundary leakage、source individuality、distance-response decay、spatial-map PCC。

- [ ] **Step 3: 解释 v1 与 v2 方法链条**

必须逐项解释：

```text
legacy prediction
= 0.80 × target_pool_anchor
+ 0.20 × legacy_residual_prediction

delta_context(i)
= Σ_j π(j | i) × (x_target(j) - x_control_anchor(j))

residual_prediction(i)
= x_source(i) + delta_context(i)

P_v2(i)
= (1 - g_i) × P_anchor(i) + g_i × P_residual(i)
```

说明旧 residual 将“纯扰动变化量”和“完整预测表达”混在一起；v2 明确拆分两者，并使用 perturbation-invariant matching。解释 calibration 是 reference-only moment alignment，不等同于机制学习。

- [ ] **Step 4: 写入结果与逐表解释**

正文至少包含：

- existing-method held-out E-distance 表；
- v2 residual/gating 多指标表；
- reference fraction × split type 敏感性表；
- G1–G7 表。

每张表后分别写“可以得出的结论”和“不能得出的结论”。必须解释：

- calibrated legacy v1 为 55.3383，但其优势高度依赖 calibration；
- explicit residual 将 residual-only 从 151.9095 改善到 69.4766；
- fixed 80/20 为 66.8581，优于 gated 的 85.9374；
- gate mean 0.051562、std 0.000948，说明 gate 贴近下界；
- SpatialProp reference-only 为 146.8739；
- 0.50 spatial-block 的 mean delta 为 +10.9391；
- G1 的 CI 跨 0，G5 的 0.50 fraction 不满足要求。

- [ ] **Step 5: 写清阶段性决策**

明确 `do_not_advance_to_graph_flow` 的含义：

- 当前轻量模块尚未跨 task 显著且 reduced-reference 不稳健；
- 暂不增加复杂模型，避免用模型容量掩盖评价与 matching 问题；
- 这不是断言 graph/flow 永远无效；
- 重新通过 G1–G7 后可以恢复 graph/flow 研究。

### Task 3: 增加入口并完成验收

**Files:**
- Modify: `repro_runs/2026-07-11-nicheflow-v2/README.md`
- Test: `repro_runs/2026-07-11-nicheflow-v2/tests/verify_chinese_progress_report.py`

- [ ] **Step 1: 在 README 增加链接**

在 Main artifacts 列表中增加：

```markdown
- Detailed Chinese progress guide: `reports/NicheFlow_v2_当前项目进展中文说明.md`
```

- [ ] **Step 2: 运行中文说明验收脚本**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/verify_chinese_progress_report.py
```

Expected: `ok`。

- [ ] **Step 3: 运行既有完整输出 verifier**

Run:

```bash
rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py --mode complete
```

Expected: `ok`。

- [ ] **Step 4: 人工可读性审计**

检查：

```bash
rtk rg -n "T[B]D|TO[D]O|待补充|见上文|显然" repro_runs/2026-07-11-nicheflow-v2/reports/NicheFlow_v2_当前项目进展中文说明.md
```

Expected: 无匹配。随后逐段确认首次出现的 held-out、calibration、residual、gate、OOD 均有中文解释。

- [ ] **Step 5: 检查并提交**

Run:

```bash
rtk git diff --check
rtk git status --short repro_runs/2026-07-11-nicheflow-v2
```

只暂存中文说明、README 和验收脚本，提交信息：

```bash
rtk git commit -m "docs: explain current NicheFlow v2 progress in Chinese"
```
