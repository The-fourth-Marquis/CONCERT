# Anchor-Guided Reference-Calibrated Pilot

日期：2026-07-20

## Protocol

每个 task/split 先把已报告的 target reference pool 再拆为 70% pseudo-fit 与
30% pseudo-validation。modulation strength 只使用 pseudo-validation 的
E-distance、delta PCC、boundary 和 individuality 选择；真实 evaluation rows
在 strength 固定后才用于一次最终评估。

选择规则为：在 pseudo-validation 同时通过四项相对 fixed 80/20 门的候选中，
取最大的安全 strength，以优先修复 boundary；若没有安全候选，则回退到
strength=0，严格保护 source pairwise geometry。evaluation-expression mutation
guard 通过。

## 首个 task

reference-only calibration 选择 strength=0.76：

| E-distance | Delta PCC | Boundary | Individuality |
| ---: | ---: | ---: | ---: |
| 31.8327 | 0.9905 | 0.5119 | 0.9829 |

相对 fixed 80/20：E-distance -21.0977，Delta PCC +0.0059，Boundary
+3.74%，Individuality -0.0170。四项机制门在该 task 的 random-stratified
split 上同时通过。

## Six-task pilot

运行 6 个代表性 task，每个 task 同时包含 random-stratified 与 spatial-block，
共 12 个 unit。全部 unit 可执行，但整体结果未通过 promotion：

- mean E-distance difference vs fixed：+17.3936；
- mean Delta PCC difference：-0.0345；
- mean boundary relative difference：+103.62%；
- mean individuality difference：+0.0027；
- all distribution wins：false；
- all delta gates：false；
- all boundary gates：false；
- all individuality gates：true。

只有 4/12 unit 的 reference pseudo-validation 找到安全非零 strength。其余
unit 回退到 strength=0，individuality 得到保护，但 global shift 单独无法定位
target distribution，尤其是 border、cross-slide、periphery 和部分
spatial-block 条件。

## 结论

本次结构改动证明了两个结论：

1. global shift + bounded local-anchor modulation 可以在部分 task 上同时兼顾
   target location 与 source individuality；这比 unrestricted flow correction
   更可控。
2. 单一 modulation family 不能跨 task 泛化。失败不再是 identity loss，而是
   shared global shift 对不同 task type 的 target relocation 估计不足，且
   pseudo-validation 只在少数 task 找到安全非零候选。

因此不进入 37-task 或多 seed。下一方法需要让 global component 至少按 niche
或 spatial block 分组，同时保持每组内共享 shift；context modulation 仍需硬
限制，并继续使用 reference-only calibration。

结果：

- `results/anchor_guided_calibrated_smoke_final/summary.json`
- `results/anchor_guided_calibrated_pilot_final/metrics.csv`
- `results/anchor_guided_calibrated_pilot_final/summary.json`
