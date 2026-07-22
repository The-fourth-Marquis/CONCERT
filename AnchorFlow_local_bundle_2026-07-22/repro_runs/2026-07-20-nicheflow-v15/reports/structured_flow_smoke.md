# Structured Source-Shift Flow Smoke

日期：2026-07-20

## 结构

本轮 decoder 采用：

    prediction
    = source expression
    + anchor-derived global perturbation shift
    + bounded centered context modulation

modulation=0 时 pairwise source geometry 严格保持。global shift 先由
reference-only context anchor 的平均位移估计，避免直接使用 evaluation target。

## 结果

任务：GSM5808054_patchclose_tumor_Jak2  
Reference fraction：0.70  
Split：random-stratified  
Seed：20260711

| 方法 | E-distance | Delta PCC | Boundary | Individuality |
| --- | ---: | ---: | ---: | ---: |
| Fixed 80/20 | 52.9304 | 0.9846 | 0.4934 | 1.0000 |
| Local anchor flow, horizon 0.25 | 38.8719 | 0.9923 | 0.4941 | 0.9255 |
| Structured source + global shift, horizon 0.25 | 81.8397 | 0.8833 | 1.2477 | 0.9989 |

structured decoder 成功保护了 individuality，但 global shift 无法替代 local anchor 的 target location 信息。local anchor flow 则相反，获得更好的 distribution matching，却扭曲 source pairwise geometry。

## 解释

当前任务同时需要：

1. target-reference 的 local spatial location；
2. source spot 的 intrinsic identity；
3. context-dependent perturbation correction。

只使用 shared global shift 会丢掉第 1 项；直接使用 local anchor 作为起点又会压缩第 2 项。bounded modulation 只能控制两者折中，不能凭空恢复未观测的 cell-level correspondence。

## 决策

当前仍不进入 6-task 或 37-task pilot：

    do_not_advance_structured_global_shift

已验证的可行方向是 local anchor 加 source residual，但还需要把 residual 定义成相对于 source-context anchor 的 identity-preserving correction，而不是直接在 target latent 上学习 endpoint。

下一步若继续，应实现：

    target local anchor
    + (source expression - source-context anchor)
    + bounded residual flow

并在 reference pseudo-validation 上选择 correction horizon。当前所有结果都只支持 Mainline A observed-perturbation relocation claim。
