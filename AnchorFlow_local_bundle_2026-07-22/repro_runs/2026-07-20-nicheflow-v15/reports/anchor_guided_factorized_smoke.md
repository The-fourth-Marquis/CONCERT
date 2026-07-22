# Anchor-Guided Factorized Decoder Smoke

日期：2026-07-20

## Decoder contract

本轮把 correction 显式拆成两部分：

```text
base = source expression + shared global shift
correction = bounded context modulation toward target local anchor
prediction = base + modulation_strength * correction
```

global shift 是 reference-only context anchor 的平均位移，对所有 query
相同。local anchor residual 只作为 context-conditioned modulation，且由
`modulation_strength in [0, 1]` 硬限制。strength=0 时 source pairwise
geometry 严格不变；实现不再使用 identity loss。

## 首个真实任务

任务：`GSM5808054_patchclose_tumor_Jak2`  
Reference fraction：0.70  
Split：random-stratified  
Seed：20260711

Fixed 80/20：E-distance 52.9304，Delta PCC 0.9846，Boundary 0.4934，
Individuality 1.0000。

| Strength | E-distance | Delta PCC | Boundary | Individuality | Pairwise geometry error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 84.2182 | 0.8833 | 1.2768 | 1.0000 | 0.0000 |
| 0.30 | 54.9239 | 0.9579 | 0.9454 | 0.9998 | 0.2925 |
| 0.40 | 46.8744 | 0.9703 | 0.8407 | 0.9994 | 0.3895 |
| 0.50 | 40.1144 | 0.9790 | 0.7404 | 0.9986 | 0.4862 |
| 0.75 | 31.8111 | 0.9902 | 0.5193 | 0.9848 | 0.7245 |
| 0.77 | 31.8893 | 0.9907 | 0.5046 | 0.9807 | 0.7438 |
| 0.80 | 32.2802 | 0.9913 | 0.4835 | 0.9717 | 0.7736 |
| 1.00 | 45.0366 | 0.9931 | 0.3908 | 0.2856 | 0.9286 |

strength=0.77 相对 fixed 的差值为：

- E-distance：-21.0411；
- Delta PCC：+0.0061；
- Boundary：+0.0111，约 +2.26%；
- Individuality：-0.01925，约在 -0.02 门内。

因此它在这个首个真实 task 上同时通过当前四项机制门。所有预测均通过
evaluation-expression mutation guard，且 `uses_evaluation_target=False`。

## 解释

单纯的 OT global shift（前一轮 factorized flow）不能提供 target local
location，E-distance 为 87.38。把 local anchor 作为无界 residual 也会损伤
individuality。当前 decoder 把 local anchor 的作用限制为一个有界比例，形成
清晰的 target-location / source-geometry trade-off：strength 从 0 增加到
0.77 时 target distribution 被拉到正确位置，而 pairwise geometry 仍可控；
strength=1 则明显破坏 individuality。

## 决策与限制

当前结果是首个 task 的 mechanism smoke，不是多任务 promotion：strength 网格
仍需在 reference-only pseudo-validation 上预先选择，不能根据 evaluation rows
调到 0.77 后直接宣称泛化。下一步是实现该 calibration，再跑 6-task、至少
random-stratified 与 spatial-block 的小规模验证；只有四项门在该验证中同时
稳定通过，才考虑 37-task、多 seed。

原始数据：`results/anchor_guided_smoke/summary.json` 与
`results/anchor_guided_fine/summary.json`。
