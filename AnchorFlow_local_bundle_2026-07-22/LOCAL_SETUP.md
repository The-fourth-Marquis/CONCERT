# AnchorFlow 本地开发包

## 这是什么

这是 2026-07-22 服务器工作区中 AnchorFlow 当前开发状态的便携快照。
科学任务边界仍是：在明确报告 target-reference pool 访问的前提下，进行
observed-perturbation spatial relocation；不是 strict unseen/OOD perturbation
prediction，也不支持逐 spot 因果反事实声明。

源工作区提交：`bede3890b4c5c475916210d19587fc71b5d11993`。
本包包含当时工作树中的当前文件，因此不能只依靠该提交重新生成。

## 目录重点

- `repro_runs/2026-07-20-nicheflow-v15/`：AnchorFlow 主代码、测试、报告和已有结果。
- `repro_runs/2026-07-22-anchorflow-submission-smoke/`：冻结确认实验及结果。
- `repro_runs/2026-07-11-nicheflow-v2/src/`：AnchorFlow 调用的数据、切分、指标和基线代码。
- `repro_runs/2026-06-22-full-repro/task_manifest_mainline_a.csv`：已改为相对路径的便携 manifest。
- `repro_runs/2026-06-22-full-repro/task_manifest_server_original.csv`：未经修改的服务器原始 manifest。
- `repro_runs/2026-06-22-full-repro/mainline_a_report/`：冻结 Mainline A 总结、中文方法说明和结果索引。
- `datasets/`、`select_cells/` 和 `surrogate_select_cells/`：37 个 Mainline A 任务的运行输入。
- `docs/superpowers/`：当前 AnchorFlow 设计和 GPU submission-smoke 执行计划。

## 本地环境

推荐 Python 3.11。先在包根目录创建独立环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-anchorflow.txt
```

`requirements-anchorflow.txt` 使用宽松的 PyTorch 下限以兼容 CPU、Apple
Silicon 和不同 CUDA 版本。若使用 NVIDIA GPU，应按本地驱动和 CUDA 版本安装
对应的 PyTorch wheel；不要机械复用服务器的 `sm_120` 运行环境。

根目录的 `pyproject.toml`、`uv.lock` 和 `requirements.txt` 是服务器项目快照，
用于追溯原环境。它们比 AnchorFlow 的最小依赖更大，而且其中的旧 PyTorch
固定版本不适合所有新 GPU。

## 首次验证

检查下载完整性和路径便携性：

```bash
python verify_bundle.py
```

运行 AnchorFlow 聚焦测试：

```bash
python repro_runs/2026-07-20-nicheflow-v15/tests/test_anchor_flow.py
python repro_runs/2026-07-20-nicheflow-v15/tests/test_anchor_flow_device.py
python repro_runs/2026-07-20-nicheflow-v15/tests/test_anchor_flow_predict.py
```

运行 v1.5 全部 24 个测试文件：

```bash
python repro_runs/2026-07-20-nicheflow-v15/tests/run_all.py
```

## 开发 smoke

CPU：

```bash
python repro_runs/2026-07-20-nicheflow-v15/run_anchor_flow_smoke.py \
  --device cpu \
  --integration-horizon 0.25 \
  --output-dir repro_runs/2026-07-20-nicheflow-v15/results/local_anchorflow_smoke
```

本地 CUDA 的第一个可见设备：

```bash
python repro_runs/2026-07-20-nicheflow-v15/run_anchor_flow_smoke.py \
  --device cuda:0 \
  --integration-horizon 0.25 \
  --output-dir repro_runs/2026-07-20-nicheflow-v15/results/local_anchorflow_cuda_smoke
```

主要实现入口：

- `src/nicheflow_v15/anchor_flow.py`
- `src/nicheflow_v15/anchor_flow_predict.py`
- `run_anchor_flow_smoke.py`

以上三个路径均相对于 `repro_runs/2026-07-20-nicheflow-v15/`。

## 冻结结果边界

`2026-07-22-anchorflow-submission-smoke/results/` 是已经完成的冻结证据，当前
决定为 `insufficient_confirmation_for_performance_claim`。AnchorFlow 在四个视图
中均优于 initial anchor，但未在总体上优于 fixed 80/20 baseline。

submission runner 的 `config.json` 故意保留服务器专用的 Python 路径、物理 GPU 2、
`CUDA_VISIBLE_DEVICES=2` 和 `sm_120` 约束，并且已有结果目录会阻止覆盖。因此本地
继续开发应使用 v1.5 smoke 或复制出新的日期目录，不要修改或覆盖这份冻结结果。

