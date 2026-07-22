#!/usr/bin/env python3
"""Verify the readable Chinese NicheFlow-v2 progress guide."""

from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "NicheFlow_v2_当前项目进展中文说明.md"

REQUIRED_HEADINGS = [
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


def verify() -> None:
    if not REPORT.exists():
        raise AssertionError(f"Missing Chinese progress report: {REPORT}")
    text = REPORT.read_text()
    if chr(9) in text:
        raise AssertionError("Report contains tab-corrupted formulas")
    required_formulas = ["P_fixed =", "Delta_context(i)", "P_residual(i)", "P_v2(i)"]
    missing_formulas = [item for item in required_formulas if item not in text]
    if missing_formulas:
        raise AssertionError(f"Missing readable formulas: {missing_formulas}")
    missing_headings = [heading for heading in REQUIRED_HEADINGS if heading not in text]
    if missing_headings:
        raise AssertionError(f"Missing required headings: {missing_headings}")

    existing = pd.read_csv(ROOT / "results/existing_method_reassessment/metrics.csv")
    v2 = pd.read_csv(ROOT / "results/residual_gating/metrics.csv")
    gates = pd.read_csv(ROOT / "results/residual_gating/gate_summary.csv")
    spatialprop = pd.read_csv(ROOT / "results/spatialprop_reference_only/metrics.csv")
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
    missing_values = {
        name: f"{value:.4f}"
        for name, value in expected_values.items()
        if f"{value:.4f}" not in text
    }
    if missing_values:
        raise AssertionError(f"Missing CSV-derived values: {missing_values}")

    required_statements = [
        "Mainline A",
        "不是严格 unseen/OOD 扰动预测",
        "do_not_advance_to_graph_flow",
    ]
    missing_statements = [item for item in required_statements if item not in text]
    if missing_statements:
        raise AssertionError(f"Missing claim or decision statements: {missing_statements}")
    missing_gates = [gate for gate in gates["gate_id"].tolist() if gate not in text]
    if missing_gates:
        raise AssertionError(f"Missing gate explanations: {missing_gates}")


if __name__ == "__main__":
    verify()
    print("ok")
