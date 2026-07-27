#!/usr/bin/env python3
"""
core/sop.py — 运行日志（无跨品类统一 Phase 表）

各品类数据与路径不同，流程见 AGENTS.md「分品类流程」。
本模块只提供 write_run_log / 可选短横幅，不再打印 Phase0–7 全表。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

PROJECT_LOG = Path(__file__).resolve().parent.parent.parent / "项目记录.md"

# 脚本阶段短名 → 中文说明（仅日志标题，不是强制 SOP）
STAGE_LABELS = {
    "quality": "初筛",
    "normalize": "规范化（遗留）",
    "clean": "规则清洗",
    "sample": "抽样",
    "analyze": "污染分析",
    "dedup": "跨批去重",
    "qc_text": "文本 QC",
    "qc_vision": "视觉 QC",
    "qc_vision_thumb": "缩略图视觉 QC",
    "qc_vision_sb": "Storyboard 视觉 QC",
    "qc_two_person": "双人对话视觉 QC",
    "resolution": "分辨率抓取",
    "yt_definition": "清晰度 hd/sd",
    "yt_definition_filter": "清晰度过滤 ≥720",
    "pipeline": "薄编排",
    # 兼容旧调用里的数字
    0: "规范化（遗留）",
    1: "初筛",
    2: "抽样",
    3: "污染分析",
    5: "规则清洗",
    9: "跨批去重",
    205: "Storyboard 视觉 QC",
}


def load_sop() -> str:
    """兼容旧 import；统一 SOP 文件已弃用。"""
    return ""


def print_banner(stage: str | int | None = None, category: str | None = None) -> None:
    """短横幅：不打印跨品类 Phase 全表。"""
    label = STAGE_LABELS.get(stage, str(stage)) if stage is not None else "pipeline"
    cat = f"  category={category}" if category else ""
    print()
    print("=" * 62)
    print(f"  {label}{cat}")
    print("  （分品类流程见 AGENTS.md，无统一 Phase0–7）")
    print("=" * 62)
    print()


def write_run_log(
    stage: str | int,
    input_path: str,
    output_dir: str,
    stats: dict | None = None,
    command: str = "",
    category: str | None = None,
) -> None:
    """追加一条运行记录到 项目记录.md，并写 output_dir/run_log.md。"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    label = STAGE_LABELS.get(stage, str(stage))

    if not PROJECT_LOG.exists():
        PROJECT_LOG.write_text(
            "# 项目记录\n\n"
            "> YouTube 元数据清洗 — 自动记录每次管道运行\n"
            "> 各品类流程不同，见 AGENTS.md\n\n"
            "---\n\n",
            encoding="utf-8",
        )

    lines = [
        "",
        f"## {label}",
        "",
        f"**时间:** {ts}",
        f"**阶段:** {label} (`{stage}`)",
    ]
    if category:
        lines.append(f"**品类:** `{category}`")
    lines.append(f"**输入:** `{input_path}`")
    lines.append(f"**输出:** `{output_dir}/`")
    if command:
        lines.append(f"**命令:** `{command}`")

    if stats:
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        for k, v in stats.items():
            if isinstance(v, int):
                lines.append(f"| {k} | {v:,} |")
            elif isinstance(v, float):
                lines.append(f"| {k} | {v:.1f} |")
            else:
                lines.append(f"| {k} | {v} |")
    lines.append("")

    with open(PROJECT_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir.rstrip("/"), "run_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Run Log — {label}\n\n")
        f.write(f"**Time:** {ts}\n")
        f.write(f"**Stage:** `{stage}`\n")
        if category:
            f.write(f"**Category:** `{category}`\n")
        f.write(f"**Input:** `{input_path}`\n")
        f.write(f"**Command:** `{command}`\n\n")
        if stats:
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in stats.items():
                if isinstance(v, int):
                    f.write(f"| {k} | {v:,} |\n")
                elif isinstance(v, float):
                    f.write(f"| {k} | {v:.1f} |\n")
                else:
                    f.write(f"| {k} | {v} |\n")

    print(f"[LOG] {PROJECT_LOG}")
    print(f"[LOG] {log_path}")
