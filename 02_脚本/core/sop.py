#!/usr/bin/env python3
"""
core/sop.py -- SOP 加载 + 运行日志

每个 phase 脚本调用 write_run_log() 自动追加到 项目记录.md。
"""

import os, time
from pathlib import Path

SOP_PATH = Path(__file__).resolve().parent.parent.parent / "SOP.md"
PROJECT_LOG = Path(__file__).resolve().parent.parent.parent / "项目记录.md"

PHASES = {
    0: "数据规范化 — 字段标准化、去重、时长过滤 → baseline.parquet",
    1: "基线数据集 — 数据质量处理 + 统计 → baseline_stats.md",
    2: "随机抽样质检 — 抽样供人工/LLM 标注",
    3: "污染分析 — 高频词/频道/类别 → pollution_analysis_v1.md",
    4: "规则生成 — 从质检证据生成 → rules_v1.toml",
    5: "规则清洗 — Pass1 统计 + Pass2 过滤 → clean_*.parquet",
    6: "效果验证 — 误杀率/漏检率/污染率 → evaluation_report_v1.md",
    7: "迭代 — 返回 Phase 2 继续闭环",
}


def load_sop() -> str:
    if SOP_PATH.exists():
        return SOP_PATH.read_text(encoding="utf-8")
    return ""


def print_banner(phase: int | None = None):
    print()
    print("=" * 62)
    print("  SOP -- 语言教学视频数据清洗流程")
    print("=" * 62)
    for i, desc in PHASES.items():
        marker = "  <-" if i == phase else ""
        print(f"  Phase {i}: {desc}{marker}")
    print("=" * 62)
    print()


def write_run_log(phase: int, input_path: str, output_dir: str,
                  stats: dict | None = None, command: str = ""):
    """追加一条运行记录到 项目记录.md。"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    # Ensure project log exists
    if not PROJECT_LOG.exists():
        PROJECT_LOG.write_text(
            "# 项目记录\n\n"
            "> teach -- YouTube 语言教学视频数据清洗\n"
            "> 自动记录每次管道运行。\n\n"
            "---\n\n",
            encoding="utf-8"
        )

    lines.append("")
    lines.append(f"## Phase {phase} -- {PHASES[phase].split(chr(8212))[0].strip()}")
    lines.append("")
    lines.append(f"**时间:** {ts}")
    lines.append(f"**阶段:** Phase {phase}")
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

    log_path = os.path.join(output_dir.rstrip("/"), "run_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Run Log -- Phase {phase}\n\n")
        f.write(f"**Time:** {ts}\n")
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
