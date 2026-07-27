#!/usr/bin/env python3
"""
pipeline/context.py — PipelineContext 管道上下文

管理一次管道运行的共享状态，包括：
  - 运行目录的 progress.json 读写
  - run_log.md 写入
  - 项目级 项目记录.md 追加

替代各 phase 脚本中分散调用的 mark_done() / write_run_log()。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataclean.core.logging import log

# 复用旧 core 模块的 SOP 定义（过渡期，Stage 3 移入本包）
try:
    from core.sop import PHASES as _PHASES
except ImportError:
    from dataclean.core.logging import log as _log
    _log("无法导入 core.sop.PHASES，使用默认阶段定义", level="WARN")
    _PHASES = {}

_PROJECT_LOG = Path(__file__).resolve().parent.parent.parent.parent / "项目记录.md"


@dataclass
class RunRecord:
    """一次阶段运行的记录。"""
    phase: int
    phase_name: str
    input_path: str
    output_dir: str
    command: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    stats: dict[str, Any] = field(default_factory=dict)
    status: str = "done"


class PipelineContext:
    """管道运行上下文。

    封装一次管道运行的目录管理和进度追踪。

    用法:
        ctx = PipelineContext("data/runs/film_tv/")
        ctx.mark_start(0, input_path, output_dir)
        ...
        ctx.mark_done(0, total_in=58000, total_out=42000, elapsed_sec=5.3)
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── progress.json ──

    def _progress_path(self) -> Path:
        return self.run_dir / "progress.json"

    def read_progress(self) -> dict[str, Any] | None:
        """读取当前进度文件。"""
        p = self._progress_path()
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def write_progress(self, phase: int, status: str = "running", **extra) -> None:
        """写入进度文件。"""
        data = {
            "phase": phase,
            "status": status,
            "timestamp": time.strftime("%H:%M:%S"),
            "updated_at": time.time(),
            **extra,
        }
        with open(self._progress_path(), "w") as f:
            json.dump(data, f, ensure_ascii=False)

    def mark_start(self, phase: int, input_path: str = "", **extra) -> None:
        """标记阶段开始。"""
        self.write_progress(phase, status="running", input=input_path, **extra)
        log(f"progress.json → phase={phase} running")

    def mark_done(self, phase: int, **extra) -> None:
        """标记阶段完成。"""
        self.write_progress(phase, status="done", **extra)
        log(f"progress.json → phase={phase} done")

    # ── run_log.md ──

    def write_run_log(self, record: RunRecord) -> None:
        """写入本次目录的 run_log.md 并追加到项目记录。"""
        self._write_local_log(record)
        self._append_project_log(record)

    def _write_local_log(self, record: RunRecord) -> None:
        p = self.run_dir / "run_log.md"
        phase_label = _PHASES.get(record.phase, f"Phase {record.phase}")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"# Run Log -- {phase_label}\n\n")
            f.write(f"**Time:** {record.timestamp}\n")
            f.write(f"**Input:** `{record.input_path}`\n")
            f.write(f"**Command:** `{record.command}`\n\n")
            if record.stats:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for k, v in record.stats.items():
                    if isinstance(v, int):
                        f.write(f"| {k} | {v:,} |\n")
                    elif isinstance(v, float):
                        f.write(f"| {k} | {v:.1f} |\n")
                    else:
                        f.write(f"| {k} | {v} |\n")
        log(f"run_log.md 已写入: {p}")

    def _append_project_log(self, record: RunRecord) -> None:
        if not _PROJECT_LOG.exists():
            _PROJECT_LOG.write_text(
                "# 项目记录\n\n"
                "> YouTube 视频元数据清洗管道\n"
                "> 自动记录每次管道运行。\n\n"
                "---\n\n",
                encoding="utf-8",
            )

        phase_label = _PHASES.get(record.phase, f"Phase {record.phase}")
        lines = [
            "",
            f"## {phase_label}",
            "",
            f"**时间:** {record.timestamp}",
            f"**阶段:** Phase {record.phase}",
            f"**输入:** `{record.input_path}`",
            f"**输出:** `{record.output_dir}/`",
        ]
        if record.command:
            lines.append(f"**命令:** `{record.command}`")

        if record.stats:
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for k, v in record.stats.items():
                if isinstance(v, int):
                    lines.append(f"| {k} | {v:,} |")
                elif isinstance(v, float):
                    lines.append(f"| {k} | {v:.1f} |")
                else:
                    lines.append(f"| {k} | {v} |")
        lines.append("")

        with open(_PROJECT_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log(f"项目记录已追加: {_PROJECT_LOG}")


# ── 便捷函数（兼容旧 API） ──


def quick_mark_done(output_dir: str, phase: int, **extra) -> None:
    """快速标记阶段完成（不创建完整的 PipelineContext）。

    兼容旧 core/progress.py 的 mark_done() API。
    """
    ctx = PipelineContext(output_dir)
    ctx.mark_done(phase, **extra)


def quick_write_log(phase: int, input_path: str, output_dir: str,
                    stats: dict | None = None, command: str = "") -> None:
    """快速写入运行日志（不创建完整的 PipelineContext）。

    兼容旧 core/sop.py 的 write_run_log() API。
    """
    ctx = PipelineContext(output_dir)
    record = RunRecord(
        phase=phase,
        phase_name=_PHASES.get(phase, f"Phase {phase}"),
        input_path=input_path,
        output_dir=output_dir,
        command=command,
        stats=stats or {},
    )
    ctx.write_run_log(record)
