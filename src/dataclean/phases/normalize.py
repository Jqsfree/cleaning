#!/usr/bin/env python3
"""
phases/normalize.py — Phase 0: 数据规范化

封装 phase0_normalize.py 的核心逻辑为 Phase 子类。
"""

from __future__ import annotations

import os
import sys

from dataclean.pipeline.base import Phase, PhaseResult
from dataclean.core.logging import log, banner, ensure_input_exists


class NormalizePhase(Phase):
    """Phase 0 — 数据规范化：字段标准化、去重、损坏/时长过滤 → baseline.parquet"""

    phase_id = 0
    phase_name = "normalize"

    def run(self, input_path: str, output_dir: str, **kwargs) -> PhaseResult:
        min_duration = kwargs.get("min_duration", 10)

        ensure_input_exists(input_path)
        os.makedirs(output_dir, exist_ok=True)

        # 委托给旧 phase0_normalize 模块
        # 过渡期：直接导入旧模块；Stage 4+ 移入本包
        try:
            import phase0_normalize
        except ImportError:
            log("无法导入 phase0_normalize，请确保 02_脚本/ 在 sys.path 中", level="ERROR")
            raise

        # 模拟旧的 CLI 调用
        old_args = type("Args", (), {})()
        old_args.input = input_path
        old_args.output_dir = output_dir
        old_args.min_duration = min_duration

        # phase0_normalize.main() 内部使用 argparse，直接调用需要重构
        # 此处提供薄包装：直接 import 旧模块的 main
        from phase0_normalize import main as old_main

        # 临时覆盖 sys.argv 以兼容 argparse
        old_argv = sys.argv[:]
        sys.argv = [
            "phase0_normalize",
            input_path,
            "-o", output_dir,
            "--min-duration", str(min_duration),
        ]
        try:
            old_main()
        finally:
            sys.argv = old_argv

        # 从 progress.json 读取统计
        from core.progress import read as read_progress
        progress = read_progress(output_dir) or {}

        return PhaseResult(
            phase_id=0,
            phase_name="normalize",
            total_in=progress.get("raw_rows", 0),
            total_out=progress.get("final", 0),
            stats={
                "过滤(null)": progress.get("null_filter", 0),
                "过滤(损坏)": progress.get("damaged", 0),
                "过滤(时长)": progress.get("duration_filter", 0),
            },
            output_files=[
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.endswith(".parquet") or f.endswith(".md")
            ],
        )
