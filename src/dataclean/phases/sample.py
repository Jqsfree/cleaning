#!/usr/bin/env python3
"""
phases/sample.py — Phase 2: 统计学抽样

封装 phase2_sample.py 的核心逻辑为 Phase 子类。
"""

from __future__ import annotations

import os
import sys

from dataclean.pipeline.base import Phase, PhaseResult
from dataclean.core.logging import log, banner, ensure_input_exists


class SamplePhase(Phase):
    """Phase 2 — 统计学抽样 → audit_sample.parquet"""

    phase_id = 2
    phase_name = "sample"

    def run(self, input_path: str, output_dir: str, **kwargs) -> PhaseResult:
        sample_size = kwargs.get("sample_size")
        seed = kwargs.get("seed", 42)

        ensure_input_exists(input_path)
        os.makedirs(output_dir, exist_ok=True)

        # 委托给旧 phase2_sample 模块
        old_argv = sys.argv[:]
        cmd = ["phase2_sample", input_path, "-o", output_dir]
        if sample_size:
            cmd.extend(["-n", str(sample_size)])
        cmd.extend(["--seed", str(seed)])
        sys.argv = cmd
        try:
            import phase2_sample
            phase2_sample.main()
        finally:
            sys.argv = old_argv

        from core.progress import read as read_progress
        progress = read_progress(output_dir) or {}

        return PhaseResult(
            phase_id=2,
            phase_name="sample",
            total_out=progress.get("final", 0),
            stats={"样本量": progress.get("final", 0)},
            output_files=[
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.endswith(".parquet") or f.endswith(".csv")
            ],
        )
