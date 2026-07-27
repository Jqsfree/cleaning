#!/usr/bin/env python3
"""
phases/clean.py — Phase 5: 规则清洗

封装 phase5_clean.py 的核心逻辑为 Phase 子类。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dataclean.pipeline.base import Phase, PhaseResult
from dataclean.pipeline.registry import load_cleaner
from dataclean.core.logging import log, banner, ensure_input_exists


class CleanPhase(Phase):
    """Phase 5 — 规则清洗：对 baseline parquet 应用黑/白名单规则。

    通过 --category 参数选择目标类别。
    """

    phase_id = 5
    phase_name = "clean"

    def run(self, input_path: str, output_dir: str, **kwargs) -> PhaseResult:
        category = kwargs.get("category", "language_teaching")
        run_name = kwargs.get("run", "run01")
        keep_score = kwargs.get("keep_score")
        gray_low = kwargs.get("gray_low")
        med_min = kwargs.get("med_min")
        no_medium = kwargs.get("no_medium", False)

        ensure_input_exists(input_path)
        os.makedirs(output_dir, exist_ok=True)

        raw_stem = os.path.splitext(os.path.basename(input_path))[0]
        stem = raw_stem.replace("_raw", "") if "_raw" in raw_stem else raw_stem

        clean_func = load_cleaner(category)

        clean_kwargs = dict(
            input_path=input_path,
            stem=stem,
            output_dir=output_dir,
            raw_name=stem,
            run=run_name,
            keep_score=keep_score,
            gray_low=gray_low,
            med_min=med_min,
            no_medium=no_medium,
        )

        log(f"调用 {category} cleaner...")
        summary = clean_func(**clean_kwargs)

        return PhaseResult(
            phase_id=5,
            phase_name=f"clean-{category}",
            total_in=summary["total_rows"],
            total_out=summary["total_keep"],
            stats={
                "保留(high)": summary.get("total_keep_high", 0),
                "保留(medium)": summary.get("total_keep_medium", 0),
                "移除": summary["total_drop"],
                "保留率": f"{summary['retention_pct']}%",
            },
            output_files=[
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.endswith(".parquet") or f.endswith(".json")
            ],
            elapsed_sec=summary["elapsed_sec"],
        )
