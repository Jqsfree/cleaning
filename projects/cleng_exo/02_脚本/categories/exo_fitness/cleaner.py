#!/usr/bin/env python3
"""categories/exo_fitness/cleaner.py — exo健身训练 文本黑名单（certain-noise）。

不默认挂 02_clean；直接调用 categories.exo_fitness.cleaner.clean(...)。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(input_path, stem="exo_fitness", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category="exo_fitness",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
