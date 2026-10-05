#!/usr/bin/env python3
"""categories/exo_dance/cleaner.py — exo 单人舞蹈 文本黑名单（非舞蹈主题闸门）。

不默认挂 02_clean；直接调用 categories.exo_dance.cleaner.clean(...)。
闸门口径见 rules/blacklist.toml 顶部注释（channel 硬闸 + pass2 软闸 + rescue 豁免 + r2 硬闸）。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(input_path, stem="exo_dance", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category="exo_dance",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
