#!/usr/bin/env python3
"""categories/exo_entertainment/cleaner.py — exo 娱乐表演 文本黑名单（非表演主题闸门）。

不默认挂 02_clean；直接调用 categories.exo_entertainment.cleaner.clean(...)。
丢与真人现场/街头/才艺/排练表演无关的串台（宁杀勿放）；
街头表演、舞蹈排练、才艺秀、马戏、现场乐队等不进黑名单。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(input_path, stem="exo_entertainment", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category="exo_entertainment",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
