#!/usr/bin/env python3
"""categories/exo_parent/cleaner.py — exo 亲子互动 文本黑名单（非亲子主题闸门）。

不默认挂 02_clean；直接调用 categories.exo_parent.cleaner.clean(...)。
只丢与亲子互动无关的确定串台；亲子陪玩/育儿劳作实拍等不进黑名单。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(input_path, stem="exo_parent", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category="exo_parent",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
