#!/usr/bin/env python3
"""categories/exo_cook/cleaner.py — exo 烹饪教学 文本黑名单（非餐饮/烹饪主题闸门）。

不默认挂 02_clean；直接调用 categories.exo_cook.cleaner.clean(...)。
只丢与餐饮、烹饪无关的确定串台；菜谱/厨艺教学/后厨/餐厅服务等不进黑名单。
v0.4+：``channel_pass2`` 金标纯 F 频道（见 ``rules/channel_blacklist_pure_f.csv``）。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(input_path, stem="exo_cook", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category="exo_cook",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
