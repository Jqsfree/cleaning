#!/usr/bin/env python3
"""categories/exo_medical/cleaner.py — exo 医疗实验 文本黑名单（非医疗实验主题闸门）。

不默认挂 02_clean；直接调用 categories.exo_medical.cleaner.clean(...)。
只丢与实验室实操/实验演示无关的确定串台；细菌培养/切片/ELISA 等不进黑名单。
v0.5+：``channel_pass2`` 高密度纯串台频道（见 blacklist.toml）。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(input_path, stem="exo_medical", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category="exo_medical",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
