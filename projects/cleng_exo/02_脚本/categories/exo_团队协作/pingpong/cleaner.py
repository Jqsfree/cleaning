#!/usr/bin/env python3
"""categories/exo_团队协作/pingpong/cleaner.py — 团队协作·双人乒乓 文本黑名单。

不默认挂 02_clean；直接调用::

    from categories.exo_团队协作.pingpong.cleaner import clean

只丢与乒乓球/桌球无关的确定串台（电竞/舞蹈/开箱等）；
WTT/ITTF/双打/混双实拍与比赛不进黑名单。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"
_CATEGORY = "exo_团队协作"


def clean(input_path, stem="双人乒乓", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category=_CATEGORY,
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
