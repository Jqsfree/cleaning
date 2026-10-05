#!/usr/bin/env python3
"""categories/exo_团队协作/teamgames/cleaner.py — 团队协作·团队游戏 文本黑名单。

口径：保留真人团队游戏/协作挑战；丢电竞/电子游戏、体育赛事、卡通综艺等。
不默认挂 02_clean；直接调用 categories.exo_团队协作.teamgames.cleaner.clean(...)。
"""

from pathlib import Path

from core.certain_noise_clean import run_clean

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(input_path, stem="团队游戏", output_dir="output", raw_name="", run="run01", **kwargs):
    return run_clean(
        category="exo_团队协作",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
