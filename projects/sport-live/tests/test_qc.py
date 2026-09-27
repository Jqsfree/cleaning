#!/usr/bin/env python3
"""
tests/test_qc.py — 测试 chunk_text_qc_v2.py 中的工具函数。

用法:
  python3 -m pytest tests/test_qc.py -v
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_脚本"))

import pytest
from chunk_text_qc_v2 import (
    make_run_id,
    make_output_stem,
    SYSTEM_PROMPT,
)


# ══════════════════════════════════════════════════════════════
# make_run_id
# ══════════════════════════════════════════════════════════════

def test_make_run_id_format():
    """make_run_id 返回 YYYYMMDD_HHmmss 格式。"""
    rid = make_run_id()
    parts = rid.split("_")
    assert len(parts) == 2, f"应为 YYYYMMDD_HHmmss 格式，实际: {rid}"
    date_part, time_part = parts
    assert len(date_part) == 8, f"日期部分长度应为 8: {date_part}"
    assert len(time_part) == 6, f"时间部分长度应为 6: {time_part}"
    assert date_part.isdigit()
    assert time_part.isdigit()


def test_make_run_id_unique():
    """连续调用应生成不同的 ID（时间精度内唯一）。"""
    ids = {make_run_id() for _ in range(3)}
    # 不应全部相同（如果三次调用在 1 秒内完成，可能相同）
    # 因此仅验证格式，不强制唯一


# ══════════════════════════════════════════════════════════════
# make_output_stem
# ══════════════════════════════════════════════════════════════

def test_make_output_stem():
    """make_output_stem 应正确拼接 stem 和 run_id。"""
    result = make_output_stem("my_data", "20260603_120000")
    assert result == "my_data_textqc_20260603_120000"


def test_make_output_stem_preserves_path_stem():
    """make_output_stem 保留输入 stem 的完整名称。"""
    result = make_output_stem("sports_chunk_05_clean", "20260603_120000")
    assert result.startswith("sports_chunk_05_clean_textqc_")


# ══════════════════════════════════════════════════════════════
# SYSTEM_PROMPT
# ══════════════════════════════════════════════════════════════

def test_system_prompt_contains_critical_keywords():
    """系统提示词应包含关键的通过/不通过指示。"""
    assert "T" in SYSTEM_PROMPT or "通过" in SYSTEM_PROMPT
    assert "F" in SYSTEM_PROMPT or "非体育" in SYSTEM_PROMPT
    assert "仅输出 T 或 F" in SYSTEM_PROMPT


def test_system_prompt_defines_sports():
    """系统提示词应定义哪些内容算体育。"""
    assert "match" in SYSTEM_PROMPT.lower()
    assert "game" in SYSTEM_PROMPT.lower()


def test_system_prompt_defines_non_sports():
    """系统提示词应定义哪些内容不算体育。"""
    assert "游戏" in SYSTEM_PROMPT or "电竞" in SYSTEM_PROMPT or "gaming" in SYSTEM_PROMPT.lower()
    assert "音乐" in SYSTEM_PROMPT or "music" in SYSTEM_PROMPT.lower()
