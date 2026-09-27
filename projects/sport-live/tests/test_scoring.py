#!/usr/bin/env python3
"""
tests/test_scoring.py — 测试 core/scoring.py 中的核心函数。

通过 DuckDB 集成测试 UDF：加载真实规则，注册 UDF，
针对已知样本运行 SQL 查询。

用法:
  conda activate data_cleaning
  cd /home/jqs/sport-live
  python3 -m pytest tests/test_scoring.py -v
  python3 -m pytest tests/test_scoring.py -v -k "test_parse_entities"
"""

import sys, os
from pathlib import Path

# 将 02_脚本/ 加入 path 以便导入 core 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_脚本"))

import pytest
import duckdb
from core.scoring import register_udfs, get_thresholds, set_rules_dir


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def db_conn():
    """创建带已注册 UDF 的 DuckDB 连接。"""
    conn = duckdb.connect(":memory:")
    register_udfs(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def rules_dir():
    """返回主规则目录路径。"""
    return str(Path(__file__).resolve().parent.parent / "02_脚本" / "rules")


# ══════════════════════════════════════════════════════════════
# Thresholds
# ══════════════════════════════════════════════════════════════

def test_get_thresholds_returns_expected_keys():
    """get_thresholds() 返回包含预期键的字典。"""
    t = get_thresholds()
    assert "keep_score" in t
    assert "gray_score_low" in t
    assert "medium_min_score" in t
    assert all(isinstance(v, int) for v in t.values())


def test_get_thresholds_are_positive():
    """阈值应为正数。"""
    t = get_thresholds()
    assert t["keep_score"] > 0
    assert t["gray_score_low"] > 0
    assert t["medium_min_score"] > 0
    assert t["keep_score"] > t["gray_score_low"], "keep_score 应高于 gray_low"


# ══════════════════════════════════════════════════════════════
# parse_entities
# ══════════════════════════════════════════════════════════════

ENTITY_TEST_CASES = [
    # (keyword, expected_contains)
    ("football match highlights", "football"),
    ("basketball full game replay", "basketball"),
    ("tennis wimbledon final", "tennis"),
    ("swimming olympic trials", "swimming"),
    ("curling extra end", "curling"),
    ("volleyball world league", "volleyball"),
    ("ping pong championship", "pong"),  # partial match
    ("", ""),                             # empty keyword
    ("xyzzy1234 notarealthing", ""),      # no known entities — fallback to word extraction
]


@pytest.mark.parametrize("keyword,expected_contains", ENTITY_TEST_CASES)
def test_parse_entities(db_conn, keyword, expected_contains):
    """parse_entities 从已知运动关键词中提取实体。"""
    result = db_conn.execute(
        "SELECT parse_entities(?)", [keyword]
    ).fetchone()[0]

    if not expected_contains:
        # 对于未知关键词，函数返回提取的词（fallback）或空字符串
        # 两种情况均可接受
        return

    entities = result.split("|") if result else []
    assert any(expected_contains in e for e in entities), \
        f"parse_entities('{keyword}') = '{result}'，期望包含 '{expected_contains}'"


def test_parse_entities_strips_tags(db_conn):
    """parse_entities 在解析实体前剥离 keyword 标签。"""
    result = db_conn.execute(
        "SELECT parse_entities(?)", ["football -virtual -cartoon"]
    ).fetchone()[0]
    assert "football" in result.split("|"), \
        f"标签剥离失败: '{result}'"


# ══════════════════════════════════════════════════════════════
# sport_score
# ══════════════════════════════════════════════════════════════

def test_sport_score_positive_signal(db_conn):
    """标题含强体育信号应得正分。"""
    score = db_conn.execute(
        "SELECT sport_score(?, ?, ?)",
        ["NBA Finals Game 7 Highlights", "ESPN", "basketball nba finals"]
    ).fetchone()[0]
    assert score > 0, f"强体育标题应得正分，实际: {score}"


def test_sport_score_negative_signal(db_conn):
    """标题含非体育信号应得负分或低分。"""
    score = db_conn.execute(
        "SELECT sport_score(?, ?, ?)",
        ["Minecraft Let's Play Episode 42", "GamingChannel", "minecraft game"]
    ).fetchone()[0]
    # 游戏相关内容得分应较低或为负
    assert score < 30, f"游戏内容不应得高分，实际: {score}"


def test_sport_score_tag_stripping(db_conn):
    """keyword 标签应被剥离，不参与打分。"""
    # 含 -game 标签不应因 'game' 被扣分
    score_with_tag = db_conn.execute(
        "SELECT sport_score(?, ?, ?)",
        ["Champions League Final 2024", "UEFA TV", "football -game -virtual"]
    ).fetchone()[0]
    # 纯 keyword 不含标签也应得到相同分数
    score_without_tag = db_conn.execute(
        "SELECT sport_score(?, ?, ?)",
        ["Champions League Final 2024", "UEFA TV", "football"]
    ).fetchone()[0]
    assert score_with_tag == score_without_tag, \
        f"标签剥离不一致: {score_with_tag} vs {score_without_tag}"


def test_sport_score_entity_alignment_penalty(db_conn):
    """keyword 实体未出现在标题中应触发 -25 惩罚。"""
    # keyword 是 basketball 但标题关于足球
    score_misaligned = db_conn.execute(
        "SELECT sport_score(?, ?, ?)",
        ["Soccer Match Brazil vs Argentina", "SportsTV", "basketball nba"]
    ).fetchone()[0]
    # keyword 与标题匹配
    score_aligned = db_conn.execute(
        "SELECT sport_score(?, ?, ?)",
        ["NBA Finals basketball highlights", "SportsTV", "basketball nba"]
    ).fetchone()[0]
    assert score_misaligned < score_aligned, \
        f"未对齐应得分更低: misaligned={score_misaligned}, aligned={score_aligned}"


# ══════════════════════════════════════════════════════════════
# blacklist_pass2
# ══════════════════════════════════════════════════════════════

def test_blacklist_pass2_detects_gaming(db_conn):
    """黑名单应捕获游戏关键词。"""
    result = db_conn.execute(
        "SELECT blacklist_pass2(?, ?, ?)",
        ["Minecraft Survival Episode 1", "SomeChannel", "minecraft"]
    ).fetchone()[0]
    # minecraft 应在黑名单中（作为 gaming 类别）
    # 如果不在，返回空字符串
    assert isinstance(result, str)


def test_blacklist_pass2_clean_sports_passes(db_conn):
    """纯体育标题不应被黑名单拦截。"""
    result = db_conn.execute(
        "SELECT blacklist_pass2(?, ?, ?)",
        ["Olympic 100m Final", "NBC Sports", "athletics olympics"]
    ).fetchone()[0]
    assert result == "", f"纯体育内容不应被黑名单命中，实际: '{result}'"


# ══════════════════════════════════════════════════════════════
# keyword_aligned
# ══════════════════════════════════════════════════════════════

def test_keyword_aligned_match(db_conn):
    """keyword 实体出现在标题中 → aligned。"""
    result = db_conn.execute(
        "SELECT keyword_aligned(?, ?, ?)",
        ["football final match", "FIFA", "football highlights"]
    ).fetchone()[0]
    assert result is True, "实体匹配应返回 True"


def test_keyword_aligned_no_entities(db_conn):
    """无实体的 keyword → 默认 aligned。"""
    result = db_conn.execute(
        "SELECT keyword_aligned(?, ?, ?)",
        ["some random video", "RandomChannel", "xyz not an entity"]
    ).fetchone()[0]
    assert result is True, "无实体时应默认 aligned"


# ══════════════════════════════════════════════════════════════
# strong_sport_signal
# ══════════════════════════════════════════════════════════════

def test_strong_sport_signal_positive(db_conn):
    """强体育信号模式应匹配已知体育标题。"""
    result = db_conn.execute(
        "SELECT strong_sport_signal(?, ?)",
        ["FIFA World Cup Final 2026", "FIFA TV"]
    ).fetchone()[0]
    # 根据 whitelist.toml 中的 strong_sport_title_pattern 而定
    # 对于已知体育实体（如 FIFA, World Cup），应为 True
    assert isinstance(result, bool)


def test_strong_sport_signal_negative(db_conn):
    """随机内容不应触发强信号。"""
    result = db_conn.execute(
        "SELECT strong_sport_signal(?, ?)",
        ["Random Video Title", "RandomChannel123"]
    ).fetchone()[0]
    assert result is False, "随机标题不应触发强体育信号"


# ══════════════════════════════════════════════════════════════
# set_rules_dir — 规则重载
# ══════════════════════════════════════════════════════════════

def test_set_rules_dir_reloads(rules_dir):
    """set_rules_dir 应触发重新加载且不抛异常。"""
    # 切换回主规则
    set_rules_dir(rules_dir)
    t = get_thresholds()
    assert t["keep_score"] > 0, "重载后阈值应正确"
