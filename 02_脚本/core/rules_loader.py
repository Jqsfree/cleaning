#!/usr/bin/env python3
"""
core/rules_loader.py -- 通用规则加载（不绑定任何类别）

从 TOML 规则目录加载黑名单、白名单、实体定义。
所有函数接收 rules_dir 参数，可被任何类别复用。

TOML 约定：
  {rules_dir}/
    blacklist.toml   -- [[title_pass2]], [[title_r3]], [[channel_pass2]],
                        [[pass2]], [[r2]]，每项含 pattern:string
    whitelist.toml   -- [meta], [[positive]], [[negative]], strong_*_pattern:string
    entities.toml    -- (可选) 类别专用实体定义。不存在则返回空字典
"""

import json
import re
import tomllib
from pathlib import Path

_HIT_CACHE_FILE = ".rule_hits_cache.json"


def load_blacklist(rules_dir: Path) -> dict[str, str]:
    """加载 title_pass2 / title_r3 / channel_pass2 / pass2 / r2 正则。

    若存在 ``.rule_hits_cache.json``，按历史命中数降序拼接 pattern
    （高命中优先，尽早过滤）。

    返回:
      {"title_pass2": "...", "title_r3": "...", "channel_pass2": "...",
       "pass2": "...", "r2": "..."}
      未配置的 section 对应 ``\\b\\B``（永不匹配；勿与文档旧哨兵 (?!x)x 混用）。
    """
    rules = load_blacklist_individual(rules_dir)
    hit_cache = load_hit_cache(rules_dir)
    result = {}
    for section in ("title_pass2", "title_r3", "channel_pass2", "pass2", "r2"):
        items = list(rules.get(section, []))
        section_hits = hit_cache.get(section) or {}
        if section_hits:
            items.sort(
                key=lambda r: section_hits.get(r.get("category", "?"), 0),
                reverse=True,
            )
        patterns = [r["pattern"] for r in items]
        result[section] = "|".join(patterns) if patterns else r"\b\B"
    return result


def load_blacklist_individual(rules_dir: Path) -> dict[str, list[dict[str, str]]]:
    """加载各 blacklist section 的逐条规则（保留 category 名）。

    返回:
      {"pass2": [{"category": "anime_cartoon", "pattern": "..."}, ...],
       "r2":    [{"category": "documentary", "pattern": "..."}, ...]}
    """
    bl_path = rules_dir / "blacklist.toml"
    if not bl_path.exists():
        return {"pass2": [], "r2": []}

    bl = tomllib.loads(bl_path.read_text("utf-8"))
    result: dict[str, list[dict[str, str]]] = {}
    for section in ("title_pass2", "title_r3", "channel_pass2", "pass2", "r2"):
        items = []
        for item in bl.get(section, []):
            pat = item.get("pattern", "")
            if pat:
                items.append({"category": item.get("category", "?"), "pattern": pat})
        result[section] = items
    return result


# ── 强信号 ───────────────────────────────────────────────

def load_strong_pattern(rules_dir: Path) -> str | None:
    """从 whitelist.toml 加载 strong_*_pattern 键值。

    各类别键名不同（strong_lang_teaching_title_pattern / strong_beauty_title_pattern），
    匹配规则: 键名以 "strong_" 开头且以 "_pattern" 结尾。
    返回正则字符串，不存在则返回 None。
    """
    wl_path = rules_dir / "whitelist.toml"
    if not wl_path.exists():
        return None

    wl = tomllib.loads(wl_path.read_text("utf-8"))
    for key, val in wl.items():
        if isinstance(val, str) and key.startswith("strong_") and key.endswith("_pattern"):
            return val
    return None


# ── 阈值 ─────────────────────────────────────────────────

def load_thresholds(rules_dir: Path) -> dict[str, int]:
    """从 whitelist.toml [meta] 加载阈值。

    返回:
      {"keep_score": 35, "gray_score_low": 15, "medium_min_score": 15}
      未配置则使用默认值。
    """
    wl_path = rules_dir / "whitelist.toml"
    if not wl_path.exists():
        return {"keep_score": 35, "gray_score_low": 15, "medium_min_score": 15}

    wl = tomllib.loads(wl_path.read_text("utf-8"))
    meta = wl.get("meta", {})
    return {
        "keep_score": meta.get("keep_score", 35),
        "gray_score_low": meta.get("gray_score_low", 15),
        "medium_min_score": meta.get("medium_min_score", 15),
    }


# ── 评分规则 ─────────────────────────────────────────────

def load_scoring_rules(
    rules_dir: Path,
) -> dict[str, list[tuple[re.Pattern, int]]]:
    """从 whitelist.toml 加载评分信号（正/负分），编译为正则。

    返回:
      {
        "positive": [(compiled_re, score), ...],
        "negative": [(compiled_re, score), ...],
      }
    """
    wl_path = rules_dir / "whitelist.toml"
    if not wl_path.exists():
        return {"positive": [], "negative": []}

    wl = tomllib.loads(wl_path.read_text("utf-8"))

    def _compile(items):
        return [(re.compile(item["pattern"], re.I), item["score"]) for item in items]

    return {
        "positive": _compile(wl.get("positive", [])),
        "negative": _compile(wl.get("negative", [])),
    }


# ── 实体字典 ─────────────────────────────────────────────

def load_entities(rules_dir: Path) -> dict:
    """从 entities.toml 加载实体定义。不存在则返回空字典。

    返回原始 dict，由类别专用 scorer 解释字段语义。
    """
    ent_path = rules_dir / "entities.toml"
    if not ent_path.exists():
        return {}

    return tomllib.loads(ent_path.read_text("utf-8"))


# ── 规则排序优化（缓存命中统计） ──────────────────────


def _hit_cache_path(rules_dir: Path) -> Path:
    return rules_dir / _HIT_CACHE_FILE


def load_hit_cache(rules_dir: Path) -> dict[str, dict[str, int]]:
    """加载规则命中统计缓存。"""
    p = _hit_cache_path(rules_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_hit_cache(rules_dir: Path, stats: dict[str, dict[str, int]]) -> None:
    """保存规则命中统计缓存（合并已有数据取最大值，防一次性偏差）。
    对标: FONDUE (2025) — 基于历史命中率的选择性优化。
    """
    existing = load_hit_cache(rules_dir)
    merged: dict[str, dict[str, int]] = {}
    for section in set(list(existing.keys()) + list(stats.keys())):
        merged[section] = {}
        for cat in set(list(existing.get(section, {}).keys()) + list(stats.get(section, {}).keys())):
            merged[section][cat] = max(
                existing.get(section, {}).get(cat, 0),
                stats.get(section, {}).get(cat, 0),
            )
    _hit_cache_path(rules_dir).write_text(json.dumps(merged, indent=2), "utf-8")


def compute_and_save_rule_stats(db, rules_dir: Path,
                                section_table_map: dict | None = None,
                                text_col: str = "title_channel") -> dict:
    """各 cleaner 共享的规则命中统计 + 缓存写入。"""
    from core.sql_builder import count_rule_hits
    if section_table_map is None:
        section_table_map = {"pass2": "step1", "r2": "step1b_r2"}
    bl_individual = load_blacklist_individual(rules_dir)
    stats: dict[str, dict[str, int]] = {}
    for section, table in section_table_map.items():
        rules = bl_individual.get(section, [])
        if not rules:
            continue
        hits = count_rule_hits(db, table, rules, text_col=text_col)
        if hits:
            stats[section] = hits
    if stats:
        try:
            save_hit_cache(rules_dir, stats)
        except OSError:
            pass
    return stats
