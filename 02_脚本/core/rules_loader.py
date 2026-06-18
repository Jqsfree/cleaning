#!/usr/bin/env python3
"""
core/rules_loader.py -- 通用规则加载（不绑定任何类别）

从 TOML 规则目录加载黑名单、白名单、实体定义。
所有函数接收 rules_dir 参数，可被任何类别复用。

TOML 约定：
  {rules_dir}/
    blacklist.toml   -- [[pass2]], [[r2]]  每个 item 含 pattern:string
    whitelist.toml   -- [meta], [[positive]], [[negative]], strong_*_pattern:string
    entities.toml    -- (可选) 类别专用实体定义。不存在则返回空字典
"""

import re
import tomllib
from pathlib import Path


# ── 黑名单 ───────────────────────────────────────────────

def load_blacklist(rules_dir: Path) -> dict[str, str]:
    """从 blacklist.toml 加载 pass2 / r2 正则。

    返回:
      {"pass2": "pat1|pat2|...", "r2": "pat1|pat2|..."}
      未配置的 section 对应 "(?!x)x"（永不匹配）。
    """
    bl_path = rules_dir / "blacklist.toml"
    if not bl_path.exists():
        return {"pass2": r"(?!x)x", "r2": r"(?!x)x"}

    bl = tomllib.loads(bl_path.read_text("utf-8"))
    result = {}
    for section in ("pass2", "r2"):
        patterns = [item["pattern"] for item in bl.get(section, [])]
        result[section] = "|".join(patterns) if patterns else r"(?!x)x"
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
