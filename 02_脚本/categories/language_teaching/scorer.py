#!/usr/bin/env python3
"""
categories/language_teaching/scorer.py -- 语言教学专用打分 UDF

基于 rules_loader 加载规则，注册 5 个 DuckDB UDF：
  - blacklist_pass2 / blacklist_r2: 黑名单匹配（含频道白名单豁免）
  - strong_lang_signal: 强语言教学信号
  - lang_teaching_score: 综合打分（正/负分 + 实体对齐惩罚）
  - parse_lang_entities: 从 keyword 解析语言实体
  - keyword_aligned: keyword 实体是否与 title/channel 对齐
"""

import re
from pathlib import Path

from core.rules_loader import (
    load_blacklist,
    load_strong_pattern,
    load_scoring_rules,
    load_entities,
)

RULES_DIR = Path(__file__).resolve().parent / "rules"

# ── 模块级缓存（只加载一次） ──
_loaded = False
_pass2_re = None
_r2_re = None
_strong_re = None
_positive = []
_negative = []
_languages = []
_languages_sorted = []
_synonyms = {}
_weak_entities = set()
_channel_wl_re = None


def _ensure_loaded():
    global _loaded, _pass2_re, _r2_re, _strong_re
    global _positive, _negative
    global _languages, _languages_sorted, _synonyms, _weak_entities, _channel_wl_re

    if _loaded:
        return

    bl = load_blacklist(RULES_DIR)
    _pass2_re = re.compile(bl["pass2"], re.I)
    _r2_re = re.compile(bl["r2"], re.I)

    strong = load_strong_pattern(RULES_DIR)
    _strong_re = re.compile(strong, re.I) if strong else re.compile(r"(?!x)x")

    scoring = load_scoring_rules(RULES_DIR)
    _positive = scoring["positive"]
    _negative = scoring["negative"]

    ent = load_entities(RULES_DIR)
    _languages = ent.get("languages", [])
    _languages_sorted = sorted(_languages, key=len, reverse=True)
    _synonyms = ent.get("synonyms", {})
    _weak_entities = set(ent.get("weak_entities", []))

    wl_rx = ent.get("channel_whitelist_regex", {})
    _channel_wl_re = (
        re.compile(wl_rx.get("pattern", r"(?!x)x"), re.I)
        if wl_rx else None
    )

    _loaded = True


def get_thresholds() -> dict:
    from core.rules_loader import load_thresholds
    return load_thresholds(RULES_DIR)


# ── UDF 注册 ─────────────────────────────────────────────

def register_udfs(conn):
    """向 DuckDB 连接注册所有语言教学清洗 UDF。"""
    _ensure_loaded()

    # ── blacklist_pass2 ──
    def blacklist_pass2(title, channel, keyword):
        kw_clean = _strip_keyword_tags(keyword or "")
        if _channel_wl_re and _channel_wl_re.search(
            f"{title or ''} {channel or ''} {kw_clean or ''}"
        ):
            return ""
        text = f"{title or ''} {channel or ''} {keyword or ''}"
        m = _pass2_re.search(text)
        return m.group(0) if m else ""

    def blacklist_r2(title, channel, keyword):
        kw_clean = _strip_keyword_tags(keyword or "")
        if _channel_wl_re and _channel_wl_re.search(
            f"{title or ''} {channel or ''} {kw_clean or ''}"
        ):
            return ""
        text = f"{title or ''} {channel or ''} {keyword or ''}"
        m = _r2_re.search(text)
        return m.group(0) if m else ""

    conn.create_function(
        "blacklist_pass2", blacklist_pass2, ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR"
    )
    conn.create_function(
        "blacklist_r2", blacklist_r2, ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR"
    )

    # ── strong_lang_signal ──
    def strong_lang_signal(title, channel):
        return bool(_strong_re.search(f"{title or ''} {channel or ''}"))

    conn.create_function(
        "strong_lang_signal", strong_lang_signal, ["VARCHAR", "VARCHAR"], "BOOLEAN"
    )

    # ── lang_teaching_score ──
    def lang_teaching_score(title, channel, keyword):
        kw_clean = _strip_keyword_tags(keyword)
        text = f"{title or ''} {channel or ''} {kw_clean or ''}"
        score = 0
        for pat, pts in _positive:
            if pat.search(text):
                score += pts
        for pat, pts in _negative:
            if pat.search(text):
                score += pts
        # 语言实体对齐惩罚
        entities_str = parse_lang_entities(keyword)
        if entities_str:
            entities = entities_str.split("|")
            title_lower = (title or "").lower()
            aligned = any(e in title_lower for e in entities)
            if not aligned:
                for ent in entities:
                    for key, syns in _synonyms.items():
                        if key in ent or ent in key:
                            for s in syns:
                                if s in title_lower:
                                    aligned = True
                                    break
                        if aligned:
                            break
                    if aligned:
                        break
            if not aligned and not _strong_re.search(title or ""):
                score -= 25
        return score

    conn.create_function(
        "lang_teaching_score",
        lang_teaching_score,
        ["VARCHAR", "VARCHAR", "VARCHAR"],
        "INTEGER",
    )

    # ── parse_lang_entities ──
    def parse_lang_entities(keyword):
        if not keyword:
            return ""
        kw = keyword.strip().strip('"').lower()
        parts = re.split(r"\s+-\s*", kw)
        core = parts[0] if parts else kw
        entities = []
        for term in _languages_sorted:
            if term in core:
                entities.append(term)
        if not entities:
            words = re.findall(r"[a-z一-鿿぀-ゟ゠-ヿ가-힯]{3,}", core)
            context_terms = {
                "lesson", "class", "course", "learn", "study", "teach",
                "grammar", "vocabulary", "pronunciation", "conversation",
                "beginner", "intermediate", "advanced", "tutorial", "practice",
                "speaking", "listening", "reading", "writing", "language",
                "training", "lecture", "coaching", "school", "academy",
            }
            ctx = [w for w in words if w.lower() in context_terms]
            if ctx:
                entities = ctx
            else:
                stop = {
                    "full", "match", "video", "race", "final", "live", "stream",
                    "commentary", "broadcast", "tournament", "championship",
                    "league", "contest", "open", "professional", "amateur",
                    "national", "international", "world", "replay", "footage",
                    "unedited", "season", "highlights", "historical", "veteran",
                    "rookie", "legendary", "masters", "diamond", "series",
                    "episode", "channel", "official",
                }
                entities = [w for w in words if w.lower() not in stop][:3]
        return "|".join(entities)

    conn.create_function(
        "parse_lang_entities", parse_lang_entities, ["VARCHAR"], "VARCHAR"
    )

    # ── keyword_aligned ──
    def keyword_aligned(keyword, title, channel):
        entities_str = parse_lang_entities(keyword)
        if not entities_str:
            return True
        entities = entities_str.split("|")
        text = f"{title or ''} {channel or ''}".lower()
        for e in entities:
            if e in text:
                return True
            for key, syns in _synonyms.items():
                if key in e or e in key:
                    for s in syns:
                        if s in text:
                            return True
        return False

    conn.create_function(
        "keyword_aligned",
        keyword_aligned,
        ["VARCHAR", "VARCHAR", "VARCHAR"],
        "BOOLEAN",
    )

    return conn


# ── 工具函数 ─────────────────────────────────────────────

def _strip_keyword_tags(kw):
    """剥离 keyword 中 - 开头的标签 token。"""
    if not kw:
        return kw
    kw = kw.strip().strip('"').lower()
    parts = re.split(r"\s+-\s*", kw)
    return parts[0] if parts else kw
