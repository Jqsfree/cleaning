#!/usr/bin/env python3
"""
core/scoring.py -- DuckDB UDF 注册

直接从 rules/*.toml 加载规则，注册为 DuckDB UDF，
供 Pass1/Pass2 SQL 调用。

规则在首次调用 register_udfs() 或 get_thresholds() 时延迟加载，
避免 import 时的副作用和文件缺失错误。
"""

import sys, os, re, tomllib
from pathlib import Path

_DEFAULT_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"

# 可通过环境变量或函数调用覆盖
_RULES_DIR = Path(os.environ.get("SPORT_RULES_DIR", str(_DEFAULT_RULES_DIR)))

# ── 延迟加载状态 ──
_loaded = False

# 阈值
KEEP_SCORE_THRESHOLD = 35
GRAY_SCORE_LOW = 15
MEDIUM_MIN_SCORE = 15

# 编译后的规则（延迟初始化）
SPORT_POSITIVE = []
SPORT_NEGATIVE = []
BL_PASS2_RE = None
BL_R2_RE = None
STRONG_RE = None
LEXICON = []
LEXICON_SORTED = []
SYNONYMS = {}
WEAK_ENTITIES = set()
CHANNEL_WL_RE = None


def _ensure_loaded():
    """延迟加载规则文件。幂等 — 多次调用只加载一次。"""
    global _loaded, KEEP_SCORE_THRESHOLD, GRAY_SCORE_LOW, MEDIUM_MIN_SCORE
    global SPORT_POSITIVE, SPORT_NEGATIVE, BL_PASS2_RE, BL_R2_RE, STRONG_RE
    global LEXICON, LEXICON_SORTED, SYNONYMS, WEAK_ENTITIES, CHANNEL_WL_RE

    if _loaded:
        return

    _bl = tomllib.load(open(_RULES_DIR / "blacklist.toml", "rb"))
    _wl = tomllib.load(open(_RULES_DIR / "whitelist.toml", "rb"))
    _ent = tomllib.load(open(_RULES_DIR / "entities.toml", "rb"))

    # 阈值 — 单一来源: whitelist.toml [meta]
    T = _wl.get("meta", {})
    KEEP_SCORE_THRESHOLD = T.get("keep_score", 35)
    GRAY_SCORE_LOW = T.get("gray_score_low", 15)
    MEDIUM_MIN_SCORE = T.get("medium_min_score", 15)

    # 编译正/负信号
    _pos_raw = [(item["pattern"], item["score"]) for item in _wl.get("positive", [])]
    _neg_raw = [(item["pattern"], item["score"]) for item in _wl.get("negative", [])]

    SPORT_POSITIVE = [(re.compile(p, re.I), s) for p, s in _pos_raw]
    SPORT_NEGATIVE = [(re.compile(p, re.I), s) for p, s in _neg_raw]

    # 黑名单
    _bl_p2_patterns = [item["pattern"] for item in _bl.get("pass2", [])]
    _bl_r2_patterns = [item["pattern"] for item in _bl.get("r2", [])]
    BL_PASS2_RE = re.compile("|".join(_bl_p2_patterns), re.I) if _bl_p2_patterns else re.compile(r"(?!x)x")
    BL_R2_RE = re.compile("|".join(_bl_r2_patterns), re.I) if _bl_r2_patterns else re.compile(r"(?!x)x")

    # 强体育标题
    STRONG_RE = re.compile(_wl.get("strong_sport_title_pattern", r"(?!x)x"), re.I)

    # 词典
    LEXICON = _ent.get("sports", [])
    LEXICON_SORTED = sorted(LEXICON, key=len, reverse=True)
    SYNONYMS = _ent.get("synonyms", {})
    WEAK_ENTITIES = set(_ent.get("weak_entities", []))

    # 频道白名单
    _wl_rx = _ent.get("channel_whitelist_regex", {})
    CHANNEL_WL_RE = re.compile(_wl_rx.get("pattern", r"(?!x)x"), re.I) if _wl_rx else None

    _loaded = True


def get_thresholds() -> dict:
    """返回规则中定义的阈值，供 cleaner.py 等模块使用。"""
    _ensure_loaded()
    return {
        "keep_score": KEEP_SCORE_THRESHOLD,
        "gray_score_low": GRAY_SCORE_LOW,
        "medium_min_score": MEDIUM_MIN_SCORE,
    }


def set_rules_dir(path: str):
    """切换规则目录并触发重新加载。"""
    global _RULES_DIR, _loaded
    _RULES_DIR = Path(path)
    _loaded = False
    _ensure_loaded()


def register_udfs(conn):
    """向 DuckDB 连接注册所有清洗 UDF。"""
    _ensure_loaded()

    # ── blacklist_pass2 ──
    def blacklist_pass2(title, channel, keyword):
        text = f"{title or ''} {channel or ''} {keyword or ''}"
        # channel 白名单豁免：命中白名单则跳过黑名单检查
        if CHANNEL_WL_RE and CHANNEL_WL_RE.search(text):
            return ""
        m = BL_PASS2_RE.search(text)
        return m.group(0) if m else ""

    def blacklist_r2(title, channel, keyword):
        text = f"{title or ''} {channel or ''} {keyword or ''}"
        if CHANNEL_WL_RE and CHANNEL_WL_RE.search(text):
            return ""
        m = BL_R2_RE.search(text)
        return m.group(0) if m else ""

    conn.create_function("blacklist_pass2", blacklist_pass2, ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR")
    conn.create_function("blacklist_r2", blacklist_r2, ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR")

    # ── strong_sport_signal ──
    def strong_sport_signal(title, channel):
        return bool(STRONG_RE.search(f"{title or ''} {channel or ''}"))
    conn.create_function("strong_sport_signal", strong_sport_signal, ["VARCHAR", "VARCHAR"], "BOOLEAN")

    # ── sport_score ──
    def _strip_keyword_tags(kw):
        """剥离 keyword 中 - 开头的标签 token，只保留主词部分用于打分。"""
        if not kw: return kw
        kw = kw.strip().strip('"').lower()
        parts = re.split(r"\s+-\s*", kw)
        return parts[0] if parts else kw

    def sport_score(title, channel, keyword):
        kw_clean = _strip_keyword_tags(keyword)
        text = f"{title or ''} {channel or ''} {kw_clean or ''}"
        score = 0
        for pat, pts in SPORT_POSITIVE:
            if pat.search(text):
                score += pts
        for pat, pts in SPORT_NEGATIVE:
            if pat.search(text):
                score += pts
        # 关键词实体对齐惩罚: keyword 有实体但没出现在 title 中 → -25
        entities_str = parse_entities(keyword)
        if entities_str:
            entities = entities_str.split("|")
            title_lower = (title or '').lower()
            aligned = any(e in title_lower for e in entities)
            if not aligned:
                for ent in entities:
                    for key, syns in SYNONYMS.items():
                        if key in ent or ent in key:
                            for s in syns:
                                if s in title_lower:
                                    aligned = True
                                    break
                        if aligned: break
                    if aligned: break
            if not aligned:
                score -= 25
        return score
    conn.create_function("sport_score", sport_score, ["VARCHAR", "VARCHAR", "VARCHAR"], "INTEGER")

    # ── parse_entities ──
    def parse_entities(keyword):
        if not keyword: return ""
        kw = keyword.strip().strip('"').lower()
        parts = re.split(r"\s+-\s*", kw)
        core = parts[0] if parts else kw
        entities = []
        for term in LEXICON_SORTED:
            if term in core:
                entities.append(term)
        if not entities:
            words = re.findall(r"[a-z]{4,}", core)
            stop = {"full","match","video","race","final","live","stream","commentary",
                    "broadcast","tournament","championship","league","contest","open",
                    "professional","amateur","national","international","world","replay",
                    "footage","unedited","ranked","season","break","career","highlights",
                    "historical","veteran","rookie","legendary","masters","diamond"}
            entities = [w for w in words if w not in stop][:3]
        return "|".join(entities)
    conn.create_function("parse_entities", parse_entities, ["VARCHAR"], "VARCHAR")

    # ── keyword_aligned ──
    def keyword_aligned(keyword, title, channel):
        entities_str = parse_entities(keyword)
        if not entities_str: return True
        entities = entities_str.split("|")
        text = f"{title or ''} {channel or ''}".lower()
        for e in entities:
            if e in text: return True
            for key, syns in SYNONYMS.items():
                if key in e or e in key:
                    for s in syns:
                        if s in text: return True
        return False
    conn.create_function("keyword_aligned", keyword_aligned, ["VARCHAR", "VARCHAR", "VARCHAR"], "BOOLEAN")

    return conn
