#!/usr/bin/env python3
"""
core/scoring.py -- DuckDB UDF 注册（语言教学版）

从 rules/*.toml 加载规则，注册为 DuckDB UDF，供清洗 SQL 调用。
规则在首次调用 register_udfs() 或 get_thresholds() 时延迟加载。
"""

import sys, os, re, tomllib
from pathlib import Path

_DEFAULT_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
_RULES_DIR = Path(os.environ.get("TEACH_RULES_DIR", str(_DEFAULT_RULES_DIR)))

# ── 延迟加载状态 ──
_loaded = False

# 阈值
KEEP_SCORE_THRESHOLD = 35
GRAY_SCORE_LOW = 15
MEDIUM_MIN_SCORE = 15

# 编译后的规则（延迟初始化）
LANG_POSITIVE = []
LANG_NEGATIVE = []
BL_PASS2_PATTERNS = []
BL_R2_PATTERNS = []
BL_PASS2_RE = None
BL_R2_RE = None
STRONG_RE = None
LANGUAGES = []
LANGUAGES_SORTED = []
SYNONYMS = {}
WEAK_ENTITIES = set()
CHANNEL_WL_RE = None


def _compile_pass2():
    global BL_PASS2_RE
    BL_PASS2_RE = re.compile("|".join(BL_PASS2_PATTERNS), re.I) if BL_PASS2_PATTERNS else re.compile(r"(?!x)x")


def _compile_r2():
    global BL_R2_RE
    BL_R2_RE = re.compile("|".join(BL_R2_PATTERNS), re.I) if BL_R2_PATTERNS else re.compile(r"(?!x)x")


def _ensure_loaded():
    """延迟加载规则文件。幂等 — 多次调用只加载一次。"""
    global _loaded, KEEP_SCORE_THRESHOLD, GRAY_SCORE_LOW, MEDIUM_MIN_SCORE
    global LANG_POSITIVE, LANG_NEGATIVE, BL_PASS2_PATTERNS, BL_R2_PATTERNS, BL_PASS2_RE, BL_R2_RE, STRONG_RE
    global LANGUAGES, LANGUAGES_SORTED, SYNONYMS, WEAK_ENTITIES, CHANNEL_WL_RE

    if _loaded:
        return

    bl_path = _RULES_DIR / "current" / "blacklist.toml"
    wl_path = _RULES_DIR / "current" / "whitelist.toml"
    ent_path = _RULES_DIR / "current" / "entities.toml"

    _bl = tomllib.loads(bl_path.read_text("utf-8")) if bl_path.exists() else {}
    _wl = tomllib.loads(wl_path.read_text("utf-8")) if wl_path.exists() else {}
    _ent = tomllib.loads(ent_path.read_text("utf-8")) if ent_path.exists() else {}

    # 阈值 — 来自 whitelist.toml [meta]
    T = _wl.get("meta", {})
    KEEP_SCORE_THRESHOLD = T.get("keep_score", 35)
    GRAY_SCORE_LOW = T.get("gray_score_low", 15)
    MEDIUM_MIN_SCORE = T.get("medium_min_score", 15)

    # 编译正/负信号
    _pos_raw = [(item["pattern"], item["score"]) for item in _wl.get("positive", [])]
    _neg_raw = [(item["pattern"], item["score"]) for item in _wl.get("negative", [])]

    LANG_POSITIVE = [(re.compile(p, re.I), s) for p, s in _pos_raw]
    LANG_NEGATIVE = [(re.compile(p, re.I), s) for p, s in _neg_raw]

    # 黑名单 — 存 pattern 列表避免 split("|") bug
    BL_PASS2_PATTERNS = [item["pattern"] for item in _bl.get("pass2", [])]
    BL_R2_PATTERNS = [item["pattern"] for item in _bl.get("r2", [])]
    _compile_pass2()
    _compile_r2()

    # 强语言教学信号
    STRONG_RE = re.compile(
        _wl.get("strong_lang_teaching_title_pattern", r"(?!x)x"), re.I
    )

    # 语言词典
    LANGUAGES = _ent.get("languages", [])
    LANGUAGES_SORTED = sorted(LANGUAGES, key=len, reverse=True)
    SYNONYMS = _ent.get("synonyms", {})
    WEAK_ENTITIES = set(_ent.get("weak_entities", []))

    # 频道白名单
    _wl_rx = _ent.get("channel_whitelist_regex", {})
    CHANNEL_WL_RE = (
        re.compile(_wl_rx.get("pattern", r"(?!x)x"), re.I) if _wl_rx else None
    )

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
    """向 DuckDB 连接注册所有语言教学清洗 UDF。"""
    _ensure_loaded()

    # ── blacklist_pass2 ──
    def blacklist_pass2(title, channel, keyword):
        kw_clean = _strip_keyword_tags(keyword or "")
        if CHANNEL_WL_RE and CHANNEL_WL_RE.search(
            f"{title or ''} {channel or ''} {kw_clean or ''}"
        ):
            return ""
        text = f"{title or ''} {channel or ''} {keyword or ''}"
        m = BL_PASS2_RE.search(text)
        return m.group(0) if m else ""

    def blacklist_r2(title, channel, keyword):
        kw_clean = _strip_keyword_tags(keyword or "")
        if CHANNEL_WL_RE and CHANNEL_WL_RE.search(
            f"{title or ''} {channel or ''} {kw_clean or ''}"
        ):
            return ""
        text = f"{title or ''} {channel or ''} {keyword or ''}"
        m = BL_R2_RE.search(text)
        return m.group(0) if m else ""

    conn.create_function(
        "blacklist_pass2", blacklist_pass2, ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR"
    )
    conn.create_function(
        "blacklist_r2", blacklist_r2, ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR"
    )

    # ── strong_lang_signal ──
    def strong_lang_signal(title, channel):
        return bool(STRONG_RE.search(f"{title or ''} {channel or ''}"))

    conn.create_function(
        "strong_lang_signal", strong_lang_signal, ["VARCHAR", "VARCHAR"], "BOOLEAN"
    )

    # ── lang_teaching_score ──
    def _strip_keyword_tags(kw):
        """剥离 keyword 中 - 开头的标签 token。"""
        if not kw:
            return kw
        kw = kw.strip().strip('"').lower()
        parts = re.split(r"\s+-\s*", kw)
        return parts[0] if parts else kw

    def lang_teaching_score(title, channel, keyword):
        kw_clean = _strip_keyword_tags(keyword)
        text = f"{title or ''} {channel or ''} {kw_clean or ''}"
        score = 0
        for pat, pts in LANG_POSITIVE:
            if pat.search(text):
                score += pts
        for pat, pts in LANG_NEGATIVE:
            if pat.search(text):
                score += pts
        # 语言实体对齐惩罚: keyword 有语言实体但没出现在 title 中 → -25
        entities_str = parse_lang_entities(keyword)
        if entities_str:
            entities = entities_str.split("|")
            title_lower = (title or "").lower()
            aligned = any(e in title_lower for e in entities)
            if not aligned:
                for ent in entities:
                    for key, syns in SYNONYMS.items():
                        if key in ent or ent in key:
                            for s in syns:
                                if s in title_lower:
                                    aligned = True
                                    break
                        if aligned:
                            break
                    if aligned:
                        break
            if not aligned and not STRONG_RE.search(title or ""):
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
        for term in LANGUAGES_SORTED:
            if term in core:
                entities.append(term)
        if not entities:
            words = re.findall(r"[a-z一-鿿぀-ゟ゠-ヿ가-힯]{3,}", core)
            # 教学语境词
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
            for key, syns in SYNONYMS.items():
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
