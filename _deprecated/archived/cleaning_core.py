#!/usr/bin/env python3
"""
cleaning_core.py — 清洗核心模块 v2

从 blacklist.toml / whitelist.toml / entities.toml 加载规则，
暴露评分、黑名单、实体解析等函数，内置命中统计。

供 clean_sports_chunk_v2.py 和 clean_sports_chunk_refine.py 共用。
"""

import re, sys, json, time
from collections import defaultdict
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("[ERROR] 需要 tomli: pip install tomli")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 规则加载
# ══════════════════════════════════════════════════════════════

def _resolve(path: str) -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here / 'rules' / path, here / path]:
        if candidate.exists():
            return candidate
    return Path(path)

_rules_dir = Path(__file__).resolve().parent

_bl = tomllib.load(open(_resolve('blacklist.toml'), 'rb'))
_wl = tomllib.load(open(_resolve('whitelist.toml'), 'rb'))
_ent = tomllib.load(open(_resolve('entities.toml'), 'rb'))


# ── 阈值 ──
T = _wl['meta']
KEEP_SCORE_THRESHOLD  = T.get('keep_score', 35)
GRAY_SCORE_LOW        = T.get('gray_score_low', 15)
MEDIUM_MIN_SCORE      = T.get('medium_min_score', 15)
PLAYLIST_MIN_SAMPLES  = T.get('playlist_min_samples', 5)
PLAYLIST_MIN_HIT_RATE = T.get('playlist_min_hit_rate', 0.10)

# ── 黑名单 ──
BLACKLIST_PASS2 = [(item['pattern'], item.get('category', '')) for item in _bl['pass2']]
BLACKLIST_PASS2_RE = re.compile("|".join(p for p, _ in BLACKLIST_PASS2), re.I)

BLACKLIST_R2 = [(item['pattern'], item.get('category', '')) for item in _bl['r2']]
BLACKLIST_R2_RE = re.compile("|".join(p for p, _ in BLACKLIST_R2), re.I)

# ── 计分 ──
SPORT_POSITIVE = [
    (re.compile(item['pattern'], re.I), item['score'], item.get('category', ''))
    for item in _wl['positive']
]
SPORT_NEGATIVE = [
    (re.compile(item['pattern'], re.I), item['score'], item.get('category', ''))
    for item in _wl['negative']
]

# ── 强体育标题 ──
STRONG_SPORT_TITLE_RE = re.compile(_wl['strong_sport_title_pattern'], re.I)

# ── 词典 ──
SPORT_LEXICON = _ent['sports']
SPORT_LEXICON_SORTED = sorted(SPORT_LEXICON, key=len, reverse=True)
SPORT_SYNONYMS = _ent['synonyms']
WEAK_ENTITY_WORDS = set(_ent.get('weak_entities', []))

# ── 精炼上下文 ──
WEAK_TITLE_RE = re.compile(
    "|".join(item['pattern'] for item in _wl.get('weak_titles', [])), re.I
)
WEAK_CHANNEL_RE = re.compile(
    "|".join(item['pattern'] for item in _wl.get('weak_channels', [])), re.I
)


# ══════════════════════════════════════════════════════════════
# 命中统计
# ══════════════════════════════════════════════════════════════

class HitTracker:
    """线程安全？单线程够用。记录每条规则的命中次数和样本。"""
    def __init__(self):
        self.blacklist_hits: dict[str, int] = defaultdict(int)
        self.positive_hits: dict[str, int] = defaultdict(int)
        self.negative_hits: dict[str, int] = defaultdict(int)
        self.entity_hits: dict[str, int] = defaultdict(int)
        self.r2_blacklist_hits: dict[str, int] = defaultdict(int)
        self.r2_context_drops: int = 0
        self.r2_weak_entity_drops: int = 0
        self.total_rows: int = 0
        self._bl_idx: dict[str, int] = {}  # pattern -> compiled group index
        self._pos_idx: dict[str, int] = {}
        self._neg_idx: dict[str, int] = {}

    def record_blacklist(self, match_text: str):
        self.blacklist_hits[match_text[:80]] += 1

    def record_positive(self, pattern: str):
        self.positive_hits[pattern[:80]] += 1

    def record_negative(self, pattern: str):
        self.negative_hits[pattern[:80]] += 1

    def record_entity(self, entity: str):
        self.entity_hits[entity] += 1

    def record_r2_blacklist(self, match_text: str):
        self.r2_blacklist_hits[match_text[:80]] += 1

    def record_r2_context(self):
        self.r2_context_drops += 1

    def record_r2_weak_entity(self):
        self.r2_weak_entity_drops += 1

    def export_hits(self, output_dir: Path, stem: str):
        """导出 rule_hits.csv 和 entity_hits.csv"""
        import csv
        # rule_hits.csv
        with open(output_dir / f"{stem}_rule_hits.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rule_type", "hit_count", "sample_match"])
            for k, v in sorted(self.blacklist_hits.items(), key=lambda x: -x[1]):
                w.writerow(["blacklist_pass2", v, k])
            for k, v in sorted(self.positive_hits.items(), key=lambda x: -x[1]):
                w.writerow(["positive", v, k])
            for k, v in sorted(self.negative_hits.items(), key=lambda x: -x[1]):
                w.writerow(["negative", v, k])
            for k, v in sorted(self.r2_blacklist_hits.items(), key=lambda x: -x[1]):
                w.writerow(["blacklist_r2", v, k])
            w.writerow(["r2_context", self.r2_context_drops, "weak_title + weak_channel"])
            w.writerow(["r2_weak_entity", self.r2_weak_entity_drops, "all generic entities"])
        # entity_hits.csv
        with open(output_dir / f"{stem}_entity_hits.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entity", "hit_count"])
            for k, v in sorted(self.entity_hits.items(), key=lambda x: -x[1]):
                w.writerow([k, v])

    def export_samples(self, df, output_dir: Path, stem: str, n: int = 200):
        """导出 keep_sample.csv 和 drop_sample.csv"""
        import pandas as pd
        keep = df[df["clean_label"] == "keep"]
        drop = df[df["clean_label"] == "drop"]
        keep_n = min(n, len(keep))
        drop_n = min(n, len(drop))
        if keep_n > 0:
            keep.sample(keep_n, random_state=42).to_csv(
                output_dir / f"{stem}_keep_sample.csv", index=False, encoding="utf-8-sig"
            )
        if drop_n > 0:
            drop.sample(drop_n, random_state=42).to_csv(
                output_dir / f"{stem}_drop_sample.csv", index=False, encoding="utf-8-sig"
            )


_hits = HitTracker()

def get_hits() -> HitTracker:
    return _hits

def reset_hits():
    global _hits
    _hits = HitTracker()

def export_hits(output_dir, stem):
    _hits.export_hits(output_dir, stem)

def export_samples(df, output_dir, stem, n=200):
    _hits.export_samples(df, output_dir, stem, n)


# ══════════════════════════════════════════════════════════════
# 核心函数（带命中统计）
# ══════════════════════════════════════════════════════════════

def parse_keyword_entities(keyword: str, track: bool = True) -> list[str]:
    if not keyword or not isinstance(keyword, str):
        return []
    kw = keyword.strip().strip('"').lower()
    parts = re.split(r"\s+-\s*", kw)
    core = parts[0] if parts else kw
    entities = []
    for term in SPORT_LEXICON_SORTED:
        if term in core:
            entities.append(term)
    if not entities:
        words = re.findall(r"[a-z]{4,}", core)
        stop = {
            "full","match","video","race","final","live","stream","commentary",
            "broadcast","tournament","championship","league","contest","open",
            "professional","amateur","national","international","world","replay",
            "footage","unedited","ranked","season","break","career","highlights",
            "historical","veteran","rookie","legendary","masters","diamond",
        }
        entities = [w for w in words if w not in stop][:3]
    if track:
        for e in entities:
            _hits.record_entity(e)
    return entities


def get_alignment_terms(entities: list[str]) -> set[str]:
    terms = set()
    for e in entities:
        terms.add(e)
        for key, syns in SPORT_SYNONYMS.items():
            if key in e or e in key:
                terms.update(syns)
    return terms


def keyword_title_aligned(keyword: str, title: str, channel: str) -> tuple[bool, list[str]]:
    entities = parse_keyword_entities(keyword, track=False)
    if not entities:
        return True, entities
    terms = get_alignment_terms(entities)
    text = f"{title or ''} {channel or ''}".lower()
    matched = [t for t in terms if t in text]
    return len(matched) > 0, entities


def strong_sport_signal(title: str, channel: str) -> bool:
    return bool(STRONG_SPORT_TITLE_RE.search(f"{title or ''} {channel or ''}"))


def blacklist_hit_pass2(title: str, channel: str) -> str | None:
    m = BLACKLIST_PASS2_RE.search(f"{title or ''} {channel or ''}")
    if m:
        _hits.record_blacklist(m.group(0))
        return m.group(0)
    return None


def blacklist_hit_r2(title: str, channel: str) -> str | None:
    m = BLACKLIST_R2_RE.search(f"{title or ''} {channel or ''}")
    if m:
        _hits.record_r2_blacklist(m.group(0))
        return m.group(0)
    return None


def sport_score(title: str, channel: str, keyword: str) -> int:
    text = f"{title or ''} {channel or ''} {keyword or ''}"
    score = 0
    for pat, pts, _ in SPORT_POSITIVE:
        if pat.search(text):
            _hits.record_positive(pat.pattern)
            score += pts
    for pat, pts, _ in SPORT_NEGATIVE:
        if pat.search(text):
            _hits.record_negative(pat.pattern)
            score += pts
    return score


def refine_row(row: dict) -> tuple[str, str]:
    title   = str(row.get("title") or "")
    channel = str(row.get("channel") or "")
    kw_entities_str = str(row.get("kw_entities") or "")

    m = BLACKLIST_R2_RE.search(f"{title} {channel}")
    if m:
        _hits.record_r2_blacklist(m.group(0))
        return ("drop", f"r2_blacklist:{m.group(0)[:50]}")

    weak_title   = bool(WEAK_TITLE_RE.search(title)) if title else True
    weak_channel = bool(WEAK_CHANNEL_RE.search(channel)) if channel else True
    if weak_title and weak_channel:
        _hits.record_r2_context()
        return ("drop", "r2_context:weak_title_and_channel")

    if kw_entities_str:
        entities = [e.strip() for e in kw_entities_str.split("|") if e.strip()]
        strong = [e for e in entities if e.lower() not in WEAK_ENTITY_WORDS]
        if not strong and entities:
            _hits.record_r2_weak_entity()
            return ("drop", f"r2_weak_entity:all_generic:{','.join(entities[:3])}")

    return ("keep", "")


def decide_keep(sc: int, aligned: bool, strong: bool, medium_recovered: bool) -> tuple[bool, str]:
    if sc >= KEEP_SCORE_THRESHOLD:
        return True, "high_score"
    if aligned and sc >= GRAY_SCORE_LOW:
        return True, "gray_aligned"
    if medium_recovered and strong and sc >= MEDIUM_MIN_SCORE:
        return True, "medium_strong_signal"
    if not aligned and not strong and sc < GRAY_SCORE_LOW:
        return False, "low_score_no_signal"
    if aligned and sc < GRAY_SCORE_LOW:
        return False, "aligned_low_score"
    if medium_recovered and not strong:
        return False, "medium_no_strong_signal"
    if medium_recovered and sc < MEDIUM_MIN_SCORE:
        return False, "medium_low_score"
    return False, "default_drop"


# ══════════════════════════════════════════════════════════════
# 报告生成
# ══════════════════════════════════════════════════════════════

def generate_report(summary: dict, output_dir: Path, stem: str):
    """生成 pipeline_report.md"""
    lines = []
    lines.append(f"# Pipeline Report — {stem}")
    lines.append(f"")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Version:** {summary.get('version', 'N/A')}")
    lines.append(f"")
    lines.append(f"## Overview")
    lines.append(f"")
    total = summary.get('total_in', 0)
    keep = summary.get('total_keep', 0)
    drop = summary.get('total_drop', 0)
    keep_high = summary.get('total_keep_high', 0)
    keep_medium = summary.get('total_keep_medium', 0)
    pct = keep / max(total, 1) * 100
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total input | {total:,} |")
    lines.append(f"| Keep | {keep:,} ({pct:.1f}%) |")
    lines.append(f"| — high | {keep_high:,} |")
    lines.append(f"| — medium | {keep_medium:,} |")
    lines.append(f"| Drop | {drop:,} |")
    lines.append(f"")
    
    steps = summary.get('steps', {})
    if steps:
        lines.append(f"## Drop Breakdown")
        lines.append(f"")
        lines.append(f"| Step | Count |")
        lines.append(f"|------|-------|")
        for step, info in steps.items():
            if isinstance(info, dict):
                lines.append(f"| {step} | {info.get('dropped', info.get('kept', 0)):,} |")
        lines.append(f"")

    # Rule hits
    h = _hits
    lines.append(f"## Rule Hit Statistics")
    lines.append(f"")
    lines.append(f"### Blacklist Pass2 (Top 20)")
    lines.append(f"")
    lines.append(f"| Hits | Match |")
    lines.append(f"|------|-------|")
    for k, v in sorted(h.blacklist_hits.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"| {v:,} | `{k[:60]}` |")
    lines.append(f"")

    lines.append(f"### Positive Scoring (Top 20)")
    lines.append(f"")
    lines.append(f"| Hits | Pattern |")
    lines.append(f"|------|---------|")
    for k, v in sorted(h.positive_hits.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"| {v:,} | `{k[:60]}` |")
    lines.append(f"")

    lines.append(f"### Negative Scoring")
    lines.append(f"")
    lines.append(f"| Hits | Pattern |")
    lines.append(f"|------|---------|")
    for k, v in sorted(h.negative_hits.items(), key=lambda x: -x[1]):
        lines.append(f"| {v:,} | `{k[:60]}` |")
    lines.append(f"")

    lines.append(f"### Entity Hits (Top 20)")
    lines.append(f"")
    lines.append(f"| Hits | Entity |")
    lines.append(f"|------|--------|")
    for k, v in sorted(h.entity_hits.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"| {v:,} | `{k}` |")
    lines.append(f"")

    report_path = output_dir / f"{stem}_pipeline_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report: {report_path}")


# ── 保留旧接口兼容 ──
def _load_rules():
    return {}
_rules = {}
