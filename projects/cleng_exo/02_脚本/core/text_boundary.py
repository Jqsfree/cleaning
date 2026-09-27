#!/usr/bin/env python3
"""文本过滤边界判定（certain-noise only）。

对齐 exo_service machine_0813 停点：
  - 可写规则: 累计标注 ≥min_f 条 F，且 0 T / 0 U（ERROR 不计）
  - 相对现有 blacklist 仍有 ≥min_f 条增量 F
  - 文本边界: 同一 keep 上连续两份独立新鲜样本，可写规则数均为 0

不把 LLM-QC F% / keep% 当停点或交付 KPI。U 不自动丢。
特征只用 title+channel，不用采集 keyword。
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.rules_loader import load_blacklist_individual

LABEL_COL = "qc_text_result"
MIN_F_DEFAULT = 3
CONSECUTIVE_DEFAULT = 2

# 跨品类常见确定噪声（提案库；必须过样本闸门才可写入 blacklist）
GENERIC_DROP_CANDIDATES: list[tuple[str, str]] = [
    ("gameplay", r"(?i)\b(gameplay|minecraft|roblox|fortnite|lets?\s*play|valorant|gta\s*[45v]?)\b"),
    ("official_mv_trailer", r"(?i)(official\s*(music\s*)?(video|audio|mv)|official\s*trailer|lyric\s*video|teaser\s*trailer)"),
    ("podcast_interview", r"(?i)(\bpodcast\b|\binterview\b|talk\s*show|late\s*night)"),
    ("anime_cartoon_kids", r"(?i)(\banime\b|\bcartoon\b|nursery\s*rhyme|peppa\s*pig|cocomelon|baby\s*shark|paw\s*patrol)"),
    ("ted_webinar_course", r"(?i)(\btedx?\b|ted\s*talk|\bwebinar\b|\bonline\s*course\b|\bmasterclass\b)"),
    ("news_hard", r"(?i)(\bbreaking\s*news\b|\bbbc\s*news\b|\babc\s*news\b|\bcnn\b\s*news)"),
    ("mukbang_recipe", r"(?i)(\bmukbang\b|how\s*to\s*cook|\brecipe\b|#cooking\b)"),
    ("asmr_eat_drink", r"(?i)(\basmr\b.{0,40}(mukbang|eating|drink)|eating\s*show)"),
    ("full_match_sport", r"(?i)(full\s*match|match\s*highlights|\bnba\b|\bnfl\b|\bufc\b)"),
    ("how_to_tutorial", r"(?i)(\bhow\s*to\b|\btutorial\b|step\s*by\s*step|beginner'?s?\s*guide)"),
]

_LATIN = re.compile(r"[a-z0-9']{4,}", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]{2,}")
_HANGUL = re.compile(r"[\uac00-\ud7af]{2,}")
_STOP = {
    "this", "that", "with", "from", "your", "have", "what", "when", "were",
    "they", "them", "will", "just", "about", "into", "over", "after", "been",
    "video", "official", "channel", "watch", "youtube", "https", "http",
    "full", "live", "best", "make", "made", "like", "love", "real", "time",
    "part", "episode", "season", "first", "last", "next", "daily", "vlog",
    "music", "power", "world", "review", "good", "play", "free", "hard",
    "night", "show", "beat", "beats", "trap", "cake", "horse", "cook",
    "business", "birthday", "bakery", "drawing", "beginners", "great",
    "life", "home", "kids", "baby", "family", "workout", "training",
}


@dataclass
class PatternScore:
    name: str
    pattern: str
    n_f: int
    n_t: int
    n_u: int
    n_error: int
    incremental_f: int
    examples: list[str] = field(default_factory=list)

    @property
    def addable(self) -> bool:
        return self.n_t == 0 and self.n_u == 0 and self.incremental_f >= MIN_F_DEFAULT

    def as_dict(self) -> dict:
        d = asdict(self)
        d["addable"] = self.addable
        return d


def row_text(row: dict) -> str:
    return f"{row.get('title') or ''} {row.get('channel') or ''}"


def label_of(row: dict) -> str:
    return str(row.get(LABEL_COL) or "").strip().upper()


def load_qc_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                vid = str(row.get("video_id") or "")
                key = vid or f"{row_text(row)}|{label_of(row)}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def count_labels(rows: list[dict]) -> dict[str, int]:
    c = Counter(label_of(r) for r in rows)
    return {
        "rows": len(rows),
        "T": int(c.get("T", 0)),
        "F": int(c.get("F", 0)),
        "U": int(c.get("U", 0)),
        "ERROR": int(c.get("ERROR", 0)),
    }


def compile_blacklist(rules_dir: Path | None) -> list[re.Pattern[str]]:
    if rules_dir is None or not (rules_dir / "blacklist.toml").exists():
        return []
    items = load_blacklist_individual(rules_dir)
    pats: list[re.Pattern[str]] = []
    for sec in ("title_pass2", "title_r3", "channel_pass2", "pass2", "r2"):
        for item in items.get(sec, []):
            pat = item.get("pattern") or ""
            if not pat:
                continue
            try:
                pats.append(re.compile(pat))
            except re.error:
                continue
    return pats


def blacklist_hit(text: str, pats: list[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in pats)


def score_pattern(
    rows: list[dict],
    name: str,
    pattern: str,
    existing: list[re.Pattern[str]] | None = None,
    *,
    min_f: int = MIN_F_DEFAULT,
    n_examples: int = 3,
) -> PatternScore:
    try:
        rx = re.compile(pattern)
    except re.error:
        return PatternScore(name, pattern, 0, 0, 0, 0, 0)
    existing = existing or []
    n_f = n_t = n_u = n_e = inc = 0
    examples: list[str] = []
    for row in rows:
        lab = label_of(row)
        if lab not in {"T", "F", "U", "ERROR"}:
            continue
        text = row_text(row)
        if not rx.search(text):
            continue
        if lab == "T":
            n_t += 1
        elif lab == "F":
            n_f += 1
            if not blacklist_hit(text, existing):
                inc += 1
                if len(examples) < n_examples:
                    examples.append((row.get("title") or "")[:120])
        elif lab == "U":
            n_u += 1
        else:
            n_e += 1
    return PatternScore(name, pattern, n_f, n_t, n_u, n_e, inc, examples)


def _tokens(text: str) -> list[str]:
    low = text.lower()
    out: list[str] = []
    for m in _LATIN.findall(low):
        tok = m.strip("'")
        if tok and tok not in _STOP:
            out.append(tok)
    out.extend(_CJK.findall(text))
    out.extend(_HANGUL.findall(text))
    return out


def mine_ngrams(
    rows: list[dict],
    existing: list[re.Pattern[str]],
    *,
    min_f: int = MIN_F_DEFAULT,
    max_candidates: int = 40,
) -> list[PatternScore]:
    """从 F 标题挖 unigram/bigram；仅保留过闸门的。"""
    f_rows = [r for r in rows if label_of(r) == "F"]
    gram_docs: dict[str, list[str]] = {}
    for row in f_rows:
        text = row_text(row)
        if blacklist_hit(text, existing):
            continue
        toks = _tokens(text)
        grams: set[str] = set()
        for tok in toks:
            # 拉丁 unigram 太容易误杀 keep；只挖 CJK/韩文词，或长度≥10 的专名
            if re.fullmatch(r"[a-z0-9']+", tok, re.I):
                if len(tok) >= 10:
                    grams.add(tok)
            else:
                grams.add(tok)
        grams.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
        for g in grams:
            gram_docs.setdefault(g, []).append(text)

    scored: list[PatternScore] = []
    for gram, _docs in gram_docs.items():
        if " " in gram:
            pat = r"(?i)" + r"\s+".join(re.escape(p) for p in gram.split())
            name = "ngram_" + gram.replace(" ", "_")[:40]
        else:
            if re.fullmatch(r"[a-z0-9']+", gram, re.I):
                pat = r"(?i)\b" + re.escape(gram) + r"\b"
            else:
                pat = re.escape(gram)
            name = "term_" + gram[:40]
        s = score_pattern(rows, name, pat, existing, min_f=min_f)
        if s.n_t == 0 and s.n_u == 0 and s.incremental_f >= min_f:
            scored.append(s)
    scored.sort(key=lambda x: (-x.incremental_f, -x.n_f, x.name))
    return scored[:max_candidates]


def propose_rules(
    rows: list[dict],
    rules_dir: Path | None,
    *,
    min_f: int = MIN_F_DEFAULT,
    extra_candidates: list[tuple[str, str]] | None = None,
    mine: bool = True,
    skip_names: set[str] | None = None,
) -> list[PatternScore]:
    existing = compile_blacklist(rules_dir)
    skip = skip_names or set()
    cands = list(GENERIC_DROP_CANDIDATES)
    if extra_candidates:
        cands.extend(extra_candidates)
    out: list[PatternScore] = []
    seen_pat: set[str] = set()
    for name, pat in cands:
        if name in skip:
            continue
        s = score_pattern(rows, name, pat, existing, min_f=min_f)
        if s.n_t == 0 and s.n_u == 0 and s.incremental_f >= min_f:
            out.append(s)
            seen_pat.add(s.pattern)
    if mine:
        for s in mine_ngrams(rows, existing, min_f=min_f):
            if s.name in skip or s.pattern in seen_pat:
                continue
            out.append(s)
            seen_pat.add(s.pattern)
    out.sort(key=lambda x: (-x.incremental_f, -x.n_f, x.name))
    return out


def residual_f_unmatched(rows: list[dict], rules_dir: Path | None) -> dict[str, int]:
    pats = compile_blacklist(rules_dir)
    f_rows = [r for r in rows if label_of(r) == "F"]
    unmatched = [r for r in f_rows if not blacklist_hit(row_text(r), pats)]
    return {
        "f_total": len(f_rows),
        "f_matched": len(f_rows) - len(unmatched),
        "f_unmatched": len(unmatched),
    }


def evaluate_sample(
    rows: list[dict],
    rules_dir: Path | None,
    *,
    min_f: int = MIN_F_DEFAULT,
    mine: bool = True,
    skip_names: set[str] | None = None,
) -> dict:
    labels = count_labels(rows)
    proposed = propose_rules(
        rows, rules_dir, min_f=min_f, mine=mine, skip_names=skip_names
    )
    residual = residual_f_unmatched(rows, rules_dir)
    return {
        "labels": labels,
        "n_addable": len(proposed),
        "addable": [p.as_dict() for p in proposed],
        "residual_f": residual,
        "mine": mine,
        "skip_names": sorted(skip_names or []),
    }


def declare_boundary(
    fresh_evals: list[dict],
    *,
    consecutive: int = CONSECUTIVE_DEFAULT,
) -> dict:
    """fresh_evals: 按时间顺序的独立新鲜样本评估（已含各自 n_addable）。"""
    zeros = 0
    for ev in reversed(fresh_evals):
        if int(ev.get("n_addable") or 0) == 0:
            zeros += 1
        else:
            break
    reached = zeros >= consecutive and consecutive > 0
    status = "text_boundary" if reached else "not_boundary"
    reason = []
    if reached:
        reason.append(f"连续 {zeros} 份独立新鲜样本可写规则=0（门限 {consecutive}）")
    else:
        reason.append(
            f"最近连续零提案样本 {zeros}/{consecutive}；"
            f"最新 n_addable={fresh_evals[-1]['n_addable'] if fresh_evals else 'n/a'}"
        )
    return {
        "status": status,
        "consecutive_zero": zeros,
        "required_consecutive": consecutive,
        "reason": reason,
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def proposed_to_toml(proposed: list[PatternScore] | list[dict]) -> str:
    lines = ["# eval_text_boundary 提案（过闸门后人工确认再合并）", ""]
    for item in proposed:
        d = item.as_dict() if isinstance(item, PatternScore) else item
        cat = str(d.get("name") or "term").replace('"', "")
        pat = str(d.get("pattern") or "").replace("\\", "\\\\").replace('"', '\\"')
        lines.append("[[pass2]]")
        lines.append(f'category = "{cat}"')
        lines.append(f'pattern = "{pat}"')
        lines.append(
            f"# F={d.get('n_f')} T={d.get('n_t')} U={d.get('n_u')} "
            f"incremental_f={d.get('incremental_f')}"
        )
        lines.append("")
    return "\n".join(lines)
