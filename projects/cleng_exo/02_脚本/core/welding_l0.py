"""电焊视觉 QC L0：标题/时长规则预判，跳过 storyboard + API。"""

from __future__ import annotations

import re

from core.regex_patterns import (
    ANIME_PATTERNS,
    DRAMA_PATTERNS,
    FAN_IDOL_PATTERNS,
    LECTURE_PATTERNS,
    LIVE_POSTER_PATTERNS,
    MUSIC_PATTERNS,
    NEWS_PATTERNS,
    PLATFORM_PATTERNS,
    SPORTS_PATTERNS,
    VARIETY_PATTERNS,
)

# 明显非焊接教学频道（零成本跳过）
CHANNEL_BLACKLIST = frozenset({
    "ABS-CBN Entertainment", "TEDx Talks", "FailArmy", "WWE",
    "The Tonight Show Starring Jimmy Fallon", "CookieSwirlC",
})

MIN_DURATION_SEC = 60
MAX_DURATION_SEC = 6 * 3600

_L0_RULES: list[tuple[str, list[str]]] = [
    ("anime", ANIME_PATTERNS),
    ("music", MUSIC_PATTERNS),
    ("platform", PLATFORM_PATTERNS),
    ("variety", VARIETY_PATTERNS),
    ("drama", DRAMA_PATTERNS),
    ("news", NEWS_PATTERNS),
    ("sports", SPORTS_PATTERNS),
    ("lecture", LECTURE_PATTERNS),
    ("live_poster", LIVE_POSTER_PATTERNS),
    ("fan_idol", FAN_IDOL_PATTERNS),
]

_WELDING_TITLE = re.compile(
    r"\b(weld(?:ing)?|mig|tig|stick|electrode|solder|焊|焊接)\b",
    re.IGNORECASE,
)


def _parse_duration_sec(duration_str: str) -> float:
    s = (duration_str or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def welding_l0_prefilter(
    title: str = "",
    channel: str = "",
    duration_str: str = "",
) -> str | None:
    """
    L0 预判：返回 None 表示需走 vision；否则返回 reason（如 l0:title:music）。
    命中则直接判 F，不写 ERROR。
    """
    ch = (channel or "").strip()
    if ch in CHANNEL_BLACKLIST:
        return "l0:channel_blacklist"

    dur = _parse_duration_sec(duration_str)
    if dur > 0 and dur < MIN_DURATION_SEC:
        return f"l0:too_short:{int(dur)}s"
    if dur > 0 and dur > MAX_DURATION_SEC:
        return f"l0:too_long:{int(dur)}s"

    t = (title or "").lower()
    if _WELDING_TITLE.search(title or ""):
        return None

    for category, patterns in _L0_RULES:
        for pat in patterns:
            if re.search(pat, t, re.IGNORECASE):
                return f"l0:title:{category}"

    return None
