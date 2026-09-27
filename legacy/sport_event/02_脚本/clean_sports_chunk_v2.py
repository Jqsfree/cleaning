#!/usr/bin/env python3
"""
v2 清洗管道 — 多 chunk 通用版本。

流程:
  Pass 1: playlist 命中率分析
  Pass 2: 四步规则清洗 → high / medium / drop 三分

用法:
  python3 clean_sports_chunk_v2.py <input.csv>
  python3 clean_sports_chunk_v2.py ../data/sports_chunk_01.csv
"""

import csv, json, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = "v2"
OUTPUT_DIR = Path("output")

# ── 配置 ────────────────────────────────────────────────
PLAYLIST_MIN_SAMPLES = 5
PLAYLIST_MIN_HIT_RATE = 0.10
KEEP_SCORE_THRESHOLD = 35
GRAY_SCORE_LOW = 15
MEDIUM_MIN_SCORE = 15

# ── 黑名单 ──────────────────────────────────────────────
BLACKLIST_PATTERNS = [
    r"\b(fmv|fan\s*cam|fanmeet|fan\s*meet)\b",
    r"\b(k[\s-]?pop|kpop|snsd|taeyeon|taeny|bts\b|blackpink|twice\b|exo\b|nct\b|stray\s*kids)\b",
    r"\b(minecraft|roblox|fortnite|gameplay|let'?s\s*play|walkthrough|mod\s*apk)\b",
    r"\b(peppa\s*pig|cartoon|anime\b|nursery\s*rhyme|kids\s*channel)\b",
    r"\b(worship|sermon|prayer|bible|gospel|pastor|church\s*service)\b",
    r"\b(recipe|cooking|mukbang|asmr\b|eat\s*with\s*me)\b",
    r"\b(unboxing|haul\b|try\s*on|makeup\s*tutorial)\b",
    r"\b(prank|vlog\b|daily\s*vlog|mukbang)\b",
    r"\b(podcast|talk\s*show|stand\s*up\s*comedy|comedy\s*special)\b",
    r"\b(netflix|love\s*is\s*blind|movieclips|official\s*trailer|teaser\s*trailer)\b",
    r"\b(graduation|wedding\s*recap|birthday\s*party)\b",
    r"\b(how\s*to\b|tutorial\b|training\s*tips|workout\s*routine)\b",
    r"\b(reaction\s*video|reacts?\s*to)\b",
    r"\b(lofi|chill\s*beats|music\s*video|official\s*audio|lyrics\s*video)\b",
    r"\b(naruto|one\s*piece|dragon\s*ball|demon\s*slayer)\b",
    r"\b(la\s*la\s*land|suicide\s*squad|movie\s*scene)\b",
    # ── QC 2026-05-31 扩展 ──
    r"\b(wwe|smackdown|wwe\s*raw|wrestlemania)\b",
    r"\b(john\s*cena|undertaker)\b",
    r"\b(subway\s*surfers|street\s*fighter|mario\s*kart|starcraft|free\s*fire|garena)\b",
    r"\b(call\s*of\s*duty|\bcod\b|black\s*ops|\bmlg\b)\b",
    r"\b(pga\s*tour\s*2k|pro\s*yakyuu\s*spirits|alpine\s*ski\s*racing\s*20\d|nba\s*2k|nhl\s*2k|madden\s*nfl|full\s*game\s*movie)\b",
    r"\b(lol\s*esports|blink\s*esports|gaming\s*mason|all\s*nintendo\s*music|esports\s*vod)\b",
    r"\b(rap\s*battle|url\s*rap|chess\s*vs|stickman\s*vs|stick\s*figure\s*vs)\b",
    r"\b(drum\s*battle|percussion\s*vs|food\s*vs|lego\s*vs|animal\s*vs|caucasian\s*shepherd\s*vs|wolf\s*vs|20\s*women\s*vs)\b",
    r"\b(drag\s*race.*(rupaul|crave|tongue\s*pop))\b",
    r"\b(ford\s*explorer\s*vs\s*dodge|gtr\s*vs\s*z06|suv\s*drag\s*race|car\s*vs\s*car)\b",
    r"\b(snowboard\s*ski\s*goggles\s*comparison|helmet\s*vs|goggle\s*vs|gear\s*comparison)\b",
    r"\b(surf\s*photography|wave\s*pool\s*review|surf\s*skate|old\s*and\s*rare\s*footage|historical\s*footage|50\s*years\s*of)\b",
    r"\b(chairlift\s*goodness|steilsten\s*skipisten|deadliest\s*waves)\b",
]
BLACKLIST_RE = re.compile("|".join(BLACKLIST_PATTERNS), re.I)

# ── 体育正信号 ──────────────────────────────────────────
SPORT_POSITIVE = [
    (r"\b(full\s*)?(match|game|race|replay|coverage)\b", 25),
    (r"\b(highlights?|extended\s*highlights)\b", 20),
    (r"\b(live\s*stream|live\s*broadcast|\blive\b)\b", 18),
    (r"\b(commentary|play[\s-]?by[\s-]?play|broadcast)\b", 22),
    (r"\b(final|semi[\s-]?final|quarter[\s-]?final)\b", 20),
    (r"\b(championship|tournament|cup\b|league\b|open\b)\b", 15),
    (r"\b(olympic|asian\s*games|world\s*cup|diamond\s*league)\b", 20),
    (r"\b(vs\.?|\svs\s|versus)\b", 18),
    (r"\b(round\s*\d|set\s*\d|period\s*\d|heat\s*\d|final\s*[a-z])\b", 12),
    (r"\b(score|goal|touchdown|home\s*run|knockout|bout)\b", 12),
    (r"\b(regatta|rowing|surfing|pickleball|skiing|athletics|fencing|taekwondo|sanda|snooker)\b", 15),
    (r"\b(nba|nfl|mlb|nhl|fifa|atp|wta|wsl|ppa\b|ncaa)\b", 18),
    (r"\b(recap|highlights?\s*reel)\b", 10),
    (r"\b(shootout|overtime|extra\s*time|penalty\s*shoot)\b", 15),
    (r"\b(squash|indoor\s*bowls|\bbowls\b)\b", 15),
    (r"\b(quarterfinals?|bronze\s*playoff|A[-\s]?Final|B[-\s]?Final|C[-\s]?Final)\b", 12),
    (r"\b(sculls?|sweep\s*oar|coxed|coxless)\b", 10),
    (r"\b(HS\b|high\s*school)\b.*\b(boys|girls|basketball|football|volleyball|soccer|hockey|lacrosse|wrestling|baseball|softball)\b", 12),
    (r"\b(varsity|junior\s*varsity|\bjv\b)\b.*\b(match|game|final|championship|team|vs)\b", 12),
    (r"\b(boys'?|girls'?|junior)\b.*\b(basketball|football|volleyball|hockey|championship|final|match|game)\b", 10),
]
SPORT_POSITIVE_RE = [(re.compile(p, re.I), s) for p, s in SPORT_POSITIVE]

SPORT_NEGATIVE = [
    (r"\b(documentary|behind\s*the\s*scenes|interview|press\s*conference)\b", -12),
    (r"\b(training\s*session|practice\s*drill|warm[\s-]?up)\b", -10),
    (r"\b(trailer|teaser|preview)\b", -15),
    (r"\b(compilation|best\s*of|top\s*10)\b", -8),
    (r"\b(gear\s*review|equipment\s*review)\b", -12),
    (r"\b(sparring\s*session|sparring\s*&?\s*discussion)\b", -10),
    (r"\b(ceremony|closing\s*ceremony|opening\s*ceremony|weigh[\s-]?in)\b", -12),
    (r"\b(podcast|talk\s*show|sunday\s*supplement)\b", -15),
    (r"\b(drag\s*race)\b.*(?!nascar|nhra|f1\b|formula|motorsport)", -15),
]
SPORT_NEGATIVE_RE = [(re.compile(p, re.I), s) for p, s in SPORT_NEGATIVE]

# ── 强体育标题 ──────────────────────────────────────────
STRONG_SPORT_TITLE_RE = re.compile(
    r"\b(match|final|semi[\s-]?final|quarter[\s-]?final|"
    r"highlights|full\s*(match|game|race|replay|coverage)|"
    r"vs\.?|\svs\s|versus|championship|tournament|"
    r"olympic|world\s*cup|grand\s*prix|diamond\s*league|"
    r"commentary|play[\s-]?by[\s-]?play|broadcast|live\s*stream|"
    r"round\s*\d|bout|regatta|grand\s*final|shootout|overtime|"
    r"national\s*games|varsity|campeonato|"
    r"@\s*\w+|flames|canadiens|lakers|celtics)\b", re.I)

# ── 体育词典 ────────────────────────────────────────────
SPORT_LEXICON = [
    "rowing", "regatta", "boat race", "henley", "oxford cambridge",
    "surfing", "surf", "wsl", "big wave",
    "pickleball", "ppa", "app tour",
    "skiing", "alpine", "slalom", "downhill", "super-g",
    "athletics", "track and field", "high jump", "long jump", "sprint",
    "fencing", "epee", "foil", "sabre",
    "taekwondo", "tkd", "sanda", "wushu",
    "snooker", "billiards", "pool",
    "basketball", "nba", "ncaa basketball",
    "football", "soccer", "fifa", "premier league", "champions league",
    "tennis", "atp", "wta", "wimbledon", "roland garros",
    "volleyball", "beach volleyball",
    "rugby", "cricket", "baseball", "mlb",
    "hockey", "nhl", "ice hockey",
    "golf", "pga", "lpga",
    "boxing", "mma", "ufc", "wrestling",
    "cycling", "tour de france",
    "swimming", "diving", "gymnastics",
    "badminton", "table tennis", "squash",
    "asian games", "olympics", "olympic", "paralympic",
    "world cup", "diamond league", "grand prix",
    "cross country", "marathon", "triathlon",
    "motorsport", "formula 1", "f1", "motogp", "nascar",
    "lacrosse", "handball", "water polo",
    "judo", "karate", "bjj", "grappling",
    # ── QC 2026-05-31 扩展 ──
    "indoor bowls", "bowls", "lawn bowls",
    "squash", "squash tv", "glass court",
    "nine-ball", "10-ball", "8-ball", "chinese pool",
    "lacrosse", "nll", "buffalo bandits", "box lacrosse",
    "horse racing", "turfway park", "keeneland", "cheltenham",
    "darts", "pdc", "world darts",
    "floorball", "unihockey", "salibandy",
    "bowling", "pba", "tenpin",
    "weightlifting", "iwf",
    "climbing", "ifsc", "sport climbing",
    "netball", "netball world cup",
    "ultimate", "wfdf", "ultimate frisbee",
    "american football", "nfl", "cfb", "super bowl",
]
SPORT_LEXICON_SORTED = sorted(SPORT_LEXICON, key=len, reverse=True)

SPORT_SYNONYMS = {
    "rowing": ["rowing", "rower", "regatta", "scull", "crew", "boat race", "hocr"],
    "surfing": ["surfing", "surf", "wsl", "wave", "surfer"],
    "pickleball": ["pickleball", "ppa", "dink", "kitchen"],
    "skiing": ["skiing", "ski", "alpine", "slalom", "downhill", "super-g", "giant slalom"],
    "athletics": ["athletics", "track", "field", "sprint", "hurdles", "decathlon"],
    "fencing": ["fencing", "epee", "foil", "sabre", "fencer"],
    "taekwondo": ["taekwondo", "tkd", "kicking"],
    "sanda": ["sanda", "sanshou", "wushu"],
    "snooker": ["snooker", "billiards", "147"],
    "basketball": ["basketball", "nba", "dunk", "hoops"],
    "football": ["football", "soccer", "fifa", "premier league", "mls"],
    "tennis": ["tennis", "atp", "wta", "wimbledon"],
    "asian games": ["asian games", "asiad", "hangzhou", "jakarta"],
    "olympics": ["olympic", "olympics", "paris 2024", "tokyo 2020", "beijing 2022"],
    "hockey": ["hockey", "nhl", "flames", "canadiens", "bruins", "rangers"],
    "squash": ["squash", "squash tv", "glass court", "boast"],
    "lacrosse": ["lacrosse", "nll", "box lacrosse", "bandits"],
    "indoor bowls": ["indoor bowls", "bowls", "lawn bowls", "lawn green"],
    "darts": ["darts", "pdc", "bullseye"],
    "floorball": ["floorball", "unihockey", "salibandy"],
    "american football": ["nfl", "cfb", "super bowl", "touchdown"],
    "cricket": ["cricket", "ipl", "bcci", "wicket"],
}


# ══════════════════════════════════════════════════════════
def parse_keyword_entities(keyword: str) -> list[str]:
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
        stop = {"full", "match", "video", "race", "final", "live", "stream", "commentary",
                "broadcast", "tournament", "championship", "league", "contest", "open",
                "professional", "amateur", "national", "international", "world", "replay",
                "footage", "unedited", "ranked", "season", "break", "career", "highlights",
                "historical", "veteran", "rookie", "legendary", "masters", "diamond"}
        entities = [w for w in words if w not in stop][:3]
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
    entities = parse_keyword_entities(keyword)
    if not entities:
        return True, entities
    terms = get_alignment_terms(entities)
    text = f"{title or ''} {channel or ''}".lower()
    matched = [t for t in terms if t in text]
    return len(matched) > 0, entities


def strong_sport_signal(title: str, channel: str) -> bool:
    return bool(STRONG_SPORT_TITLE_RE.search(f"{title or ''} {channel or ''}"))


def blacklist_hit(title: str, channel: str) -> str | None:
    m = BLACKLIST_RE.search(f"{title or ''} {channel or ''}")
    return m.group(0) if m else None


def sport_score(title: str, channel: str, keyword: str) -> int:
    text = f"{title or ''} {channel or ''} {keyword or ''}"
    score = 0
    for pat, pts in SPORT_POSITIVE_RE:
        if pat.search(text):
            score += pts
    for pat, pts in SPORT_NEGATIVE_RE:
        if pat.search(text):
            score += pts
    return score


# ══════════════════════════════════════════════════════════
def pass1_playlist_stats(input_path: Path) -> set[str]:
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "hits": 0})
    print("Pass 1: analyzing playlist hit rates...")
    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            pref = row.get("source_ref") or ""
            if not pref:
                continue
            title = row.get("title") or ""
            channel = row.get("channel") or ""
            keyword = row.get("keyword") or ""
            sc = sport_score(title, channel, keyword)
            aligned, _ = keyword_title_aligned(keyword, title, channel)
            strong = strong_sport_signal(title, channel)
            if sc >= GRAY_SCORE_LOW or aligned or strong:
                stats[pref]["hits"] += 1
            stats[pref]["total"] += 1
            if i % 500_000 == 0:
                print(f"  scanned {i:,} rows...")
    polluted = set()
    for pref, s in stats.items():
        if s["total"] >= PLAYLIST_MIN_SAMPLES and s["hits"] / s["total"] < PLAYLIST_MIN_HIT_RATE:
            polluted.add(pref)
    print(f"  playlists: {len(stats):,}, polluted: {len(polluted):,}")
    return polluted


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


# ══════════════════════════════════════════════════════════
def pass2_clean(input_path: Path, polluted: set, stem: str) -> dict:
    out_high = OUTPUT_DIR / f"{stem}_clean_{VERSION}_high.csv"
    out_medium = OUTPUT_DIR / f"{stem}_clean_{VERSION}_medium.csv"
    out_all = OUTPUT_DIR / f"{stem}_clean_{VERSION}_all.csv"
    out_drop = OUTPUT_DIR / f"{stem}_dropped_{VERSION}.csv"
    out_ids_high = OUTPUT_DIR / f"{stem}_clean_{VERSION}_high_ids.csv"
    out_ids_medium = OUTPUT_DIR / f"{stem}_clean_{VERSION}_medium_ids.csv"
    out_ids_all = OUTPUT_DIR / f"{stem}_clean_{VERSION}_all_ids.csv"
    out_summary = OUTPUT_DIR / f"{stem}_clean_{VERSION}_summary.json"

    summary = {
        "version": VERSION, "input": str(input_path), "stem": stem,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {
            "step1_blacklist": {"dropped": 0},
            "step3_no_signal": {"dropped": 0},
            "step3_medium_recovered": {"kept": 0},
            "step4_playlist": {"dropped": 0},
            "step_keep_high": {"kept": 0},
            "step_keep_medium": {"kept": 0},
            "step_other_drop": {"dropped": 0},
        },
        "total_in": 0, "total_keep": 0, "total_keep_high": 0, "total_keep_medium": 0, "total_drop": 0,
    }

    extra_fields = [
        "clean_label", "confidence_tier", "drop_step", "drop_reason",
        "sport_score", "kw_entities", "kw_aligned", "strong_sport_signal",
    ]

    print(f"\nPass 2: cleaning {VERSION}...")
    with input_path.open(newline="", encoding="utf-8", errors="replace") as fin, \
         out_high.open("w", newline="", encoding="utf-8") as fhigh, \
         out_medium.open("w", newline="", encoding="utf-8") as fmed, \
         out_all.open("w", newline="", encoding="utf-8") as fall, \
         out_drop.open("w", newline="", encoding="utf-8") as fdrop, \
         out_ids_high.open("w", encoding="utf-8") as fidh, \
         out_ids_medium.open("w", encoding="utf-8") as fidm, \
         out_ids_all.open("w", encoding="utf-8") as fida:

        reader = csv.DictReader(fin)
        base_fields = reader.fieldnames or []
        out_fields = base_fields + [f for f in extra_fields if f not in base_fields]

        writers = {
            "high": csv.DictWriter(fhigh, fieldnames=out_fields, extrasaction="ignore"),
            "medium": csv.DictWriter(fmed, fieldnames=out_fields, extrasaction="ignore"),
            "all": csv.DictWriter(fall, fieldnames=out_fields, extrasaction="ignore"),
            "drop": csv.DictWriter(fdrop, fieldnames=out_fields, extrasaction="ignore"),
        }
        for w in writers.values():
            w.writeheader()
        fidh.write("video_id\n")
        fidm.write("video_id\n")
        fida.write("video_id\n")

        for i, row in enumerate(reader, 1):
            summary["total_in"] += 1
            title = row.get("title") or ""
            channel = row.get("channel") or ""
            keyword = row.get("keyword") or ""
            pref = row.get("source_ref") or ""

            entities = parse_keyword_entities(keyword)
            row["kw_entities"] = "|".join(entities)

            bl = blacklist_hit(title, channel)
            if bl:
                row.update(clean_label="drop", confidence_tier="", drop_step="step1_blacklist",
                           drop_reason=f"blacklist:{bl}", sport_score="", kw_aligned="", strong_sport_signal="")
                writers["drop"].writerow(row)
                summary["steps"]["step1_blacklist"]["dropped"] += 1
                summary["total_drop"] += 1
                continue

            aligned, _ = keyword_title_aligned(keyword, title, channel)
            strong = strong_sport_signal(title, channel)
            medium_recovered = False
            row["kw_aligned"] = str(aligned)
            row["strong_sport_signal"] = str(strong)

            if entities and not aligned:
                if strong:
                    medium_recovered = True
                    summary["steps"]["step3_medium_recovered"]["kept"] += 1
                else:
                    sc = sport_score(title, channel, keyword)
                    row.update(clean_label="drop", confidence_tier="", drop_step="step3_no_signal",
                               drop_reason=f"no_match_no_signal:{','.join(entities[:3])}",
                               sport_score=str(sc))
                    writers["drop"].writerow(row)
                    summary["steps"]["step3_no_signal"]["dropped"] += 1
                    summary["total_drop"] += 1
                    continue

            if pref in polluted:
                sc = sport_score(title, channel, keyword)
                row.update(clean_label="drop", confidence_tier="", drop_step="step4_playlist",
                           drop_reason="low_playlist_hit_rate", sport_score=str(sc))
                writers["drop"].writerow(row)
                summary["steps"]["step4_playlist"]["dropped"] += 1
                summary["total_drop"] += 1
                continue

            sc = sport_score(title, channel, keyword)
            row["sport_score"] = str(sc)
            keep, reason = decide_keep(sc, aligned, strong, medium_recovered)

            if not keep:
                row.update(clean_label="drop", confidence_tier="", drop_step="step_other", drop_reason=reason)
                writers["drop"].writerow(row)
                summary["steps"]["step_other_drop"]["dropped"] += 1
                summary["total_drop"] += 1
                continue

            if medium_recovered:
                tier = "medium"
                summary["steps"]["step_keep_medium"]["kept"] += 1
                summary["total_keep_medium"] += 1
            else:
                tier = "high"
                summary["steps"]["step_keep_high"]["kept"] += 1
                summary["total_keep_high"] += 1

            row.update(clean_label="keep", confidence_tier=tier, drop_step="", drop_reason="")
            writers[tier].writerow(row)
            writers["all"].writerow(row)
            vid = row.get("video_id", "")
            if tier == "high":
                fidh.write(f"{vid}\n")
            else:
                fidm.write(f"{vid}\n")
            fida.write(f"{vid}\n")
            summary["total_keep"] += 1

            if i % 500_000 == 0:
                print(f"  {i:,} | keep {summary['total_keep']:,} "
                      f"(H {summary['total_keep_high']:,} M {summary['total_keep_medium']:,}) "
                      f"| drop {summary['total_drop']:,}")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["outputs"] = {
        "high": str(out_high),
        "medium": str(out_medium),
        "all": str(out_all),
        "dropped": str(out_drop),
        "ids_high": str(out_ids_high),
        "ids_medium": str(out_ids_medium),
        "ids_all": str(out_ids_all),
    }
    return summary


# ══════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("用法: python3 clean_sports_chunk_v2.py <input.csv>")
        print("示例: python3 clean_sports_chunk_v2.py ../data/sports_chunk_01.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"=== {VERSION} CLEANING ===")
    print(f"  Input:  {input_path}")
    print(f"  Output: {OUTPUT_DIR}/")

    polluted = pass1_playlist_stats(input_path)
    summary = pass2_clean(input_path, polluted, stem)
    summary["playlist_analysis"] = {"polluted_playlists": len(polluted), "min_hit_rate": PLAYLIST_MIN_HIT_RATE}

    out_summary = OUTPUT_DIR / f"{stem}_clean_{VERSION}_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    t = summary["total_in"]
    print(f"\n=== {VERSION} CLEANING COMPLETE ===")
    print(f"  Input:   {t:,}")
    print(f"  Keep:    {summary['total_keep']:,} ({100*summary['total_keep']/max(t,1):.1f}%)")
    print(f"    high:   {summary['total_keep_high']:,}")
    print(f"    medium: {summary['total_keep_medium']:,}")
    print(f"  Drop:    {summary['total_drop']:,}")
    print(f"\n  Outputs:")
    for v in summary["outputs"].values():
        print(f"    {v}")


if __name__ == "__main__":
    main()
