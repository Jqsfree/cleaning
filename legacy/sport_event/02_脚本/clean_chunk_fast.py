#!/usr/bin/env python3
"""Fast cleaning: step1 already done by duckdb, run steps 3-5 + playlist on survivors."""
import csv, json, re, sys, time
from collections import defaultdict
from pathlib import Path

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sports_chunk_merged_post_bl.csv")
PREFIX = INPUT.stem.replace("_post_bl", "")
OUT_HIGH = Path(f"{PREFIX}_clean_v2_high.csv")
OUT_MEDIUM = Path(f"{PREFIX}_clean_v2_medium.csv")
OUT_ALL = Path(f"{PREFIX}_clean_v2_all.csv")
OUT_DROPPED = Path(f"{PREFIX}_dropped_v2.csv")
OUT_SUMMARY = Path(f"{PREFIX}_clean_v2_summary.json")

PLAYLIST_MIN_SAMPLES, PLAYLIST_MIN_HIT_RATE = 5, 0.10
KEEP_SCORE, GRAY_SCORE, MEDIUM_MIN_SCORE = 35, 15, 15

# ── Scoring regexes (same as v2-qc) ──
SPORT_POS = [(re.compile(p, re.I), s) for p, s in [
    (r"\b(full\s*(match|game|race|replay|coverage))\b", 25),
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
    (r"\b(quarterfinals?|bronze\s*playoff|A[-\\s]?Final|B[-\\s]?Final|C[-\\s]?Final)\b", 12),
    (r"\b(sculls?|sweep\s*oar|coxed|coxless)\b", 10),
]]

SPORT_NEG = [(re.compile(p, re.I), s) for p, s in [
    (r"\b(documentary|behind\s*the\s*scenes|interview|press\s*conference)\b", -12),
    (r"\b(training\s*session|practice\s*drill|warm[\s-]?up)\b", -10),
    (r"\b(trailer|teaser|preview)\b", -15),
    (r"\b(compilation|best\s*of|top\s*10)\b", -8),
    (r"\b(gear\s*review|equipment\s*review)\b", -12),
    (r"\b(sparring\s*session)\b", -10),
    (r"\b(ceremony|weigh[\s-]?in)\b", -12),
]]

STRONG_RE = re.compile(
    r"\b(match|final|semi[\s-]?final|quarter[\s-]?final|"
    r"highlights|full\s*(match|game|race|replay|coverage)|"
    r"vs\.?|\svs\s|versus|championship|tournament|"
    r"olympic|world\s*cup|grand\s*prix|diamond\s*league|"
    r"commentary|play[\s-]?by[\s-]?play|broadcast|live\s*stream|"
    r"round\s*\d|bout|regatta|grand\s*final|shootout|overtime|"
    r"national\s*games|varsity|campeonato)\b", re.I)

SPORT_LEXICON_SORTED = sorted([
    "rowing","regatta","boat race","henley","surfing","surf","wsl","big wave",
    "pickleball","ppa","app tour","skiing","alpine","slalom","downhill","super-g",
    "athletics","track and field","high jump","long jump","sprint",
    "fencing","epee","foil","sabre","taekwondo","tkd","sanda","wushu",
    "snooker","billiards","pool","basketball","nba","ncaa basketball",
    "football","soccer","fifa","premier league","champions league",
    "tennis","atp","wta","wimbledon","roland garros",
    "volleyball","beach volleyball","rugby","cricket","baseball","mlb",
    "hockey","nhl","ice hockey","golf","pga","lpga",
    "boxing","mma","ufc","wrestling","cycling","tour de france",
    "swimming","diving","gymnastics","badminton","table tennis","squash",
    "asian games","olympics","olympic","paralympic",
    "world cup","diamond league","grand prix",
    "cross country","marathon","triathlon",
    "motorsport","formula 1","f1","motogp","nascar",
    "lacrosse","handball","water polo","judo","karate","bjj","grappling",
    "indoor bowls","bowls","lawn bowls","nine-ball","10-ball","8-ball","chinese pool",
    "lacrosse","nll","buffalo bandits","box lacrosse",
    "horse racing","darts","pdc","floorball","unihockey","salibandy",
    "bowling","pba","tenpin","weightlifting","iwf",
    "climbing","ifsc","sport climbing","netball",
    "ultimate","wfdf","ultimate frisbee",
    "american football","nfl","cfb","super bowl",
], key=len, reverse=True)

SPORT_SYNONYMS = {
    "rowing": ["rowing","rower","regatta","scull","crew","boat race","hocr"],
    "surfing": ["surfing","surf","wsl","wave","surfer"],
    "pickleball": ["pickleball","ppa","dink","kitchen"],
    "skiing": ["skiing","ski","alpine","slalom","downhill","super-g","giant slalom"],
    "athletics": ["athletics","track","field","sprint","hurdles","decathlon"],
    "fencing": ["fencing","epee","foil","sabre","fencer"],
    "taekwondo": ["taekwondo","tkd","kicking"],
    "sanda": ["sanda","sanshou","wushu"],
    "snooker": ["snooker","billiards","147"],
    "basketball": ["basketball","nba","dunk","hoops"],
    "football": ["football","soccer","fifa","premier league","mls"],
    "tennis": ["tennis","atp","wta","wimbledon"],
    "asian games": ["asian games","asiad","hangzhou","jakarta"],
    "olympics": ["olympic","olympics","paris 2024","tokyo 2020","beijing 2022"],
    "hockey": ["hockey","nhl","flames","canadiens","bruins","rangers"],
    "cricket": ["cricket","ipl","bcci","wicket"],
    "squash": ["squash","squash tv","glass court","boast"],
    "lacrosse": ["lacrosse","nll","box lacrosse","bandits"],
    "indoor bowls": ["indoor bowls","bowls","lawn bowls","lawn green"],
}

def sport_score(title, channel, keyword):
    text = f"{title or ''} {channel or ''} {keyword or ''}"
    s = 0
    for pat, pts in SPORT_POS:
        if pat.search(text): s += pts
    for pat, pts in SPORT_NEG:
        if pat.search(text): s += pts
    return s

def parse_entities(keyword):
    if not keyword: return []
    kw = str(keyword).strip().strip('"').lower()
    core = kw.split(" - ")[0]
    entities = [t for t in SPORT_LEXICON_SORTED if t in core]
    return entities

def keyword_aligned(keyword, title, channel):
    entities = parse_entities(keyword)
    if not entities: return True, entities
    terms = set(entities)
    for e in entities:
        for k, syns in SPORT_SYNONYMS.items():
            if k in e or e in k: terms.update(syns)
    text = f"{title or ''} {channel or ''}".lower()
    return any(t in text for t in terms), entities

def strong_signal(title, channel):
    return bool(STRONG_RE.search(f"{title or ''} {channel or ''}"))

# ── Pass1: playlist stats ──
t0 = time.perf_counter()
print("Pass1: playlist analysis...", flush=True)
stats = defaultdict(lambda: {"total":0,"hits":0})
with INPUT.open(newline="", encoding="utf-8", errors="replace") as f:
    for i, row in enumerate(csv.DictReader(f), 1):
        pref = row.get("source_ref") or ""
        if not pref: continue
        title, channel, keyword = row.get("title","") or "", row.get("channel","") or "", row.get("keyword","") or ""
        sc = sport_score(title, channel, keyword)
        aligned, _ = keyword_aligned(keyword, title, channel)
        strong = strong_signal(title, channel)
        if sc >= GRAY_SCORE or aligned or strong:
            stats[pref]["hits"] += 1
        stats[pref]["total"] += 1
        if i % 1000000 == 0: print(f"  scanned {i:,}...", flush=True)

polluted = {p for p, s in stats.items() if s["total"] >= PLAYLIST_MIN_SAMPLES and s["hits"]/s["total"] < PLAYLIST_MIN_HIT_RATE}
print(f"  playlists: {len(stats):,}, polluted: {len(polluted):,} ({time.perf_counter()-t0:.1f}s)", flush=True)

# ── Pass2: clean ──
print("Pass2: cleaning...", flush=True)
extra = ["clean_label","confidence_tier","drop_step","drop_reason","sport_score","kw_entities","kw_aligned","strong_sport_signal"]
s = {"total":0,"keep":0,"keep_high":0,"keep_medium":0,"drop":0,"step3_no_signal":0,"step3_medium":0,"step4_playlist":0,"step_other":0}

with INPUT.open(newline="", encoding="utf-8", errors="replace") as fin, \
     OUT_HIGH.open("w", newline="", encoding="utf-8") as fh, \
     OUT_MEDIUM.open("w", newline="", encoding="utf-8") as fm, \
     OUT_ALL.open("w", newline="", encoding="utf-8") as fa, \
     OUT_DROPPED.open("w", newline="", encoding="utf-8") as fd:

    reader = csv.DictReader(fin)
    base = reader.fieldnames or []
    out_fields = base + [f for f in extra if f not in base]
    writers = {"high": csv.DictWriter(fh, out_fields, extrasaction="ignore"),
               "medium": csv.DictWriter(fm, out_fields, extrasaction="ignore"),
               "all": csv.DictWriter(fa, out_fields, extrasaction="ignore"),
               "drop": csv.DictWriter(fd, out_fields, extrasaction="ignore")}
    for w in writers.values(): w.writeheader()

    for i, row in enumerate(reader, 1):
        s["total"] += 1
        title, channel, keyword = row.get("title","") or "", row.get("channel","") or "", row.get("keyword","") or ""
        pref = row.get("source_ref") or ""

        entities = parse_entities(keyword)
        row["kw_entities"] = "|".join(entities)

        aligned, _ = keyword_aligned(keyword, title, channel)
        strong = strong_signal(title, channel)
        medium_recovered = False
        row["kw_aligned"] = str(aligned)
        row["strong_sport_signal"] = str(strong)

        # Step3: keyword mismatch
        if entities and not aligned:
            if strong:
                medium_recovered = True
                s["step3_medium"] += 1
            else:
                sc = sport_score(title, channel, keyword)
                row.update(clean_label="drop", confidence_tier="", drop_step="step3_no_signal",
                           drop_reason=f"no_match:{','.join(entities[:3])}", sport_score=str(sc))
                writers["drop"].writerow(row)
                s["drop"] += 1; s["step3_no_signal"] += 1
                continue

        # Step4: playlist
        if pref in polluted:
            sc = sport_score(title, channel, keyword)
            row.update(clean_label="drop", confidence_tier="", drop_step="step4_playlist",
                       drop_reason="low_playlist_hit_rate", sport_score=str(sc))
            writers["drop"].writerow(row)
            s["drop"] += 1; s["step4_playlist"] += 1
            continue

        sc = sport_score(title, channel, keyword)
        row["sport_score"] = str(sc)

        # Decide keep/drop
        keep = False; reason = ""
        if sc >= KEEP_SCORE:
            keep = True; reason = "high_score"
        elif aligned and sc >= GRAY_SCORE:
            keep = True; reason = "gray_aligned"
        elif medium_recovered and strong and sc >= MEDIUM_MIN_SCORE:
            keep = True; reason = "medium_strong"
        else:
            reason = "low_score"

        if not keep:
            row.update(clean_label="drop", confidence_tier="", drop_step="step_other", drop_reason=reason)
            writers["drop"].writerow(row)
            s["drop"] += 1; s["step_other"] += 1
            continue

        tier = "medium" if medium_recovered else "high"
        row.update(clean_label="keep", confidence_tier=tier, drop_step="", drop_reason="")
        writers[tier].writerow(row)
        writers["all"].writerow(row)
        s["keep"] += 1
        if tier == "high": s["keep_high"] += 1
        else: s["keep_medium"] += 1

        if i % 500000 == 0:
            print(f"  {i:,} | keep {s['keep']:,} (H{s['keep_high']:,} M{s['keep_medium']:,}) | drop {s['drop']:,}", flush=True)

elapsed = time.perf_counter() - t0
t = s["total"]
print(f"\n=== CLEANING DONE ({elapsed:.0f}s) ===", flush=True)
print(f"Input:  {t:,}")
print(f"Keep:   {s['keep']:,} ({100*s['keep']/t:.1f}%)")
print(f"  high:  {s['keep_high']:,}")
print(f"  medium:{s['keep_medium']:,}")
print(f"Drop:   {s['drop']:,}")
print(f"  step3_no_signal: {s['step3_no_signal']:,}")
print(f"  step3_medium:    {s['step3_medium']:,} (recovered)")
print(f"  step4_playlist:  {s['step4_playlist']:,}")
print(f"  step_other:      {s['step_other']:,}")
