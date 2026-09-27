#!/usr/bin/env python3
"""
共享精炼规则 — 被 clean_sports_chunk_refine.py / clean_sports_chunk_refine_v2.py 引用。

三组规则:
  r2_blacklist   — 扩展黑名单（补充第一轮漏掉的模式）
  r2_context     — 上下文过滤（title + channel + keyword 联合判断）
  r2_weak_entity — 弱实体剔除（keyword 实体过弱/模糊）
"""

import re

# ── r2_blacklist: 第二轮黑名单 ──────────────────────────
# 补充第一轮黑名单漏掉的模式，通常更精确以避免过度误杀
R2_BLACKLIST_PATTERNS = [
    # 游戏 — 精确匹配，避免误杀体育游戏内容
    r"\b(walkthrough|let'?s\s*play|gameplay\s*part\s*\d+)\b",
    r"\b(mod\s*apk|apk\s*mod|hack\s*tool|cheat\s*engine)\b",
    r"\b(gta\s*[45v]|grand\s*theft\s*auto|cyberpunk)\b",
    r"\b(dota\s*2|league\s*of\s*legends|valorant|overwatch)\b",
    r"\b(minecraft\s*(server|survival|bedwars|skyblock))\b",
    # 纯娱乐/影视
    r"\b(official\s*trailer|teaser\s*trailer)\b.*\b(movie|film|hd)\b",
    r"\b(full\s*episode\s*(season|episode)?\s*\d+)\b",
    r"\b(behind\s*the\s*scenes|making\s*of|bloopers?)\b",
    # 纯音乐
    r"\b(official\s*(music\s*)?video|lyrics?\s*video|audio\s*only)\b",
    r"\b(lofi|chill\s*(beats|hop|mix)|relaxing\s*music)\b",
    # 宗教/灵修
    r"\b(bible\s*study|sermon\s*series|worship\s*(night|live))\b",
    # 教育/教程 — 不含体育词汇时
    r"\b(crash\s*course|full\s*course|masterclass\b|bootcamp\b)\b",
    # 政治/新闻 — 非体育
    r"\b(press\s*conference|white\s*house|presidential|election)\b",
    # ── QC 2026-05-31 扩展 ──
    # WWE/职业摔角
    r"\b(wwe\s*(smackdown|raw|raw\s*\d+)|kenny\s*omega|bryan\s*danielson|ring\s*of\s*honor|roh\s*bound)\b",
    # 体育游戏冒充
    r"\b(pga\s*tour\s*2k|nhl\s*2k|alpine\s*ski\s*racing\s*20\d|pro\s*yakyuu\s*spirits|nba\s*2k|madden\s*nfl|mlb\s*the\s*show|ea\s*sports\s*fc)\b",
    r"\b(fifa\s*\d+|fifa\s*(ultimate|career|manager))\b",
    # 谈话/播客/球迷节目（频道级）
    r"\b(barstool\s*sports|theflightmike|fullridenation|baseball\s*isn'?t\s*boring|sunday\s*supplement)\b",
    # 训练/对练/健身教学
    r"\b(sparring\s*session|sparring\s*&?\s*discussion|pro\s*boxing\s*sparring|follow\s*along\s*rowing|fairway\s*woods\s*and\s*hybrids)\b",
    r"\b(golf\s*lessons|wisdom\s*in\s*golf|fighttips|rowalong)\b",
    # 体育游戏/电竞频道
    r"\b(lol\s*esports|blink\s*esports|gaming\s*mason|all\s*nintendo\s*music|esports\s*vod)\b",
    # 仪式/颁奖/纪录片
    r"\b(closing\s*ceremony|opening\s*ceremony|ceremonial\s*weigh[\s-]?in|big\s*wave\s*awards)\b",
    r"\b(50\s*years\s*of\s*title\s*ix|entry\s*of\s*the\s*year)\b",
    r"\b(what\s*is\s*(henley|the\s*[a-z]+\s*(royal|regatta|championship|grand|world)))\b",
    # 非体育 vs（动漫/影视/游戏角色 vs）
    r"\b(din\s*djarin\s*vs|loki\s*vs\s*russell|ippo\s*vs\s*sendo|dashie\s*vs\s*lamarr)\b",
    r"\b(tongue\s*pop\s*by|sub\s*focus\s*&?\s*wilkinson\s*vs|drum\s*&?\s*bass\s*vs)\b",
    # 汽车对比/直线加速（非 motorsport 赛事）
    r"\b(ford\s*explorer\s*vs\s*dodge|gtr\s*vs\s*z06|suv\s*drag\s*race)\b",
    # 训练/教学频道
    r"\b(shawn\s*clement'?s?\s*wisdom|clement'?s?\s*wisdom\s*in\s*golf)\b",
    # ── QC Round 2 (2026-05-31) 基于二次 text QC F 样本扩展 ──
    # 体育谈话/播客/DFS博彩频道
    r"\b(cess\s*talks\s*sports|the\s*athletic\s*nba|nba\s*news\s*24h|frankly\s*hockey|letsrun|oneturf\s*universe)\b",
    r"\b(pga\s*dfs|inside\s*the\s*ropes|dfs\s*&\s*betting|fantasy\s*football|sports\s*betting|swot\s*analysis|post-match\s*analysis.*nukta)\b",
    r"\b(talksports?\s*golf|run\s*pure\s*sports|fanatics\s*view|unfiltered\s*w\b|barstool\s*sports)\b",
    # 音乐频道（YouTube Topic 自动生成）
    r"\b\w+-\s*Topic\b",
    r"\b(dj\s*mag|dj\s*set|extended\s*mix|official\s*audio.*music|music\s*video)\b",
    # 影视/剧集
    r"\b(wuxia\s*fantasy|dashing\s*youth|star\s*jalsha|gma\s*network|youku\s*english)\b",
    r"\b(full\s*story.*episode\s*\d+|episode\s*\d+.*part\s*\d|season\s*\d+.*episode\b)\b",
    # 舞蹈/说唱对战
    r"\b(red\s*bull\s*dance|dance\s*your\s*style|tahitian\s*dance\s*competition)\b",
    r"\b(bullpen\s*battle|2livecrew\s*vs|streetrunnaz|officialmemphisjookin|waackxxxy\s*vs)\b",
    r"\b(rap\s*battle.*league|battle\s*league.*rap|battle.*politics|arson\s*vs)\b",
    # 纪录片/历史频道
    r"\b(simple\s*history|topt\s*enz|archaic\s*arms|legends\s*of\s*the\s*green)\b",
    r"\b(most\s*underrated\s*ancient|terrible\s*examples\s*of\s*human|yakuza.*mafia)\b",
    # 游戏频道/游戏名
    r"\b(cohhcarnage|fuzzfinger\s*gaming|mcduffy\s*gaming|stayplation\s*gaming|epic\s*tabs|harleen\s*quinzel)\b",
    r"\b(final\s*fantasy\s*(xii|xiii|xiv|xv|xvi|union)|baldur'?s\s*gate|wii\s*sports\s*resort|star\s*wars.*clone\s*wars)\b",
    r"\b(clash\s*of\s*clans|clash\s*royale|totally\s*accurate\s*battle|king\s*of\s*fighters\s*xv)\b",
    r"\b(trials\s*of\s*osiris|destiny\s*2|flawless\s*pov|boss\s*fight|esper\s*boss|starcraft.*alpha)\b",
    # 非体育频道（新闻/考试/政治/生活）
    r"\b(sakshi\s*tv|aaradhya\s*ras\s*academy|narendra\s*modi|daiwik\s*funtime|mayor\s*sylvester\s*turner)\b",
    r"\b(scammers\s*looting|upi\s*scammers|rajasthan\s*history|gk\s*gs|bol\s*network|game\s*show\s*aisay)\b",
    # 滑雪/冲浪 POV/生活类
    r"\b(first\s*person\s*view\s*gopro|boise\s*river\s*surf|winter\s*sesh|rough\s*cut)\b",
    r"\b(let'?s\s*ski!|gopro\s*(ski|surf|snow)|campfire\s*at\s*sunset|4k\s*12h|alpine.*peak\s*to\s*peaks|sunset\b)\b",
    # 非赛事奥运内容
    r"\b(olympic\s*rings\s*in\s*kew|gymnastics\s*gala\s*performance|throwback\s*thursday|paralympic.*torch)\b",
    # DIY/教程
    r"\b(dishwasher\s*for\s*beginners|install\s*dishwasher|clay\s*making|diy\s*replace)\b",
    r"\b(treasure\s*coast\s*house\s*hunters|colton\s*crump\s*diy|real\s*estate.*home\s*beats)\b",
    # 新闻/政治人物
    r"\b(donald\s*trump\s*skipped|pm\s*modi\s*live|mann\s*ki\s*baat|women\s*protest\s*for)\b",
    # 武术示范/非比赛
    r"\b(protaekwondo\s*en\s*santa|alto\s*impacto\s*marcial|gun.*firing|hk\s*gmg|mk19)\b",
    # ── QC Round 3 (2026-05-31) 持续扩展 ──
    # 军事/战斗频道
    r"\b(funker\s*530|warleaks|military\s*blog|combat\s*footage|veteran\s*community|ukrainain\s*forces)\b",
    # 摔角/AEW
    r"\b(all\s*elite\s*wrestling|aew\s*dynamite|owen\s*cup|aew\b.*wrestling)\b",
    r"\b(willow\s*nightingale|mariah\s*may|#\s*aew|#\s*aew\s*dynamite)\b",
    # 喜剧/播客频道
    r"\b(kill\s*tony|jamar\s*neighbors|brian\s*moses|overtime\s*with\s*matt\s*moscona)\b",
    # 游戏频道
    r"\b(game\s*grumps|arekkz\s*gaming|decathlon\s*gamer|bilsavage|races\s*and\s*fun)\b",
    r"\b(mario\s*&\s*sonic\s*at\s*the\s*olympic|olympic\s*games\s*tokyo\s*2020.*gamer)\b",
    r"\b(golf\s*clash|mlbb\b|m3\s*world|mobile\s*legends|hot\s*wheels.*race|matchbox\s*tournament)\b",
    r"\b(pcm\s*yeti|cycling.*game.*ep\s*\d+|metal\s*gear\s*solid|mother\s*base)\b",
    # 军事/武器
    r"\b(hk\s*models.*review|b17g\s*review|model\s*kit.*\d+/\d+)\b",
    # 非赛事奥运
    r"\b(beijing\s*2022.*torch|olympic\s*torch|paris\s*olympic.*bronze\s*medal.*news)\b",
    r"\b(manu\s*bhaker|dehradun\s*connection|jai\s*bharat\s*tv)\b",
    # 器材销售/评测
    r"\b(viking\s*pickleball\s*paddles|now\s*on\s*sale.*pickleball|\$\d+\s*vs\s*\$\d+\s*karate)\b",
    r"\b(karate\s*gi|martial\s*arts.*equipment|pickleball\s*warehouse)\b",
    # 滑雪/冲浪教程/生活
    r"\b(ski\s*prep|ankle\s*flexion|c.mo\s*guardar\s*los\s*esqu.s|session\s*de\s*surf\s*en)\b",
    r"\b(surf\s*outside.*restaurant|best\s*watched\s*on\s*hd|beach\s*house\s*restaurant)\b",
    # 汽车/科技频道
    r"\b(motor\s*trend\s*channel|roadkill\s*garage|cheap\s*upgrades\s*on\s*our)\b",
    r"\b(remote\s*control\s*your\s*computer|teamviewer|android\s*device.*remote)\b",
    # 音乐频道
    r"\b(asian\s*kung-fu\s*generation|haruka\s*kanata|official\s*youtube.*band)\b",
    # 体育设备/教学
    r"\b(tournament\s*lessons.*every\s*shot\s*counts|postgame\s*comments\s*u18|usa\s*hockey.*postgame)\b",
    # 新闻/其他
    r"\b(qatar\s*sentenced.*indians|spying\s*on\s*israel|adeel\s*azhar)\b",
    r"\b(remote\s*control\s*your|florida\s*state.*it\s*starts\s*here|bowserfsu)\b",
]
R2_BLACKLIST_RE = re.compile("|".join(R2_BLACKLIST_PATTERNS), re.I)

# ── r2_context: 弱上下文字段 ───────────────────────────
# 用于判断 title 是否过于简短/空泛，搭配弱 channel 时剔除
WEAK_TITLE_PATTERNS = [
    r"^(untitled|no\s*title|video\s*\d+|clip\s*\d+)$",
    r"^(stream\s*\d+|live\s*\d+|broadcast\s*\d+)$",
    r"^[A-Za-z0-9_\-]{1,8}$",  # 极短随机字符串
]
WEAK_TITLE_RE = re.compile("|".join(WEAK_TITLE_PATTERNS), re.I)

WEAK_CHANNEL_PATTERNS = [
    r"\b(user|channel|account|page)\b",
    r"\b(temp|test|demo|sample)\b",
]
WEAK_CHANNEL_RE = re.compile("|".join(WEAK_CHANNEL_PATTERNS), re.I)

# ── r2_weak_entity: 弱 keyword 实体 ─────────────────────
# keyword 提取出的实体过于模糊/通用，不是明确体育项目
WEAK_ENTITY_WORDS = {
    "sports", "sport", "game", "games", "match", "matches",
    "play", "playing", "player", "players",
    "video", "videos", "clip", "clips",
    "stream", "streams", "live", "broadcast",
    "event", "events", "competition",
    "tournament", "tourney", "contest",
    "race", "racing", "running",
    "team", "teams", "club", "league",
    "score", "scoring", "goal", "goals",
    "win", "winner", "champion", "champions",
    "final", "finals", "semi", "prelim",
    "season", "series", "episode",
    "full", "hd", "best", "top",
    "highlights", "highlight", "recap",
    "commentary", "comment", "review",
    "coverage", "analysis", "breakdown",
    "vs", "versus", "showdown", "faceoff",
}


# ══════════════════════════════════════════════════════════
# 核心函数
# ══════════════════════════════════════════════════════════

def refine_row(row: dict) -> tuple[str, str]:
    """
    对已清洗的行做第二轮机审精炼。

    返回 (label, reason)，其中:
      label:  "keep" 或 "drop"
      reason: 剔除原因标识（如 "r2_blacklist:xxx"）
    """
    title = str(row.get("title") or "")
    channel = str(row.get("channel") or "")
    keyword = str(row.get("keyword") or "")
    kw_entities_str = str(row.get("kw_entities") or "")

    # ── Rule 1: r2_blacklist ──
    text_full = f"{title} {channel}"
    m = R2_BLACKLIST_RE.search(text_full)
    if m:
        return ("drop", f"r2_blacklist:{m.group(0)[:50]}")

    # ── Rule 2: r2_context ──
    # 弱 title + 弱 channel 同时出现 → 几乎没有可判断上下文
    weak_title = bool(WEAK_TITLE_RE.search(title)) if title else True
    weak_channel = bool(WEAK_CHANNEL_RE.search(channel)) if channel else True
    if weak_title and weak_channel:
        return ("drop", "r2_context:weak_title_and_channel")

    # ── Rule 3: r2_weak_entity ──
    if kw_entities_str:
        entities = [e.strip() for e in kw_entities_str.split("|") if e.strip()]
        strong_entities = [e for e in entities if e.lower() not in WEAK_ENTITY_WORDS]
        if not strong_entities and entities:
            return ("drop", f"r2_weak_entity:all_generic:{','.join(entities[:3])}")

    return ("keep", "")
