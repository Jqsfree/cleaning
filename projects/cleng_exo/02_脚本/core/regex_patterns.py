#!/usr/bin/env python3
"""Common regex pattern lists used across QC scripts."""

# ============================================================
# ANIME / CARTOON
# ============================================================
ANIME_PATTERNS = [
    r'\banime\b', r'\bcartoon\b', r'マンガ', r'アニメ',
    r'\banimated\b', r'\b3d\s*animation\b', r'\bmanga\b',
    r'\bpixar\b', r'\btoon\b', r'\bstop\s*motion\b',
    r'\bdonghua\b', r'애니', r'만화',
]

# ============================================================
# MUSIC / MTV
# ============================================================
MUSIC_PATTERNS = [
    r'\bofficial\s+(music\s+)?video\b',
    r'\bmusic\s+video\b', r'\blyric\b', r'\bofficial\s+audio\b',
    r'\bvevo\b', r'\bmtv\b', r'\bconcert\b',
    r'\blive\s+performance\b',
    r'\bMV\b', r'\bM/V\b', r'\[MV\]', r'\(MV\)', r'[（）(MV））]',
    r'\bsong\b', r'\balbum\b', r'\bdance\b', r'\bdancer\b',
    r'\bperformance\b', r'\bperforms?\b', r'\bband\b',
    r'\bgrammy\b', r'\bbeatles?\b',
]

# ============================================================
# PLATFORM / WATERMARK
# ============================================================
PLATFORM_PATTERNS = [
    r'抖音', r'快手', r'tiktok', r'小红书',
    r'B站', r'bilibili', r'微博', r'微视',
    r'youtube\s+shorts?', r'shopee',
]

# ============================================================
# VARIETY / BTS / ENTERTAINMENT
# ============================================================
VARIETY_PATTERNS = [
    r'综艺', r'选秀', r'真人秀', r'花絮', r'幕后',
    r'幕后花絮', r'男团', r'女团', r'偶像', r'选秀节目', r'才艺',
    r'练习生', r'圆桌派', r'毛雪汪', r'喜剧大会', r'综艺节目',
    r'精華', r'精彩片段', r'完整版', r'抢先看', r'網路獨家',
    r'新說唱', r'煮戰', r'綜藝玩很大',
    r'\bbts\b', r'\bbehind\s?the\s?scenes?\b', r'\btalent\s?show\b',
    r'\bvariety\s?show\b', r'\breality\s?(tv|show)\b', r'\btrainee\b',
    r'\bclip\b', r'\bsub\b',
]

# ============================================================
# DRAMA / CLIPS / MOVIE
# ============================================================
DRAMA_PATTERNS = [
    # 中文
    r'电视剧', r'网剧', r'短剧', r'剧照', r'电影',
    r'剪辑', r'片段', r'预告', r'情景剧', r'情景喜剧', r'微电影',
    r'完整版', r'精華', r'抢先看',
    # 英文通用
    r'\bdrama\b', r'\bweb\s?series\b', r'\bmovie\b', r'\bfilm\b',
    r'\bskit\b', r'\bsitcom\b', r'\bcomedy\s?sketch\b',
    # 印度剧 — 集数模式（3–4 位数，避免误杀播客 EP.12）
    r'\bepisode\s*-?\s*\d{3,}\b',
    r'\bep\.?\s*-?\s*\d{3,}\b',
    r'[-|]\s*\d{3,4}\s*[-|]',
    # 印度剧专有词
    r'\btelecast\b', r'\bserial\b',
    r'\bsaas\b', r'\bbahu\b', r'\brishta\b', r'\bpyaar\b',
    r'\bishq\b', r'\bzindagi\b', r'\bkumkum\b', r'\bnaagin\b', r'\bbegusarai\b',
    # 印度综艺/真人秀
    r'\bbigg\s*boss\b', r'\bkaun\s*banega\b', r'\bnach\s*baliye\b',
    r'\bdance\s*india\b', r'\bsuperstar\s*singer\b', r'\bindian\s*idol\b',
    # 片段剪辑标志
    r'\bfull\s+episode\b', r'\bnext\s+episode\b', r'\blatest\s+episode\b',
    r'\bwatch\s+online\b', r'\bpromo\b',
]

# ============================================================
# NEWS / PRESS / POLITICS
# ============================================================
NEWS_PATTERNS = [
    r'新闻', r'发布会', r'记者会', r'政论', r'时政',
    r'时事', r'政治', r'国会', r'议会', r'白宫',
    r'时事评论', r'政经', r'时评',
    r'特朗普', r'拜登', r'习近平', r'赖清德',
    r'总统', r'首相', r'总理',
    r'前進新台灣', r'三立', r'年代向錢看', r'主播', r'新聞台',
    r'\bnews\b', r'\bpress\s?conference\b', r'\bbriefing\b',
]

# ============================================================
# SPORTS
# ============================================================
SPORTS_PATTERNS = [
    r'英超', r'足球', r'球賽', r'聯賽', r'賽季', r'球證', r'拳擊',
    r'籃球', r'棒球', r'排球', r'網球', r'體育', r'賽事',
    r'\bsports?', r'\bnba', r'\bmlb', r'\bfifa',
    r'\bnfl', r'\bnhl', r'\bepl',
    r'\bboxing\b', r'\bfight\b', r'\bmatch\b', r'\bhighlights?\b',
    r'\bwrestling\b', r'\bwwe\b', r'\bufc\b', r'\bdazn\b',
    r'\btraining\b', r'\bchampionship\b', r'\btournament\b',
]

# ============================================================
# LECTURE / SEMINAR / COURSE
# ============================================================
LECTURE_PATTERNS = [
    r'讲座', r'講座', r'股市', r'線上課程', r'教你', r'理財',
    r'長壽', r'健康2\.0',
    r'\bseminar\b', r'\blecture\b',
]

# ============================================================
# LIVE STREAM / POSTER / MARKETING
# ============================================================
LIVE_POSTER_PATTERNS = [
    r'直播', r'大字报', r'广告', r'营销', r'促销',
    r'带货', r'秒杀', r'抢购', r'优惠',
    r'\blive\s?stream\b', r'\bpromo\b', r'\bpromotion\b', r'\bsponsor\b',
]

# ============================================================
# FAN / IDOL
# ============================================================
FAN_IDOL_PATTERNS = [
    r'粉丝', r'饭圈', r'爱豆', r'应援', r'打榜', r'投票', r'应援色',
    r'\bfan\s?made\b', r'\bfan\s?edit\b', r'\bfancam\b',
    r'\bidol\b', r'\bstan\b', r'\bfandom\b',
]

# ============================================================
# SOLO / VLOG / NON-DIALOGUE
# ============================================================
SOLO_PATTERNS = [
    r'\bgaming\b', r'\bgameplay\b', r"\blet's\s+play\b",
    r'\bwalkthrough\b', r'\btutorial\b', r'\bhow[\s-]to\b',
    r'\bunboxing\b', r'\breview\b', r'\bproduct\s+(demo|review)\b',
    r'\bprank\b', r'\bchallenge\b', r'\bshorts?\b', r'#shorts',
    r'\basmr\b', r'\bmeditation\b', r'\bworkout\b',
    r'\bcooking\b', r'\brecipe\b', r'\broutine\b',
    r'\bcompilation\b', r'\btrailer\b', r'\bteaser\b',
    r'\bvlog\b', r'\b24/7\b',
    # 教学/教程/占卜/第一人称
    r'\bhindi\b', r'\bsong\b', r'\bclass\b', r'\blearn\b',
    r'\btips\b', r'\bhoroscope\b', r'\bastrology\b', r'\btarot\b',
    r'\bstep\s+by\s+step\b', r'\bscammer[s]?\b', r'\blyrics?\b', r'\bsir\b',
    r'\bi\s+infiltrated\b', r'\bi\s+spent\b', r'\bi\s+tried\b',
    # 技术/开发
    r'\blinux\b', r'\bcommand\b', r'\bpython\b', r'\bjavascript\b',
    r'\bprogramming\b', r'\bcoding\b', r'\bcoder\b', r'\bdeveloper\b',
    r'\bsoftware\b', r'\bapp\b', r'\bapi\b', r'\bweb\s+development\b',
    # 游戏/直播
    r'\broblox\b', r'\bminecraft\b', r'\bfortnite\b', r'\bfree\s+fire\b',
    r'\bstreaming\b', r'\bstreamer\b', r'\bgame\b', r'\bgamer\b',
    r'\bplaythrough\b', r'\blet\'s\s+play\b', r'\blive\s+stream\b',
    # 练习/技能
    r'\bpractice\b', r'\bexercise\b', r'\btest\b', r'\bexam\b',
    r'\bproblem\b', r'\bsolution\b',
]

# ============================================================
# DIALOGUE / INTERVIEW / PODCAST (positive signals)
# ============================================================
DIALOGUE_PATTERNS = [
    r'\bpodcast\b', r'\bepisode\b', r'\bep[\.\s]?\d+',
    r'\binterview\b', r'\binterviews?\b', r'\bwith\s', r'\bft\.?\b', r'\bfeaturing\b',
    r'\bvs\.?\b', r'\bversus\b', r'\bconversation\b',
    r'\bdialogue\b', r'\bdiscussion\b', r'\bdebate\b',
    r'\btalk\s+show\b', r'\bpanel\b', r'\broundtable\b',
    r'\bguest\b', r'\bhost\b', r'\bchat\b', r'\bfireside\b',
    r'\bspeaks?\s+(with|to|about)\b', r'\btalks?\s+(with|to|about)\b',
    r'\bqa\b', r'\bq\s*&\s*a\b',
    '專訪', '訪談', '對談', '對話',
    '访谈', '对话', '面对面',
    '대담', '인터뷰', '토크',
]
