"""棒球视频数据清洗 — 快速去除重复和无效数据。

改进点（vs 原 main.py）：
1. 去掉 100% 空的列（6列）
2. 过滤 [Deleted video] / [Private video]
3. 过滤不含比赛关键词的标题（非棒球/非比赛内容）
4. 过滤时长异常：< 60s（剪辑）或 > 72000s（异常）
5. 可选：只保留含 FULL REPLAY / full game 关键词的真·完整比赛
"""

import pandas as pd

INPUT = "/home/jqs/tiyu/data/体育解说38-棒球_5af5c257_records.csv"
OUTPUT = "/home/jqs/tiyu/dispose_data/processed_baseball_videos.csv"

# ── 1. 读取 ──────────────────────────────────────────
df = pd.read_csv(INPUT)
n0 = len(df)
print(f"[1] 原始: {n0} 行, {len(df.columns)} 列")

# ── 2. 去掉全空列 ────────────────────────────────────
empty_cols = df.columns[df.isnull().all()].tolist()
df = df.drop(columns=empty_cols)
print(f"[2] 去除全空列: {empty_cols} → 剩 {len(df.columns)} 列")

# ── 3. 去重（video_id + url 双保险）──────────────────
df = df.drop_duplicates(subset=['video_id'])
df = df.drop_duplicates(subset=['url'])
print(f"[3] 去重后: {len(df)} 行 (video_id + url)")

# ── 4. 过滤空值 ──────────────────────────────────────
for col in ['title', 'channel', 'duration_seconds', 'view_count']:
    df = df.dropna(subset=[col])
    df = df[df[col].astype(str).str.strip() != '']
print(f"[4] 去关键列空值后: {len(df)} 行")

# ── 5. 类型标准化 ────────────────────────────────────
df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce').astype('Int64')
df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').astype('Int64')
df['title'] = df['title'].str.strip()
df['channel'] = df['channel'].str.strip()

# 再删一次转数值失败产生的 NaN
df = df.dropna(subset=['duration_seconds', 'view_count'])
print(f"[5] 数值标准化后: {len(df)} 行")

# ── 6. 过滤无效内容 ──────────────────────────────────

# 6a. [Deleted video] / [Private video] — 完全无用
mask_deleted = df['title'].str.contains(
    r'\[Deleted video\]|\[Private video\]', case=False, na=False
)
print(f"  └ 删除已删除/私密视频: {mask_deleted.sum()} 条")
df = df[~mask_deleted]

# 6b. 棒球专项过滤 — title 或 channel 必须含棒球关键词
#     通用比赛词（vs/at/Game/Final）会引入大量非棒球内容（WWE/音乐/电影）
#     所以分两层：先匹配比赛结构，再确认是棒球

# 第一层：比赛结构关键词（排除纯娱乐/音乐/电影）
MATCH_KEYWORDS = (
    r'vs\.? |VS\.? | at | @ |Series|Final|Game|Championship|'
    r'NCAA|College|Baseball|MLB|CWS|World Series|Playoff|'
    r'FULL REPLAY|full game|Highlights|Regional|Super Regional|'
    r'Opening Day|Spring Training|All[- ]Star'
)
mask_struct = df['title'].str.contains(MATCH_KEYWORDS, case=False, na=False, regex=True)

# 第二层：棒球专项关键词（title 或 channel）
BASEBALL_TERMS = (
    r'baseball|棒球|mlb|cws|college world series|diamond|hitter|pitcher|'
    r'inning|home run|homerun|strikeout|dugout|bullpen|batting|pitching|'
    r'ncaa baseball|minor league|major league|world series|yankees|dodgers|'
    r'red sox|cubs|giants|cardinals|braves|mets|astros|padres|blue jays|'
    r'orioles|rays|rangers|mariners|twins|guardians|royals|tigers|'
    r'white sox|angels|athletics|rockies|diamondbacks|marlins|brewers|'
    r'pirates|reds|phillies|nationals|baseball'
)
mask_baseball_title = df['title'].str.contains(BASEBALL_TERMS, case=False, na=False, regex=True)
mask_baseball_channel = df['channel'].str.contains(BASEBALL_TERMS, case=False, na=False, regex=True)
mask_baseball = mask_baseball_title | mask_baseball_channel

# 最终：必须同时满足比赛结构 + 棒球专项
mask_game = mask_struct & mask_baseball
print(f"  └ 不含比赛关键词: {(~mask_game).sum()} 条 → 剔除")
df = df[mask_game]

# 6c. 时长过滤：短于 60s 是剪辑/短片，长于 20h 是异常
mask_dur_ok = (df['duration_seconds'] >= 60) & (df['duration_seconds'] <= 72000)
print(f"  └ 时长异常 (<60s 或 >20h): {(~mask_dur_ok).sum()} 条 → 剔除")
df = df[mask_dur_ok]

# 6d. view_count 为 0 且无意义（可选，暂时保留但标记）
zero_views = (df['view_count'] == 0).sum()
if zero_views > 0:
    print(f"  └ view_count=0: {zero_views} 条（保留）")

# ── 7. 按 view_count 降序排列 ────────────────────────
df = df.sort_values('view_count', ascending=False).reset_index(drop=True)

print(f"\n{'='*50}")
print(f"最终: {len(df)} 行 (剔除 {n0 - len(df)} 行, {(n0 - len(df))/n0:.1%})")
print(f"列: {list(df.columns)}")
print(f"view_count 范围: {df['view_count'].min():,} ~ {df['view_count'].max():,}")
print(f"时长范围: {df['duration_seconds'].min()}s ~ {df['duration_seconds'].max()}s")

# ── 8. 保存 ──────────────────────────────────────────
df.to_csv(OUTPUT, index=False)
print(f"\n保存至: {OUTPUT}")
