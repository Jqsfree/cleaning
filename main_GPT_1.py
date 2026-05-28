import pandas as pd

# 读取数据
df = pd.read_csv("/home/jqs/tiyu/data/live sports-pre_0528.csv")

# =====================
# 1. 去重
# =====================

df = df.drop_duplicates(subset=["video_id"])

# =====================
# 2. 基础清洗
# =====================

df = df.dropna(
    subset=[
        "title",
        "channel",
        "duration_seconds",
        "view_count"
    ]
)

# 转换类型
df["duration_seconds"] = pd.to_numeric(
    df["duration_seconds"],
    errors="coerce"
)

df["view_count"] = pd.to_numeric(
    df["view_count"],
    errors="coerce"
)

# 删除异常值
df = df[df["duration_seconds"] > 30]
df = df[df["view_count"] > 100]

# =====================
# 3. 文本标准化
# =====================

df["title"] = (
    df["title"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# =====================
# 4. 过滤会员/私密/下架视频
# =====================

BLOCK_KEYWORDS = [
    "private video",
    "deleted video",
    "members only",
    "subscriber only",
    "video unavailable"
]

pattern = "|".join(BLOCK_KEYWORDS)

df = df[
    ~df["title"].str.contains(
        pattern,
        case=False,
        na=False
    )
]

# =====================
# 5. 保存结果
# =====================

df.to_csv(
    "/home/jqs/tiyu/dispose_data/live sports-pre_0528.csv",
    index=False
)

print("清洗完成")
print("剩余数据量:", len(df))
