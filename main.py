import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
import time

# =====================
# DashScope / Qwen 配置
# =====================

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MODEL_NAME = "qwen3.6-flash"

# =====================
# 读取数据
# =====================

df = pd.read_csv("/home/jqs/tiyu/dispose_data/sample_videos.csv")

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
# 5. AI质检
# =====================

SYSTEM_PROMPT = """
分析该视频的内容是否属于体育赛事解说。

仅输出：
T
或
F
"""

def ai_qc(text):

    if not text:
        return "F"

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=1
        )

        result = (
            response.choices[0]
            .message.content
            .strip()
            .upper()
        )

        if result not in ["T", "F"]:
            return "F"

        return result

    except Exception as e:

        print("AI ERROR:", e)

        time.sleep(2)

        return "F"

# =====================
# 批量AI质检
# =====================

qc_results = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    result = ai_qc(row["title"])

    qc_results.append(result)

df["qc_ai"] = qc_results

# =====================
# 6. 最终过滤
# =====================

df = df[df["qc_ai"] == "T"]

# =====================
# 7. 保存结果
# =====================

df.to_csv(
    "/home/jqs/tiyu/dispose_data/final_baseball_videos.csv",
    index=False
)

print("清洗 + AI质检完成")
print("最终数量:", len(df))
