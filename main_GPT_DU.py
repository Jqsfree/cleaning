import pandas as pd
from tqdm import tqdm
import os

INPUT_FILE = "/home/jqs/tiyu/data/live sports-pre_0528.csv"

OUTPUT_FILE = "/home/jqs/tiyu/dispose_data/live sports-pre_0528.csv"

CHUNK_SIZE = 100000

BLOCK_KEYWORDS = [
    "private",
    "deleted"
]

# =====================
# 获取文件大小
# =====================

file_size = os.path.getsize(INPUT_FILE)

# 已处理字节
processed_bytes = 0

# video_id 去重集合
seen_video_ids = set()

# 是否首次写入
first_write = True

print("开始处理...")

with tqdm(
    total=file_size,
    unit="B",
    unit_scale=True,
    desc="处理进度"
) as pbar:

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE
    ):

        # =====================
        # 统计进度
        # =====================

        chunk_memory = chunk.memory_usage(
            deep=True
        ).sum()

        processed_bytes += chunk_memory

        pbar.update(chunk_memory)

        # =====================
        # title 标准化
        # =====================

        chunk["title"] = (
            chunk["title"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # =====================
        # 过滤 private/deleted
        # =====================

        pattern = "|".join(BLOCK_KEYWORDS)

        chunk = chunk[
            ~chunk["title"].str.contains(
                pattern,
                case=False,
                na=False
            )
        ]

        # =====================
        # video_id 去重
        # =====================

        chunk = chunk[
            ~chunk["video_id"].isin(
                seen_video_ids
            )
        ]

        # 更新去重集合
        seen_video_ids.update(
            chunk["video_id"].tolist()
        )

        # =====================
        # 写入结果
        # =====================

        chunk.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=first_write
        )

        first_write = False

        # =====================
        # 实时状态
        # =====================

        pbar.set_postfix({
            "保留数据": len(seen_video_ids)
        })

print("处理完成")
print("最终数量:", len(seen_video_ids))
