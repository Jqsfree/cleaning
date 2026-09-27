"""
merge_records.py
递归搜索指定根目录下所有 record.csv，合并成一个文件。

用法：
    python merge_records.py                        # 默认从当前目录搜索
    python merge_records.py /home/jqs/tiyu         # 指定根目录
    python merge_records.py /data/a /data/b /data/c  # 指定多个目录
"""

import sys
import os
import pandas as pd
from pathlib import Path

# =====================
# 配置
# =====================

# 输出文件路径
OUTPUT_CSV = "merged_records.csv"

# 是否在合并结果中增加来源列（记录每行来自哪个文件）
ADD_SOURCE_COLUMN = True
SOURCE_COLUMN_NAME = "_source_file"

# 重复行处理：None=不去重, "all"=全字段去重, ["col1","col2"]=按指定列去重
DEDUP_SUBSET = None

# =====================
# 搜索目录
# =====================

if len(sys.argv) > 1:
    search_roots = [Path(p) for p in sys.argv[1:]]
else:
    search_roots = [Path(".")]

# =====================
# 递归查找所有 record.csv
# =====================

csv_files = []
for root in search_roots:
    found = sorted(root.rglob("records.csv"))
    csv_files.extend(found)

if not csv_files:
    print("未找到任何 record.csv 文件，请检查目录路径。")
    sys.exit(1)

print(f"找到 {len(csv_files)} 个 record.csv：")
for f in csv_files:
    print(f"  {f}")

# =====================
# 逐文件读取并合并
# =====================

frames = []
error_files = []

for fpath in csv_files:
    try:
        df = pd.read_csv(fpath)

        if ADD_SOURCE_COLUMN:
            df[SOURCE_COLUMN_NAME] = str(fpath)

        frames.append(df)
        print(f"  ✓ {fpath}  ({len(df)} 行, {len(df.columns)} 列)")

    except Exception as e:
        error_files.append((fpath, str(e)))
        print(f"  ✗ {fpath}  读取失败: {e}")

if not frames:
    print("所有文件读取失败，退出。")
    sys.exit(1)

# =====================
# 合并
# =====================

df_merged = pd.concat(frames, ignore_index=True)
print(f"\n合并后总行数: {len(df_merged)}")

# =====================
# 去重（可选）
# =====================

if DEDUP_SUBSET is not None:
    before = len(df_merged)
    subset = None if DEDUP_SUBSET == "all" else DEDUP_SUBSET
    df_merged = df_merged.drop_duplicates(subset=subset)
    print(f"去重后行数: {len(df_merged)}  (移除 {before - len(df_merged)} 条重复)")

# =====================
# 保存
# =====================

df_merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存至: {os.path.abspath(OUTPUT_CSV)}")

if error_files:
    print(f"\n以下 {len(error_files)} 个文件读取失败：")
    for fpath, err in error_files:
        print(f"  {fpath}: {err}")
