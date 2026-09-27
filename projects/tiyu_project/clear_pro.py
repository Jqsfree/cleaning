"""
clean_records.py
对合并后的 records.csv 进行：
  1. 按 video_id 去重
  2. 过滤 Private / Deleted 视频

使用 DuckDB，不全量加载进内存，适合大文件。

用法：
    python3 clean_records.py
    python3 clean_records.py /path/to/merged.csv
"""

import sys
import duckdb

# =====================
# 配置
# =====================

INPUT_CSV  = sys.argv[1] if len(sys.argv) > 1 else "merged.csv"
OUTPUT_CSV = "cleaned_records.csv"

# 过滤关键词（title 字段，不区分大小写）
BLOCK_KEYWORDS = [
    "private video",
    "deleted video",
    "private",
    "deleted",
]

# =====================
# 构建 SQL
# =====================

# 拼出 NOT ILIKE 过滤条件
block_conditions = "\n      AND ".join(
    f"title NOT ILIKE '%{kw}%'" for kw in BLOCK_KEYWORDS
)

sql = f"""
COPY (
    WITH deduped AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY rowid) AS rn
        FROM read_csv_auto('{INPUT_CSV}', header=true)
    )
    SELECT * EXCLUDE (rn)
    FROM deduped
    WHERE rn = 1
      AND {block_conditions}
)
TO '{OUTPUT_CSV}' (HEADER, DELIMITER ',')
"""

# =====================
# 执行
# =====================

print(f"输入: {INPUT_CSV}")
print("执行去重 + 初筛...")

con = duckdb.connect()
con.execute(sql)

# 输出结果统计
total_raw   = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{INPUT_CSV}')").fetchone()[0]
total_clean = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{OUTPUT_CSV}')").fetchone()[0]

print(f"原始行数:   {total_raw:,}")
print(f"清洗后行数: {total_clean:,}  (移除 {total_raw - total_clean:,} 条)")
print(f"输出: {OUTPUT_CSV}")
