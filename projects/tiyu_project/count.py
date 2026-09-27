import pandas as pd

INPUT_FILE = "/home/jqs/tiyu/dispose_data/live sports-pre_0528.csv"

CHUNK_SIZE = 100000

less_30s = 0
greater_6h = 0
greater_12h = 0

total = 0

print("开始统计...")

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    # 转换类型
    chunk["duration_seconds"] = pd.to_numeric(
        chunk["duration_seconds"],
        errors="coerce"
    )

    # 删除 NaN
    chunk = chunk.dropna(
        subset=["duration_seconds"]
    )

    total += len(chunk)

    # 小于30秒
    less_30s += (
        chunk["duration_seconds"] < 30
    ).sum()

    # 大于6小时
    greater_6h += (
        chunk["duration_seconds"] > 21600
    ).sum()

    # 大于12小时
    greater_12h += (
        chunk["duration_seconds"] > 43200
    ).sum()

print("\n统计完成")

print(f"总记录数: {total:,}")

print(f"小于30秒: {less_30s:,}")

print(f"大于6小时: {greater_6h:,}")

print(f"大于12小时: {greater_12h:,}")
