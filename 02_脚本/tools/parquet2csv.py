#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
parquet2csv.py — parquet → csv 转换

用法:
  python3 parquet2csv.py input.parquet
  python3 parquet2csv.py input.parquet -o output.csv
"""
import sys, argparse, duckdb

parser = argparse.ArgumentParser(description="parquet → csv")
parser.add_argument("input", help="输入 parquet")
parser.add_argument("-o", "--output", default=None, help="输出 csv（默认同目录同名 .csv）")
args = parser.parse_args()

out = args.output or args.input.replace(".parquet", ".csv")
duckdb.connect().execute(f"COPY (SELECT * FROM '{args.input}') TO '{out}' (FORMAT CSV, HEADER true)")
print(f"✅ {args.input} → {out}")
