#!/usr/bin/env python3
"""
phase_merge.py — 合并 clean_sports_v3 产物

将 run 目录下的 keep_high + keep_medium 合并为 keep.parquet，
从 baseline 反推 drop.parquet。

用法:
  python3 phase_merge.py run_dir/ -b baseline.parquet
  python3 phase_merge.py run_dir/ -b baseline.parquet --run run01
"""

import sys, os, argparse, glob
from pathlib import Path
import duckdb


def main():
    parser = argparse.ArgumentParser(description="合并 clean 产物")
    parser.add_argument("run_dir", help="clean 输出目录")
    parser.add_argument("-b", "--baseline", required=True, help="baseline parquet")
    parser.add_argument("--run", default=None, help="run 编号")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"[ERROR] 目录不存在: {args.run_dir}")
        sys.exit(1)

    run = args.run or os.path.basename(args.run_dir.rstrip("/"))
    high_files = glob.glob(os.path.join(args.run_dir, f"*{run}_keep_high.parquet"))
    med_files = glob.glob(os.path.join(args.run_dir, f"*{run}_keep_medium.parquet"))

    if not high_files:
        print(f"[ERROR] 未找到 keep_high.parquet")
        sys.exit(1)

    high_path = high_files[0]
    med_path = med_files[0] if med_files else None
    stem = os.path.basename(high_path).replace(f"_{run}_keep_high.parquet", "")
    keep_out = os.path.join(args.run_dir, f"{stem}_{run}_keep.parquet")

    con = duckdb.connect()
    n_high = con.execute(f"SELECT COUNT(*) FROM read_parquet('{high_path}')").fetchone()[0]
    n_med = 0
    if med_path:
        n_med = con.execute(f"SELECT COUNT(*) FROM read_parquet('{med_path}')").fetchone()[0]
        con.execute(f"COPY (SELECT * FROM read_parquet('{high_path}') UNION ALL SELECT * FROM read_parquet('{med_path}')) TO '{keep_out}' (FORMAT PARQUET)")
    else:
        con.execute(f"COPY (SELECT * FROM read_parquet('{high_path}')) TO '{keep_out}' (FORMAT PARQUET)")

    n_keep = con.execute(f"SELECT COUNT(*) FROM read_parquet('{keep_out}')").fetchone()[0]
    print(f"keep: {n_keep:,} (H={n_high:,} M={n_med:,})")

    drop_out = os.path.join(args.run_dir, f"{stem}_{run}_drop.parquet")
    con.execute(f"COPY (SELECT b.* FROM read_parquet('{args.baseline}') b WHERE b.video_id NOT IN (SELECT video_id FROM read_parquet('{keep_out}'))) TO '{drop_out}' (FORMAT PARQUET)")
    n_drop = con.execute(f"SELECT COUNT(*) FROM read_parquet('{drop_out}')").fetchone()[0]
    print(f"drop: {n_drop:,}")
    print(f"产物: {keep_out}")
    print(f"      {drop_out}")
    con.close()


if __name__ == "__main__":
    main()

