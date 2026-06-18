#!/usr/bin/env python3
"""
dedup_new_batch.py — 新批次 CSV vs 旧基线去重

从新批次 CSV 中移除旧基线已有的 video_id，输出去重后 CSV。
输出文件名：{原名}_dedup.csv，保存在与输入相同的目录。

用法:
  python3 dedup_new_batch.py new_batch.csv baseline.parquet
  python3 dedup_new_batch.py new_batch.csv baseline.parquet -o custom_output.csv
"""

import sys, os, argparse
from pathlib import Path
import duckdb


def main():
    parser = argparse.ArgumentParser(description="新批次 CSV vs 旧基线去重")
    parser.add_argument("new_csv", help="新批次 CSV 文件")
    parser.add_argument("baseline", help="旧基线 parquet 文件")
    parser.add_argument("-o", "--output", default=None,
                        help="输出路径（默认：与输入同目录，{原名}_dedup.csv）")
    args = parser.parse_args()

    if not os.path.exists(args.new_csv):
        print(f"[ERROR] 新批次文件不存在: {args.new_csv}")
        sys.exit(1)
    if not os.path.exists(args.baseline):
        print(f"[ERROR] 旧基线文件不存在: {args.baseline}")
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        in_dir = os.path.dirname(os.path.abspath(args.new_csv))
        stem = Path(args.new_csv).stem
        out_path = os.path.join(in_dir, f"{stem}_dedup.csv")

    db = duckdb.connect()

    n_new = db.execute(f"""
        SELECT COUNT(*) FROM read_csv_auto('{args.new_csv}',
            header=true, all_varchar=true, sample_size=-1)
    """).fetchone()[0]

    n_old = db.execute(f"SELECT COUNT(*) FROM '{args.baseline}'").fetchone()[0]

    print(f"新批次: {n_new:,} 行")
    print(f"旧基线: {n_old:,} 行")

    db.execute(f"""
        COPY (
            SELECT n.*
            FROM read_csv_auto('{args.new_csv}',
                header=true, all_varchar=true, sample_size=-1) n
            ANTI JOIN '{args.baseline}' o ON n.video_id = o.video_id
        ) TO '{out_path}' (HEADER, DELIMITER ',')
    """)

    n_out = db.execute(f"""
        SELECT COUNT(*) FROM read_csv_auto('{out_path}',
            header=true, all_varchar=true, sample_size=-1)
    """).fetchone()[0]
    size_mb = os.path.getsize(out_path) / 1024 / 1024

    print(f"重复: {n_new - n_out:,} ({(n_new - n_out)/max(n_new,1)*100:.1f}%)")
    print(f"去重后: {n_out:,} 行 ({size_mb:.1f} MB)")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
