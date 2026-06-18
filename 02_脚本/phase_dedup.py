#!/usr/bin/env python3
"""
phase_dedup.py — 新批次 vs 旧批次去重

从 baseline 中移除旧批次已交付的 video_id。

用法:
  # 单个旧交付
  python3 phase_dedup.py baseline.parquet -d old_deliver/keep_final.csv -o deduped.parquet

  # 多个旧交付（跨运动去重）
  python3 phase_dedup.py baseline.parquet -d data/runs/data_ONE/*/deliver/ -o deduped.parquet

  # 指定旧批次目录（自动找 deliver/）
  python3 phase_dedup.py baseline.parquet --old-run data/runs/data_ONE/curling_one/ -o deduped.parquet
"""

import sys, os, argparse, glob
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sop import write_run_log


def find_deliver_csvs(source: str) -> list:
    """从目录或文件路径解析交付 CSV 列表。"""
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        deliver = os.path.join(source, "deliver")
        if os.path.isdir(deliver):
            csvs = glob.glob(os.path.join(deliver, "*_keep_final.csv"))
            if csvs:
                return csvs
        csvs = glob.glob(os.path.join(source, "*_keep_final.csv"))
        if csvs:
            return csvs
    return []


def main():
    parser = argparse.ArgumentParser(description="新批次 vs 旧批次去重")
    parser.add_argument("input", help="新批次 baseline.parquet")
    parser.add_argument("-d", "--deliveries", nargs="+", default=[],
                        help="旧批次交付 CSV 或目录（可多个）")
    parser.add_argument("--old-run", nargs="+", default=[],
                        help="旧批次 run 目录")
    parser.add_argument("-o", "--output", required=True,
                        help="输出去重后的 parquet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] baseline 不存在: {args.input}")
        sys.exit(1)

    old_csvs = []
    for d in args.deliveries:
        old_csvs.extend(find_deliver_csvs(d))
    for r in args.old_run:
        old_csvs.extend(find_deliver_csvs(r))

    if not old_csvs:
        print("[ERROR] 未找到旧交付。用 -d 或 --old-run 指定。")
        sys.exit(1)

    print(f"旧交付: {len(old_csvs)} 个")
    for c in old_csvs:
        print(f"  {c}")

    con = duckdb.connect()
    n_bl = con.execute(f"SELECT COUNT(*) FROM read_parquet('{args.input}')").fetchone()[0]
    print(f"\nbaseline: {n_bl:,}")

    parts = []
    for c in old_csvs:
        parts.append(f"SELECT video_id FROM read_csv_auto('{c}', header=true, all_varchar=true, sample_size=-1)")
    sql = " UNION ALL ".join(parts)
    con.execute(f"CREATE TEMP TABLE old_ids AS SELECT DISTINCT video_id FROM ({sql}) t")
    n_old = con.execute("SELECT COUNT(*) FROM old_ids").fetchone()[0]
    print(f"旧 ID 去重后: {n_old:,}")

    con.execute(f"""
        CREATE TEMP TABLE deduped AS
        SELECT b.* FROM read_parquet('{args.input}') b
        WHERE b.video_id NOT IN (SELECT video_id FROM old_ids)
    """)
    n_out = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
    print(f"baseline: {n_bl:,} → 去重后: {n_out:,} (移除 {n_bl-n_out:,})")

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        con.execute(f"COPY (SELECT * FROM deduped) TO '{args.output}' (FORMAT PARQUET)")
        print(f"已写出: {args.output}")
    else:
        print("[dry-run]")

    con.close()


if __name__ == "__main__":
    main()

