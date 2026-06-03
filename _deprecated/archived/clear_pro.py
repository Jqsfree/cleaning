#!/usr/bin/env python3
"""
clear_pro.py — 去重 + 初筛（DuckDB）

阶段:
  1. 去重 — 按 video_id PARTITION BY rowid，保留首条
  2. 初筛 — 过滤 Private / Deleted 视频

每阶段输出筛掉条数 + DuckDB 内置进度条。

用法:
    python3 clear_pro.py
    python3 clear_pro.py /path/to/merged.csv
    python3 clear_pro.py merged.csv -o cleaned.csv -p       # 显示进度条
"""

import sys, os, time, argparse, textwrap
import duckdb

# ── 配置 ──
BLOCK_KEYWORDS = [
    "private video",
    "deleted video",
    "private",
    "deleted",
]

BLOCK_CONDITIONS = "\n      AND ".join(
    f"title NOT ILIKE '%{kw}%'" for kw in BLOCK_KEYWORDS
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="去重 + 初筛（DuckDB），逐阶段统计筛掉条数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python3 clear_pro.py merged.csv
              python3 clear_pro.py merged.csv -o clean.csv -p
        """),
    )
    parser.add_argument("input", nargs="?", default="merged.csv",
                        help="输入 CSV (默认: merged.csv)")
    parser.add_argument("-o", "--output", default="cleaned_records.csv",
                        help="输出 CSV (默认: cleaned_records.csv)")
    parser.add_argument("-p", "--progress", action="store_true",
                        help="显示 DuckDB 进度条")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    t0 = time.perf_counter()
    con = duckdb.connect()

    if args.progress:
        con.execute("SET enable_progress_bar = true")

    # ── Stage 0: 原始行数 ──
    log(f"读取: {args.input}")
    n_raw = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{args.input}', header=true)"
    ).fetchone()[0]
    log(f"  原始行数: {n_raw:,}")

    # ── Stage 1: 去重 ──
    log("Stage 1/2: 按 video_id 去重 ...")
    t1 = time.perf_counter()

    con.execute(f"""
        CREATE TEMP TABLE deduped AS
        SELECT * EXCLUDE (rn)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY rowid) AS rn
            FROM read_csv_auto('{args.input}', header=true)
        )
        WHERE rn = 1
    """)

    n_dedup = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
    n_dropped_dedup = n_raw - n_dedup
    log(f"  去重后:   {n_dedup:,}  ← 移除 {n_dropped_dedup:,} 条重复 "
        f"({n_dropped_dedup/max(n_raw,1)*100:.1f}%) "
        f"[{time.perf_counter()-t1:.1f}s]")

    # ── Stage 2: 初筛 ──
    log("Stage 2/2: 过滤 Private / Deleted 视频 ...")
    t2 = time.perf_counter()

    con.execute(f"""
        COPY (
            SELECT * FROM deduped
            WHERE {BLOCK_CONDITIONS}
        )
        TO '{args.output}' (HEADER, DELIMITER ',')
    """)

    n_final = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{args.output}', header=true)"
    ).fetchone()[0]
    n_dropped_filter = n_dedup - n_final
    log(f"  筛选后:   {n_final:,}  ← 移除 {n_dropped_filter:,} 条 "
        f"({n_dropped_filter/max(n_dedup,1)*100:.1f}%) "
        f"[{time.perf_counter()-t2:.1f}s]")

    con.close()

    # ── 汇总 ──
    elapsed = time.perf_counter() - t0
    total_dropped = n_raw - n_final

    print()
    print("=" * 62)
    print(f"  去重 + 初筛 完成")
    print("=" * 62)
    print(f"  原始:       {n_raw:>12,}")
    print(f"  去重移除:   {n_dropped_dedup:>12,}  ({n_dropped_dedup/max(n_raw,1)*100:5.1f}%)")
    print(f"  初筛移除:   {n_dropped_filter:>12,}  ({n_dropped_filter/max(n_dedup,1)*100:5.1f}%)")
    print(f"  {'─'*54}")
    print(f"  最终保留:   {n_final:>12,}  ({n_final/max(n_raw,1)*100:5.1f}%)")
    print(f"  总移除:     {total_dropped:>12,}  ({total_dropped/max(n_raw,1)*100:5.1f}%)")
    print(f"  耗时:       {elapsed:>11.1f}s")
    print(f"  输出:       {args.output}")
    print("=" * 62)


if __name__ == "__main__":
    main()
