#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
phase0_normalize.py -- SOP Phase 0/1: 数据规范化 → baseline 产物

仅做通用数据质量处理，不做任何体育相关的过滤/打分。

输出:
  {output_dir}/{原始文件名}_raw.parquet
  {output_dir}/baseline_stats.md

用法:
  python3 phase0_normalize.py input.csv -o runs/001_baseline/
  python3 phase0_normalize.py input.csv -o runs/001_baseline/ --min-duration 10
"""

import sys, os, time, argparse, textwrap
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.sop import load_sop, print_banner
from core.progress import update, mark_done
from core.sop import write_run_log

BLOCK_KEYWORDS = ["private video", "deleted video", "private", "deleted"]
BLOCK_CONDITIONS = "\n      AND ".join(
    f"title NOT ILIKE '%{kw}%'" for kw in BLOCK_KEYWORDS
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def total_hours(con, table: str) -> float:
    """计算指定表的总时长（小时）"""
    try:
        secs = con.execute(
            f"SELECT SUM(TRY_CAST(duration_seconds AS DOUBLE)) FROM {table}"
        ).fetchone()[0]
        return (secs or 0) / 3600.0
    except Exception:
        return 0.0


def fmt_hours(h: float) -> str:
    if h >= 10000:
        return f"{h/10000:.1f}万h"
    return f"{h:,.1f}h"


def main():
    sop_text = load_sop()
    if sop_text:
        print(sop_text[:800])
        print("...\n")
    print_banner(0)

    parser = argparse.ArgumentParser(
        description="SOP Phase 0/1: 数据规范化 → baseline.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python3 phase0_normalize.py input.csv -o runs/001_baseline/
              python3 phase0_normalize.py input.csv -o runs/001_baseline/ --min-duration 10 -p
        """),
    )
    parser.add_argument("input", help="输入 CSV")
    parser.add_argument("-o", "--output-dir", default="data/runs/001_baseline",
                        help="输出目录 (默认: runs/001_baseline)")
    parser.add_argument("--min-duration", type=int, default=0)
    parser.add_argument("--max-duration", type=int, default=0)
    parser.add_argument("--keep-null", action="store_true",
                        help="保留空 title/keyword 行（跳过空值过滤）")
    parser.add_argument("-p", "--progress", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 路径不存在: {args.input}")
        sys.exit(1)

    # 目录支持：自动 glob *.csv
    if os.path.isdir(args.input):
        csv_glob = os.path.join(args.input, "*.csv")
        log(f"输入目录: {args.input} → {csv_glob}")
    else:
        csv_glob = args.input

    out_dir = args.output_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    con = duckdb.connect()
    if args.progress:
        con.execute("SET enable_progress_bar = true")

    stats = {}
    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    out_parquet = os.path.join(out_dir, f"{raw_stem}_raw.parquet")
    out_stats   = os.path.join(out_dir, "baseline_stats.md")

    # Stage 0
    log(f"读取: {csv_glob}")
    n_raw = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{csv_glob}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)"
    ).fetchone()[0]
    raw_h = con.execute(
        f"SELECT SUM(TRY_CAST(duration_seconds AS DOUBLE)) FROM read_csv_auto('{csv_glob}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)"
    ).fetchone()[0] or 0
    raw_h /= 3600.0
    stats["raw"] = n_raw
    log(f"  原始行数: {n_raw:,} ({fmt_hours(raw_h)})")

    # Stage 1: null
    if args.keep_null:
        log("Stage 1/5: 空值过滤 — 跳过 (--keep-null)")
        con.execute(f"""
            CREATE TEMP TABLE stage1 AS
            SELECT * FROM read_csv_auto('{csv_glob}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)
        """)
        n1 = n_raw
        stats["null_keep"] = n1
        stats["null_drop"] = 0
        h1 = raw_h
    else:
        log("Stage 1/5: 空值过滤 ...")
        t1 = time.perf_counter()
        con.execute(f"""
            CREATE TEMP TABLE stage1 AS
            SELECT * FROM read_csv_auto('{csv_glob}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)
            WHERE title IS NOT NULL AND title != ''
              AND keyword IS NOT NULL AND keyword != ''
        """)
        n1 = con.execute("SELECT COUNT(*) FROM stage1").fetchone()[0]
        stats["null_keep"] = n1
        stats["null_drop"] = n_raw - n1
        h1 = total_hours(con, "stage1")
        log(f"  保留: {n1:,} ({fmt_hours(h1)})  ← 移除 {n_raw-n1:,} ({n1/max(n_raw,1)*100:.1f}%) [{time.perf_counter()-t1:.1f}s]")
    update(out_dir, 0, stage="null_filter", done=n1, total=n_raw, pct=round(n1/max(n_raw,1)*100,1), hours=round(h1,1))

    # Stage 2: dedup
    log("Stage 2/5: 去重 ...")
    t2 = time.perf_counter()
    con.execute("""
        CREATE TEMP TABLE stage2 AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY rowid) AS rn
            FROM stage1
        ) WHERE rn = 1
    """)
    n2 = con.execute("SELECT COUNT(*) FROM stage2").fetchone()[0]
    stats["dedup_keep"] = n2
    stats["dedup_drop"] = n1 - n2
    h2 = total_hours(con, "stage2")
    log(f"  保留: {n2:,} ({fmt_hours(h2)})  ← 移除 {n1-n2:,} ({n2/max(n1,1)*100:.1f}%) [{time.perf_counter()-t2:.1f}s]")
    update(out_dir, 0, stage="dedup", done=n2, total=n1, pct=round(n2/max(n1,1)*100,1), hours=round(h2,1))

    # Stage 3: damaged
    log("Stage 3/5: 损坏数据 (Private/Deleted) ...")
    t3 = time.perf_counter()
    con.execute(f"""
        CREATE TEMP TABLE stage3 AS
        SELECT * FROM stage2 WHERE {BLOCK_CONDITIONS}
    """)
    n3 = con.execute("SELECT COUNT(*) FROM stage3").fetchone()[0]
    stats["damg_keep"] = n3
    stats["damg_drop"] = n2 - n3
    h3 = total_hours(con, "stage3")
    log(f"  保留: {n3:,} ({fmt_hours(h3)})  ← 移除 {n2-n3:,} ({n3/max(n2,1)*100:.1f}%) [{time.perf_counter()-t3:.1f}s]")
    update(out_dir, 0, stage="damaged", done=n3, total=n2, pct=round(n3/max(n2,1)*100,1), hours=round(h3,1))

    # Stage 4: duration
    if args.min_duration > 0 or args.max_duration > 0:
        log("Stage 4/5: 时长过滤 ...")
        t4 = time.perf_counter()
        conds = []
        if args.min_duration > 0:
            conds.append(f"TRY_CAST(duration_seconds AS DOUBLE) >= {args.min_duration}")
        if args.max_duration > 0:
            conds.append(f"TRY_CAST(duration_seconds AS DOUBLE) <= {args.max_duration}")
        con.execute(f"CREATE TEMP TABLE stage4 AS SELECT * FROM stage3 WHERE {' AND '.join(conds)}")
        n4 = con.execute("SELECT COUNT(*) FROM stage4").fetchone()[0]
        h4 = total_hours(con, "stage4")
        stats["dur_keep"] = n4
        stats["dur_drop"] = n3 - n4
        log(f"  保留: {n4:,} ({fmt_hours(h4)})  ← 移除 {n3-n4:,} ({n4/max(n3,1)*100:.1f}%) [{time.perf_counter()-t4:.1f}s]")
        src = "stage4"
    else:
        log("Stage 4/5: 时长过滤 — 跳过")
        stats["dur_keep"] = n3
        stats["dur_drop"] = 0
        src = "stage3"

    # Stage 5: write parquet + csv
    out_csv = out_parquet.replace(".parquet", ".csv")
    log(f"Stage 5/5: 写出 {raw_stem}_raw.parquet + .csv ...")
    t5 = time.perf_counter()
    con.execute(f"COPY (SELECT * FROM {src}) TO '{out_parquet}' (FORMAT PARQUET)")
    t5a = time.perf_counter()
    con.execute(f"COPY (SELECT * FROM {src}) TO '{out_csv}' (FORMAT CSV, HEADER true)")
    n_final = con.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
    final_h = total_hours(con, src)
    log(f"  parquet: {n_final:,} 行 ({fmt_hours(final_h)}) [{t5a-t5:.1f}s]")
    log(f"  csv:     {n_final:,} 行 ({fmt_hours(final_h)}) [{time.perf_counter()-t5a:.1f}s]")
    log(f"  总写出:  {time.perf_counter()-t5:.1f}s")
    con.close()

    # baseline_stats.md
    elapsed = time.perf_counter() - t0
    with open(out_stats, "w") as f:
        f.write(f"# Baseline Statistics\n\n")
        f.write(f"**Input:** `{args.input}`\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Elapsed:** {elapsed:.1f}s\n\n")
        f.write(f"| Stage | Retained | Dropped |\n")
        f.write(f"|-------|----------|--------|\n")
        f.write(f"| Raw input | {n_raw:,} | -- |\n")
        f.write(f"| Null filter | {stats['null_keep']:,} | {stats['null_drop']:,} |\n")
        f.write(f"| Dedup | {stats['dedup_keep']:,} | {stats['dedup_drop']:,} |\n")
        f.write(f"| Damaged | {stats['damg_keep']:,} | {stats['damg_drop']:,} |\n")
        f.write(f"| Duration | {stats['dur_keep']:,} | {stats['dur_drop']:,} |\n")
        f.write(f"| **Final** | **{n_final:,}** | **{n_raw-n_final:,}** |\n\n")
        f.write(f"**Retention:** {n_final/max(n_raw,1)*100:.1f}%\n")
        f.write(f"**Total Duration:** {fmt_hours(final_h)}\n\n")
        f.write(f"**Next:** Phase 2\n```bash\n")
        f.write(f"python3 phase2_sample.py {out_parquet} -o runs/002_audit/\n```\n")

    # summary
    print()
    print("=" * 62)
    print(f"  Phase 0/1 -- Baseline 完成")
    print("=" * 62)
    print(f"  原始:       {n_raw:>12,}  ({fmt_hours(raw_h)})")
    print(f"  空值:       {stats['null_drop']:>12,}")
    print(f"  去重:       {stats['dedup_drop']:>12,}")
    print(f"  损坏:       {stats['damg_drop']:>12,}")
    print(f"  时长过滤:   {stats['dur_drop']:>12,}")
    print(f"  {'─'*54}")
    print(f"  保留:       {n_final:>12,}  ({n_final/max(n_raw,1)*100:5.1f}%)  {fmt_hours(final_h)}")
    print(f"  耗时:       {elapsed:>11.1f}s")
    print(f"  产物:       {out_dir}/")
    print(f"              {raw_stem}_raw.parquet")
    print(f"              {raw_stem}_raw.csv")
    print(f"              baseline_stats.md")
    print("=" * 62)

    # 运行日志
    mark_done(out_dir, 0, final=n_final, retention_pct=round(n_final/max(n_raw,1)*100,1), elapsed_sec=round(elapsed,1))
    write_run_log(0, args.input, out_dir,
                  stats={"raw_rows": n_raw, "null_dropped": stats['null_drop'],
                         "dedup_dropped": stats['dedup_drop'], "damaged_dropped": stats['damg_drop'],
                         "duration_dropped": stats['dur_drop'], "final_rows": n_final,
                         "retention_pct": round(n_final/max(n_raw,1)*100, 1), "elapsed_sec": round(elapsed, 1)},
                  command=f"phase0_normalize.py {args.input} -o {out_dir}")


if __name__ == "__main__":
    main()
