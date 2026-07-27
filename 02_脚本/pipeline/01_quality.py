#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
初筛脚本 — 对原始 CSV 做质量过滤 + 批内去重 + 时长过滤，不涉及任何内容规则。

功能:
  - 移除空标题、空 video_id
  - 移除已删除/私享/会员/不可用视频（标题关键词）
  - 移除零时长或负时长
  - 批内去重：同一 video_id 保留第一条
  - 时长范围过滤（默认 1 分钟 – 4 小时）

用法:
  python3 02_脚本/pipeline/01_quality.py raw/xxx.csv -o data/runs/001_quality/
  python3 02_脚本/pipeline/01_quality.py raw/xxx.csv --min 60 --max 14400 -o data/runs/001_quality/
"""

import os, sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
from core.log import log
from core.progress import mark_done, update
from core.sop import print_banner, write_run_log
from core.sql_builder import load_raw_table, sql_escape


def quality_check(
    input_path: str,
    output_dir: str,
    stem: str = "",
    min_duration: int = 60,
    max_duration: int = 14400,
):
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    update(output_dir, "quality", status="running", input=input_path)

    if not stem:
        raw_name = os.path.splitext(os.path.basename(input_path))[0]
        stem = raw_name.replace("_raw", "") if "_raw" in raw_name else raw_name

    db = duckdb.connect(":memory:")

    # ── Step 0: 加载 ──
    n_total = load_raw_table(db, input_path)
    log(f"rows: {n_total:,}")

    # ── Step 1: 数据质量 ──
    db.execute("""
        CREATE TEMP TABLE drop_quality AS
        SELECT *, 'quality' AS drop_step,
               CASE
                   WHEN title IS NULL OR title = '' THEN 'empty_title'
                   WHEN video_id IS NULL OR video_id = '' THEN 'empty_video_id'
                   WHEN regexp_matches(LOWER(COALESCE(title,'')),
                       'deleted video|private video|members only|unavailable|removed|this video is|video unavailable|content unavailable|account terminated')
                       THEN 'deleted_or_private'
                   WHEN TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) IS NULL THEN 'invalid_duration'
                   WHEN TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) <= 0 THEN 'zero_duration'
                   ELSE 'other'
               END AS drop_reason
        FROM raw
        WHERE (title IS NULL OR title = '')
           OR (video_id IS NULL OR video_id = '')
           OR regexp_matches(LOWER(COALESCE(title,'')),
               'deleted video|private video|members only|unavailable|removed|this video is|video unavailable|content unavailable|account terminated')
           OR TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) <= 0
           OR TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) IS NULL
    """)
    n_quality = db.execute("SELECT COUNT(*) FROM drop_quality").fetchone()[0]
    log(f"质量过滤: 移除 {n_quality:,} 条 (空标题/已删除/零时长)")

    db.execute("""
        CREATE TEMP TABLE after_quality AS
        SELECT * FROM raw
        WHERE (title IS NOT NULL AND title != '')
          AND (video_id IS NOT NULL AND video_id != '')
          AND NOT regexp_matches(LOWER(COALESCE(title,'')),
              'deleted video|private video|members only|unavailable|removed|this video is|video unavailable|content unavailable|account terminated')
          AND TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) > 0
          AND TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) IS NOT NULL
    """)

    # ── Step 2: 批内去重 ──
    n_before_dedup = db.execute("SELECT COUNT(*) FROM after_quality").fetchone()[0]

    db.execute("""
        CREATE TEMP TABLE after_dedup AS
        SELECT * FROM after_quality
        WHERE rowid IN (SELECT min(rowid) FROM after_quality GROUP BY video_id)
    """)
    n_after_dedup = db.execute("SELECT COUNT(*) FROM after_dedup").fetchone()[0]
    n_dedup = n_before_dedup - n_after_dedup

    db.execute("""
        CREATE TEMP TABLE drop_dedup AS
        SELECT *, 'dedup' AS drop_step, 'duplicate_video_id' AS drop_reason
        FROM after_quality
        WHERE rowid NOT IN (SELECT min(rowid) FROM after_quality GROUP BY video_id)
    """)

    if n_dedup > 0:
        log(f"批内去重: 移除 {n_dedup:,} 条 (重复 video_id)")

    # ── Step 3: 时长过滤 ──
    db.execute(f"""
        CREATE TEMP TABLE drop_duration AS
        SELECT *, 'duration' AS drop_step,
               CASE WHEN TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) < {min_duration}
                    THEN 'duration<{min_duration}s'
                    ELSE 'duration>{max_duration}s'
               END AS drop_reason
        FROM after_dedup
        WHERE TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) < {min_duration}
           OR TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) > {max_duration}
    """)
    n_dur = db.execute("SELECT COUNT(*) FROM drop_duration").fetchone()[0]
    log(f"时长过滤 ({min_duration}s–{max_duration}s): 移除 {n_dur:,} 条")

    db.execute(f"""
        CREATE TEMP TABLE keep AS
        SELECT * FROM after_dedup
        WHERE TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) >= {min_duration}
          AND TRY_CAST(COALESCE(duration_seconds,'0') AS DOUBLE) <= {max_duration}
    """)
    n_keep = db.execute("SELECT COUNT(*) FROM keep").fetchone()[0]

    # ── 输出 ──
    date_tag = time.strftime("%m%d")
    out_keep = os.path.join(output_dir, f"{stem}_quality_{date_tag}.csv")
    out_drop = os.path.join(output_dir, f"{stem}_quality_drop_{date_tag}.csv")

    db.execute(f"COPY keep TO '{sql_escape(out_keep)}' (FORMAT CSV, HEADER true)")

    db.execute("CREATE TEMP TABLE drop_all AS SELECT * FROM drop_quality UNION ALL SELECT * FROM drop_dedup UNION ALL SELECT * FROM drop_duration")
    db.execute(f"COPY drop_all TO '{sql_escape(out_drop)}' (FORMAT CSV, HEADER true)")

    elapsed = time.perf_counter() - t0

    print()
    log(f"初筛完成: {n_total:,} → {n_keep:,} ({n_keep/max(1,n_total)*100:.1f}%)  |  质量-{n_quality:,} + 去重-{n_dedup:,} + 时长-{n_dur:,}  |  {elapsed:.1f}s")
    print(f"  keep: {out_keep}")
    print(f"  drop: {out_drop}")

    db.close()
    stats = {
        "total_rows": n_total,
        "keep": n_keep,
        "drop_quality": n_quality,
        "drop_dedup": n_dedup,
        "drop_duration": n_dur,
        "retention_pct": round(n_keep / max(1, n_total) * 100, 1),
        "elapsed_sec": round(elapsed, 1),
        "keep_path": out_keep,
        "drop_path": out_drop,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="初筛：质量过滤 + 批内去重 + 时长过滤（唯一默认入口）")
    parser.add_argument("input", help="原始 CSV 文件")
    parser.add_argument("-o", "--output-dir", default="data/runs/film_tv/01_quality/",
                        help="输出目录（示例默认偏 film_tv；其它品类请显式指定）")
    parser.add_argument("--min", type=int, default=60, help="最小时长(秒)")
    parser.add_argument("--max", type=int, default=14400, help="最大时长(秒)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log(f"文件不存在: {args.input}", level="ERROR")
        sys.exit(1)

    print_banner("quality")
    stats = quality_check(
        args.input, args.output_dir,
        min_duration=args.min, max_duration=args.max,
    )
    mark_done(
        args.output_dir, "quality",
        final=stats["keep"], total=stats["total_rows"],
        retain_pct=stats["retention_pct"], elapsed_sec=stats["elapsed_sec"],
    )
    write_run_log(
        "quality", args.input, args.output_dir, stats=stats,
        command=f"01_quality.py {args.input} -o {args.output_dir}",
    )


if __name__ == "__main__":
    main()
