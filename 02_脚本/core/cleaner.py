#!/usr/bin/env python3
"""
core/cleaner.py — Pass 2: DuckDB 多步过滤 + Parquet 输出

流程:
  Step 1: 黑名单 drop
  Step 2: 弱信号 drop（无实体对齐 + 无强信号 + 低分）
  Step 3: 污染 playlist drop
  Step 4: 计分决定 high / medium / drop

输出:
  - {stem}_high.parquet  /  .csv
  - {stem}_medium.parquet / .csv
  - {stem}_all.parquet    / .csv
  - {stem}_dropped.parquet / .csv
  - summary JSON
"""

import time, json, os
from pathlib import Path
import duckdb
import pandas as pd

from .scoring import register_udfs, get_thresholds


def clean(input_path: str, polluted: set, stem: str = "clean",
          output_dir: str = "output",
          raw_name: str = "", run: str = "run01",
          keep_score: int | None = None, gray_low: int | None = None,
          med_min: int | None = None,
          fmt: str = "parquet") -> dict:
    """
    Pass 2: DuckDB 多步清洗。

    fmt: "parquet" | "csv" | "both"
    """
    # Load thresholds from rules if not explicitly provided
    thresholds = get_thresholds()
    keep_score = keep_score if keep_score is not None else thresholds["keep_score"]
    gray_low = gray_low if gray_low is not None else thresholds["gray_score_low"]
    med_min = med_min if med_min is not None else thresholds["medium_min_score"]

    t0 = time.perf_counter()
    print(f"Pass 2: cleaning (DuckDB, fmt={fmt})...")

    os.makedirs(output_dir, exist_ok=True)

    db = duckdb.connect(":memory:")
    register_udfs(db)

    # 污染 playlist 表
    db.execute("CREATE TEMP TABLE polluted_playlists(source_ref VARCHAR)")
    if polluted:
        for p in polluted:
            db.execute("INSERT INTO polluted_playlists VALUES (?)", [p])

    # ── 主查询：多步 CTE 流水线 ──
    # 为了避免 DuckDB 的 SQL 太长，分步查询
    ext = os.path.splitext(input_path)[1].lower()
    reader = "read_parquet" if ext == ".parquet" else "read_csv_auto"
    reader_args = f"('{input_path}')" if ext == ".parquet" else f"('{input_path}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)"

    db.execute(f"""
        CREATE TEMP TABLE raw AS
        SELECT *,
               blacklist_pass2(title, channel, keyword) AS bl_match,
               sport_score(title, channel, keyword) AS s_score,
               keyword_aligned(keyword, title, channel) AS kw_aligned,
               strong_sport_signal(title, channel) AS strong_sig,
               parse_entities(keyword) AS kw_entities
        FROM {reader}{reader_args}
    """)

    n_total = db.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"  rows: {n_total:,}")

    # Step 1: blacklist
    db.execute("""
        CREATE TEMP TABLE step1 AS
        SELECT *, 'step1_blacklist' AS drop_step, bl_match AS drop_reason
        FROM raw WHERE bl_match != ''
    """)
    n_bl = db.execute("SELECT COUNT(*) FROM step1").fetchone()[0]

    db.execute("""
        CREATE TEMP TABLE after_bl AS
        SELECT * FROM raw WHERE bl_match = ''
    """)

    # Step 2: no signal
    db.execute(f"""
        CREATE TEMP TABLE step2_no_signal AS
        SELECT *, 'step2_no_signal' AS drop_step,
               'no_align_no_signal:' || COALESCE(kw_entities, '') AS drop_reason
        FROM after_bl
        WHERE kw_entities != ''
          AND NOT kw_aligned
          AND NOT strong_sig
    """)
    n_no_sig = db.execute("SELECT COUNT(*) FROM step2_no_signal").fetchone()[0]

    db.execute("""
        CREATE TEMP TABLE after_signal AS
        SELECT * FROM after_bl
        WHERE NOT (kw_entities != '' AND NOT kw_aligned AND NOT strong_sig)
    """)

    # Step 3: polluted playlist
    db.execute("""
        CREATE TEMP TABLE step4_playlist AS
        SELECT a.*, 'step4_playlist' AS drop_step, 'low_playlist_hit_rate' AS drop_reason
        FROM after_signal a
        INNER JOIN polluted_playlists p ON a.source_ref = p.source_ref
    """)
    n_pl = db.execute("SELECT COUNT(*) FROM step4_playlist").fetchone()[0]

    db.execute("""
        CREATE TEMP TABLE after_pl AS
        SELECT a.* FROM after_signal a
        WHERE a.source_ref NOT IN (SELECT source_ref FROM polluted_playlists)
           OR a.source_ref IS NULL
           OR a.source_ref = ''
    """)

    # Step 4: scoring decision
    db.execute(f"""
        CREATE TEMP TABLE scored AS
        SELECT *,
               CASE
                   WHEN s_score >= {keep_score} THEN 'high_score'
                   WHEN kw_aligned AND s_score >= {gray_low} THEN 'gray_aligned'
                   WHEN (kw_entities != '' AND NOT kw_aligned AND strong_sig AND s_score >= {med_min})
                        THEN 'medium_strong_signal'
                   WHEN NOT kw_aligned AND NOT strong_sig AND s_score < {gray_low}
                        THEN 'low_score_no_signal'
                   WHEN kw_aligned AND s_score < {gray_low}
                        THEN 'aligned_low_score'
                   WHEN kw_entities != '' AND NOT kw_aligned AND NOT strong_sig
                        THEN 'medium_no_strong_signal'
                   WHEN kw_entities != '' AND NOT kw_aligned AND strong_sig AND s_score < {med_min}
                        THEN 'medium_low_score'
                   ELSE 'default_drop'
               END AS reason,
               CASE
                   WHEN s_score >= {keep_score} THEN 'high'
                   WHEN kw_aligned AND s_score >= {gray_low} THEN 'high'
                   WHEN (kw_entities != '' AND NOT kw_aligned AND strong_sig AND s_score >= {med_min})
                        THEN 'medium'
                   ELSE 'drop'
               END AS tier
        FROM after_pl
    """)

    # 分离 high/medium/drop
    db.execute("""
        CREATE TEMP TABLE keep_high AS
        SELECT * FROM scored WHERE tier = 'high'
    """)
    db.execute("""
        CREATE TEMP TABLE keep_medium AS
        SELECT * FROM scored WHERE tier = 'medium'
    """)
    db.execute("""
        CREATE TEMP TABLE dropped AS
        SELECT *, 'step_other' AS drop_step, reason AS drop_reason
        FROM scored WHERE tier = 'drop'
        UNION ALL
        SELECT *, drop_step, drop_reason FROM step1
        UNION ALL
        SELECT *, drop_step, drop_reason FROM step2_no_signal
        UNION ALL
        SELECT *, drop_step, drop_reason FROM step4_playlist
    """)

    n_high = db.execute("SELECT COUNT(*) FROM keep_high").fetchone()[0]
    n_medium = db.execute("SELECT COUNT(*) FROM keep_medium").fetchone()[0]
    n_drop = db.execute("SELECT COUNT(*) FROM dropped").fetchone()[0]
    n_keep = n_high + n_medium

    print(f"  keep: {n_keep:,} (H={n_high:,} M={n_medium:,}) | drop: {n_drop:,}")

    # ── 输出 ──
    base = raw_name if raw_name else stem
    out_high    = os.path.join(output_dir, f"{base}_{run}_keep_high.parquet")
    out_medium  = os.path.join(output_dir, f"{base}_{run}_keep_medium.parquet")
    out_all     = os.path.join(output_dir, f"{base}_{run}_keep.parquet")
    out_dropped = os.path.join(output_dir, f"{base}_{run}_drop.parquet")

    # 内部辅助列，不输出到最终文件
    _AUX_COLS = {'bl_match', 's_score', 'kw_aligned', 'strong_sig', 'kw_entities', 'drop_step', 'drop_reason', 'tier', 'reason'}
    select_cols = db.execute("SELECT * FROM keep_high LIMIT 0").description
    all_col_names = [c[0] for c in select_cols]

    def write_table(table_name, base_path):
        cols = [c for c in all_col_names if c not in _AUX_COLS]
        col_str = ", ".join(f'"{c}"' for c in cols)
        sql = f'SELECT {col_str} FROM {table_name}'
        db.execute(f"COPY ({sql}) TO '{base_path}' (FORMAT PARQUET)")

    write_table("keep_high", out_high)
    write_table("keep_medium", out_medium)


    # ── summary ──
    elapsed = time.perf_counter() - t0
    summary = {
        "step": "clean_v3",
        "engine": "duckdb",
        "input": os.path.abspath(input_path),
        "total_rows": n_total,
        "total_keep": n_keep,
        "total_keep_high": n_high,
        "total_keep_medium": n_medium,
        "total_drop": n_drop,
        "elapsed_sec": round(elapsed, 1),
        "steps": {
            "step1_blacklist": {"dropped": n_bl},
            "step2_no_signal": {"dropped": n_no_sig},
            "step4_playlist": {"dropped": n_pl},
        },
    }

    summary_path = os.path.join(output_dir, "clean_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    db.close()

    print(f"  output: {output_dir}/ ({fmt})")
    return summary
