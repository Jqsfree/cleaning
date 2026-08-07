#!/usr/bin/env python3
"""
categories/film_tv/cleaner.py — 影视剧规则清洗（纯黑名单）

流程:
  Step 0: 加载数据（输入由 01_quality 确保已做质量+时长过滤）
  Step 1: pass2 黑名单
  Step 1b: r2 黑名单
  → 幸存行 = keep / 命中行 = drop

输出（CSV，有意为之；其它品类多为 parquet）:
  - {stem}_clean_{MMDD}.csv
  - {stem}_clean_drop_{MMDD}.csv
  - clean_summary.json
"""

import os
import time
from pathlib import Path

import duckdb

from core.log import log
from core.rules_loader import load_blacklist, compute_and_save_rule_stats
from core.sql_builder import (
    sql_escape,
    load_raw_table,
    add_search_text,
    write_summary_json,
    KEEP_EXCLUDE_DEFAULT,
)

_RULES_DIR = Path(__file__).resolve().parent / "rules"


def clean(
    input_path: str,
    stem: str = "clean",
    output_dir: str = "output",
    raw_name: str = "",
    run: str = "run01",
    fmt: str = "parquet",
    **kwargs,
) -> dict:
    t0 = time.perf_counter()
    log("影视剧规则清洗...")

    os.makedirs(output_dir, exist_ok=True)

    # ── 加载规则 ──
    rules = load_blacklist(_RULES_DIR)
    pass2_re = sql_escape(rules["pass2"])
    r2_re = sql_escape(rules["r2"])

    db = duckdb.connect(":memory:")

    # ── Step 0: 加载 ──
    n_total = load_raw_table(db, input_path)
    log(f"  rows: {n_total:,}")
    add_search_text(db)

    # ── Step 1: pass2 ──
    db.execute(f"""
        CREATE TEMP TABLE step1 AS
        SELECT *, 'step1_blacklist' AS drop_step,
               regexp_extract(title_channel, '{pass2_re}', 0) AS drop_reason
        FROM raw_text
        WHERE regexp_matches(title_channel, '{pass2_re}', 'i')
    """)
    n_bl = db.execute("SELECT COUNT(*) FROM step1").fetchone()[0]
    log(f"  pass2 drop: {n_bl:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_bl AS
        SELECT * FROM raw_text
        WHERE NOT regexp_matches(title_channel, '{pass2_re}', 'i')
    """)

    # ── Step 2: r2 ──
    db.execute(f"""
        CREATE TEMP TABLE step1b_r2 AS
        SELECT *, 'step1b_r2' AS drop_step,
               regexp_extract(title_channel, '{r2_re}', 0) AS drop_reason
        FROM after_bl
        WHERE regexp_matches(title_channel, '{r2_re}', 'i')
    """)
    n_r2 = db.execute("SELECT COUNT(*) FROM step1b_r2").fetchone()[0]
    log(f"  r2 drop:   {n_r2:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_r2 AS
        SELECT * FROM after_bl
        WHERE NOT regexp_matches(title_channel, '{r2_re}', 'i')
    """)

    n_keep = db.execute("SELECT COUNT(*) FROM after_r2").fetchone()[0]

    # ── 合并 drop ──
    db.execute("""
        CREATE TEMP TABLE dropped AS
        SELECT * EXCLUDE (search_text, title_channel, drop_step, drop_reason),
               drop_step, drop_reason
        FROM step1
        UNION ALL
        SELECT * EXCLUDE (search_text, title_channel, drop_step, drop_reason),
               drop_step, drop_reason
        FROM step1b_r2
    """)
    n_drop = db.execute("SELECT COUNT(*) FROM dropped").fetchone()[0]

    log(f"  keep: {n_keep:,} | drop: {n_drop:,}")

    # ── 规则命中统计 ──
    rule_stats = compute_and_save_rule_stats(db, _RULES_DIR)

    # ── 输出 ──
    base = raw_name if raw_name else stem
    date_tag = time.strftime("%m%d")
    out_keep = os.path.join(output_dir, f"{base}_clean_{date_tag}.csv")
    out_drop = os.path.join(output_dir, f"{base}_clean_drop_{date_tag}.csv")

    keep_exclude = {"search_text", "title_channel"}
    keep_cols = ','.join(f'"{c}"' for c in keep_exclude)
    db.execute(f"COPY (SELECT * EXCLUDE ({keep_cols}) FROM after_r2) TO '{sql_escape(out_keep)}' (FORMAT CSV, HEADER true)")
    db.execute(f"COPY dropped TO '{sql_escape(out_drop)}' (FORMAT CSV, HEADER true)")

    elapsed = time.perf_counter() - t0
    summary = {
        "engine": "film_tv-blacklist",
        "category": "film_tv",
        "input": os.path.abspath(input_path),
        "total_rows": n_total,
        "total_keep": n_keep,
        "total_drop": n_drop,
        "retention_pct": round(n_keep / max(n_total, 1) * 100, 1),
        "elapsed_sec": round(elapsed, 1),
        "steps": {
            "step1_pass2": {"dropped": n_bl},
            "step2_r2": {"dropped": n_r2},
        },
        "rule_stats": rule_stats,
        "keep_path": out_keep,
        "drop_path": out_drop,
    }

    write_summary_json(output_dir, summary)
    db.close()

    print(f"\n  总行数: {n_total:>12,}")
    print(f"  保留:   {n_keep:>12,} ({summary['retention_pct']}%)")
    print(f"  移除:   {n_drop:>12,}")
    print(f"  耗时:   {elapsed:>11.1f}s")
    print(f"  产物:   {output_dir}/")
    return summary
