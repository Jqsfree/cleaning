#!/usr/bin/env python3
"""
categories/beauty/cleaner.py — 美妆黑名单清洗

纯 SQL 过滤（pass2 + r2 → keep/drop），无打分 UDF、无 tier 分档。

流程:
  Step 0: 加载 raw 数据
  Step 1: pass2 黑名单 drop
  Step 2: r2 黑名单 drop
  → 幸存行全部 keep

输出:
  - {stem}_{run}_keep.parquet
  - {stem}_{run}_drop.parquet
  - clean_summary.json
"""

import os
import time
from pathlib import Path

import duckdb

from core.rules_loader import load_blacklist
from core.sql_builder import (
    sql_escape,
    load_raw_table,
    add_search_text,
    write_parquet_with_excludes,
    write_summary_json,
)

RULES_DIR = Path(__file__).resolve().parent / "rules"

_KEEP_EXCLUDE = {"search_text", "title_channel", "drop_step", "drop_reason"}
_DROP_EXCLUDE = {"search_text", "title_channel"}


def clean(
    input_path: str,
    stem: str = "clean",
    output_dir: str = "output",
    raw_name: str = "",
    run: str = "run01",
    fmt: str = "parquet",
) -> dict:
    t0 = time.perf_counter()
    print("美妆规则清洗 (pass2 + r2 黑名单)...")

    os.makedirs(output_dir, exist_ok=True)

    # ── 加载规则 ──
    rules = load_blacklist(RULES_DIR)
    pass2_re = sql_escape(rules["pass2"])
    r2_re = sql_escape(rules["r2"])

    db = duckdb.connect(":memory:")

    # ── Step 0: 加载 raw ──
    n_total = load_raw_table(db, input_path)
    print(f"  rows: {n_total:,}")

    add_search_text(db)

    # ── Step 1: pass2 黑名单 ──
    db.execute(f"""
        CREATE TEMP TABLE step1 AS
        SELECT *, 'step1_blacklist' AS drop_step,
               regexp_extract(search_text, '{pass2_re}') AS drop_reason
        FROM raw_text
        WHERE regexp_matches(search_text, '{pass2_re}', 'i')
    """)
    n_bl = db.execute("SELECT COUNT(*) FROM step1").fetchone()[0]
    print(f"  pass2 drop: {n_bl:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_bl AS
        SELECT * FROM raw_text
        WHERE NOT regexp_matches(search_text, '{pass2_re}', 'i')
    """)

    # ── Step 2: r2 黑名单 ──
    db.execute(f"""
        CREATE TEMP TABLE step2 AS
        SELECT *, 'step2_r2' AS drop_step,
               regexp_extract(search_text, '{r2_re}') AS drop_reason
        FROM after_bl
        WHERE regexp_matches(search_text, '{r2_re}', 'i')
    """)
    n_r2 = db.execute("SELECT COUNT(*) FROM step2").fetchone()[0]
    print(f"  r2 drop:   {n_r2:,}")

    # 幸存行
    db.execute(f"""
        CREATE TEMP TABLE after_r2 AS
        SELECT * FROM after_bl
        WHERE NOT regexp_matches(search_text, '{r2_re}', 'i')
    """)
    n_surviving = db.execute("SELECT COUNT(*) FROM after_r2").fetchone()[0]
    print(f"  幸存:     {n_surviving:,}")

    # 合并 drop
    db.execute("""
        CREATE TEMP TABLE dropped AS
        SELECT * EXCLUDE (search_text, title_channel, drop_step, drop_reason),
               drop_step, drop_reason
        FROM step1
        UNION ALL
        SELECT * EXCLUDE (search_text, title_channel, drop_step, drop_reason),
               drop_step, drop_reason
        FROM step2
    """)
    n_drop = db.execute("SELECT COUNT(*) FROM dropped").fetchone()[0]

    # ── 输出 ──
    base = raw_name if raw_name else stem
    out_keep = os.path.join(output_dir, f"{base}_{run}_keep.parquet")
    out_drop = os.path.join(output_dir, f"{base}_{run}_drop.parquet")

    write_parquet_with_excludes(db, "after_r2", out_keep, _KEEP_EXCLUDE)
    write_parquet_with_excludes(db, "dropped", out_drop, _DROP_EXCLUDE)

    elapsed = time.perf_counter() - t0

    n_keep = n_surviving
    total_drop = n_drop
    print(f"  keep: {n_keep:,} | drop: {total_drop:,} ({total_drop/max(n_total,1)*100:.1f}%)")

    summary = {
        "engine": "beauty-blacklist",
        "category": "beauty",
        "input": os.path.abspath(input_path),
        "total_rows": n_total,
        "total_keep": n_keep,
        "total_drop": total_drop,
        "retention_pct": round(n_keep / max(n_total, 1) * 100, 1),
        "elapsed_sec": round(elapsed, 1),
        "steps": {
            "step1_pass2_blacklist": {"dropped": n_bl},
            "step2_r2_blacklist": {"dropped": n_r2},
        },
    }

    write_summary_json(output_dir, summary)

    print(f"  output: {output_dir}/ ({elapsed:.1f}s)")
    print()
    print(f"  keep:  {out_keep}")
    print(f"  drop:  {out_drop}")
    db.close()

    return summary
