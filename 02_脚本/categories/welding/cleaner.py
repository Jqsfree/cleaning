#!/usr/bin/env python3
"""
categories/welding/cleaner.py — 焊接黑名单清洗

纯 SQL pass2 过滤，无 UDF 打分。
r2 规则已定义但默认不启用（需 --with-r2）。

流程:
  Step 0: 加载 raw 数据
  Step 1: pass2 黑名单 drop
  Step 2: (可选) r2 黑名单 drop
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

from core.rules_loader import load_blacklist, load_blacklist_individual, save_hit_cache
from core.sql_builder import (
    sql_escape,
    load_raw_table,
    add_search_text,
    write_parquet_with_excludes,
    write_summary_json,
    count_rule_hits,
)

RULES_DIR = Path(__file__).resolve().parent / "rules"

_KEEP_EXCLUDE = {"search_text", "title_channel", "drop_step", "drop_reason"}
_DROP_EXCLUDE = {"search_text", "title_channel"}


def _compute_rule_stats(db, rules_dir):
    """统计各规则的命中数。"""
    bl_individual = load_blacklist_individual(rules_dir)
    stats = {}
    pass2_rules = bl_individual.get("pass2", [])
    if pass2_rules:
        p2_hits = count_rule_hits(db, "step1", pass2_rules)
        if p2_hits:
            stats["pass2"] = p2_hits
    # welding 的 r2 drop 表可能不存在（不启用 --with-r2 时）
    r2_rules = bl_individual.get("r2", [])
    if r2_rules:
        try:
            r2_hits = count_rule_hits(db, "step2", r2_rules)
            if r2_hits:
                stats["r2"] = r2_hits
        except Exception:
            pass
    return stats


def clean(
    input_path: str,
    stem: str = "clean",
    output_dir: str = "output",
    raw_name: str = "",
    run: str = "run01",
    with_r2: bool = False,
    fmt: str = "parquet",
    **kwargs,
) -> dict:
    t0 = time.perf_counter()
    mode = "pass2 + r2" if with_r2 else "pass2 only"
    print(f"焊接规则清洗 ({mode})...")

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
        SELECT *, 'step1_pass2' AS drop_step,
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

    # ── Step 2: r2 黑名单 (可选) ──
    n_r2 = 0
    if with_r2:
        db.execute(f"""
            CREATE TEMP TABLE step2 AS
            SELECT *, 'step2_r2' AS drop_step,
                   regexp_extract(search_text, '{r2_re}') AS drop_reason
            FROM after_bl
            WHERE regexp_matches(search_text, '{r2_re}', 'i')
        """)
        n_r2 = db.execute("SELECT COUNT(*) FROM step2").fetchone()[0]
        print(f"  r2 drop:   {n_r2:,}")

        db.execute(f"""
            CREATE TEMP TABLE after_r2 AS
            SELECT * FROM after_bl
            WHERE NOT regexp_matches(search_text, '{r2_re}', 'i')
        """)
        survivor_table = "after_r2"
    else:
        print(f"  r2 drop:   跳过 (需 --with-r2 启用)")
        survivor_table = "after_bl"

    n_surviving = db.execute(f"SELECT COUNT(*) FROM {survivor_table}").fetchone()[0]
    print(f"  幸存:     {n_surviving:,}")

    # 合并 drop
    if with_r2:
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
    else:
        db.execute("""
            CREATE TEMP TABLE dropped AS
            SELECT * EXCLUDE (search_text, title_channel, drop_step, drop_reason),
                   drop_step, drop_reason
            FROM step1
        """)
    n_drop = db.execute("SELECT COUNT(*) FROM dropped").fetchone()[0]

    # ── 输出 ──
    base = raw_name if raw_name else stem
    out_keep = os.path.join(output_dir, f"{base}_{run}_keep.parquet")
    out_drop = os.path.join(output_dir, f"{base}_{run}_drop.parquet")

    write_parquet_with_excludes(db, survivor_table, out_keep, _KEEP_EXCLUDE)
    write_parquet_with_excludes(db, "dropped", out_drop, _DROP_EXCLUDE)

    # ── 规则命中统计 ──
    rule_stats = _compute_rule_stats(db, RULES_DIR)
    if rule_stats:
        save_hit_cache(RULES_DIR, rule_stats)

    elapsed = time.perf_counter() - t0

    n_keep = n_surviving
    print(f"  keep: {n_keep:,} | drop: {n_drop:,} ({n_drop/max(n_total,1)*100:.1f}%)")

    summary = {
        "engine": "welding-blacklist",
        "category": "welding",
        "mode": mode,
        "input": os.path.abspath(input_path),
        "total_rows": n_total,
        "total_keep": n_keep,
        "total_drop": n_drop,
        "retention_pct": round(n_keep / max(n_total, 1) * 100, 1),
        "elapsed_sec": round(elapsed, 1),
        "steps": {
            "step1_pass2_blacklist": {"dropped": n_bl},
            "step2_r2_blacklist": {"dropped": n_r2},
        },
        "rule_stats": rule_stats,
        "keep_path": out_keep,
        "drop_path": out_drop,
    }

    write_summary_json(output_dir, summary)

    print(f"  output: {output_dir}/ ({elapsed:.1f}s)")
    print()
    print(f"  keep:  {out_keep}")
    print(f"  drop:  {out_drop}")
    db.close()

    return summary
