#!/usr/bin/env python3
"""
categories/live_sell/cleaner.py — 直播带货黑名单清洗

纯 SQL 过滤（pass2 + r2 → keep/drop），无打分 UDF。
规则：certain-noise only（见 rules/blacklist.toml）。
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
    bl_individual = load_blacklist_individual(rules_dir)
    stats = {}
    for section, table in [("pass2", "step1"), ("r2", "step2")]:
        rules = bl_individual.get(section, [])
        if rules:
            hits = count_rule_hits(db, table, rules)
            if hits:
                stats[section] = hits
    return stats


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
    print("直播带货规则清洗 (pass2 + r2 黑名单)...")

    os.makedirs(output_dir, exist_ok=True)

    rules = load_blacklist(RULES_DIR)
    pass2_re = sql_escape(rules["pass2"])
    r2_re = sql_escape(rules["r2"])

    db = duckdb.connect(":memory:")

    n_total = load_raw_table(db, input_path)
    print(f"  rows: {n_total:,}")

    add_search_text(db)

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
    n_surviving = db.execute("SELECT COUNT(*) FROM after_r2").fetchone()[0]
    print(f"  幸存:     {n_surviving:,}")

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

    base = raw_name if raw_name else stem
    out_keep = os.path.join(output_dir, f"{base}_{run}_keep.parquet")
    out_drop = os.path.join(output_dir, f"{base}_{run}_drop.parquet")

    write_parquet_with_excludes(db, "after_r2", out_keep, _KEEP_EXCLUDE)
    write_parquet_with_excludes(db, "dropped", out_drop, _DROP_EXCLUDE)

    rule_stats = _compute_rule_stats(db, RULES_DIR)
    if rule_stats:
        save_hit_cache(RULES_DIR, rule_stats)

    elapsed = time.perf_counter() - t0
    n_keep = n_surviving
    summary = {
        "engine": "live_sell-blacklist",
        "category": "live_sell",
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
    print(f"  keep: {n_keep:,} | drop: {n_drop:,} ({n_drop/max(n_total,1)*100:.1f}%)")
    print(f"  output: {output_dir}/ ({elapsed:.1f}s)")
    db.close()
    return summary
