#!/usr/bin/env python3
"""
categories/exo/cleaner.py — 农业采摘第三人称黑名单清洗

title_pass2 + channel_pass2 + pass2 + r2（certain-noise only）。
"""

from __future__ import annotations

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
    for section, table, text_col in [
        ("title_pass2", "title_drop", "title"),
        ("channel_pass2", "channel_drop", "channel"),
        ("pass2", "step1", "search_text"),
        ("r2", "step2", "search_text"),
    ]:
        rules = bl_individual.get(section, [])
        if rules:
            hits = count_rule_hits(db, table, rules, text_col=text_col)
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
    print("exo 规则清洗 (title + channel + pass2/r2)...")

    os.makedirs(output_dir, exist_ok=True)

    rules = load_blacklist(RULES_DIR)
    title_pass2_re = sql_escape(rules["title_pass2"])
    channel_pass2_re = sql_escape(rules["channel_pass2"])
    pass2_re = sql_escape(rules["pass2"])
    r2_re = sql_escape(rules["r2"])

    db = duckdb.connect(":memory:")
    n_total = load_raw_table(db, input_path)
    print(f"  rows: {n_total:,}")
    add_search_text(db)

    db.execute(f"""
        CREATE TEMP TABLE title_drop AS
        SELECT *, 'title_blacklist' AS drop_step,
               regexp_extract(COALESCE(title, ''), '{title_pass2_re}') AS drop_reason
        FROM raw_text
        WHERE regexp_matches(COALESCE(title, ''), '{title_pass2_re}', 'i')
    """)
    n_title = db.execute("SELECT COUNT(*) FROM title_drop").fetchone()[0]
    print(f"  title drop: {n_title:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_title AS
        SELECT * FROM raw_text
        WHERE NOT regexp_matches(COALESCE(title, ''), '{title_pass2_re}', 'i')
    """)

    db.execute(f"""
        CREATE TEMP TABLE channel_drop AS
        SELECT *, 'channel_blacklist' AS drop_step,
               regexp_extract(COALESCE(channel, ''), '{channel_pass2_re}') AS drop_reason
        FROM after_title
        WHERE regexp_matches(COALESCE(channel, ''), '{channel_pass2_re}', 'i')
    """)
    n_channel = db.execute("SELECT COUNT(*) FROM channel_drop").fetchone()[0]
    print(f"  channel:    {n_channel:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_channel AS
        SELECT * FROM after_title
        WHERE NOT regexp_matches(COALESCE(channel, ''), '{channel_pass2_re}', 'i')
    """)

    db.execute(f"""
        CREATE TEMP TABLE step1 AS
        SELECT *, 'step1_blacklist' AS drop_step,
               regexp_extract(search_text, '{pass2_re}') AS drop_reason
        FROM after_channel
        WHERE regexp_matches(search_text, '{pass2_re}', 'i')
    """)
    n_pass2 = db.execute("SELECT COUNT(*) FROM step1").fetchone()[0]
    print(f"  pass2:      {n_pass2:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_pass2 AS
        SELECT * FROM after_channel
        WHERE NOT regexp_matches(search_text, '{pass2_re}', 'i')
    """)

    db.execute(f"""
        CREATE TEMP TABLE step2 AS
        SELECT *, 'step2_r2' AS drop_step,
               regexp_extract(search_text, '{r2_re}') AS drop_reason
        FROM after_pass2
        WHERE regexp_matches(search_text, '{r2_re}', 'i')
    """)
    n_r2 = db.execute("SELECT COUNT(*) FROM step2").fetchone()[0]
    print(f"  r2:         {n_r2:,}")

    db.execute(f"""
        CREATE TEMP TABLE keep_tbl AS
        SELECT * FROM after_pass2
        WHERE NOT regexp_matches(search_text, '{r2_re}', 'i')
    """)
    n_keep = db.execute("SELECT COUNT(*) FROM keep_tbl").fetchone()[0]
    print(f"  keep:       {n_keep:,}")

    db.execute("""
        CREATE TEMP TABLE drop_all AS
        SELECT * FROM title_drop
        UNION ALL BY NAME SELECT * FROM channel_drop
        UNION ALL BY NAME SELECT * FROM step1
        UNION ALL BY NAME SELECT * FROM step2
    """)
    n_drop = db.execute("SELECT COUNT(*) FROM drop_all").fetchone()[0]

    keep_path = os.path.join(output_dir, f"{stem}_{run}_keep.parquet")
    drop_path = os.path.join(output_dir, f"{stem}_{run}_drop.parquet")
    write_parquet_with_excludes(db, "keep_tbl", keep_path, _KEEP_EXCLUDE)
    write_parquet_with_excludes(db, "drop_all", drop_path, _DROP_EXCLUDE)

    rule_stats = _compute_rule_stats(db, RULES_DIR)
    save_hit_cache(RULES_DIR, rule_stats)

    summary = {
        "category": "exo",
        "input": input_path,
        "run": run,
        "n_total": n_total,
        "n_keep": n_keep,
        "n_drop": n_drop,
        "drop_title": n_title,
        "drop_channel": n_channel,
        "drop_pass2": n_pass2,
        "drop_r2": n_r2,
        "keep_path": keep_path,
        "drop_path": drop_path,
        "rule_hits": rule_stats,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    write_summary_json(output_dir, summary)
    print(f"  done in {summary['elapsed_sec']}s → {keep_path}")
    db.close()
    return summary
