#!/usr/bin/env python3
"""
categories/language_teaching/cleaner.py — 语言教学 DuckDB 多步清洗

流程:
  Step 0: 加载 raw 数据
  Step 1: pass2 黑名单 drop  — 纯 SQL regexp_matches
  Step 1b: r2 黑名单 drop   — 纯 SQL regexp_matches
  Step 2: 强语言教学信号      — 纯 SQL
  Step 3: 对幸存行调用打分 UDF，决定 high / medium / drop

输出:
  - {stem}_{run}_keep_high.parquet
  - {stem}_{run}_keep_medium.parquet
  - {stem}_{run}_keep.parquet
  - {stem}_{run}_drop.parquet
  - clean_summary.json
"""

import os
import time
from pathlib import Path

import duckdb

from core.rules_loader import load_blacklist, load_blacklist_individual, load_strong_pattern, save_hit_cache
from core.sql_builder import (
    sql_escape,
    load_raw_table,
    add_search_text,
    write_parquet_with_excludes,
    write_summary_json,
    count_rule_hits,
)
from categories.language_teaching.scorer import register_udfs, get_thresholds

_RULES_DIR = Path(__file__).resolve().parent / "rules"

# 输出时排除的辅助列（keep 文件用 _KEEP_EXCLUDE，drop 文件用 _DROP_EXCLUDE）
_KEEP_EXCLUDE = {
    "search_text", "title_channel",
    "lt_score", "kw_aligned", "strong_sig", "kw_entities",
    "drop_step", "drop_reason", "tier",
}
_DROP_EXCLUDE = _KEEP_EXCLUDE - {"drop_step", "drop_reason"}


def _compute_rule_stats(db: duckdb.DuckDBPyConnection, rules_dir: Path) -> dict:
    """统计各规则的命中数。"""
    bl_individual = load_blacklist_individual(rules_dir)
    stats: dict[str, dict[str, int]] = {}

    # pass2 命中
    pass2_rules = bl_individual.get("pass2", [])
    if pass2_rules:
        p2_hits = count_rule_hits(db, "step1", pass2_rules)
        if p2_hits:
            stats["pass2"] = p2_hits

    # r2 命中
    r2_rules = bl_individual.get("r2", [])
    if r2_rules:
        r2_hits = count_rule_hits(db, "step1b_r2", r2_rules)
        if r2_hits:
            stats["r2"] = r2_hits

    return stats


def _compute_auto_thresholds(db, table, score_col, high_pct=80, med_pct=50):
    """基于分数分位数计算自适应阈值。

    对标: DemandClean (VLDB 2025) — 动态阈值替代静态硬编码。
    """
    result = db.execute(f"""
        SELECT
            CAST(approx_quantile({score_col}, {high_pct / 100}) AS INTEGER),
            CAST(approx_quantile({score_col}, {med_pct / 100}) AS INTEGER)
        FROM {table}
    """).fetchone()
    keep = max(10, result[0] or 30)
    med = max(5, result[1] or 10)
    return {"keep_score": keep, "gray_score_low": med, "medium_min_score": med}


def clean(
    input_path: str,
    stem: str = "clean",
    output_dir: str = "output",
    raw_name: str = "",
    run: str = "run01",
    keep_score: int | None = None,
    gray_low: int | None = None,
    med_min: int | None = None,
    no_medium: bool = False,
    auto_threshold: bool = False,
    fmt: str = "parquet",
    **kwargs,
) -> dict:
    thresholds = get_thresholds()
    keep_score = keep_score if keep_score is not None else thresholds["keep_score"]
    gray_low = gray_low if gray_low is not None else thresholds["gray_score_low"]
    med_min = med_min if med_min is not None else thresholds["medium_min_score"]
    if no_medium:
        med_min = 9999

    t0 = time.perf_counter()
    mode = "no-medium" if no_medium else "high+medium"
    print(f"语言教学清洗 (DuckDB SQL-first, {mode})...")

    os.makedirs(output_dir, exist_ok=True)

    # ── 加载规则 ──
    rules = load_blacklist(_RULES_DIR)
    pass2_re = sql_escape(rules["pass2"])
    r2_re = sql_escape(rules["r2"])
    strong = load_strong_pattern(_RULES_DIR)
    strong_re = sql_escape(strong) if strong else r"(?!x)x"

    db = duckdb.connect(":memory:")

    # ── Step 0: 加载 raw ──
    n_total = load_raw_table(db, input_path)
    print(f"  rows: {n_total:,}")

    add_search_text(db)

    # ── Step 1: pass2 黑名单（纯 SQL） ──
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

    # ── Step 1b: r2 黑名单（纯 SQL） ──
    db.execute(f"""
        CREATE TEMP TABLE step1b_r2 AS
        SELECT *, 'step1b_r2' AS drop_step,
               regexp_extract(search_text, '{r2_re}') AS drop_reason
        FROM after_bl
        WHERE regexp_matches(search_text, '{r2_re}', 'i')
    """)
    n_r2 = db.execute("SELECT COUNT(*) FROM step1b_r2").fetchone()[0]
    print(f"  r2 drop:   {n_r2:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_r2 AS
        SELECT * FROM after_bl
        WHERE NOT regexp_matches(search_text, '{r2_re}', 'i')
    """)

    # ── Step 2: 强信号（纯 SQL） ──
    db.execute(f"""
        CREATE TEMP TABLE after_r2_sig AS
        SELECT *,
               regexp_matches(title_channel, '{strong_re}', 'i') AS strong_sig
        FROM after_r2
    """)

    # ── Step 3: UDF 打分 ──
    n_surviving = db.execute("SELECT COUNT(*) FROM after_r2_sig").fetchone()[0]
    print(f"  幸存行:   {n_surviving:,} → 调 UDF 打分...")

    register_udfs(db)

    db.execute("""
        CREATE TEMP TABLE scored AS
        SELECT *,
               lang_teaching_score(title, channel, keyword) AS lt_score,
               keyword_aligned(keyword, title, channel) AS kw_aligned,
               parse_lang_entities(keyword) AS kw_entities
        FROM after_r2_sig
    """)

    # ── 自适应阈值 ──
    if auto_threshold:
        auto_t = _compute_auto_thresholds(db, "scored", "lt_score")
        keep_score = auto_t["keep_score"]
        gray_low = auto_t["gray_score_low"]
        med_min = max(med_min, auto_t["medium_min_score"])
        print(f"  auto-threshold: keep={keep_score} gray={gray_low} med={med_min}")

    # ── Step 4: tier 分档 ──
    db.execute(f"""
        CREATE TEMP TABLE scored_tier AS
        SELECT *,
               CASE
                   WHEN lt_score >= {keep_score} THEN 'high'
                   WHEN kw_aligned AND lt_score >= {gray_low} THEN 'high'
                   WHEN (kw_entities != '' AND NOT kw_aligned AND strong_sig AND lt_score >= {med_min})
                        THEN 'medium'
                   ELSE 'drop'
               END AS tier
        FROM scored
    """)

    db.execute("CREATE TEMP TABLE keep_high AS SELECT * FROM scored_tier WHERE tier = 'high'")
    db.execute("CREATE TEMP TABLE keep_medium AS SELECT * FROM scored_tier WHERE tier = 'medium'")

    # 合并 drop 来源：打分 drop + 黑名单 drop
    db.execute("""
        CREATE TEMP TABLE dropped AS
        SELECT *, 'step_score' AS drop_step, tier AS drop_reason
        FROM scored_tier WHERE tier = 'drop'
        UNION ALL
        SELECT * EXCLUDE (drop_step, drop_reason),
               NULL::BOOLEAN AS strong_sig,
               NULL::INTEGER AS lt_score,
               NULL::BOOLEAN AS kw_aligned,
               NULL::VARCHAR AS kw_entities,
               NULL::VARCHAR AS tier,
               drop_step, drop_reason
        FROM step1
        UNION ALL
        SELECT * EXCLUDE (drop_step, drop_reason),
               NULL::BOOLEAN AS strong_sig,
               NULL::INTEGER AS lt_score,
               NULL::BOOLEAN AS kw_aligned,
               NULL::VARCHAR AS kw_entities,
               NULL::VARCHAR AS tier,
               drop_step, drop_reason
        FROM step1b_r2
    """)

    n_high = db.execute("SELECT COUNT(*) FROM keep_high").fetchone()[0]
    n_medium = db.execute("SELECT COUNT(*) FROM keep_medium").fetchone()[0]
    n_drop = db.execute("SELECT COUNT(*) FROM dropped").fetchone()[0]
    n_keep = n_high + n_medium

    print(f"  keep: {n_keep:,} (H={n_high:,} M={n_medium:,}) | drop: {n_drop:,}")

    # ── 输出 ──
    base = raw_name if raw_name else stem
    out_high = os.path.join(output_dir, f"{base}_{run}_keep_high.parquet")
    out_medium = os.path.join(output_dir, f"{base}_{run}_keep_medium.parquet")
    out_all = os.path.join(output_dir, f"{base}_{run}_keep.parquet")
    out_dropped = os.path.join(output_dir, f"{base}_{run}_drop.parquet")

    write_parquet_with_excludes(db, "keep_high", out_high, _KEEP_EXCLUDE)
    write_parquet_with_excludes(db, "keep_medium", out_medium, _KEEP_EXCLUDE)

    db.execute("CREATE TEMP TABLE keep_all AS SELECT * FROM keep_high UNION ALL SELECT * FROM keep_medium")
    write_parquet_with_excludes(db, "keep_all", out_all, _KEEP_EXCLUDE)
    write_parquet_with_excludes(db, "dropped", out_dropped, _DROP_EXCLUDE)

    # ── 规则命中统计 ──
    rule_stats = _compute_rule_stats(db, _RULES_DIR)
    if rule_stats:
        save_hit_cache(_RULES_DIR, rule_stats)

    elapsed = time.perf_counter() - t0
    summary = {
        "engine": "duckdb-sql-first",
        "category": "language_teaching",
        "input": os.path.abspath(input_path),
        "total_rows": n_total,
        "total_keep": n_keep,
        "total_keep_high": n_high,
        "total_keep_medium": n_medium,
        "total_drop": n_drop,
        "retention_pct": round(n_keep / max(n_total, 1) * 100, 1),
        "elapsed_sec": round(elapsed, 1),
        "steps": {
            "step1_pass2_blacklist": {"dropped": n_bl},
            "step1b_r2_blacklist": {"dropped": n_r2},
            "step3_score_drop": {"dropped": max(0, n_drop - n_bl - n_r2)},
        },
        "rule_stats": rule_stats,
        "keep_path": out_all,
        "keep_high_path": out_high,
        "keep_medium_path": out_medium,
        "drop_path": out_dropped,
    }

    write_summary_json(output_dir, summary)

    db.close()

    print(f"  output: {output_dir}/ ({elapsed:.1f}s)")
    return summary
