#!/usr/bin/env python3
"""
core/cleaner.py — DuckDB 多步清洗（语言教学版）

流程:
  Step 0: 加载 raw 数据（无 UDF）
  Step 1: pass2 黑名单 drop  — 纯 SQL regexp_matches
  Step 1b: r2 黑名单 drop   — 纯 SQL regexp_matches
  Step 2: 弱信号 drop       — 纯 SQL
  Step 3: 对幸存行调用打分 UDF，决定 high / medium / drop

输出:
  - {stem}_{run}_keep_high.parquet
  - {stem}_{run}_keep_medium.parquet
  - {stem}_{run}_keep.parquet
  - {stem}_{run}_drop.parquet
  - clean_summary.json
"""

import time, json, os, tomllib
from pathlib import Path
import duckdb

from .scoring import register_udfs, get_thresholds

_RULES_DIR = Path(__file__).resolve().parent.parent / "rules" / "current"


def _load_regex_list(section: str) -> str:
    """从 blacklist.toml 加载指定 section 的 pattern，拼成一个正则"""
    bl_path = _RULES_DIR / "blacklist.toml"
    if not bl_path.exists():
        return r"(?!x)x"
    bl = tomllib.loads(bl_path.read_text("utf-8"))
    patterns = [item["pattern"] for item in bl.get(section, [])]
    return "|".join(patterns) if patterns else r"(?!x)x"


def _load_strong_pattern() -> str:
    """从 whitelist.toml 加载强语言教学信号正则"""
    wl_path = _RULES_DIR / "whitelist.toml"
    if not wl_path.exists():
        return r"(?!x)x"
    wl = tomllib.loads(wl_path.read_text("utf-8"))
    return wl.get("strong_lang_teaching_title_pattern", r"(?!x)x")


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
    fmt: str = "parquet",
) -> dict:
    thresholds = get_thresholds()
    keep_score = keep_score if keep_score is not None else thresholds["keep_score"]
    gray_low = gray_low if gray_low is not None else thresholds["gray_score_low"]
    med_min = med_min if med_min is not None else thresholds["medium_min_score"]
    if no_medium:
        # 把 medium 门槛拉到 infinity，只保留 high
        med_min = 9999

    t0 = time.perf_counter()
    mode = "no-medium" if no_medium else "high+medium"
    print(f"语言教学清洗 (DuckDB SQL-first, {mode})...")

    os.makedirs(output_dir, exist_ok=True)

    # 预编译正则（纯 SQL 用，不加 Python 边界开销）
    # TOML 中的 \b ' 等字符放入 SQL 字符串需转义
    def _sql_escape(s: str) -> str:
        """仅转义 SQL 单引号。反斜杠不转义：DuckDB SQL 字面量中反斜杠无特殊含义，
        RE2 正则引擎直接接收原样字符，如 \\b 即为单词边界。"""
        return s.replace("'", "''")

    pass2_re = _sql_escape(_load_regex_list("pass2"))
    r2_re = _sql_escape(_load_regex_list("r2"))
    strong_re = _sql_escape(_load_strong_pattern())

    db = duckdb.connect(":memory:")

    # ── Step 0: 加载 raw，不加任何 UDF ──
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        reader = f"read_parquet('{input_path}')"
    else:
        reader = f"read_csv_auto('{input_path}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)"

    db.execute(f"CREATE TEMP TABLE raw AS SELECT * FROM {reader}")
    n_total = db.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"  rows: {n_total:,}")

    # 拼接文本列（只拼一次）
    # keyword 剥离 -xxx 否定标签，避免黑名单误命中
    # "spanish class -vlog -cartoon" → "spanish class"
    db.execute("""
        CREATE TEMP TABLE raw_text AS
        SELECT *,
               COALESCE(title,'') || ' ' || COALESCE(channel,'') || ' ' ||
               regexp_replace(COALESCE(keyword,''), '((^|\\s)-[a-zA-Z0-9*?]+)+$', '') AS search_text,
               COALESCE(title,'') || ' ' || COALESCE(channel,'') AS title_channel
        FROM raw
    """)

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

    # ── Step 2: 强语言教学信号（纯 SQL） ──
    db.execute(f"""
        CREATE TEMP TABLE after_r2_sig AS
        SELECT *,
               regexp_matches(title_channel, '{strong_re}', 'i') AS strong_sig
        FROM after_r2
    """)

    # ── Step 3: 对幸存行调 Python UDF（打分 + 实体解析） ──
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

    # ── Step 4: 计分分类 ──
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

    _AUX_COLS = {
        "search_text", "title_channel",
        "lt_score", "kw_aligned", "strong_sig", "kw_entities",
        "drop_step", "drop_reason", "tier",
    }
    select_cols = db.execute("SELECT * FROM dropped LIMIT 0").description  # dropped 是列超集（含 drop_step/drop_reason）
    all_col_names = [c[0] for c in select_cols]

    # keep/drop 输出分开控制：keep 文件去掉所有辅助列，drop 文件保留 drop_step, drop_reason 供评估用
    _KEEP_EXCLUDE = _AUX_COLS
    _DROP_EXCLUDE = _AUX_COLS - {"drop_step", "drop_reason"}

    def _write_cols(table_name, base_path, exclude):
        cols = [c for c in all_col_names if c not in exclude]
        col_str = ", ".join(f'"{c}"' for c in cols)
        db.execute(f"COPY (SELECT {col_str} FROM {table_name}) TO '{base_path}' (FORMAT PARQUET)")

    _write_cols("keep_high", out_high, _KEEP_EXCLUDE)
    _write_cols("keep_medium", out_medium, _KEEP_EXCLUDE)

    db.execute("CREATE TEMP TABLE keep_all AS SELECT * FROM keep_high UNION ALL SELECT * FROM keep_medium")
    _write_cols("keep_all", out_all, _KEEP_EXCLUDE)
    _write_cols("dropped", out_dropped, _DROP_EXCLUDE)

    elapsed = time.perf_counter() - t0
    summary = {
        "engine": "duckdb-sql-first",
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
    }

    summary_path = os.path.join(output_dir, "clean_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    db.close()

    print(f"  output: {output_dir}/ ({elapsed:.1f}s)")
    return summary
