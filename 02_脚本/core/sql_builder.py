#!/usr/bin/env python3
"""
core/sql_builder.py -- 通用 DuckDB 操作（不绑定任何类别）

提供:
  - sql_escape: DuckDB SQL 字面量转义
  - load_raw_table: 自动判断 csv/parquet，创建 temp 表
  - add_search_text: 拼接 title + channel + keyword（剥离否定标签）
  - write_parquet_with_excludes: 写 parquet 并排除辅助列
  - write_summary_json: 写 clean_summary.json
"""

import json
import os
import duckdb

# ── 共享常量 ──
KEEP_EXCLUDE_DEFAULT = {"search_text", "title_channel", "drop_step", "drop_reason"}
DROP_EXCLUDE_DEFAULT = {"search_text", "title_channel"}
REQUIRED_RAW_COLUMNS = ("video_id", "title")


def sql_escape(s: str) -> str:
    """转义 DuckDB SQL 字面量中的单引号。

    注意: 不转义反斜杠 — DuckDB SQL 字面量中反斜杠无特殊含义，
    RE2 正则引擎直接接收原样字符（如 \\b 即为单词边界）。
    """
    return s.replace("'", "''")


def table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {c[0] for c in conn.execute(f"SELECT * FROM {table} LIMIT 0").description}


def require_columns(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    required: tuple[str, ...] | list[str],
) -> None:
    """缺列时抛 ValueError（清晰 schema 错误，避免下游 SQL 难读失败）。"""
    cols = table_columns(conn, table)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"表 {table} 缺少必填列: {missing}；现有列: {sorted(cols)}"
        )


# ── 数据加载 ─────────────────────────────────────────────

def load_raw_table(
    conn: duckdb.DuckDBPyConnection,
    input_path: str,
    *,
    ignore_errors: bool = True,
    require: tuple[str, ...] | list[str] | None = REQUIRED_RAW_COLUMNS,
) -> int:
    """从 csv/parquet 加载 raw 表，返回行数。

    创建临时表 ``raw``。CSV 在 ignore_errors 时对比文件行数并 WARN。
    读入口统一走 ``core.io.duckdb_reader``。
    """
    from core.io import duckdb_reader, warn_csv_row_skew

    reader = duckdb_reader(input_path, ignore_errors=ignore_errors)
    conn.execute(f"CREATE TEMP TABLE raw AS SELECT * FROM {reader}")
    n = conn.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".csv", ".tsv") and ignore_errors:
        try:
            from core.log import log
            warn_csv_row_skew(input_path, n, log_fn=log)
        except Exception:
            pass
    if require:
        require_columns(conn, "raw", require)
    return n


# ── 文本拼接 ─────────────────────────────────────────────

def add_search_text(
    conn: duckdb.DuckDBPyConnection,
    from_table: str = "raw",
    to_table: str = "raw_text",
) -> None:
    """拼接搜索文本列，剥离 keyword 中 - 开头的否定标签。

    在 from_table 基础上创建 to_table，新增两列:
      - search_text:   title + channel + keyword（去除 -xxx 标签）
      - title_channel: title + channel

    keyword 否定标签例: "spanish class -vlog -cartoon" → "spanish class"
    """
    cols = table_columns(conn, from_table)
    for col in ("title", "channel", "keyword"):
        if col not in cols:
            # 缺列用空串表达式，避免硬炸；title 仍建议上游 require_columns
            pass
    title_expr = "COALESCE(title,'')" if "title" in cols else "''"
    channel_expr = "COALESCE(channel,'')" if "channel" in cols else "''"
    if "keyword" in cols:
        kw_expr = (
            "regexp_replace(COALESCE(keyword,''), "
            "'((^|\\s)-[a-zA-Z0-9*?]+)+$', '')"
        )
    else:
        kw_expr = "''"
    conn.execute(f"""
        CREATE TEMP TABLE {to_table} AS
        SELECT *,
               {title_expr} || ' ' || {channel_expr} || ' ' || {kw_expr}
                   AS search_text,
               {title_expr} || ' ' || {channel_expr} AS title_channel
        FROM {from_table}
    """)


# ── 输出 ─────────────────────────────────────────────────

def write_parquet_with_excludes(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    output_path: str,
    exclude_cols: set[str],
) -> None:
    """从 DuckDB 表写 parquet，自动排除指定列。

    从表中读取列名，移除 exclude_cols 中的列，执行 COPY TO。
    """
    all_cols = [c[0] for c in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
    keep_cols = [c for c in all_cols if c not in exclude_cols]
    col_str = ", ".join(f'"{c}"' for c in keep_cols)
    conn.execute(
        f"COPY (SELECT {col_str} FROM {table}) TO '{sql_escape(output_path)}' (FORMAT PARQUET)"
    )


def write_summary_json(output_dir: str, summary: dict) -> str:
    """写 clean_summary.json，返回文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clean_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


# ── 规则命中统计 ──────────────────────────────────────────

def count_rule_hits(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    rules: list[dict[str, str]],
    text_col: str = "search_text",
    fallback_col: str = "title_channel",
) -> dict[str, int]:
    """统计每条黑名单规则的命中数（用于规则效果评估）。

    对已标记为 drop 的表，逐条统计每个规则 category 的命中数。

    Args:
        conn: DuckDB 连接
        table: 已筛选的 drop 表（如 step1、step1b_r2）
        rules: load_blacklist_individual() 返回的规则列表
        text_col: 主搜索文本列
        fallback_col: 备选搜索文本列

    Returns:
        {category: hit_count} 字典
    """
    hits: dict[str, int] = {}
    if not rules:
        return hits

    # 检查 text_col 是否存在，不存在则用 fallback_col
    cols = {c[0] for c in conn.execute(f"SELECT * FROM {table} LIMIT 0").description}
    search_col = text_col if text_col in cols else fallback_col

    errors: list[str] = []
    for rule in rules:
        pattern = rule["pattern"].replace("'", "''")
        category = rule.get("category", "?")
        try:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE regexp_matches(\"{search_col}\", '{pattern}', 'i')"
            ).fetchone()[0]
        except Exception as e:
            errors.append(f"{category}: {e}")
            continue
        if n > 0:
            hits[category] = hits.get(category, 0) + n

    if errors:
        preview = "; ".join(errors[:5])
        more = f" …(+{len(errors) - 5})" if len(errors) > 5 else ""
        raise RuntimeError(
            f"规则命中统计失败 {len(errors)}/{len(rules)} 条: {preview}{more}"
        )

    return hits
