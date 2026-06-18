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


def sql_escape(s: str) -> str:
    """转义 DuckDB SQL 字面量中的单引号。

    注意: 不转义反斜杠 — DuckDB SQL 字面量中反斜杠无特殊含义，
    RE2 正则引擎直接接收原样字符（如 \\b 即为单词边界）。
    """
    return s.replace("'", "''")


# ── 数据加载 ─────────────────────────────────────────────

def load_raw_table(conn: duckdb.DuckDBPyConnection, input_path: str) -> int:
    """从 csv/parquet 加载 raw 表，返回行数。

    创建临时表 ``raw``。
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        reader = f"read_parquet('{input_path}')"
    else:
        reader = (
            f"read_csv_auto('{input_path}', header=true, all_varchar=true, "
            f"sample_size=-1, ignore_errors=true)"
        )

    conn.execute(f"CREATE TEMP TABLE raw AS SELECT * FROM {reader}")
    return conn.execute("SELECT COUNT(*) FROM raw").fetchone()[0]


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
    conn.execute(f"""
        CREATE TEMP TABLE {to_table} AS
        SELECT *,
               COALESCE(title,'') || ' ' || COALESCE(channel,'') || ' ' ||
               regexp_replace(COALESCE(keyword,''), '((^|\\s)-[a-zA-Z0-9*?]+)+$', '')
                   AS search_text,
               COALESCE(title,'') || ' ' || COALESCE(channel,'') AS title_channel
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
        f"COPY (SELECT {col_str} FROM {table}) TO '{output_path}' (FORMAT PARQUET)"
    )


def write_summary_json(output_dir: str, summary: dict) -> str:
    """写 clean_summary.json，返回文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clean_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path
