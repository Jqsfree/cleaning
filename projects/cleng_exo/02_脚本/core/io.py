#!/usr/bin/env python3
"""
core/io.py — 统一文件读写与路径小助手

提供跨 CSV/Parquet 的 DuckDB reader，以及 stem / -o 目录解析。
"""

from __future__ import annotations

import csv
import os
import re

from core.sql_builder import sql_escape

_STEM_SUFFIX_RE = re.compile(
    r"_(?:raw|quality|quality_drop|clean|clean_drop|sample|keep|keep_high|keep_medium|"
    r"drop|thumb_qc|textqc|resolution)(?:_\d{4,8})?$",
    re.IGNORECASE,
)


def duckdb_reader(path: str, *, ignore_errors: bool = True) -> str:
    """根据文件扩展名返回 DuckDB reader 表达式。

    CSV 默认 ignore_errors=true（脏行可进管道），调用方应在加载后用
    ``warn_csv_row_skew`` 对比文件行数与表行数。
    """
    ext = os.path.splitext(path)[1].lower()
    safe = sql_escape(path)
    if ext in (".csv", ".tsv"):
        flag = "true" if ignore_errors else "false"
        return (
            f"read_csv_auto('{safe}', header=true, all_varchar=true, "
            f"sample_size=-1, ignore_errors={flag})"
        )
    return f"read_parquet('{safe}')"


def count_csv_data_lines(path: str) -> int | None:
    """逻辑数据行数（按 CSV 记录计，引号内换行不计；已减表头）。失败返回 None。

    不能用 ``\\n`` 计数：``title`` 等字段内含引号包裹的换行会把物理行数抬高，
    使正常文件被误判为「丢行」。

    但 CSV 解析本身也可能失败（如字段含 NUL 字节），此时退回物理行数近似：
    宁可粗略告警，也不要静默放过真实丢行。
    """
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except OSError:
        return None
    except csv.Error:
        pass
    # csv 解析失败：退回物理行数（可能含引号内换行，仅作兜底告警）
    try:
        with open(path, "rb") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return None


def warn_csv_row_skew(
    path: str,
    loaded_rows: int,
    *,
    log_fn=None,
) -> int:
    """
    若 CSV 逻辑行数明显大于已加载行，WARN 可能因 ignore_errors 丢行。
    返回估计跳过行数（>=0）。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".csv", ".tsv"):
        return 0
    file_rows = count_csv_data_lines(path)
    if file_rows is None:
        return 0
    skipped = file_rows - loaded_rows
    if skipped <= 0:
        return 0
    msg = (
        f"CSV 可能静默丢行: file≈{file_rows:,} loaded={loaded_rows:,} "
        f"skew≈{skipped:,} ({path})；检查引号/坏行或改 ignore_errors=false"
    )
    if log_fn:
        log_fn(msg, level="WARN")
    else:
        print(f"[WARN] {msg}", flush=True)
    return skipped


def is_parquet(path: str) -> bool:
    """判断文件是否为 Parquet 格式。"""
    return os.path.splitext(path)[1].lower() in (".parquet", ".pq")


def strip_stem(name: str) -> str:
    """剥常见管道后缀（_raw/_quality/_clean/_sample/… 及日期尾巴）。

    接受带扩展名的文件名或纯 stem。
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    prev = None
    while prev != stem:
        prev = stem
        stem = _STEM_SUFFIX_RE.sub("", stem)
    return stem


def resolve_output_dir(arg: str | None, input_path: str) -> str:
    """解析 -o：空 → 输入同目录；文件路径 → 取其 dirname；否则当作目录。

    尚未落盘的 ``*.csv`` / ``*.parquet`` 等也按文件路径处理，避免
    ``makedirs(output.csv)`` 把结果文件名建成空目录（vision_thumb 踩过）。
    """
    if not arg:
        return os.path.dirname(os.path.abspath(input_path)) or "."
    if os.path.isfile(arg):
        return os.path.dirname(os.path.abspath(arg)) or "."
    if os.path.isdir(arg):
        return arg
    # 显式文件扩展名（含未创建）→ 父目录；勿把 path/foo.csv 当目录 makedirs
    ext = os.path.splitext(arg.rstrip(os.sep))[1].lower()
    if ext in {".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".txt"}:
        return os.path.dirname(os.path.abspath(arg)) or "."
    return arg
