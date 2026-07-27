#!/usr/bin/env python3
"""
core/io.py — 统一文件读写与路径小助手

提供跨 CSV/Parquet 的 DuckDB reader，以及 stem / -o 目录解析。
"""

from __future__ import annotations

import os
import re

_STEM_SUFFIX_RE = re.compile(
    r"_(?:raw|quality|quality_drop|clean|clean_drop|sample|keep|keep_high|keep_medium|"
    r"drop|thumb_qc|textqc|resolution)(?:_\d{4,8})?$",
    re.IGNORECASE,
)


def duckdb_reader(path: str) -> str:
    """根据文件扩展名返回 DuckDB reader 表达式。

    用于 COPY / CREATE TABLE AS SELECT * FROM {reader} 等场景。

    >>> duckdb_reader("data.parquet")
    "read_parquet('data.parquet')"
    >>> duckdb_reader("data.csv")
    "read_csv_auto('data.csv', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)"
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return (
            f"read_csv_auto('{path}', header=true, all_varchar=true, "
            f"sample_size=-1, ignore_errors=true)"
        )
    return f"read_parquet('{path}')"


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
    """解析 -o：空 → 输入同目录；已有文件 → 取其 dirname；否则当作目录。"""
    if not arg:
        return os.path.dirname(os.path.abspath(input_path)) or "."
    if os.path.isfile(arg):
        return os.path.dirname(os.path.abspath(arg)) or "."
    return arg
