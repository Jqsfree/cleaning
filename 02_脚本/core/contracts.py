#!/usr/bin/env python3
"""
core/contracts.py — 层边界数据契约（轻量，无 Great Expectations 依赖）

在 quality / clean / deliver 出口校验 schema 与行数；硬失败可阻断。
品类可覆盖：categories/<name>/contracts.toml；默认 _shared/contracts_default.toml。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import duckdb

from core.io import duckdb_reader

_CATEGORIES = Path(__file__).resolve().parent.parent / "categories"
_DEFAULT = _CATEGORIES / "_shared" / "contracts_default.toml"


def load_contracts(category: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if _DEFAULT.is_file():
        data = tomllib.loads(_DEFAULT.read_text(encoding="utf-8"))
    if category:
        path = _CATEGORIES / category / "contracts.toml"
        if path.is_file():
            overlay = tomllib.loads(path.read_text(encoding="utf-8"))
            for k, v in overlay.items():
                if isinstance(v, dict) and isinstance(data.get(k), dict):
                    merged = dict(data[k])
                    merged.update(v)
                    data[k] = merged
                else:
                    data[k] = v
    return data


def _layer_cfg(contracts: dict[str, Any], layer: str) -> dict[str, Any]:
    return dict(contracts.get(layer) or contracts.get("default") or {})


def validate_table(
    path: str | Path,
    *,
    layer: str,
    category: str | None = None,
    upstream_rows: int | None = None,
) -> list[str]:
    """
    返回问题列表（空=通过）。
    hard 问题以 'HARD:' 前缀；其余为 WARN。
    """
    path = Path(path)
    issues: list[str] = []
    if not path.is_file():
        return [f"HARD: 文件不存在: {path}"]

    cfg = _layer_cfg(load_contracts(category), layer)
    required = list(cfg.get("required_columns") or ["video_id", "title"])
    min_rows = int(cfg.get("min_rows", 1))
    max_empty_id = float(cfg.get("max_empty_video_id_rate", 0.0))
    min_retention = cfg.get("min_retention_vs_upstream")
    hard_on_empty = bool(cfg.get("hard_on_fail", True))

    con = duckdb.connect()
    try:
        reader = duckdb_reader(str(path))
        n = con.execute(f"SELECT COUNT(*) FROM {reader}").fetchone()[0]
        cols = {c[0] for c in con.execute(f"SELECT * FROM {reader} LIMIT 0").description}
        missing = [c for c in required if c not in cols]
        if missing:
            msg = f"缺少必填列 {missing}"
            issues.append(f"HARD: {msg}" if hard_on_empty else f"WARN: {msg}")

        if n < min_rows:
            msg = f"行数 {n} < min_rows {min_rows}"
            issues.append(f"HARD: {msg}" if hard_on_empty else f"WARN: {msg}")

        if "video_id" in cols and n > 0:
            empty = con.execute(
                f"SELECT COUNT(*) FROM {reader} "
                f"WHERE video_id IS NULL OR CAST(video_id AS VARCHAR) = ''"
            ).fetchone()[0]
            rate = empty / n
            if rate > max_empty_id:
                msg = f"空 video_id 比例 {rate:.2%} > {max_empty_id:.2%}"
                issues.append(f"HARD: {msg}" if hard_on_empty else f"WARN: {msg}")

        if upstream_rows is not None and min_retention is not None and upstream_rows > 0:
            ret = n / upstream_rows
            thr = float(min_retention)
            if ret < thr:
                issues.append(
                    f"WARN: 相对上游留存 {ret:.1%} < {thr:.1%} "
                    f"(n={n}, upstream={upstream_rows})"
                )
    except Exception as e:
        issues.append(f"HARD: 契约校验失败: {e}")
    finally:
        con.close()
    return issues


def assert_contracts(
    path: str | Path,
    *,
    layer: str,
    category: str | None = None,
    upstream_rows: int | None = None,
    soft: bool = False,
) -> list[str]:
    """打印问题；若有 HARD 且 not soft → SystemExit(2)。"""
    issues = validate_table(
        path, layer=layer, category=category, upstream_rows=upstream_rows,
    )
    for msg in issues:
        level = "ERROR" if msg.startswith("HARD:") else "WARN"
        print(f"[CONTRACT {level}] {msg}", flush=True)
    hard = [m for m in issues if m.startswith("HARD:")]
    if hard and not soft:
        raise SystemExit(2)
    return issues
