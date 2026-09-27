#!/usr/bin/env python3
"""
core/provenance.py — 输入/规则/工具版本指纹（写入 manifest stage）
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path, *, limit: int = 32 * 1024 * 1024) -> str | None:
    """文件 sha256；过大只哈希前 limit 字节 + 文件大小后缀。"""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    size = p.stat().st_size
    with p.open("rb") as f:
        remaining = limit
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    digest = h.hexdigest()
    if size > limit:
        return f"{digest}:size={size}:partial={limit}"
    return digest


def dir_sha256(path: str | Path, *, patterns: tuple[str, ...] = ("*.toml",)) -> str | None:
    """目录下匹配文件内容的稳定哈希（按相对路径排序）。"""
    root = Path(path)
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    files: list[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    for fp in sorted(set(files), key=lambda x: str(x.relative_to(root))):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(root)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(fp.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def tool_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    try:
        import duckdb
        out["duckdb"] = getattr(duckdb, "__version__", "?")
    except Exception:
        pass
    try:
        import pandas as pd
        out["pandas"] = getattr(pd, "__version__", "?")
    except Exception:
        pass
    return out


def build_provenance(
    *,
    input_path: str | Path | None = None,
    rules_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """供 update_stage(..., provenance=...)。"""
    prov: dict[str, Any] = {"tools": tool_versions()}
    if input_path:
        p = Path(input_path)
        prov["input"] = str(p)
        digest = file_sha256(p)
        if digest:
            prov["input_sha256"] = digest
    if rules_dir:
        rd = Path(rules_dir)
        prov["rules_dir"] = str(rd)
        digest = dir_sha256(rd)
        if digest:
            prov["rules_sha256"] = digest
    if extra:
        prov.update(extra)
    return prov
