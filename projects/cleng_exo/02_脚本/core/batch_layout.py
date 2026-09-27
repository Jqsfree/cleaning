#!/usr/bin/env python3
"""
core/batch_layout.py — 批次路径契约（data/runs/{category}/{source}_{batch}/）

独立脚本默认不得落到旧式 001_quality / 002_audit 等根下路径。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SOURCE_BATCH_RE = re.compile(r"^(human|machine)_.+", re.IGNORECASE)


def looks_like_batch_root(path: str | Path) -> bool:
    """目录名是否形如 human_0724 / machine_0727。"""
    name = Path(path).name
    return bool(_SOURCE_BATCH_RE.match(name))


def infer_batch_root(path: str | Path) -> Path | None:
    """
    从阶段目录或文件向上推断批次根。

    识别 …/{source}_{batch}/01_quality|02_sample|03_qc|04_rules|05_clean|06_tools|07_deliver/…
    """
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    stage_names = {
        "01_quality", "02_sample", "03_qc", "04_rules",
        "05_clean", "06_tools", "07_deliver",
    }
    cur = p
    for _ in range(6):
        if looks_like_batch_root(cur):
            return cur
        if cur.name in stage_names and looks_like_batch_root(cur.parent):
            return cur.parent
        # 05_clean/run01 → 上两级
        if cur.parent.name == "05_clean" and looks_like_batch_root(cur.parent.parent):
            return cur.parent.parent
        if cur == cur.parent:
            break
        cur = cur.parent
    return None


def require_output_dir(arg: str | None, *, flag: str = "-o") -> str:
    """要求显式输出目录；禁止依赖旧默认路径。"""
    if not arg or not str(arg).strip():
        raise SystemExit(
            f"[ERROR] 必须指定 {flag} 输出目录"
            f"（…/data/runs/{{category}}/{{source}}_{{batch}}/…）"
        )
    return str(arg).rstrip("/")


def warn_outside_batch(path: str | Path, log_fn=None) -> None:
    """若不在可推断的批次根下，打 WARN（不阻断，兼容临时目录/测试）。"""
    root = infer_batch_root(path)
    if root is not None:
        return
    msg = (
        f"输出路径不在约定批次根 data/runs/{{cat}}/{{source}}_{{batch}}/ 下: {path}"
    )
    if log_fn:
        log_fn(msg, level="WARN")
    else:
        print(f"[WARN] {msg}", flush=True)


def checklist_for_source(source: str) -> list[dict[str, str]]:
    """
    双路径检查项（薄编排配套 checklist，非自动跑）。

    status 由 evaluate_checklist 填：ok / missing / optional_missing / skip
    """
    source = source.strip().lower()
    common = [
        {"id": "quality", "path": "01_quality", "kind": "dir_nonempty",
         "required": "yes", "hint": "pipeline/run.py 或 01_quality.py"},
        {"id": "sample", "path": "02_sample", "kind": "dir_nonempty",
         "required": "yes", "hint": "03_sample.py → 02_sample/"},
        {"id": "qc", "path": "03_qc", "kind": "dir_nonempty",
         "required": "yes", "hint": "人工结果表或 qc/text.py"},
        {"id": "rules", "path": "04_rules", "kind": "dir_any",
         "required": "machine_only", "hint": "机采须 NOTES.md 等规则依据"},
        {"id": "clean", "path": "05_clean", "kind": "dir_any",
         "required": "optional", "hint": "仅需清洗时；人工合格可跳过"},
        {"id": "deliver", "path": "07_deliver", "kind": "dir_nonempty",
         "required": "optional", "hint": "batch_deliver_ge720 等"},
        {"id": "manifest", "path": "manifest.json", "kind": "file",
         "required": "yes", "hint": "run_manifest.py init / run.py"},
    ]
    # required 字段按 source 解释
    out = []
    for item in common:
        req = item["required"]
        if req == "machine_only":
            item = {**item, "required": "yes" if source == "machine" else "optional"}
        out.append(item)
    return out


def evaluate_checklist(batch_root: str | Path, source: str) -> list[dict[str, str]]:
    """对批次根跑 checklist，返回带 status 的项。"""
    root = Path(batch_root)
    rows = []
    for item in checklist_for_source(source):
        target = root / item["path"]
        kind = item["kind"]
        ok = False
        if kind == "file":
            ok = target.is_file()
        elif kind == "dir_any":
            ok = target.is_dir()
        elif kind == "dir_nonempty":
            ok = target.is_dir() and any(target.iterdir())
        required = item["required"] == "yes"
        if ok:
            status = "ok"
        elif required:
            status = "missing"
        else:
            status = "optional_missing"
        rows.append({**item, "status": status, "resolved": str(target)})
    return rows
