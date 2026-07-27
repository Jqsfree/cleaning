#!/usr/bin/env python3
"""
core/run_manifest.py — 批次索引（快速定位输入/阶段/交付路径）

落在批次根目录 manifest.json：
  data/runs/{category}/{source}_{batch}/manifest.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


MANIFEST_NAME = "manifest.json"


def manifest_path(batch_root: str | Path) -> Path:
    return Path(batch_root) / MANIFEST_NAME


def load_manifest(batch_root: str | Path) -> dict[str, Any]:
    path = manifest_path(batch_root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(batch_root: str | Path, data: dict[str, Any]) -> Path:
    root = Path(batch_root)
    root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(root)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def init_manifest(
    batch_root: str | Path,
    *,
    category: str,
    source: str,
    batch: str,
    input_path: str = "",
    notes: str = "",
) -> Path:
    """创建或覆盖批次清单骨架。"""
    source = source.strip().lower()
    if source not in ("human", "machine"):
        raise ValueError("source 必须是 human 或 machine")
    data: dict[str, Any] = {
        "category": category,
        "source": source,
        "batch": batch,
        "input": input_path,
        "notes": notes,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
        "deliver_path": "",
    }
    return save_manifest(batch_root, data)


def update_stage(
    batch_root: str | Path,
    stage: str,
    *,
    paths: dict[str, str] | None = None,
    stats: dict[str, Any] | None = None,
    deliver_path: str | None = None,
) -> Path:
    """追加/更新某一 stage 记录。"""
    data = load_manifest(batch_root)
    if not data:
        raise FileNotFoundError(
            f"无 manifest: {manifest_path(batch_root)}；请先 run_manifest.py init"
        )
    stages = data.setdefault("stages", {})
    entry: dict[str, Any] = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if paths:
        entry["paths"] = paths
    if stats:
        entry["stats"] = stats
    stages[stage] = entry
    if deliver_path:
        data["deliver_path"] = deliver_path
    return save_manifest(batch_root, data)


def format_summary(data: dict[str, Any]) -> str:
    if not data:
        return "(empty manifest)"
    lines = [
        f"category={data.get('category')}  source={data.get('source')}  batch={data.get('batch')}",
        f"input={data.get('input')}",
        f"deliver={data.get('deliver_path') or '(未登记)'}",
        f"updated_at={data.get('updated_at')}",
        "stages:",
    ]
    for name, meta in (data.get("stages") or {}).items():
        paths = meta.get("paths") or {}
        lines.append(f"  - {name}: {paths or meta}")
    return "\n".join(lines)


def format_paths_only(data: dict[str, Any]) -> str:
    """仅列出各 stage 的文件路径。"""
    if not data:
        return "(empty manifest)"
    lines = [
        f"category={data.get('category')} source={data.get('source')} batch={data.get('batch')}",
    ]
    if data.get("deliver_path"):
        lines.append(f"deliver={data['deliver_path']}")
    for name, meta in (data.get("stages") or {}).items():
        paths = meta.get("paths") or {}
        if not paths:
            continue
        for k, v in paths.items():
            lines.append(f"{name}.{k}={v}")
    return "\n".join(lines)


def iter_manifests(runs_root: str | Path) -> list[dict[str, Any]]:
    """
    扫描 runs_root 下 */*/manifest.json（约定 data/runs/{category}/{source}_{batch}/）。
    返回带 _batch_root 的 dict 列表。
    """
    root = Path(runs_root)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not data:
            continue
        data = dict(data)
        data["_batch_root"] = str(path.parent)
        data["_manifest"] = str(path)
        rows.append(data)
    return rows


def find_deliver_paths(
    runs_root: str | Path,
    *,
    category: str,
    batch: str | None = None,
    source: str | None = None,
) -> list[dict[str, str]]:
    """
    按品类（及可选 batch/source）查找交付路径。
    优先 manifest.deliver_path；否则 glob 批次 07_deliver/*。
    """
    root = Path(runs_root)
    cat_dir = root / category
    results: list[dict[str, str]] = []
    if not cat_dir.is_dir():
        return results

    for path in sorted(cat_dir.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        batch_root = path.parent
        src = (data.get("source") or "").lower()
        bat = str(data.get("batch") or "")
        # 目录名兜底：human_0724
        dirname = batch_root.name
        if not src and "_" in dirname:
            src = dirname.split("_", 1)[0].lower()
        if not bat and "_" in dirname:
            bat = dirname.split("_", 1)[1]

        if source and src != source.lower():
            continue
        if batch and bat != batch and batch not in dirname:
            continue

        deliver = (data.get("deliver_path") or "").strip()
        paths: list[str] = []
        if deliver:
            paths.append(deliver)
        else:
            ddir = batch_root / "07_deliver"
            if ddir.is_dir():
                paths.extend(
                    str(p) for p in sorted(ddir.iterdir())
                    if p.is_file() and not p.name.startswith(".")
                )

        for dp in paths:
            results.append({
                "category": category,
                "source": src,
                "batch": bat,
                "batch_root": str(batch_root),
                "deliver_path": dp,
            })
    return results


def format_list_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no manifests)"
    header = f"{'category':<16} {'source':<8} {'batch':<12} {'updated':<20} deliver"
    lines = [header, "-" * len(header)]
    for r in rows:
        deliver = r.get("deliver_path") or ""
        if len(deliver) > 48:
            deliver = "…" + deliver[-47:]
        lines.append(
            f"{str(r.get('category') or ''):<16} "
            f"{str(r.get('source') or ''):<8} "
            f"{str(r.get('batch') or ''):<12} "
            f"{str(r.get('updated_at') or ''):<20} "
            f"{deliver or '(未登记)'}"
        )
    return "\n".join(lines)
